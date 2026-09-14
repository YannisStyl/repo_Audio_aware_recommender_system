import os
# --- 1. WINDOWS CRASH PREVENTION ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"

import datasets
from datasets import Dataset
datasets.disable_caching()

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, set_seed
import random
from trl import GRPOTrainer, GRPOConfig
import re
import pandas as pd
import pickle
import numpy as np
import sys
import collections
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, TaskType
import wandb
from dotenv import load_dotenv
import datetime
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from Data_setup.generate_reward_models import NormalizedPreferenceDensity
from Training_Scripts.parse_utils import parse_completion_to_coord
from Training_Scripts.prompts import SYSTEM_PROMPT_DUAL_AUDIO, SYSTEM_PROMPT_NO_AUDIO


# =====================================================================
# 1. THE Q-FORMER / CROSS-ATTENTION ADAPTER MODULE
# =====================================================================
class QFormerAdapter(nn.Module):
    """
    Generic Q-Former cross-attention adapter.  Used for both the FX stream
    (enc_dim=2048, from Fx-Encoder++ conv6) and the Voice stream
    (enc_dim=768, from Auden-Voice Zipformer last_hidden_state).

    The enc_dim parameter is the only architectural difference between the
    two adapter instances; everything else (num_queries, num_heads, llm_dim)
    is identical, so no code duplication is needed.
    """
    def __init__(self, enc_dim=2048, llm_dim=1024, num_queries=16, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.llm_dim = llm_dim

        # Linear projection layer W (enc_dim -> llm_dim)
        self.proj = nn.Linear(enc_dim, llm_dim)

        # 16 Learnable Query Embeddings (initialized with standard dev 0.02)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, llm_dim) * 0.02)

        # Multi-Head Cross Attention Block.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln_attn = nn.LayerNorm(llm_dim)

        # Feed-Forward Network (FFN).
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim * 4, llm_dim),
            nn.Dropout(dropout),
        )
        self.ln_ffn = nn.LayerNorm(llm_dim)

    def forward(self, x_padded, key_padding_mask):
        """
        x_padded:         [Batch, T_max, enc_dim]
        key_padding_mask: [Batch, T_max]  (True = padding, to be ignored)
        Returns:          [Batch, num_queries, llm_dim]
        """
        batch_size = x_padded.size(0)

        # 1. Project encoder features to LLM embedding dimension
        proj_x = self.proj(x_padded)  # [Batch, T_max, llm_dim]

        # 2. Expand learnable queries to batch
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [Batch, 16, llm_dim]

        # 3. Cross-Attention: queries attend to projected encoder frames
        attn_out, _ = self.cross_attn(
            query=queries,
            key=proj_x,
            value=proj_x,
            key_padding_mask=key_padding_mask
        )

        # 4. Residual + LayerNorm
        x = self.ln_attn(queries + attn_out)

        # 5. FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        out = self.ln_ffn(x + ffn_out)

        return out  # [Batch, 16, llm_dim]


# =====================================================================
# 2. TEXT MASK SCHEDULE (MODALITY DROPOUT)
# =====================================================================
class TextMaskSchedule:
    """
    Controls the probability of zeroing user-turn text embeddings during
    training forward passes (modality dropout / text masking).

    Forcing the model to predict without text embeddings creates gradient
    pressure on both Q-Formers to extract a signal from their respective
    audio features alone, breaking the text-shortcut that causes
    train/eval generalisation gaps when each training prompt maps to
    exactly one audio file.

    The probability decays on a cosine curve from `max_prob` at step 0 to
    `min_prob` at `max_steps`, mirroring the ScheduleState pattern for
    diversity reward decay.

    Parameters
    ----------
    max_steps : int
        Total training steps.
    max_prob : float
        Masking probability at step 0. Set to 0.0 to disable entirely.
    min_prob : float
        Floor probability at max_steps.
    """
    def __init__(self, max_steps: int = 1000, max_prob: float = 0.35, min_prob: float = 0.0):
        self.global_step: int = 0
        self._max_steps = max_steps
        self._max_prob = max_prob
        self._min_prob = min_prob

    def get_prob(self) -> float:
        """Current masking probability given the cosine schedule and global_step."""
        progress = min(self.global_step / max(self._max_steps, 1), 1.0)
        return self._min_prob + 0.5 * (self._max_prob - self._min_prob) * (
            1.0 + math.cos(math.pi * progress)
        )


# =====================================================================
# 3. THE DUAL-STREAM MULTIMODAL WRAPPER (GRPO COMPATIBLE)
# =====================================================================
class DualStreamConditionedQwen(nn.Module):
    """
    Dual-stream audio-conditioned wrapper around a PEFT-wrapped Qwen model.

    Two independent QFormerAdapter instances bridge two audio encoders into
    Qwen's embedding space:

      FX stream  (music/effects):
        Fx-Encoder++ conv6 features  [T_fx,  2048]  →  fx_adapter  → 16 embeddings
        Injected at <|audio|> placeholder tokens (FX_TOKEN_ID)

      Voice stream (paralinguistic):
        Auden-Voice Zipformer states  [T_voice, 768] → voice_adapter → 16 embeddings
        Injected at <|voice|> placeholder tokens (VOICE_TOKEN_ID)

    Together, 32 audio tokens are injected per prompt, giving the LLM a
    rich, temporally-resolved dual-modal acoustic context for EQ prediction.

    Neither encoder is loaded during training — features are pre-extracted
    and cached in prompt_to_fx / prompt_to_voice dicts.

    NOTE ON JOIN KEYS: prompt_to_fx / prompt_to_voice are still keyed by
    prompt text (see _recover_features). fx_ids / voice_ids are optional
    companion dicts (prompt text -> stable sample_id, e.g.
    "TrackName.wav::w3") carried through purely for provenance/debugging
    and for external tooling (causal audio-swap grounding test, perceptual-
    study stimulus generation) — they are not consulted during the forward
    pass itself, since threading an ID through TRL's GRPO batch construction
    into model.forward() isn't something this wrapper can rely on without
    verifying TRL's internal batching behavior first.
    """
    def __init__(
        self,
        qwen_model,
        use_audio: bool = True,
        tokenizer=None,
        fx_token_id: int | None = None,
        voice_token_id: int | None = None,
        prompt_to_fx: dict | None = None,
        prompt_to_voice: dict | None = None,
        fx_ids: dict | None = None,
        voice_ids: dict | None = None,
        text_mask_schedule: TextMaskSchedule | None = None,
        val_prompts=None,
    ):
        super().__init__()
        self.qwen = qwen_model
        self.config = qwen_model.config
        self.use_audio = use_audio

        # Cache validation prompts for debug output
        self.val_prompts = set(val_prompts) if val_prompts is not None else set()

        # Tokenizer and token IDs stored as instance state (not globals) so the
        # class is importable / usable outside this exact script.
        self.tokenizer = tokenizer
        self.fx_token_id = fx_token_id
        self.voice_token_id = voice_token_id
        self.prompt_to_fx = prompt_to_fx if prompt_to_fx is not None else {}
        self.prompt_to_voice = prompt_to_voice if prompt_to_voice is not None else {}

        # Provenance-only companion dicts — not used for lookup during
        # training, see class docstring.
        self.fx_ids = fx_ids if fx_ids is not None else {}
        self.voice_ids = voice_ids if voice_ids is not None else {}

        self.text_mask_schedule = text_mask_schedule

        if use_audio and (
            tokenizer is None
            or fx_token_id is None
            or voice_token_id is None
        ):
            raise ValueError(
                "DualStreamConditionedQwen(use_audio=True) requires "
                "`tokenizer`, `fx_token_id`, and `voice_token_id` to be passed explicitly."
            )

        # Cache chat-template delimiter IDs once at construction.
        self._im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>') if tokenizer else None
        self._im_end_id   = tokenizer.convert_tokens_to_ids('<|im_end|>')   if tokenizer else None

        hidden_dim = self.config.hidden_size
        device     = self.qwen.device

        # FX Q-Former: Fx-Encoder++ conv6 output dim = 2048
        self.fx_adapter = QFormerAdapter(
            enc_dim=2048,
            llm_dim=hidden_dim,
            num_queries=16,
            num_heads=8,
        ).to(device, dtype=torch.bfloat16)
        for p in self.fx_adapter.parameters():
            p.requires_grad = True

        # Voice Q-Former: Auden-Voice Zipformer hidden dim = 768
        self.voice_adapter = QFormerAdapter(
            enc_dim=768,
            llm_dim=hidden_dim,
            num_queries=16,
            num_heads=8,
        ).to(device, dtype=torch.bfloat16)
        for p in self.voice_adapter.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Attribute delegation to the inner Qwen model (PEFT compatibility)
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        # Delegate unknown attributes to the inner Qwen model.
        # Uses super().__getattr__ first (catches nn.Module submodules/parameters),
        # then falls back to getattr(self.qwen, ...).
        #
        # IMPORTANT: nn.Module stores registered submodules in self._modules
        # (set via object.__setattr__ during nn.Module.__init__), NOT in
        # self.__dict__ directly. object.__getattribute__(self, "qwen") would
        # therefore always raise AttributeError even when self.qwen is properly
        # registered. We guard by checking self._modules instead, which IS in
        # __dict__ and is safe to access with object.__getattribute__.
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        try:
            modules = object.__getattribute__(self, "_modules")
        except AttributeError:
            # _modules itself not yet set — nn.Module.__init__ hasn't run
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"(nn.Module not yet initialized)"
            )
        if "qwen" not in modules:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"(self.qwen is not yet registered)"
            )
        return getattr(modules["qwen"], name)

    # ------------------------------------------------------------------
    # Feature recovery
    # ------------------------------------------------------------------
    def _recover_features(self, input_ids):
        """
        Decode input_ids, match each sequence to its pre-extracted FX and
        Voice feature tensors, then batch-pad and generate key padding masks
        for both streams independently.

        Lookup strategy: exact match only, tried in order —
          1. regex-extracted user-turn body from the full decoded text
          2. text-tail split on "user\\n" (handles cases where the regex's
             closing-delimiter assumption doesn't match, e.g. truncation)
        Both are deterministic reconstructions of the same underlying prompt
        text used as the pickle's key at extraction time, so an exact match
        on either is trustworthy. A prior version of this method also
        included a substring-scan fallback ("does any known prompt appear
        as a substring of the decoded text") — that has been removed: it
        could silently match the wrong prompt whenever one prompt's text is
        a substring of another's (a real risk given how similar paraphrases
        are within this dataset), attaching the wrong track's audio
        features to a training example with no error raised. Failing loudly
        via the RuntimeError below is preferable to a silent wrong match.

        Returns
        -------
        padded_fx    : [B, T_fx_max,    2048], bfloat16
        fx_mask      : [B, T_fx_max],    bool  (True = padding)
        padded_voice : [B, T_voice_max,  768],  bfloat16
        voice_mask   : [B, T_voice_max], bool  (True = padding)
        """
        if not self.use_audio:
            B = input_ids.size(0)
            dev = input_ids.device
            return (
                torch.zeros(B, 1, 2048, device=dev, dtype=torch.bfloat16),
                torch.zeros(B, 1, dtype=torch.bool, device=dev),
                torch.zeros(B, 1,  768, device=dev, dtype=torch.bfloat16),
                torch.zeros(B, 1, dtype=torch.bool, device=dev),
            )

        batch_fx    = []
        batch_voice = []
        decoded     = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        for text in decoded:
            found_fx    = None
            found_voice = None

            # Primary lookup: extract the user-turn body via regex
            match = re.search(
                r'<\|im_start\|>user\n(.*?)(?:<\|im_end\|>)?\n<\|im_start\|>assistant',
                text, re.DOTALL
            )
            if match:
                user_request = match.group(1).strip()
                found_fx    = self.prompt_to_fx.get(user_request)
                found_voice = self.prompt_to_voice.get(user_request)

            # Secondary lookup: split on "user\n" tail (exact match only —
            # no substring fallback, see docstring above)
            if found_fx is None or found_voice is None:
                text_tail = text.split("user\n")[-1].strip()
                if found_fx is None:
                    found_fx = self.prompt_to_fx.get(text_tail)
                if found_voice is None:
                    found_voice = self.prompt_to_voice.get(text_tail)

            if found_fx is None:
                raise RuntimeError(
                    "Extraction mismatch: could not match a decoded prompt to any "
                    f"entry in prompt_to_fx. Decoded tail was: {text[-300:]!r}"
                )
            if found_voice is None:
                raise RuntimeError(
                    "Extraction mismatch: could not match a decoded prompt to any "
                    f"entry in prompt_to_voice. Decoded tail was: {text[-300:]!r}"
                )

            batch_fx.append(found_fx)
            batch_voice.append(found_voice)

        B   = len(batch_fx)
        dev = input_ids.device

        # ---- Pad FX features ----
        max_T_fx     = max(t.size(0) for t in batch_fx)
        padded_fx    = torch.zeros(B, max_T_fx, 2048, device=dev, dtype=torch.bfloat16)
        fx_mask      = torch.ones(B, max_T_fx, dtype=torch.bool, device=dev)   # True = ignored
        for i, feat in enumerate(batch_fx):
            T = feat.size(0)
            padded_fx[i, :T, :]  = feat.to(dev, dtype=torch.bfloat16)
            fx_mask[i,   :T]     = False   # Unmask valid frames

        # ---- Pad Voice features ----
        max_T_voice  = max(t.size(0) for t in batch_voice)
        padded_voice = torch.zeros(B, max_T_voice, 768, device=dev, dtype=torch.bfloat16)
        voice_mask   = torch.ones(B, max_T_voice, dtype=torch.bool, device=dev)
        for i, feat in enumerate(batch_voice):
            T = feat.size(0)
            padded_voice[i, :T, :] = feat.to(dev, dtype=torch.bfloat16)
            voice_mask[i,   :T]    = False

        return padded_fx, fx_mask, padded_voice, voice_mask

    # ------------------------------------------------------------------
    # Text modality dropout helper
    # ------------------------------------------------------------------
    def _get_user_turn_mask(self, token_ids_1d: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask of shape [T] that is True for every token
        inside the user-turn body (between the second <|im_start|> and its
        closing <|im_end|>), with BOTH audio placeholder token types forced
        back to False so they are never zeroed out.

        System-prompt tokens, delimiter tokens, and the assistant prefix are
        always False (never masked).
        """
        mask = torch.zeros_like(token_ids_1d, dtype=torch.bool)

        if self._im_start_id is None or self._im_end_id is None:
            return mask

        im_start_positions = (token_ids_1d == self._im_start_id).nonzero(as_tuple=True)[0]
        if len(im_start_positions) < 2:
            return mask     # Malformed sequence — skip silently

        user_start = im_start_positions[1].item()

        im_end_positions = (token_ids_1d == self._im_end_id).nonzero(as_tuple=True)[0]
        candidates = im_end_positions[im_end_positions > user_start]
        if len(candidates) == 0:
            return mask

        user_end = candidates[0].item()
        mask[user_start + 1 : user_end] = True

        # Audio placeholder tokens must NOT be zeroed — their embeddings are
        # overwritten by the adapters immediately after this masking step.
        if self.fx_token_id is not None:
            fx_pos = (token_ids_1d == self.fx_token_id).nonzero(as_tuple=True)[0]
            mask[fx_pos] = False
        if self.voice_token_id is not None:
            voice_pos = (token_ids_1d == self.voice_token_id).nonzero(as_tuple=True)[0]
            mask[voice_pos] = False

        return mask

    # ------------------------------------------------------------------
    # Core embedding construction
    # ------------------------------------------------------------------
    def get_inputs_embeds(
        self,
        input_ids,
        fx_features=None,
        fx_mask=None,
        voice_features=None,
        voice_mask=None,
    ):
        if any(x is None for x in (fx_features, fx_mask, voice_features, voice_mask)):
            fx_features, fx_mask, voice_features, voice_mask = self._recover_features(input_ids)

        inputs_embeds = self.qwen.get_input_embeddings()(input_ids).clone()

        # ---- TEXT MODALITY DROPOUT ----
        # Single coin flip per forward call so all items in a GRPO rollout
        # group see the same conditioning (consistent group-relative advantage).
        if self.training and self.text_mask_schedule is not None:
            mask_prob = self.text_mask_schedule.get_prob()
            if mask_prob > 0.0 and torch.rand(1).item() < mask_prob:
                for batch_idx in range(input_ids.size(0)):
                    user_mask = self._get_user_turn_mask(input_ids[batch_idx])
                    if user_mask.any():
                        inputs_embeds[batch_idx, user_mask] = 0.0

        if self.use_audio:
            # ---- FX stream injection ----
            projected_fx = self.fx_adapter(fx_features, fx_mask)          # [B, 16, D]
            projected_fx = projected_fx.to(inputs_embeds.dtype)

            # ---- Voice stream injection ----
            projected_voice = self.voice_adapter(voice_features, voice_mask)  # [B, 16, D]
            projected_voice = projected_voice.to(inputs_embeds.dtype)

            for batch_idx in range(input_ids.size(0)):
                # Inject FX embeddings into <|audio|> placeholder positions
                fx_indices = (input_ids[batch_idx] == self.fx_token_id).nonzero(as_tuple=True)[0]
                if len(fx_indices) == self.fx_adapter.num_queries:
                    inputs_embeds[batch_idx, fx_indices] = projected_fx[batch_idx]
                else:
                    raise ValueError(
                        f"Expected {self.fx_adapter.num_queries} FX placeholder tokens "
                        f"(<|audio|>), found {len(fx_indices)} at batch index {batch_idx}."
                    )

                # Inject Voice embeddings into <|voice|> placeholder positions
                voice_indices = (input_ids[batch_idx] == self.voice_token_id).nonzero(as_tuple=True)[0]
                if len(voice_indices) == self.voice_adapter.num_queries:
                    inputs_embeds[batch_idx, voice_indices] = projected_voice[batch_idx]
                else:
                    raise ValueError(
                        f"Expected {self.voice_adapter.num_queries} Voice placeholder tokens "
                        f"(<|voice|>), found {len(voice_indices)} at batch index {batch_idx}."
                    )

        return inputs_embeds

    # ------------------------------------------------------------------
    # Standard nn.Module interface — forward / generate
    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Pop dual-stream tensors if a caller supplies them directly,
        # otherwise _recover_features() will decode-and-match from input_ids.
        fx_features    = kwargs.pop("fx_features",    None)
        fx_mask        = kwargs.pop("fx_mask",         None)
        voice_features = kwargs.pop("voice_features", None)
        voice_mask     = kwargs.pop("voice_mask",     None)

        inputs_embeds = self.get_inputs_embeds(
            input_ids, fx_features, fx_mask, voice_features, voice_mask
        )
        return self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(self, input_ids=None, **kwargs):
        # Drop any audio tensors that a caller may have passed — generate()
        # always runs _recover_features() via get_inputs_embeds.
        for key in ("fx_features", "fx_mask", "voice_features", "voice_mask"):
            kwargs.pop(key, None)

        if input_ids is not None:
            inputs_embeds = self.get_inputs_embeds(input_ids)
            generated_output = self.qwen.generate(inputs_embeds=inputs_embeds, **kwargs)

            if isinstance(generated_output, torch.Tensor):
                return torch.cat([input_ids, generated_output], dim=1)
            else:
                generated_output.sequences = torch.cat(
                    [input_ids, generated_output.sequences], dim=1
                )
                return generated_output

        return self.qwen.generate(input_ids=input_ids, **kwargs)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):
        sd = get_peft_model_state_dict(self.qwen)

        # Save FX adapter under its own prefix
        for k, v in self.fx_adapter.state_dict(*args, **kwargs).items():
            sd[f"fx_projector.{k}"] = v

        # Save Voice adapter under its own prefix
        for k, v in self.voice_adapter.state_dict(*args, **kwargs).items():
            sd[f"voice_projector.{k}"] = v

        # Deduplicate tied weights (lm_head == embed_tokens in most Qwen configs)
        keys_to_remove = [
            k for k in sd
            if "lm_head.weight" in k
            and sd.get(k.replace("lm_head.weight", "model.embed_tokens.weight")) is not None
            and sd[k].data_ptr() == sd[k.replace("lm_head.weight", "model.embed_tokens.weight")].data_ptr()
        ]
        for k in keys_to_remove:
            del sd[k]

        return sd

    def save_pretrained(self, save_directory, **kwargs):
        if hasattr(self.qwen, "save_pretrained"):
            self.qwen.save_pretrained(save_directory, **kwargs)
        torch.save(
            self.fx_adapter.state_dict(),
            os.path.join(save_directory, "fx_projector.bin")
        )
        torch.save(
            self.voice_adapter.state_dict(),
            os.path.join(save_directory, "voice_projector.bin")
        )


# =====================================================================
# 4. REWARD FUNCTIONS & CALLBACKS
# =====================================================================
def make_grpo_prediction_reward(all_reward_models, val_prompts=None):
    """
    Factory that closes over `all_reward_models`. Identical to the single-stream
    version — reward computation is purely text-based and stream-agnostic.
    """
    val_prompts_set = set(val_prompts) if val_prompts is not None else set()

    def grpo_prediction_reward(prompts: list[str], completions: list[str], **kwargs) -> list:
        all_rewards = []
        raw_prompts = kwargs["raw_prompt"]

        grouped_completions = {}
        for prompt, completion in zip(raw_prompts, completions):
            if prompt not in grouped_completions:
                grouped_completions[prompt] = []
            grouped_completions[prompt].append(completion)

        unique_prompts_in_order = list(dict.fromkeys(raw_prompts))

        for prompt in unique_prompts_in_order:
            prompt_completions = grouped_completions[prompt]
            parsed_predictions = [parse_completion_to_coord(c) for c in prompt_completions]

            try:
                prompt_density = all_reward_models[prompt]["positive_reward"]
                prompt_rewards = []
                for pred in parsed_predictions:
                    if pred is not None:
                        reward_value = float(prompt_density(pred).item())
                        prompt_rewards.append(reward_value)
                    else:
                        prompt_rewards.append(-1.0)

                all_rewards.extend(prompt_rewards)

            except KeyError:
                print(f"Warning: No reward model found for prompt: {prompt}")
                all_rewards.extend([0.0] * len(prompt_completions))
                prompt_rewards = [0.0] * len(prompt_completions)

            if prompt in val_prompts_set:
                print(f"\n[DEBUG EVAL PREDICTIONS] Validation Prompt: {prompt}", flush=True)
                print(f"[DEBUG EVAL PREDICTIONS] Model Predictions/Completions:", flush=True)
                for idx, (c, p, r) in enumerate(zip(prompt_completions, parsed_predictions, prompt_rewards)):
                    print(f"  Rollout {idx+1}: {c.strip()!r} -> Parsed: {p} | Reward: {r}", flush=True)
                print("=" * 60, flush=True)

        return all_rewards

    return grpo_prediction_reward


# Maximum possible L2 distance between two points in the [-6,6]x[-6,6] space.
_COORD_MAX_DIST = 12.0 * math.sqrt(2)


class ScheduleState:
    """
    Mutable step counter shared between reward functions and callbacks.

    Two schedule types:
      'cosine' — smooth decay from 1.0 → min_scale over max_steps.
      'step'   — pulse defined by step_window=(start, end):
                   [0, start)     → min_scale  (adapter warmup)
                   [start, end)   → 1.0         (diversity exploration)
                   [end, max_steps] → min_scale  (exploitation)
    """
    def __init__(
        self,
        max_steps: int = 1000,
        min_scale: float = 0.0,
        schedule: str = 'cosine',
        step_window: tuple[int, int] | None = None,
    ):
        if schedule not in ('cosine', 'step'):
            raise ValueError(f"schedule must be 'cosine' or 'step', got {schedule!r}")
        if schedule == 'step' and step_window is None:
            raise ValueError("schedule='step' requires step_window=(start, end).")
        if step_window is not None:
            start, end = step_window
            if not (0 <= start < end <= max_steps):
                raise ValueError(
                    f"step_window=({start}, {end}) invalid: "
                    f"require 0 <= start < end <= max_steps ({max_steps})."
                )

        self.global_step: int = 0
        self._max_steps   = max_steps
        self._min_scale   = min_scale
        self._schedule    = schedule
        self._step_window = step_window

    def get_scale(self) -> float:
        if self._schedule == 'step':
            start, end = self._step_window
            return 1.0 if start <= self.global_step < end else self._min_scale
        progress = min(self.global_step / max(self._max_steps, 1), 1.0)
        return self._min_scale + 0.5 * (1.0 - self._min_scale) * (1.0 + math.cos(math.pi * progress))


def make_grpo_diversity_reward(schedule_state: ScheduleState, uniqueness_based: bool = False):
    """
    Factory for the pairwise geometric diversity reward with schedule decay.
    Identical to the single-stream version — completions are text strings.
    """
    def grpo_diversity_reward(prompts: list[str], completions: list[str], **kwargs):
        all_diversity_rewards = []
        raw_prompts = kwargs["raw_prompt"]

        grouped_completions = collections.defaultdict(list)
        for prompt, completion in zip(raw_prompts, completions):
            grouped_completions[prompt].append(completion)

        G = len(completions) // len(grouped_completions)
        scale = schedule_state.get_scale()
        unique_prompts_in_order = list(dict.fromkeys(raw_prompts))

        for prompt in unique_prompts_in_order:
            prompt_completions = grouped_completions[prompt]
            parsed_predictions = [parse_completion_to_coord(c) for c in prompt_completions]

            prompt_rewards = [-1.0 * scale] * G
            valid = [(i, np.array(p)) for i, p in enumerate(parsed_predictions) if p is not None]

            if uniqueness_based:
                valid_rounded = np.round([c for _, c in valid])
                _, idx, counts = np.unique(valid_rounded, axis=0, return_inverse=True, return_counts=True)
                scores = 1 - 2 * (counts[idx] - 1) / (G - 1)
                for idx, (i, _) in enumerate(valid):
                    prompt_rewards[i] = scale * scores[idx]
            else:
                if len(valid) > 1:
                    coords = np.stack([c for _, c in valid])
                    for idx, (i, _) in enumerate(valid):
                        dists = np.linalg.norm(coords - coords[idx], axis=1)
                        mean_dist = dists.sum() / (len(valid) - 1)
                        prompt_rewards[i] = scale * (2.0 * mean_dist / _COORD_MAX_DIST - 1.0)
                elif len(valid) == 1:
                    i, _ = valid[0]
                    prompt_rewards[i] = scale

            all_diversity_rewards.extend(prompt_rewards)

        return all_diversity_rewards

    return grpo_diversity_reward


# =====================================================================
# 4b. CALLBACKS
# =====================================================================
class DualStreamConditioningCallback(TrainerCallback):
    """
    Logs gradient norms and weight norms for BOTH adapters (fx_adapter and
    voice_adapter) to wandb, and keeps both ScheduleState objects in sync
    with the trainer's global_step.
    """
    def __init__(
        self,
        schedule_state: ScheduleState | None = None,
        text_mask_schedule: TextMaskSchedule | None = None,
    ):
        self._grad_norms        = {}
        self._schedule_state    = schedule_state
        self._text_mask_schedule = text_mask_schedule

    def on_step_begin(self, args, state, control, **kwargs):
        if self._schedule_state is not None:
            self._schedule_state.global_step = state.global_step
        if self._text_mask_schedule is not None:
            self._text_mask_schedule.global_step = state.global_step

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        for adapter_name in ("fx_adapter", "voice_adapter"):
            adapter = getattr(model, adapter_name, None)
            if adapter is None:
                continue
            for name, param in adapter.named_parameters():
                if param.requires_grad:
                    self._grad_norms[f"adapter_grad_norm/{adapter_name}/{name}"] = (
                        param.grad.norm().item() if param.grad is not None else 0.0
                    )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        model = kwargs.get("model")
        if model is None:
            return

        logs.update(self._grad_norms)

        for adapter_name in ("fx_adapter", "voice_adapter"):
            adapter = getattr(model, adapter_name, None)
            if adapter is None:
                continue
            for name, param in adapter.named_parameters():
                if param.requires_grad:
                    logs[f"adapter_weight_norm/{adapter_name}/{name}"] = param.norm().item()

        if self._schedule_state is not None:
            logs["diversity/decay_scale"] = self._schedule_state.get_scale()

        if self._text_mask_schedule is not None:
            logs["text_masking/mask_prob"] = self._text_mask_schedule.get_prob()


class AdapterWarmupCallback(TrainerCallback):
    """
    Freezes all LoRA parameters for the first `warmup_steps` optimizer
    updates so the randomly-initialised adapters (both fx_adapter and
    voice_adapter) get a chance to learn a real signal before the LLM's
    LoRA weights start adapting around them (BLIP-2-style staged training).

    Both adapters train from step 0 regardless of this callback — only
    the LoRA parameters are frozen/unfrozen here.
    """
    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps
        self._lora_params = None
        self._unfrozen    = False

    def _collect_lora_params(self, model):
        return [
            (name, param)
            for name, param in model.qwen.named_parameters()
            if "lora_" in name
        ]

    def on_train_begin(self, args, state, control, **kwargs):
        if self.warmup_steps <= 0:
            return
        model = kwargs["model"]
        self._lora_params = self._collect_lora_params(model)
        if not self._lora_params:
            print("AdapterWarmupCallback found no lora_ parameters — check USE_LORA.")
            return
        for _, param in self._lora_params:
            param.requires_grad = False
        print(
            f"Adapter warmup: froze {len(self._lora_params)} LoRA parameters "
            f"for the first {self.warmup_steps} steps."
        )

    def on_step_begin(self, args, state, control, **kwargs):
        if self._unfrozen or not self._lora_params:
            return
        if state.global_step >= self.warmup_steps:
            for _, param in self._lora_params:
                param.requires_grad = True
            self._unfrozen = True
            print(
                f"Adapter warmup complete at step {state.global_step}: "
                f"unfroze {sum(p.numel() for _, p in self._lora_params)} LoRA parameters."
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self._lora_params:
            logs["adapter_warmup/frozen"] = 0.0 if self._unfrozen else 1.0


def create_hf_dataset(reward_models_dict, sys_prompt, tok, split_name="Dataset"):
    print(f"[{split_name}] Formatting {len(reward_models_dict)} prompts...", flush=True)

    raw_prompts       = []
    formatted_prompts = []
    for prompt in reward_models_dict.keys():
        messages = [
            {"role": "system",    "content": sys_prompt},
            {"role": "user",      "content": prompt},
        ]
        formatted_prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        raw_prompts.append(prompt)
        formatted_prompts.append(formatted_prompt)

    print(f"[{split_name}] Wrapping into HF Dataset...", flush=True)
    ds = Dataset.from_dict({"raw_prompt": raw_prompts, "prompt": formatted_prompts})
    print(f"[{split_name}] Done!", flush=True)
    return ds


class CustomGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO Trainer with a dual-phase LR schedule.
    Phase 1: Both Q-Former adapters train; LoRA is frozen.
             LR ramps up and decays, scaled by `adapter_lr_ratio`.
    Phase 2: LoRA unfreezes; full model trains.
             LR resets, ramps up to LEARNING_RATE, then cosine decays.
    """
    def __init__(self, *args, adapter_warmup_steps=500, lora_warmup_steps=100, adapter_lr_ratio=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_warmup_steps = adapter_warmup_steps
        self.lora_warmup_steps    = lora_warmup_steps
        self.adapter_lr_ratio     = adapter_lr_ratio

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            def lr_lambda(current_step):
                adapter_warmup = self.adapter_warmup_steps
                lora_warmup    = self.lora_warmup_steps
                max_steps      = num_training_steps
                ratio          = self.adapter_lr_ratio

                if adapter_warmup == 0:
                    if current_step < lora_warmup:
                        return float(current_step) / float(max(1, lora_warmup))
                    progress = float(current_step - lora_warmup) / float(max(1, max_steps - lora_warmup))
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

                if current_step < adapter_warmup:
                    phase_warmup = min(lora_warmup, max(1, adapter_warmup // 3))
                    if current_step < phase_warmup:
                        return ratio * (float(current_step) / float(max(1, phase_warmup)))
                    progress = float(current_step - phase_warmup) / float(max(1, adapter_warmup - phase_warmup))
                    return ratio * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
                else:
                    phase_step  = current_step - adapter_warmup
                    phase_total = max_steps - adapter_warmup
                    if phase_step < lora_warmup:
                        return float(phase_step) / float(max(1, lora_warmup))
                    progress = float(phase_step - lora_warmup) / float(max(1, phase_total - lora_warmup))
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            opt = self.optimizer if optimizer is None else optimizer
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return self.lr_scheduler


# =====================================================================
# 5. MAIN CONFIGURATION & EXECUTION
# =====================================================================
if __name__ == "__main__":

    # Hyperparameters
    GRPO_BETA_VALUE       = 0.1
    NUM_GEN               = 16
    LEARNING_RATE         = 1e-4
    LR_SCHEDULER          = "cosine"                    # overwritten by CustomGRPOTrainer
    MAX_STEPS             = 13500
    LORA_WARMUP_STEPS     = 1000
    ADAPTER_WARMUP_STEPS  = 3500
    ADAPTER_LR_RATIO      = 10.0
    REWARD_WEIGHTS        = [0.75, 0.25]                # [prediction_reward, diversity_reward]
    TEXT_MASK_MAX_PROB    = 0.0
    TEXT_MASK_MIN_PROB    = 0.0
    DIVERSITY_SCHEDULE    = "step"
    DIVERSITY_STEP_WINDOW = (0, MAX_STEPS)
    DIVERSITY_MIN_SCALE   = 0.0
    TEMPERATURE           = 1.0
    TOP_K                 = 20
    WEIGHT_DECAY          = 0.05
    SIZE                  = "3.5-0.8B"
    USE_LORA              = True
    USE_AUDIO_CONDITIONING= True
    ADAPTER               = "Fx_Voice_Qformer" if USE_AUDIO_CONDITIONING else "No_adapter"

    # Paths
    TRAIN_DATA_PATH = os.path.join("Data_dir", "reward_models", "reward_models_augmented_train.pkl")
    VAL_DATA_PATH   = os.path.join("Data_dir", "reward_models", "reward_models_augmented_val.pkl")
    FX_PATH         = os.path.join("Data_dir", "prompt_to_fx_windowed.pkl")
    VOICE_PATH      = os.path.join("Data_dir", "prompt_to_voice_windowed.pkl")
    dir_name        = (datetime.datetime.now().strftime("%D_%H_%M")).replace("/", "_")
    CHECKPOINTS_SAVE_DIR = (
        f"src/GRPO_models/{ADAPTER}/{dir_name}/"
        f"LR_{LEARNING_RATE}_Qwen{SIZE}-grpo_beta-{GRPO_BETA_VALUE}_numgen-{NUM_GEN}"
    )
    RUN_NAME = f"FX-Voice-{DIVERSITY_SCHEDULE}-beta_{GRPO_BETA_VALUE}-numgen_{NUM_GEN}"

    # W&B setup
    load_dotenv()
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        print("WANDB_API_KEY not found; running with WANDB_MODE=offline.")
        os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "audio-qformer-grpo")
    os.environ["WANDB_MODE"]    = os.environ.get("WANDB_MODE", "online")

    # ---- Load Base LLM ----
    model_name = f"Qwen/Qwen{SIZE}"
    print(f"Loading Base LLM: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation='sdpa',
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Register BOTH special placeholder tokens in one call so only a single
    # resize_token_embeddings() is needed (calling it twice leads to shape mismatches).
    special_tokens_dict = {'additional_special_tokens': ['<|audio|>', '<|voice|>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    base_model.resize_token_embeddings(len(tokenizer))

    FX_TOKEN_ID    = tokenizer.convert_tokens_to_ids('<|audio|>')
    VOICE_TOKEN_ID = tokenizer.convert_tokens_to_ids('<|voice|>')
    print(f"Token IDs — <|audio|>: {FX_TOKEN_ID}  <|voice|>: {VOICE_TOKEN_ID}")

    # ---- LoRA ----
    if USE_LORA:
        print("Injecting LoRA layers...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",     # Standard self-attention
                "out_proj",                                  # DeltaNet output
                "in_proj_qkv",                               # DeltaNet write encoding
                "in_proj_b",                                 # DeltaNet write strength (β)
                "gate_proj", "up_proj", "down_proj",         # MLP (all 24 layers)
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        base_model = get_peft_model(base_model, lora_config)
        base_model.print_trainable_parameters()

    # ---- Load pre-extracted features ----
    # NOTE: as of the windowed extraction pipeline, prompt_to_fx_windowed.pkl
    # and prompt_to_voice_windowed.pkl map prompt text -> a dict with keys
    # "tensor" (the actual feature tensor, used for training) and
    # "sample_id"/"track"/"window_index"/"window_start_sec" (provenance
    # metadata, not used in the forward pass — see companion
    # fx_sample_id_provenance.pkl / voice_sample_id_provenance.pkl for the
    # full reverse-lookup table, useful for the causal audio-swap test and
    # for linking perceptual-study stimuli back to specific windows).
    print("Loading pre-extracted FX features...")
    with open(FX_PATH, 'rb') as f:
        prompt_to_fx_raw = pickle.load(f)
    prompt_to_fx = {k.strip(): v["tensor"] for k, v in prompt_to_fx_raw.items()}
    fx_ids       = {k.strip(): v["sample_id"] for k, v in prompt_to_fx_raw.items()}
    print(f"  Loaded {len(prompt_to_fx)} FX feature entries.")
    for k, v in prompt_to_fx.items():
        if v.size(0) < 1:
            raise ValueError(f"Zero-length FX feature for prompt: {k!r}")

    print("Loading pre-extracted Voice features...")
    with open(VOICE_PATH, 'rb') as f:
        prompt_to_voice_raw = pickle.load(f)
    prompt_to_voice = {k.strip(): v["tensor"] for k, v in prompt_to_voice_raw.items()}
    voice_ids       = {k.strip(): v["sample_id"] for k, v in prompt_to_voice_raw.items()}
    print(f"  Loaded {len(prompt_to_voice)} Voice feature entries.")
    for k, v in prompt_to_voice.items():
        if v.size(0) < 1:
            raise ValueError(f"Zero-length Voice feature for prompt: {k!r}")

    # ---- Verify both dicts have matching key sets ----
    fx_keys    = set(prompt_to_fx.keys())
    voice_keys = set(prompt_to_voice.keys())
    missing_in_voice = fx_keys - voice_keys
    missing_in_fx    = voice_keys - fx_keys
    if missing_in_voice:
        print(
            f"{len(missing_in_voice)} prompts have FX features but NO voice features. "
            "These will raise RuntimeError during training. Run extract_voice_features.py first."
        )
    if missing_in_fx:
        print(
            f"{len(missing_in_fx)} prompts have voice features but NO FX features."
        )

    # ---- Schedules ----
    schedule_state = ScheduleState(
        max_steps=MAX_STEPS,
        min_scale=DIVERSITY_MIN_SCALE,
        schedule=DIVERSITY_SCHEDULE,
        step_window=DIVERSITY_STEP_WINDOW if DIVERSITY_SCHEDULE == 'step' else None,
    )
    text_mask_schedule = TextMaskSchedule(
        max_steps=MAX_STEPS,
        max_prob=TEXT_MASK_MAX_PROB,
        min_prob=TEXT_MASK_MIN_PROB,
    )

    # ---- Load reward models ----
    with open(TRAIN_DATA_PATH, 'rb') as f:
        train_reward_models = pickle.load(f)
    with open(VAL_DATA_PATH, 'rb') as f:
        val_reward_models = pickle.load(f)
    all_reward_models = {**train_reward_models, **val_reward_models}

    # ---- Instantiate dual-stream wrapper ----
    model = DualStreamConditionedQwen(
        base_model,
        use_audio=USE_AUDIO_CONDITIONING,
        tokenizer=tokenizer,
        fx_token_id=FX_TOKEN_ID,
        voice_token_id=VOICE_TOKEN_ID,
        prompt_to_fx=prompt_to_fx,
        prompt_to_voice=prompt_to_voice,
        fx_ids=fx_ids,
        voice_ids=voice_ids,
        text_mask_schedule=text_mask_schedule,
        val_prompts=val_reward_models.keys(),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    system_prompt = SYSTEM_PROMPT_DUAL_AUDIO if USE_AUDIO_CONDITIONING else SYSTEM_PROMPT_NO_AUDIO

    print("Preparing datasets...")
    train_dataset = create_hf_dataset(train_reward_models, system_prompt, tokenizer, "TRAIN")
    eval_dataset  = create_hf_dataset(val_reward_models,   system_prompt, tokenizer, "EVAL")

    # ---- GRPOConfig ----
    print(f"\nStarting Dual-Stream Q-Former GRPO Training\n")
    config = GRPOConfig(
        bf16=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=16,
        num_generations=NUM_GEN,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        warmup_steps=LORA_WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=50,
        beta=GRPO_BETA_VALUE,
        output_dir=CHECKPOINTS_SAVE_DIR,
        ddp_find_unused_parameters=False,
        reward_weights=REWARD_WEIGHTS,
        loss_type="grpo",
        report_to="wandb",
        run_name=RUN_NAME,
        remove_unused_columns=False,
        save_only_model=True,
        use_vllm=False,
    )

    print("--- INITIALIZING GRPO TRAINER ---", flush=True)
    grpo_trainer = CustomGRPOTrainer(
        adapter_warmup_steps=ADAPTER_WARMUP_STEPS,
        lora_warmup_steps=LORA_WARMUP_STEPS,
        adapter_lr_ratio=ADAPTER_LR_RATIO,
        model=model,
        args=config,
        reward_funcs=[
            make_grpo_prediction_reward(all_reward_models, val_prompts=val_reward_models.keys()),
            make_grpo_diversity_reward(schedule_state, uniqueness_based=True),
        ],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[
            DualStreamConditioningCallback(
                schedule_state=schedule_state,
                text_mask_schedule=text_mask_schedule,
            ),
            AdapterWarmupCallback(warmup_steps=ADAPTER_WARMUP_STEPS),
        ],
    )

    print("Executing GRPOTrainer.train()...")
    grpo_trainer.train()

    print("Training finished. Saving model weights...")
    grpo_trainer.save_model(os.path.join(CHECKPOINTS_SAVE_DIR, "final_model"))
    wandb.finish()