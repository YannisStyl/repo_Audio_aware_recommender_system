"""
visualize_attention_hybrid.py

Attention + DeltaNet visualization for AudioFusedConditionedQwen on the
hybrid Gated DeltaNet / Gated Attention architecture.

Hardcoded architecture constants (from known model layout):
  - 24 total layers
  - Standard Gated Attention at layers: 7, 11, 15, 19, 23  (5 layers)
    └─ GQA: 8 Q-heads, 2 KV-heads, head_dim=256
  - Gated DeltaNet at all other layers: 0-6, 8-10, 12-14, 16-18, 20-22  (19 layers)
    └─ 16 heads (QK and V), head_dim=128

Module naming (from model inspection):
  - Standard attention:  layers.{L}.self_attn.[q|k|v|o]_proj
  - DeltaNet:            layers.{L}.linear_attn.in_proj_[qkv|b|a|z], out_proj
  - MLP (all layers):    layers.{L}.mlp.[gate|up|down]_proj

Produces six plots:
  1. Unified layer-wise audio routing   -- attention mass at attention layers,
                                          β write strength at DeltaNet layers
  2. Per-head audio attention            -- at the peak attention layer
  3. 2D self-attention heatmap           -- best audio-routing head, peak layer
  4. Audio token breakdown               -- [user_tokens × N_audio], peak layer
  5. DeltaNet write strength             -- [audio_position × DeltaNet_layer] heatmap
  6. State influence                     -- per-layer L2 diff (real vs zero-audio)
                                          across all sequence positions
"""

import os
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from safetensors.torch import load_file

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from Training_Scripts.t2b_fx_GRPO import AudioFusedConditionedQwen
from Training_Scripts.prompts import SYSTEM_PROMPT_AUDIO


# =====================================================================
# ARCHITECTURE CONSTANTS
# =====================================================================
ATTENTION_LAYER_INDICES = [3, 7, 11, 15, 19, 23]
DELTANET_LAYER_INDICES  = [i for i in range(24) if i not in ATTENTION_LAYER_INDICES]
# Confirmed by probe: [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
DELTANET_NUM_HEADS      = 16   # QK and V heads in DeltaNet
ATTN_Q_HEADS            = 8    # GQA query heads in standard attention
ATTN_KV_HEADS           = 2    # GQA key-value heads in standard attention


# =====================================================================
# CONFIGURATION
# =====================================================================
@dataclass
class VisConfig:
    model_checkpoint: str = (
        "src/GRPO_models/FX_QFormer/"
        "LR_0.0001_Qwen3.5-0.8B-grpo_beta-0.1_numgen-16/checkpoint-2910"

    )
    base_model_name: str    = "Qwen/Qwen3.5-0.8B"
    fx_path: str            = "Data_dir/prompt_to_fx_unpooled.pkl"
    save_dir: str           = "src/Data_setup/Attention_visualizations"
    test_prompt: str | None = None
    lora_r: int             = 8
    lora_alpha: int         = 16
    lora_dropout: float     = 0.05
    # Set to True if the checkpoint contains lora_magnitude_vector keys (DoRA).
    # Mismatch between this and the checkpoint causes silent weight corruption.
    use_dora: bool          = False
    # Run a second forward pass with zeroed audio features to produce a
    # content-free baseline for state influence computation (Plot 6).
    baseline_comparison: bool = True


# =====================================================================
# UTILITIES
# =====================================================================
def find_module(root_module: torch.nn.Module, suffix: str) -> torch.nn.Module | None:
    """
    Finds the first module whose full dotted name ends with `suffix`.
    Robust to PEFT wrapper prefixes (base_model.model.model.…).
    """
    for name, module in root_module.named_modules():
        # Strip any leading PEFT prefix and match from the right.
        trimmed = name.split("base_model.model.model.")[-1]
        if trimmed == suffix or name == suffix:
            return module
    return None


def clean_token_labels(
    tokenizer,
    input_ids: list[int],
    audio_token_id: int,
) -> list[str]:
    labels = []
    audio_counter = 0
    for tid in input_ids:
        if tid == audio_token_id:
            labels.append(f"<|audio_{audio_counter}|>")
            audio_counter += 1
        else:
            tok = tokenizer.decode([tid])
            if not tok:
                tok = tokenizer.convert_ids_to_tokens([tid])[0]
                if isinstance(tok, bytes):
                    tok = tok.decode("utf-8", errors="replace")
            tok = tok.replace("\n", "\\n").replace("\t", "\\t")
            labels.append(tok)
    return labels


def get_audio_indices(input_ids: list[int], audio_token_id: int) -> list[int]:
    indices = [i for i, tid in enumerate(input_ids) if tid == audio_token_id]
    if not indices:
        raise ValueError(
            f"Audio token id {audio_token_id} not found in input_ids. "
            "Ensure the system prompt contains <|audio|> placeholder tokens."
        )
    return indices


def get_user_start_idx(tokenizer, labels: list[str]) -> int:
    """
    Returns the index of the first content token in the user turn.
    Scans forward from the second <|im_start|> past role and whitespace tokens
    rather than assuming a fixed +2 offset.
    """
    im_starts = [i for i, t in enumerate(labels) if "<|im_start|>" in t]
    if len(im_starts) < 2:
        return (im_starts[0] + 2) if im_starts else 0
    i = im_starts[1] + 1
    while i < len(labels) and labels[i].strip() in ("user", "\\n", ""):
        i += 1
    return i


# =====================================================================
# MODEL LOADING
# =====================================================================
def _detect_dora(sd: dict) -> bool:
    """Returns True if any key in sd contains lora_magnitude_vector."""
    return any("lora_magnitude_vector" in k for k in sd)


def load_model_for_inference(
    cfg: VisConfig,
    tokenizer,
    prompt_to_fx: dict,
    device: str,
) -> tuple["AudioFusedConditionedQwen", int]:
    """
    Loads base model, injects PEFT adapters, wraps with AudioFusedConditionedQwen,
    then loads checkpoint weights.

    attn_implementation='eager' is mandatory: SDPA and Flash Attention kernels
    do not return attention weight matrices even with output_attentions=True,
    producing outputs.attentions=None silently.

    DoRA detection: if the checkpoint contains lora_magnitude_vector keys but
    use_dora=False in cfg, weights will be silently dropped. The loader detects
    this and warns explicitly.
    """
    print(f"  Loading base model: {cfg.base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    base_model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=[
            # Standard attention projections (layers 7, 11, 15, 19, 23)
            "q_proj", "k_proj", "v_proj", "o_proj",
            # DeltaNet output projection (layers 0-6, 8-10, etc.)
            "out_proj",
            # DeltaNet write encoding and write strength
            "in_proj_qkv", "in_proj_b",
            # MLP (all layers)
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_dora=cfg.use_dora,
        task_type=TaskType.CAUSAL_LM,
    )
    base_model = get_peft_model(base_model, lora_config)

    audio_token_id = tokenizer.convert_tokens_to_ids("<|audio|>")
    model = AudioFusedConditionedQwen(
        base_model,
        use_audio=True,
        tokenizer=tokenizer,
        audio_token_id=audio_token_id,
        prompt_to_fx=prompt_to_fx,
    ).to(device)

    # --- Load checkpoint ---
    has_adapter_config = os.path.exists(
        os.path.join(cfg.model_checkpoint, "adapter_config.json")
    )
    if has_adapter_config:
        print("  Loading PEFT weights via PeftModel.from_pretrained...")
        model.qwen = PeftModel.from_pretrained(model.qwen, cfg.model_checkpoint).to(device)
        adapter_bin = os.path.join(cfg.model_checkpoint, "audio_projector.bin")
        if os.path.exists(adapter_bin):
            model.audio_adapter.load_state_dict(torch.load(adapter_bin, map_location=device))
            print("  audio_projector.bin loaded ✓")
        else:
            print("  audio_projector.bin not found -- adapter weights are random init.")
    else:
        st_path = os.path.join(cfg.model_checkpoint, "model.safetensors")
        if not os.path.exists(st_path):
            raise FileNotFoundError(f"Expected model.safetensors at {st_path}.")
        print(f"  Loading monolithic checkpoint: {st_path}")
        sd = load_file(st_path)

        # DoRA/LoRA mismatch guard.
        ckpt_has_dora = _detect_dora(sd)
        if ckpt_has_dora and not cfg.use_dora:
            print(
                "  Checkpoint contains lora_magnitude_vector keys (DoRA) but "
                "cfg.use_dora=False. Set use_dora=True in VisConfig to load correctly."
            )
        elif not ckpt_has_dora and cfg.use_dora:
            print(
                "  cfg.use_dora=True but checkpoint has no lora_magnitude_vector keys. "
                "Set use_dora=False in VisConfig."
            )

        adapter_sd = {
            k.removeprefix("audio_projector."): v
            for k, v in sd.items() if k.startswith("audio_projector.")
        }
        peft_sd = {k: v for k, v in sd.items() if not k.startswith("audio_projector.")}

        # PEFT key format remap (old: lora_A.weight → new: lora_A.default.weight).
        needs_remap = any(
            ".lora_A.weight" in k or ".lora_B.weight" in k for k in peft_sd
        )
        if needs_remap:
            print("  Remapping old-format PEFT keys to current format...")
            peft_sd = {
                k.replace(".lora_A.weight", ".lora_A.default.weight")
                 .replace(".lora_B.weight", ".lora_B.default.weight"): v
                for k, v in peft_sd.items()
            }

        missing, unexpected = model.qwen.load_state_dict(peft_sd, strict=False)
        real_missing = [k for k in missing if "lora_A" in k or "lora_B" in k or "magnitude" in k]
        if real_missing:
            print(f"  Missing PEFT keys ({len(real_missing)}): {real_missing[:3]} …")
        print(f"  Loaded {len(peft_sd) - len(unexpected)} / {len(peft_sd)} PEFT keys ✓")

        if adapter_sd:
            model.audio_adapter.load_state_dict(adapter_sd, strict=False)
            print(f"  Loaded {len(adapter_sd)} audio_projector keys ✓")
        else:
            print("  No audio_projector.* keys -- adapter weights are random init.")

    model.eval()
    return model, audio_token_id


# =====================================================================
# FORWARD PASS WITH HOOK-BASED EXTRACTION
# =====================================================================
class ForwardPassCapture:
    """
    Registers hooks on DeltaNet and attention layers before a forward pass,
    captures the quantities needed for visualization, then removes hooks.

    Captured per DeltaNet layer:
      beta_logits[L]    -- raw output of linear_attn.in_proj_b: [T, out_dim]
                          sigmoid applied to get write strength ∈ (0, 1)
      layer_output[L]   -- hidden states exiting linear_attn: [T, D]

    Captured per attention layer:
      Returned directly via outputs.attentions after filtering None entries.
    """
    def __init__(self, model: "AudioFusedConditionedQwen"):
        self.model          = model
        self._hooks         = []
        self.beta_logits    : dict[int, torch.Tensor] = {}
        self.layer_output   : dict[int, torch.Tensor] = {}

    def _register(self):
        for L in DELTANET_LAYER_INDICES:
            # Hook 1: in_proj_b output → write strength logits
            beta_mod = find_module(self.model.qwen, f"layers.{L}.linear_attn.in_proj_b")
            if beta_mod is not None:
                def _beta_hook(mod, inp, out, _L=L):
                    self.beta_logits[_L] = out.squeeze(0).detach().cpu().float()
                self._hooks.append(beta_mod.register_forward_hook(_beta_hook))
            else:
                print(f"  in_proj_b not found at layer {L} -- write strength unavailable.")

            # Hook 2: linear_attn output → hidden states for state influence
            lattn_mod = find_module(self.model.qwen, f"layers.{L}.linear_attn")
            if lattn_mod is not None:
                def _out_hook(mod, inp, out, _L=L):
                    h = out[0] if isinstance(out, (tuple, list)) else out
                    self.layer_output[_L] = h.squeeze(0).detach().cpu().float()
                self._hooks.append(lattn_mod.register_forward_hook(_out_hook))

    def _remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def run(
        self,
        model: "AudioFusedConditionedQwen",
        input_ids_tensor: torch.Tensor,
        audio_features: torch.Tensor,
        key_padding_mask: torch.Tensor,
        zero_audio: bool = False,
    ) -> tuple[tuple, dict, dict]:
        """
        Runs one forward pass, returns:
          attn_matrices   -- dict {true_layer_idx: [H, S, S] float32 cpu tensor}
          beta_logits     -- dict {layer_idx: [T, out_dim] float32 cpu tensor}
          layer_output    -- dict {layer_idx: [T, D] float32 cpu tensor}
        """
        self.beta_logits.clear()
        self.layer_output.clear()
        self._register()

        af = torch.zeros_like(audio_features) if zero_audio else audio_features

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_tensor,
                audio_features=af,
                key_padding_mask=key_padding_mask,
                output_attentions=True,
            )

        self._remove()

        if outputs.attentions is None:
            raise RuntimeError(
                "outputs.attentions is None. Ensure attn_implementation='eager'."
            )

        # Filter to attention layers only; DeltaNet layers return None.
        # 1. Filter out None entries and ensure we are dealing with a list of tensors
        actual_tensors = [a for a in outputs.attentions if a is not None]
        
        attn_matrices = {}
        
        # 2. Check if the model returned ONLY the attention-capable layers (Short List)
        if len(actual_tensors) == len(ATTENTION_LAYER_INDICES):
            print(f"  Detected 'short-list' attention return ({len(actual_tensors)} layers). Mapping to indices...")
            for true_idx, tensor in zip(ATTENTION_LAYER_INDICES, actual_tensors):
                attn_matrices[true_idx] = tensor.squeeze(0).cpu().float()
                
        # 3. Fallback: Check if it's a full 24-layer list (Long List)
        elif len(outputs.attentions) == 24:
            print("  Detected full-length attention return (24 layers). Extracting by index...")
            for i, a in enumerate(outputs.attentions):
                if i in ATTENTION_LAYER_INDICES and a is not None:
                    attn_matrices[i] = a.squeeze(0).cpu().float()
        
        else:
            print(f"  Warning: Unexpected attention list length: {len(outputs.attentions)}. "
                  f"Expected 6 or 24. Attempting best-effort mapping...")
            # Best effort: just pair whatever we got with the indices in order
            for i, tensor in enumerate(actual_tensors):
                if i < len(ATTENTION_LAYER_INDICES):
                    attn_matrices[ATTENTION_LAYER_INDICES[i]] = tensor.squeeze(0).cpu().float()

        n_expected = len(ATTENTION_LAYER_INDICES)
        n_got      = len(attn_matrices)
        if n_got < n_expected:
            print(
                f"  Expected attention matrices from {n_expected} layers, "
                f"got {n_got}. Missing: "
                f"{set(ATTENTION_LAYER_INDICES) - set(attn_matrices)}"
            )

        return attn_matrices, dict(self.beta_logits), dict(self.layer_output)


def prepare_inputs(
    model, tokenizer, test_prompt, prompt_to_fx, audio_token_id, device
) -> tuple:
    """Tokenises the prompt and loads audio features. Returns all tensors needed
    for ForwardPassCapture.run(), plus the raw input_ids and derived labels."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_AUDIO},
        {"role": "user",   "content": test_prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids_tensor = tokenizer(formatted, return_tensors="pt")["input_ids"].to(device)
    input_ids        = input_ids_tensor[0].tolist()

    feat              = prompt_to_fx[test_prompt]           # [T, 2048]
    audio_features    = feat.unsqueeze(0).to(device, dtype=torch.bfloat16)
    T                 = feat.size(0)
    key_padding_mask  = torch.zeros(1, T, dtype=torch.bool, device=device)

    labels     = clean_token_labels(tokenizer, input_ids, audio_token_id)
    audio_idxs = get_audio_indices(input_ids, audio_token_id)
    user_start = get_user_start_idx(tokenizer, labels)

    print(
        f"  Sequence: {len(input_ids)} tokens | "
        f"Audio span: positions {audio_idxs[0]}–{audio_idxs[-1]} "
        f"({len(audio_idxs)} tokens) | User turn starts at: {user_start}"
    )
    return input_ids_tensor, audio_features, key_padding_mask, input_ids, labels, audio_idxs, user_start


# =====================================================================
# PLOT 1: UNIFIED LAYER-WISE ROUTING
# =====================================================================
def plot_unified_layer_routing(
    attn_matrices      : dict[int, torch.Tensor],
    beta_logits        : dict[int, torch.Tensor],
    audio_idxs         : list[int],
    user_start         : int,
    save_dir           : str,
    test_prompt        : str,
    attn_matrices_base : dict[int, torch.Tensor] | None = None,
    beta_logits_base   : dict[int, torch.Tensor] | None = None,
) -> int:
    """
    Produces a single plot covering all 24 layers on a shared x-axis:
      - Attention layers (7,11,15,19,23): mean attention mass to audio span
        (mean over heads, summed over audio keys, averaged over user queries)
      - DeltaNet layers (all others): mean write strength β at audio positions
        (mean over heads and audio positions after sigmoid)

    The two y-axes are on the same scale [0, 1] so the curves are directly
    comparable: both measure "how much routing budget the model directs toward
    audio-related processing at each layer."

    Returns the peak attention layer index (true layer index, not plot index).
    """
    n_layers = 24

    # --- Attention mass at attention layers ---
    attn_mass   = {}   # {layer_idx: float}
    for L, a in attn_matrices.items():
        mean_a = a.mean(dim=0).numpy()              # [S, S]
        mass   = mean_a[user_start:, audio_idxs].sum(axis=-1).mean()
        attn_mass[L] = float(mass)

    # --- Mean write strength β at DeltaNet layers ---
    deltanet_beta = {}   # {layer_idx: float}
    for L, logits in beta_logits.items():
        # logits: [T, out_dim]; apply sigmoid, select audio positions, average
        beta = torch.sigmoid(logits)               # [T, out_dim]
        audio_beta = beta[audio_idxs, :].mean().item()
        deltanet_beta[L] = float(audio_beta)

    # Baseline curves (zero-audio forward pass)
    attn_mass_base    = {}
    deltanet_beta_base = {}
    if attn_matrices_base is not None:
        for L, a in attn_matrices_base.items():
            mean_a = a.mean(dim=0).numpy()
            attn_mass_base[L] = float(mean_a[user_start:, audio_idxs].sum(axis=-1).mean())
    if beta_logits_base is not None:
        for L, logits in beta_logits_base.items():
            beta = torch.sigmoid(logits)
            attn_mass_base[L] = float(beta[audio_idxs, :].mean().item())

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.set_style("whitegrid")
    xs = list(range(n_layers))

    # DeltaNet β points
    dn_xs   = [x for x in xs if x in deltanet_beta]
    dn_vals = [deltanet_beta[x] for x in dn_xs]
    ax.scatter(dn_xs, dn_vals, color="#5b9bd5", s=60, zorder=4,
               label="DeltaNet β (mean write strength at audio positions)")
    ax.plot(dn_xs, dn_vals, color="#5b9bd5", linewidth=1.2, linestyle="--", zorder=3)

    # Attention mass points
    at_xs   = [x for x in xs if x in attn_mass]
    at_vals = [attn_mass[x] for x in at_xs]
    ax.scatter(at_xs, at_vals, color="#d9534f", s=80, marker="D", zorder=5,
               label="Attention mass (mean fraction of attn budget on audio)")
    ax.plot(at_xs, at_vals, color="#d9534f", linewidth=1.5, zorder=4)

    # Baseline overlays
    if attn_mass_base:
        at_base_xs   = [x for x in xs if x in attn_mass_base and x in ATTENTION_LAYER_INDICES]
        at_base_vals = [attn_mass_base[x] for x in at_base_xs]
        ax.scatter(at_base_xs, at_base_vals, color="#e0a0a0", s=50, marker="D",
                   zorder=3, label="Attention mass (zero-audio baseline)")
        ax.plot(at_base_xs, at_base_vals, color="#e0a0a0", linewidth=1.0,
                linestyle=":", zorder=2)
    if deltanet_beta_base:
        dn_base_xs   = [x for x in xs if x in deltanet_beta_base and x in DELTANET_LAYER_INDICES]
        dn_base_vals = [deltanet_beta_base[x] for x in dn_base_xs]
        ax.scatter(dn_base_xs, dn_base_vals, color="#a0b8d8", s=40,
                   zorder=3, label="DeltaNet β (zero-audio baseline)")

    # Shade attention layer positions
    for L in ATTENTION_LAYER_INDICES:
        ax.axvspan(L - 0.4, L + 0.4, color="#f5e6e6", alpha=0.6, zorder=1)

    # Peak attention layer
    peak_layer = max(attn_mass, key=attn_mass.get) if attn_mass else ATTENTION_LAYER_INDICES[0]
    ax.axvline(x=peak_layer, color="#222", linestyle=":", linewidth=1.5,
               label=f"Peak attention layer ({peak_layer})", zorder=6)

    ax.set_title(
        "Unified Layer-wise Audio Routing\n"
        "DeltaNet β = write strength  |  Attention = fraction of attention budget on audio span\n"
        "Shaded bands = standard Gated Attention layers",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Routing strength toward audio [0–1]", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs], fontsize=8)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f'Prompt: "{test_prompt[:80]}"', fontsize=8, y=0.01, style="italic")
    fig.tight_layout()

    path = os.path.join(save_dir, "1_unified_layer_routing.png")
    fig.savefig(path, format="png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}  (peak attention layer: {peak_layer})")
    return peak_layer


# =====================================================================
# PLOT 2: PER-HEAD AUDIO ATTENTION AT PEAK ATTENTION LAYER
# =====================================================================
def plot_per_head_audio_attention(
    attn_matrices : dict[int, torch.Tensor],
    peak_layer    : int,
    audio_idxs    : list[int],
    user_start    : int,
    save_dir      : str,
) -> int:
    if peak_layer not in attn_matrices:
        print(f"  Peak layer {peak_layer} not in attn_matrices -- skipping Plot 2.")
        return 0

    attn_peak = attn_matrices[peak_layer]      # [H, S, S]
    num_heads  = attn_peak.shape[0]            # 8 Q-heads for GQA

    head_scores = [
        float(attn_peak[h, user_start:, :][:, audio_idxs].sum(axis=-1).mean())
        for h in range(num_heads)
    ]
    best_head = int(np.argmax(head_scores))
    colors    = ["#d9534f" if h == best_head else "#7bafd4" for h in range(num_heads)]

    fig, ax = plt.subplots(figsize=(max(8, num_heads * 0.8), 4))
    ax.bar(range(num_heads), head_scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=np.mean(head_scores), color="#555", linestyle="--",
               linewidth=1.0, label="Mean across heads")
    patch = mpatches.Patch(color="#d9534f", label=f"Best head (head {best_head})")
    ax.legend(handles=[patch, ax.lines[0]], fontsize=9)
    ax.set_title(
        f"Per-Head Audio Attention at Layer {peak_layer} (GQA: {num_heads} Q-heads)\n"
        "(sum over audio key positions, mean over user query positions)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Attention Head", fontsize=11)
    ax.set_ylabel("Mean Attention to Audio Span", fontsize=11)
    ax.set_xticks(range(num_heads))
    fig.tight_layout()

    path = os.path.join(save_dir, f"2_per_head_audio_attn_layer{peak_layer}.png")
    fig.savefig(path, format="png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}  (best head: {best_head})")
    return best_head


# =====================================================================
# PLOT 3: 2D SELF-ATTENTION HEATMAP
# =====================================================================
def plot_attention_heatmap(
    attn_matrices : dict[int, torch.Tensor],
    peak_layer    : int,
    best_head     : int,
    audio_idxs    : list[int],
    user_start    : int,
    labels        : list[str],
    save_dir      : str,
    test_prompt   : str,
) -> None:
    if peak_layer not in attn_matrices:
        print(f"  Peak layer {peak_layer} not in attn_matrices -- skipping Plot 3.")
        return

    attn = attn_matrices[peak_layer][best_head].numpy()  # [S, S]
    S    = len(labels)

    visible_attn = attn[user_start:, :]          # user-turn queries vs all keys
    row_labels   = labels[user_start:]

    fig, ax = plt.subplots(
        figsize=(max(14, S * 0.18), max(8, len(row_labels) * 0.22))
    )
    sns.heatmap(
        visible_attn, xticklabels=labels, yticklabels=row_labels,
        cmap="magma", ax=ax,
        cbar_kws={"label": f"Attn weight (head {best_head}, layer {peak_layer})"},
        linewidths=0,
    )
    ax.axvspan(audio_idxs[0], audio_idxs[-1] + 1, color="cyan", alpha=0.15,
               label="Audio span (keys)")
    audio_rows = [i - user_start for i in audio_idxs if user_start <= i < S]
    if audio_rows:
        ax.axhspan(audio_rows[0], audio_rows[-1] + 1, color="cyan", alpha=0.15,
                   label="Audio span (queries)")
    ax.set_title(
        f"Self-Attention Heatmap -- Layer {peak_layer}, Head {best_head}\n"
        f"Rows: user-turn queries  |  Audio span: positions "
        f"{audio_idxs[0]}–{audio_idxs[-1]}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Key position", fontsize=10)
    ax.set_ylabel("Query position (user turn)", fontsize=10)
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=7)
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f'Prompt: "{test_prompt[:80]}"', fontsize=8, y=0.005, style="italic")
    fig.tight_layout()

    path = os.path.join(save_dir, f"3_attn_heatmap_layer{peak_layer}_head{best_head}.png")
    fig.savefig(path, format="png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# PLOT 4: AUDIO TOKEN BREAKDOWN (N_audio > 1 only)
# =====================================================================
def plot_audio_token_breakdown(
    attn_matrices : dict[int, torch.Tensor],
    peak_layer    : int,
    audio_idxs    : list[int],
    user_start    : int,
    labels        : list[str],
    save_dir      : str,
    test_prompt   : str,
) -> None:
    n_audio = len(audio_idxs)
    if n_audio <= 1:
        print("  Plot 4 skipped (N_audio == 1).")
        return
    if peak_layer not in attn_matrices:
        print(f"  Peak layer {peak_layer} not in attn_matrices -- skipping Plot 4.")
        return

    mean_attn    = attn_matrices[peak_layer].mean(dim=0).numpy()    # [S, S]
    user_to_audio = mean_attn[user_start:, audio_idxs]              # [user_tokens, N_audio]
    row_labels   = labels[user_start:]
    col_labels   = [labels[i] for i in audio_idxs]

    fig, ax = plt.subplots(
        figsize=(max(6, n_audio * 0.55), max(5, len(row_labels) * 0.2))
    )
    sns.heatmap(
        user_to_audio, xticklabels=col_labels, yticklabels=row_labels,
        cmap="YlOrRd", ax=ax,
        cbar_kws={"label": f"Mean attn (mean over heads, layer {peak_layer})"},
        linewidths=0.3, linecolor="white",
    )
    ax.set_title(
        f"Audio Token Breakdown -- Layer {peak_layer} (mean over heads)\n"
        "Rows: user-turn query tokens  |  Columns: individual audio placeholder tokens",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Audio token slot", fontsize=10)
    ax.set_ylabel("User query token", fontsize=10)
    plt.xticks(rotation=45, fontsize=8, ha="right")
    plt.yticks(rotation=0, fontsize=7)
    fig.suptitle(f'Prompt: "{test_prompt[:80]}"', fontsize=8, y=0.005, style="italic")
    fig.tight_layout()

    path = os.path.join(save_dir, f"4_audio_token_breakdown_layer{peak_layer}.png")
    fig.savefig(path, format="png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# PLOT 5: DELTANET WRITE STRENGTH HEATMAP
# =====================================================================
def plot_deltanet_write_strength(
    beta_logits : dict[int, torch.Tensor],
    audio_idxs  : list[int],
    save_dir    : str,
    test_prompt : str,
) -> None:
    """
    Heatmap of [audio_token × DeltaNet_layer] write strength β after sigmoid.
    Each cell shows how strongly that audio token position was written into
    the recurrent memory at that DeltaNet layer, averaged over output heads.

    High β at early DeltaNet layers confirms audio content is being encoded
    into the recurrent state before it reaches the first full attention layer.
    Low β everywhere indicates audio tokens are not being written to memory --
    a critical failure mode for audio conditioning through DeltaNet layers.
    """
    if not beta_logits:
        print("  No DeltaNet beta logits captured -- skipping Plot 5.")
        return

    sorted_layers = sorted(beta_logits.keys())
    n_audio       = len(audio_idxs)

    # Build [audio_position × DeltaNet_layer] matrix
    # beta_logits[L] has shape [T, out_dim]; select audio rows, mean over out_dim
    beta_matrix = np.zeros((n_audio, len(sorted_layers)))
    for j, L in enumerate(sorted_layers):
        logits = beta_logits[L]           # [T, out_dim]
        beta   = torch.sigmoid(logits)    # [T, out_dim]
        for i, pos in enumerate(audio_idxs):
            if pos < beta.shape[0]:
                beta_matrix[i, j] = beta[pos, :].mean().item()

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_layers) * 0.45), max(4, n_audio * 0.4)))
    sns.heatmap(
        beta_matrix,
        xticklabels=[str(L) for L in sorted_layers],
        yticklabels=[f"audio_{i}" for i in range(n_audio)],
        cmap="Blues",
        ax=ax,
        vmin=0, vmax=1,
        cbar_kws={"label": "Write strength β = σ(in_proj_b output), mean over heads"},
        linewidths=0.2, linecolor="white",
    )
    # Annotate which layer block each DeltaNet layer belongs to
    block_boundaries = [i for i, L in enumerate(sorted_layers) if L % 4 == 3]
    for b in block_boundaries:
        ax.axvline(x=b + 1, color="red", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_title(
        "DeltaNet Write Strength (β) at Audio Token Positions\n"
        "Rows: audio placeholder token slots  |  Columns: DeltaNet layer index\n"
        "Red dashed lines = block boundaries (after each Gated Attention layer)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("DeltaNet Layer Index", fontsize=11)
    ax.set_ylabel("Audio Token Slot", fontsize=11)
    plt.xticks(rotation=45, fontsize=8, ha="right")
    plt.yticks(rotation=0, fontsize=8)
    fig.suptitle(f'Prompt: "{test_prompt[:80]}"', fontsize=8, y=0.005, style="italic")
    fig.tight_layout()

    path = os.path.join(save_dir, "5_deltanet_write_strength.png")
    fig.savefig(path, format="png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# PLOT 6: STATE INFLUENCE (real audio vs zero-audio baseline)
# =====================================================================
def plot_state_influence(
    layer_output      : dict[int, torch.Tensor],
    layer_output_base : dict[int, torch.Tensor],
    audio_idxs        : list[int],
    labels            : list[str],
    save_dir          : str,
    test_prompt       : str,
) -> None:
    """
    For each DeltaNet layer, computes the L2 norm of
        (real_audio_hidden_state − zero_audio_hidden_state)
    at every sequence position. This is a causal influence measure: it shows
    which tokens' representations were most changed by the presence of real
    audio content in the recurrent memory at each layer.

    Unlike attention weights (which measure where a token *looks*), this
    measures what actually *changed* in the model's computation due to audio.
    High influence at user-turn text tokens (after the audio span) confirms
    the audio information propagated forward into text processing.
    """
    if not layer_output or not layer_output_base:
        print("  Missing layer outputs for state influence -- skipping Plot 6.")
        return

    sorted_layers = sorted(set(layer_output.keys()) & set(layer_output_base.keys()))
    if not sorted_layers:
        print("  No overlapping DeltaNet layer outputs -- skipping Plot 6.")
        return

    T = layer_output[sorted_layers[0]].shape[0]    # sequence length
    influence = np.zeros((T, len(sorted_layers)))

    for j, L in enumerate(sorted_layers):
        real  = layer_output[L]           # [T, D]
        base  = layer_output_base[L]      # [T, D]
        diff  = (real - base).numpy()
        influence[:, j] = np.linalg.norm(diff, axis=-1)  # [T]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [2, 1]})

    # Top: heatmap [sequence_position × DeltaNet_layer]
    ax = axes[0]
    # Clip labels for readability
    tick_labels = [l[:10] for l in labels]
    sns.heatmap(
        influence.T,
        xticklabels=tick_labels,
        yticklabels=[str(L) for L in sorted_layers],
        cmap="Reds",
        ax=ax,
        cbar_kws={"label": "L2 influence (real − zero-audio hidden state norm)"},
        linewidths=0,
    )
    for pos in audio_idxs:
        ax.axvspan(pos, pos + 1, color="cyan", alpha=0.3)
    ax.set_title(
        "State Influence: L2 norm of (real audio − zero audio) hidden states\n"
        "Cyan columns = audio token positions  |  "
        "High values after audio span = audio content propagated to text tokens",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Sequence position", fontsize=10)
    ax.set_ylabel("DeltaNet layer", fontsize=10)
    plt.sca(ax)
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=7)

    # Bottom: mean influence per layer (collapsed over sequence)
    ax2 = axes[1]
    mean_influence_per_layer = influence.mean(axis=0)  # [n_layers]
    ax2.bar(range(len(sorted_layers)), mean_influence_per_layer,
            color="#d9534f", alpha=0.8)
    ax2.set_xticks(range(len(sorted_layers)))
    ax2.set_xticklabels([str(L) for L in sorted_layers], fontsize=8)
    ax2.set_title("Mean State Influence per DeltaNet Layer (averaged over all sequence positions)",
                  fontsize=10)
    ax2.set_xlabel("DeltaNet Layer", fontsize=10)
    ax2.set_ylabel("Mean L2 Influence", fontsize=10)

    fig.suptitle(f'Prompt: "{test_prompt[:80]}"', fontsize=8, y=0.005, style="italic")
    fig.tight_layout()

    path = os.path.join(save_dir, "6_state_influence.png")
    fig.savefig(path, format="png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# MAIN
# =====================================================================
def generate_attention_plots(cfg: VisConfig | None = None) -> None:
    if cfg is None:
        cfg = VisConfig()

    os.makedirs(cfg.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Architecture: {len(ATTENTION_LAYER_INDICES)} attention layers "
          f"at {ATTENTION_LAYER_INDICES}, "
          f"{len(DELTANET_LAYER_INDICES)} DeltaNet layers.")

    print(f"\nLoading prompt_to_fx from {cfg.fx_path}...")
    with open(cfg.fx_path, "rb") as f:
        prompt_to_fx_raw = pickle.load(f)
    prompt_to_fx = {k.strip(): v for k, v in prompt_to_fx_raw.items()}

    test_prompt = cfg.test_prompt or next(iter(prompt_to_fx))
    if test_prompt not in prompt_to_fx:
        raise KeyError(
            f"test_prompt not found in prompt_to_fx. "
            f"First key: {next(iter(prompt_to_fx))!r}"
        )
    print(f"Test prompt: {test_prompt!r}")

    print(f"\nLoading tokenizer from {cfg.model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_checkpoint, trust_remote_code=True)

    print("Loading model...")
    model, audio_token_id = load_model_for_inference(cfg, tokenizer, prompt_to_fx, device)

    print("\nPreparing inputs...")
    (input_ids_tensor, audio_features, key_padding_mask,
     input_ids, labels, audio_idxs, user_start) = prepare_inputs(
        model, tokenizer, test_prompt, prompt_to_fx, audio_token_id, device
    )

    capturer = ForwardPassCapture(model)

    # --- Real audio forward pass ---
    print("\nForward pass: real audio...")
    attn_matrices, beta_logits, layer_output = capturer.run(
        model, input_ids_tensor, audio_features, key_padding_mask, zero_audio=False
    )
    print(
        f"  Captured: {len(attn_matrices)} attention matrices, "
        f"{len(beta_logits)} DeltaNet β tensors, "
        f"{len(layer_output)} DeltaNet hidden state tensors."
    )
    # Report DeltaNet β shape for debugging
    if beta_logits:
        sample_L   = next(iter(beta_logits))
        sample_shp = beta_logits[sample_L].shape
        print(f"  Beta logit shape at layer {sample_L}: {sample_shp} "
              f"(T={sample_shp[0]}, out_dim={sample_shp[1] if len(sample_shp) > 1 else 1})")

    # --- Zero-audio baseline forward pass ---
    attn_matrices_base = None
    beta_logits_base   = None
    layer_output_base  = None
    if cfg.baseline_comparison:
        print("\nForward pass: zero-audio baseline...")
        attn_matrices_base, beta_logits_base, layer_output_base = capturer.run(
            model, input_ids_tensor, audio_features, key_padding_mask, zero_audio=True
        )

    # --- Plots ---
    print("\nGenerating plots...")
    peak_layer = plot_unified_layer_routing(
        attn_matrices, beta_logits, audio_idxs, user_start,
        cfg.save_dir, test_prompt,
        attn_matrices_base=attn_matrices_base,
        beta_logits_base=beta_logits_base,
    )
    best_head = plot_per_head_audio_attention(
        attn_matrices, peak_layer, audio_idxs, user_start, cfg.save_dir
    )
    plot_attention_heatmap(
        attn_matrices, peak_layer, best_head, audio_idxs,
        user_start, labels, cfg.save_dir, test_prompt
    )
    plot_audio_token_breakdown(
        attn_matrices, peak_layer, audio_idxs, user_start,
        labels, cfg.save_dir, test_prompt
    )
    plot_deltanet_write_strength(
        beta_logits, audio_idxs, cfg.save_dir, test_prompt
    )
    if cfg.baseline_comparison and layer_output_base is not None:
        plot_state_influence(
            layer_output, layer_output_base,
            audio_idxs, labels, cfg.save_dir, test_prompt
        )
    else:
        print("  Plot 6 skipped (baseline_comparison=False).")

    print(f"\n✓ All plots saved to: {cfg.save_dir}")


if __name__ == "__main__":
    generate_attention_plots(VisConfig())