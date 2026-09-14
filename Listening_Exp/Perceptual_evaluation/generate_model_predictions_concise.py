"""
generate_model_predictions_concise.py
======================================

Concise variant of generate_model_predictions.py for perceptual evaluation.

DIFFERENCE FROM THE FULL SCRIPT
---------------------------------
  Full script (`generate_model_predictions.py`):
    - 4 baseline models (greedy, 1 completion each)
    - 1 hero model (FX+Voice: 64 sampled rollouts -> 3 diverse modes extracted)
    - Total generated predictions per prompt: 7 coordinates

  Concise script (`generate_model_predictions_concise.py`):
    - All 5 models (no-audio, FX, Voice, FX+Voice, external) run with a single
      greedy completion (temperature=0.0, do_sample=False).
    - No sampling pool, no mode extraction.
    - Total generated predictions per prompt: 5 coordinates

MUSHRA LISTENING TEST SCREEN COMPOSITION
-----------------------------------------
  Concise screen (7 stimuli total):
    1. No-audio baseline (GRPO-trained, 0.8B, SYSTEM_PROMPT_NO_AUDIO)
    2. FX-only (greedy)
    3. Voice-only (greedy)
    4. FX+Voice (greedy, single point)
    5. External baseline (Qwen3.5-4B, non-GRPO, SYSTEM_PROMPT)
    6. Tonmeister reference (1 expert point, collected separately)
    7. Hidden reference (original unprocessed audio)

  (Compare to Full screen: 7 model points + 4 tonmeisters + hidden ref = 12 stimuli)

OUTPUT JSON KEYS PER ITEM (predictions dict)
---------------------------------------------
  "no_audio_baseline"                   -> {"raw_completion", "parsed_coord"}
  "fx_model"                            -> {"raw_completion", "parsed_coord"}
  "voice_model"                         -> {"raw_completion", "parsed_coord"}
  "fx_voice_model"                      -> {"raw_completion", "parsed_coord"}
  "external_prompt_engineered_baseline" -> {"raw_completion", "parsed_coord"}

USAGE
-----
  python generate_model_predictions_concise.py

  # Custom paths:
  python generate_model_predictions_concise.py \
      --output_json model_predictions_concise.json \
      --dual_model_dir ../../src/GRPO_models/FX_Voice_model
"""

import os
import sys
import json
import gc
import pickle
import argparse
import torch
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.abspath(os.path.join(script_dir, "..", ".."))
src_dir    = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from GRPO_models.infer_fx import load_fx_model_and_tokenizer
from GRPO_models.infer_voice import load_voice_model_and_tokenizer
from GRPO_models.infer_fx_and_voice import load_dual_model_and_tokenizer
from GRPO_models.infer_text import load_text_model_and_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from Training_Scripts.parse_utils import parse_completion_to_coord
from Training_Scripts.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_AUDIO,
    SYSTEM_PROMPT_VOICE,
    SYSTEM_PROMPT_DUAL_AUDIO,
    SYSTEM_PROMPT_NO_AUDIO,
)


# ============================================================
# HELPERS
# ============================================================

def ensure_pad_token(tokenizer, model):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, "config"):
            model.config.pad_token_id = model.config.eos_token_id


def load_plain_llm_and_tokenizer(model_name_or_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def run_greedy(
    model,
    tokenizer,
    sys_prompt: str,
    prompt_text: str,
    device: str,
    verbose: bool = True,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """
    Single greedy completion for one prompt.
    Returns {"raw_completion": str, "parsed_coord": list | None}.
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": prompt_text},
    ]
    template_kwargs = chat_template_kwargs or {}
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    encoded   = tokenizer(formatted, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    else:
        attn_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            do_sample=False,
            max_new_tokens=32,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen    = out[0][input_ids.shape[1]:]
    text   = tokenizer.decode(gen, skip_special_tokens=True).strip()
    coord  = parse_completion_to_coord(text)
    parsed = coord.tolist() if coord is not None else None

    if verbose:
        print(f"    -> {text!r} -> {parsed}", flush=True)

    return {"raw_completion": text, "parsed_coord": parsed}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Concise prediction generator: 5 models, all greedy. "
            "One coordinate per model per item -- no sampling, no tonmeister input."
        )
    )
    parser.add_argument("--prompt_link_json", type=str,
                        default=os.path.join(script_dir, "prompt_audio_link.json"))
    parser.add_argument("--fx_pkl",    type=str,
                        default=os.path.join(script_dir, "perceptual_prompt_to_fx.pkl"))
    parser.add_argument("--voice_pkl", type=str,
                        default=os.path.join(script_dir, "perceptual_prompt_to_voice.pkl"))
    parser.add_argument("--output_json", type=str,
                        default=os.path.join(script_dir, "model_predictions_concise.json"))
    parser.add_argument("--fx_model_dir",    type=str,
                        default=os.path.join(src_dir, "GRPO_models", "FX_model"))
    parser.add_argument("--voice_model_dir", type=str,
                        default=os.path.join(src_dir, "GRPO_models", "Voice_model"))
    parser.add_argument("--dual_model_dir",  type=str,
                        default=os.path.join(src_dir, "GRPO_models", "FX_Voice_model"))
    parser.add_argument("--no_audio_model_dir", type=str,
                        default=os.path.join(src_dir, "GRPO_models", "Text_only"))
    parser.add_argument("--external_model_name", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-item output.")

    args = parser.parse_args()

    print("=== Concise Perceptual Prediction Generator (greedy, 5 models) ===")
    print(f"  Device      : {args.device}")
    print(f"  Output JSON : {args.output_json}\n")

    with open(args.prompt_link_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"Loading FX features:    {args.fx_pkl}")
    with open(args.fx_pkl, "rb") as f:
        raw_fx = pickle.load(f)
    fx_dict = {
        k.strip(): (v["tensor"] if isinstance(v, dict) and "tensor" in v else v)
        for k, v in raw_fx.items()
    }

    print(f"Loading Voice features: {args.voice_pkl}")
    with open(args.voice_pkl, "rb") as f:
        raw_voice = pickle.load(f)
    voice_dict = {
        k.strip(): (v["tensor"] if isinstance(v, dict) and "tensor" in v else v)
        for k, v in raw_voice.items()
    }

    results = []
    for idx, item in enumerate(items):
        results.append({
            "id":          idx + 1,
            "prompt":      item["prompt"].strip(),
            "category":    item["category"],
            "audio_clip":  item["audio_clip"],
            "descriptor":  item.get("descriptor", ""),
            "credits":     item.get("credits", ""),
            "predictions": {},
        })

    # ------------------------------------------------------------------
    # PHASE 1/5: EXTERNAL BASELINE
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"PHASE 1/5: External Baseline ({args.external_model_name})...")
    print("=" * 60)
    ext_model, ext_tok = load_plain_llm_and_tokenizer(args.external_model_name, args.device)
    ensure_pad_token(ext_tok, ext_model)

    for i, res in enumerate(results):
        prompt_text = res["prompt"]
        print(f"[{i+1}/{len(results)}] {prompt_text[:60]!r}")
        res["predictions"]["external_prompt_engineered_baseline"] = run_greedy(
            ext_model, ext_tok, SYSTEM_PROMPT, prompt_text,
            device=args.device, verbose=not args.quiet,
            chat_template_kwargs={"enable_thinking": False},
        )

    del ext_model, ext_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PHASE 2/5: NO-AUDIO BASELINE
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2/5: No-Audio Baseline...")
    print("=" * 60)
    no_audio_model, no_audio_tok = load_text_model_and_tokenizer(
        args.no_audio_model_dir, device=args.device
    )
    ensure_pad_token(no_audio_tok, no_audio_model)

    for i, res in enumerate(results):
        prompt_text = res["prompt"]
        print(f"[{i+1}/{len(results)}] {prompt_text[:60]!r}")
        res["predictions"]["no_audio_baseline"] = run_greedy(
            no_audio_model, no_audio_tok, SYSTEM_PROMPT_NO_AUDIO, prompt_text,
            device=args.device, verbose=not args.quiet,
        )

    del no_audio_model, no_audio_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PHASE 3/5: FX MODEL
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 3/5: FX Model...")
    print("=" * 60)
    fx_model, fx_tok = load_fx_model_and_tokenizer(args.fx_model_dir, device=args.device)
    ensure_pad_token(fx_tok, fx_model)

    for i, res in enumerate(results):
        prompt_text = res["prompt"]
        fx_tensor = fx_dict.get(prompt_text, torch.zeros(13, 2048, dtype=torch.bfloat16))
        fx_model.prompt_to_fx[prompt_text] = fx_tensor.to(args.device)

        print(f"[{i+1}/{len(results)}] {prompt_text[:60]!r}")
        res["predictions"]["fx_model"] = run_greedy(
            fx_model, fx_tok, SYSTEM_PROMPT_AUDIO, prompt_text,
            device=args.device, verbose=not args.quiet,
        )

    del fx_model, fx_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PHASE 4/5: VOICE MODEL
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 4/5: Voice Model...")
    print("=" * 60)
    voice_model, voice_tok = load_voice_model_and_tokenizer(args.voice_model_dir, device=args.device)
    ensure_pad_token(voice_tok, voice_model)

    for i, res in enumerate(results):
        prompt_text = res["prompt"]
        voice_tensor = voice_dict.get(prompt_text, torch.zeros(25, 768, dtype=torch.bfloat16))
        voice_model.prompt_to_voice[prompt_text] = voice_tensor.to(args.device)

        print(f"[{i+1}/{len(results)}] {prompt_text[:60]!r}")
        res["predictions"]["voice_model"] = run_greedy(
            voice_model, voice_tok, SYSTEM_PROMPT_VOICE, prompt_text,
            device=args.device, verbose=not args.quiet,
        )

    del voice_model, voice_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PHASE 5/5: DUAL FX+VOICE MODEL -- single greedy completion
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 5/5: Dual FX+Voice Model...")
    print("=" * 60)
    dual_model, dual_tok = load_dual_model_and_tokenizer(args.dual_model_dir, device=args.device)
    ensure_pad_token(dual_tok, dual_model)

    for i, res in enumerate(results):
        prompt_text  = res["prompt"]
        fx_tensor    = fx_dict.get(prompt_text,    torch.zeros(13, 2048, dtype=torch.bfloat16))
        voice_tensor = voice_dict.get(prompt_text, torch.zeros(25, 768,  dtype=torch.bfloat16))
        dual_model.prompt_to_fx[prompt_text]    = fx_tensor.to(args.device)
        dual_model.prompt_to_voice[prompt_text] = voice_tensor.to(args.device)

        print(f"[{i+1}/{len(results)}] {prompt_text[:60]!r}")
        res["predictions"]["fx_voice_model"] = run_greedy(
            dual_model, dual_tok, SYSTEM_PROMPT_DUAL_AUDIO, prompt_text,
            device=args.device, verbose=not args.quiet,
        )

    del dual_model, dual_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Saving to: {args.output_json}")
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\nDone -- {len(results)} items written.")
    print("MUSHRA stimuli per screen: 5 rated conditions + 1 hidden reference")
    print("  1. No-audio baseline")
    print("  2. FX-only (greedy)")
    print("  3. Voice-only (greedy)")
    print("  4. FX+Voice (greedy)")
    print("  5. External baseline (Qwen3.5-4B)")
    print("=" * 60)


if __name__ == "__main__":
    main()
