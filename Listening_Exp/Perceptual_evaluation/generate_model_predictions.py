import os
import sys
import json
import gc
import pickle
import argparse
import torch
import numpy as np
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
src_dir = os.path.join(repo_root, "src")
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


def ensure_pad_token(tokenizer, model):
    """
    FIX: the original script never set tokenizer.pad_token (only the
    training script did, and only inside its own __main__ block). With
    pad_token_id=None and no attention_mask passed to generate(), HF's
    generate() can fail to resolve padding correctly for inputs_embeds-based
    generation — a likely cause of the reported hang. Mirrors the fix
    already applied in the training script.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, "config"):
            model.config.pad_token_id = model.config.eos_token_id


def load_plain_llm_and_tokenizer(model_name_or_path: str, device: str):
    """
    Loads a stock (no custom audio adapters) HF causal LM + tokenizer, for
    the no-audio baseline condition. Unlike the FX/Voice/Dual models, this
    doesn't go through a custom wrapper class — no audio features to
    inject, so a plain AutoModelForCausalLM is sufficient.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def generate_rollouts_for_model(
    model,
    tokenizer,
    sys_prompt: str,
    prompt_text: str,
    num_rollouts: int = 16,
    temperature: float = 1.0,
    top_k: int = 20,
    device: str = "cuda",
    verbose: bool = True,
    chat_template_kwargs: dict | None = None,
):
    """
    Generates `num_rollouts` completions for a single prompt using chat
    formatting. Returns (raw_completions, parsed_coords).

    chat_template_kwargs: forwarded to apply_chat_template — e.g.
    {"enable_thinking": False} for Qwen3.5, which operates in thinking mode
    by default and would otherwise prepend <think>...</think> content
    before the actual coordinate response. Only needed for the external
    4B baseline (a stock, non-GRPO-trained model) — the GRPO-trained
    FX/Voice/Dual/no-audio models should NOT have this passed, since
    changing their chat template behavior wasn't requested and may not
    even be meaningful for their checkpoints.
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt_text}
    ]
    template_kwargs = chat_template_kwargs or {}
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    encoded = tokenizer(formatted, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    # FIX: explicitly build and pass attention_mask (previously omitted).
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = torch.ones_like(input_ids, device=device)

    completions = []
    parsed_coords = []

    with torch.no_grad():
        for r in range(num_rollouts):
            if verbose:
                print(f"    rollout {r + 1}/{num_rollouts}...", end=" ", flush=True)
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                top_k=top_k if temperature > 0.0 else None,
                max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_tokens = output_ids[0][input_ids.shape[1]:]
            completion_str = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            completions.append(completion_str)

            coord = parse_completion_to_coord(completion_str)
            parsed_coords.append(coord.tolist() if coord is not None else None)
            if verbose:
                print(f"-> {completion_str!r} -> {coord.tolist() if coord is not None else None}", flush=True)

    return completions, parsed_coords


def compute_prediction_stats(parsed_coords: list):
    valid = [c for c in parsed_coords if c is not None]
    if not valid:
        return {"valid_count": 0, "mean_coord": None, "std_coord": None}

    valid_arr = np.array(valid)
    mean_coord = np.mean(valid_arr, axis=0).round(3).tolist()
    std_coord = np.std(valid_arr, axis=0).round(3).tolist()

    return {"valid_count": len(valid), "mean_coord": mean_coord, "std_coord": std_coord}


def extract_top_k_modes(parsed_coords: list, k: int = 3, min_distance: float = 0.5):
    """
    Selects up to k representative points from a sampled rollout set, using
    exact-match frequency over the raw (unrounded) parsed coordinates —
    rounding to integers was discarding real precision, since the model's
    text output already has limited decimal precision on its own, so exact
    duplicates occur naturally without needing to round.

    Selection: sort unique raw predictions by frequency (most to least
    common). Walk down that list and greedily keep a prediction only if
    it's at least `min_distance` (true L2 distance, not grid distance) from
    every prediction already selected — a near-duplicate of an
    already-kept point is skipped and we continue to the next
    less-frequent prediction, rather than being selected and defeating the
    coverage argument.

    Returns
    -------
    selected_modes : list of up to k coordinates (raw precision)
    selected_counts : frequency count for each selected mode
    all_unique_sorted : full frequency table, all unique raw predictions,
        most to least frequent — kept for transparency/debugging and for
        the coverage analysis in the perceptual study (whether the
        tonmeister's choice falls within the full empirical distribution,
        not just the 3 selected/reported modes).
    """
    valid = [tuple(c) for c in parsed_coords if c is not None]
    if not valid:
        return [], [], []

    counts = Counter(valid)
    # Sort by frequency desc; ties keep first-occurrence order (stable sort
    # over Counter's insertion-ordered items, which reflects first
    # occurrence in `valid`).
    all_unique_sorted = sorted(counts.items(), key=lambda kv: -kv[1])

    selected, selected_counts = [], []
    for pt, cnt in all_unique_sorted:
        pt_arr = np.array(pt)
        if all(np.linalg.norm(pt_arr - np.array(s)) >= min_distance for s in selected):
            selected.append(pt)
            selected_counts.append(cnt)
        if len(selected) == k:
            break

    if len(selected) < k:
        print(f"    [WARN] Only found {len(selected)} mode(s) with min_distance={min_distance}; "
              f"relaxing distance constraint to fill remaining slots.")
        for pt, cnt in all_unique_sorted:
            if len(selected) >= k:
                break
            if pt not in selected:
                selected.append(pt)
                selected_counts.append(cnt)

    selected_out = [list(p) for p in selected[:k]]
    selected_counts_out = selected_counts[:k]
    all_unique_out = [{"coord": list(pt), "count": cnt} for pt, cnt in all_unique_sorted]

    return selected_out, selected_counts_out, all_unique_out


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(description="Generate model predictions for the perceptual evaluation set.")
    parser.add_argument("--prompt_link_json", type=str, default=os.path.join(script_dir, "prompt_audio_link.json"))
    parser.add_argument("--fx_pkl", type=str, default=os.path.join(script_dir, "perceptual_prompt_to_fx.pkl"))
    parser.add_argument("--voice_pkl", type=str, default=os.path.join(script_dir, "perceptual_prompt_to_voice.pkl"))
    parser.add_argument("--output_json", type=str, default=os.path.join(script_dir, "model_predictions.json"))
    parser.add_argument("--fx_model_dir", type=str, default=os.path.join(src_dir, "GRPO_models", "FX_model"))
    parser.add_argument("--voice_model_dir", type=str, default=os.path.join(src_dir, "GRPO_models", "Voice_model"))
    parser.add_argument("--dual_model_dir", type=str, default=os.path.join(src_dir, "GRPO_models", "FX_Voice_model"))
    # Condition #7: external, non-GRPO-trained baseline, 4B, SYSTEM_PROMPT.
    parser.add_argument("--external_model_name", type=str, default="Qwen/Qwen3.5-4B")
    # Condition #1: no-audio GRPO-trained baseline (0.8B), SYSTEM_PROMPT_NO_AUDIO.
    parser.add_argument("--no_audio_model_dir", type=str, default=os.path.join(src_dir, "GRPO_models", "Text_only"))
    # FX-only / Voice-only: greedy, single completion (plan §3, conditions 2-3)
    parser.add_argument("--greedy_temperature", type=float, default=0.0)
    # FX+Voice: larger sampling pool for stable mode frequencies (plan §3)
    parser.add_argument("--fx_voice_num_samples", type=int, default=64,
                         help="Number of sampled rollouts used to extract the 3 FX+Voice modes.")
    parser.add_argument("--fx_voice_temperature", type=float, default=1.0)
    parser.add_argument("--fx_voice_top_k", type=int, default=20)
    parser.add_argument("--num_modes", type=int, default=3)
    parser.add_argument("--min_mode_distance", type=float, default=0.5,
                         help="Minimum L2 distance between the selected FX+Voice modes.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-rollout progress printing.")

    args = parser.parse_args()

    print("=== Perceptual Evaluation Prediction Generator ===")
    print(f"Device: {args.device}")
    print(f"Reading items from: {args.prompt_link_json}")

    with open(args.prompt_link_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"Loading FX features from: {args.fx_pkl}")
    with open(args.fx_pkl, "rb") as f:
        raw_fx = pickle.load(f)
    fx_dict = {k.strip(): (v["tensor"] if isinstance(v, dict) and "tensor" in v else v) for k, v in raw_fx.items()}

    print(f"Loading Voice features from: {args.voice_pkl}")
    with open(args.voice_pkl, "rb") as f:
        raw_voice = pickle.load(f)
    voice_dict = {k.strip(): (v["tensor"] if isinstance(v, dict) and "tensor" in v else v) for k, v in raw_voice.items()}

    results = []
    for idx, item in enumerate(items):
        results.append({
            "id": idx + 1,
            "prompt": item["prompt"].strip(),
            "category": item["category"],
            "audio_clip": item["audio_clip"],
            "descriptor": item.get("descriptor", ""),
            "credits": item.get("credits", ""),
            "predictions": {}
        })

    # ------------------------------------------------------------------
    # PHASE 1/5: EXTERNAL BASELINE (condition #7) — 4B stock model, no
    # GRPO training, no audio adapters, SYSTEM_PROMPT.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"PHASE 1/5: External Baseline ({args.external_model_name}, greedy)...")
    print("=" * 60)
    ext_model, ext_tok = load_plain_llm_and_tokenizer(args.external_model_name, args.device)
    ensure_pad_token(ext_tok, ext_model)

    for i, res in enumerate(results):
        prompt_text = res["prompt"]
        print(f"[{i+1}/{len(results)}] External Baseline -> \"{prompt_text[:50]}...\"")
        raw_comp, parsed = generate_rollouts_for_model(
            ext_model, ext_tok, SYSTEM_PROMPT, prompt_text,
            num_rollouts=1, temperature=args.greedy_temperature, device=args.device,
            verbose=not args.quiet,
            chat_template_kwargs={"enable_thinking": False},
        )
        res["predictions"]["external_prompt_engineered_baseline"] = {
            "raw_completion": raw_comp[0],
            "parsed_coord": parsed[0],
        }

    del ext_model, ext_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PHASE 2/5: NO-AUDIO BASELINE (condition #1) — GRPO-trained, 0.8B,
    # SYSTEM_PROMPT_NO_AUDIO, matched-size internal ablation.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2/5: No-Audio Baseline (greedy, 1 completion per prompt)...")
    print("=" * 60)
    no_audio_model, no_audio_tok = load_text_model_and_tokenizer(args.no_audio_model_dir, device=args.device)
    ensure_pad_token(no_audio_tok, no_audio_model)

    for i, res in enumerate(results):
        prompt_text = res["prompt"]
        print(f"[{i+1}/{len(results)}] No-Audio Baseline -> \"{prompt_text[:50]}...\"")
        raw_comp, parsed = generate_rollouts_for_model(
            no_audio_model, no_audio_tok, SYSTEM_PROMPT_NO_AUDIO, prompt_text,
            num_rollouts=1, temperature=args.greedy_temperature, device=args.device,
            verbose=not args.quiet,
        )
        res["predictions"]["no_audio_baseline"] = {
            "raw_completion": raw_comp[0],
            "parsed_coord": parsed[0],
        }

    del no_audio_model, no_audio_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PHASE 3/5: FX MODEL — greedy, single completion (condition #2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 3/5: FX Model (greedy, 1 completion per prompt)...")
    print("=" * 60)
    fx_model, fx_tok = load_fx_model_and_tokenizer(args.fx_model_dir, device=args.device)
    ensure_pad_token(fx_tok, fx_model)

    for i, res in enumerate(results):
        prompt_text = res["prompt"]
        fx_tensor = fx_dict.get(prompt_text, torch.zeros(13, 2048, dtype=torch.bfloat16))
        fx_model.prompt_to_fx[prompt_text] = fx_tensor.to(args.device)

        print(f"[{i+1}/{len(results)}] FX Model -> \"{prompt_text[:50]}...\"")
        raw_comp, parsed = generate_rollouts_for_model(
            fx_model, fx_tok, SYSTEM_PROMPT_AUDIO, prompt_text,
            num_rollouts=1, temperature=args.greedy_temperature, device=args.device,
            verbose=not args.quiet,
        )
        res["predictions"]["fx_model"] = {
            "raw_completion": raw_comp[0],
            "parsed_coord": parsed[0],
        }

    del fx_model, fx_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PHASE 4/5: VOICE MODEL — greedy, single completion (condition #3)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 4/5: Voice Model (greedy, 1 completion per prompt)...")
    print("=" * 60)
    voice_model, voice_tok = load_voice_model_and_tokenizer(args.voice_model_dir, device=args.device)
    ensure_pad_token(voice_tok, voice_model)

    for i, res in enumerate(results):
        prompt_text = res["prompt"]
        voice_tensor = voice_dict.get(prompt_text, torch.zeros(25, 768, dtype=torch.bfloat16))
        voice_model.prompt_to_voice[prompt_text] = voice_tensor.to(args.device)

        print(f"[{i+1}/{len(results)}] Voice Model -> \"{prompt_text[:50]}...\"")
        raw_comp, parsed = generate_rollouts_for_model(
            voice_model, voice_tok, SYSTEM_PROMPT_VOICE, prompt_text,
            num_rollouts=1, temperature=args.greedy_temperature, device=args.device,
            verbose=not args.quiet,
        )
        res["predictions"]["voice_model"] = {
            "raw_completion": raw_comp[0],
            "parsed_coord": parsed[0],
        }

    del voice_model, voice_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PHASE 5/5: DUAL FX+VOICE MODEL — sampled pool + 3-mode extraction
    # (conditions #4-6)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"PHASE 5/5: Dual FX+Voice Model ({args.fx_voice_num_samples} samples "
          f"-> top {args.num_modes} modes per prompt)...")
    print("=" * 60)
    dual_model, dual_tok = load_dual_model_and_tokenizer(args.dual_model_dir, device=args.device)
    ensure_pad_token(dual_tok, dual_model)

    for i, res in enumerate(results):
        prompt_text = res["prompt"]
        fx_tensor = fx_dict.get(prompt_text, torch.zeros(13, 2048, dtype=torch.bfloat16))
        voice_tensor = voice_dict.get(prompt_text, torch.zeros(25, 768, dtype=torch.bfloat16))
        dual_model.prompt_to_fx[prompt_text] = fx_tensor.to(args.device)
        dual_model.prompt_to_voice[prompt_text] = voice_tensor.to(args.device)

        print(f"[{i+1}/{len(results)}] Dual FX+Voice Model -> \"{prompt_text[:50]}...\"")
        raw_comp, parsed = generate_rollouts_for_model(
            dual_model, dual_tok, SYSTEM_PROMPT_DUAL_AUDIO, prompt_text,
            num_rollouts=args.fx_voice_num_samples, temperature=args.fx_voice_temperature,
            top_k=args.fx_voice_top_k, device=args.device, verbose=not args.quiet,
        )
        stats = compute_prediction_stats(parsed)
        modes, mode_counts, all_unique = extract_top_k_modes(
            parsed, k=args.num_modes, min_distance=args.min_mode_distance
        )
        res["predictions"]["fx_voice_model"] = {
            "raw_completions": raw_comp,
            "parsed_coords": parsed,
            "mean_coord": stats["mean_coord"],
            "std_coord": stats["std_coord"],
            "valid_count": stats["valid_count"],
            "selected_modes": modes,          # up to num_modes points, for conditions #4-6
            "selected_mode_counts": mode_counts,
            "all_unique_predictions": all_unique,  # full frequency table, most to least common
        }

    del dual_model, dual_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # SAVE OUTPUT JSON
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Saving all predictions to: {args.output_json}")
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    print("Done! Predictions successfully generated and saved. ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()