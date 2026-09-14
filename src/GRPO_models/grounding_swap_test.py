"""
Causal Audio-Swap Grounding Test -- FX + Voice GRPO Model
=========================================================

PURPOSE (see project_overview.md §4.1, §4.2)
---------------------------------------------
For a fixed text prompt, swap in audio features from a *different* track and
measure whether the model's predicted EQ coordinate shifts appreciably.

Three swap conditions are tested per prompt:
  1. FX-only swap    -- Voice features kept from the original track.
  2. Voice-only swap -- FX features kept from the original track.
  3. Both swapped    -- both streams replaced by the donor track's features.

STRATIFIED DOMAIN-MATCHED vs DOMAIN-MISMATCHED EVALUATION
---------------------------------------------------------
Tracks are categorized into three distinct acoustic types:
  - `pure-instrumental` : Classical, instrumental jazz, solo piano (5 tracks)
  - `pure-voice`        : Audiobook speech narration, spoken podcasts, dialogue (9 tracks)
  - `mixed`             : Vocals over music, rock/pop songs, film scenes, ambient nature (16 tracks)

For each anchor prompt:
  - **Matched Swap (In-Domain)**: Donor audio comes from the SAME category
    (e.g., Instrumental -> Instrumental). Tests track-specific sensitivity within domain.
  - **Mismatched Swap (Cross-Domain)**: Donor audio comes from the OPPOSITE category
    (e.g., Instrumental -> Voice, or Voice -> Instrumental). Tests whether the model
    detects cross-domain incompatibility (should produce larger L2 shifts and sharper density drops).

SCOPE (project_overview.md §4.1)
---------------------------------
Runs on the validation set (held-out paraphrase per concept, defined by
reward_models_augmented_val.pkl). This supports a local, in-distribution
causal grounding claim -- NOT generalisation to novel audio.

INPUTS
------
  --checkpoint        Path to the FX+Voice model directory or checkpoint-N.
  --fx_features       prompt_to_fx_windowed.pkl
  --voice_features    prompt_to_voice_windowed.pkl
  --val_reward_models reward_models_augmented_val.pkl  (defines the val set)
  --num_rollouts      Rollouts per (prompt, audio) pair.   Default: 1 (greedy).
  --temperature       Sampling temperature.  0 = greedy.   Default: 0.
  --num_donors        Donor tracks per domain condition.   Default: 3.
  --seed              RNG seed for reproducible selection. Default: 42.
  --output_json       Path to write full results JSON (optional).
  --device            cuda / cpu.
"""

# ============================================================
# 0. ENVIRONMENT
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import pickle
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# --- Module path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir    = os.path.abspath(os.path.join(script_dir, ".."))
repo_root  = os.path.abspath(os.path.join(src_dir,    ".."))
for _p in [src_dir, repo_root, os.path.join(src_dir, "Data_setup")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Training_Scripts.t2b_fx_and_voice_GRPO import DualStreamConditionedQwen   # noqa: E402
from Training_Scripts.parse_utils import parse_completion_to_coord              # noqa: E402
from Training_Scripts.prompts import SYSTEM_PROMPT_DUAL_AUDIO                  # noqa: E402
from Data_setup.generate_reward_models import NormalizedPreferenceDensity       # noqa: E402,F401 (pickle)


# ============================================================
# 1. TRACK CATEGORIZATION DICTIONARY
# ============================================================

TRACK_CATEGORIES: dict[str, str] = {
    # pure-instrumental (5 tracks)
    "AlexandreDesplatTheI.wav": "pure-instrumental",
    "ESQUIVELOyeNegra.wav":     "pure-instrumental",
    "QuartangoMilongaDiab.wav": "pure-instrumental",
    "TresRythme.wav":           "pure-instrumental",
    "VladimirAshkenazyJSB.wav": "pure-instrumental",

    # pure-voice (9 tracks)
    "dev228.wav":               "pure-voice",
    "dev276.wav":               "pure-voice",
    "dev1197.wav":              "pure-voice",
    "dev13101311.wav":          "pure-voice",
    "ArtnCompany.wav":          "pure-voice",
    "BorderlessPodcast.wav":    "pure-voice",
    "LinearDigressions.wav":    "pure-voice",
    "CosmosLaundromat1.wav":    "pure-voice",
    "Sintel4.wav":              "pure-voice",

    # mixed (16 tracks)
    "BobbyMcFerrinATrainL.wav": "mixed",
    "ClubForFiveBrothersi.wav": "mixed",
    "GlennHughesYoungLust.wav": "mixed",
    "KingsOfLeonSexonFire.wav": "mixed",
    "LaBottineSourianteLa.wav": "mixed",
    "MilkyChanceStolenDan.wav": "mixed",
    "MobyPorcelain.wav":        "mixed",
    "OneLoveInMyLifetime.wav":  "mixed",
    "PassengerLetHerGo.wav":    "mixed",
    "Penguins.wav":             "mixed",
    "PentatonixHavana.wav":     "mixed",
    "Tearsofsteel5.wav":        "mixed",
    "ZhuFaded.wav":             "mixed",
    "ambientforestsoundsc.wav": "mixed",
    "calmzenriverflowing2.wav": "mixed",
    "seaandseagullwave593.wav": "mixed",
}


def get_track_category(track_name: str) -> str:
    """Return category for track: 'pure-instrumental', 'pure-voice', or 'mixed'."""
    clean = os.path.basename(track_name).strip()
    if clean in TRACK_CATEGORIES:
        return TRACK_CATEGORIES[clean]
    # heuristic fallbacks
    if clean.startswith("dev"):
        return "pure-voice"
    return "mixed"


# ============================================================
# 2. MODEL LOADING
# ============================================================

def _resolve_checkpoint(checkpoint_path: str) -> str:
    """Return the numerically latest checkpoint-N subdirectory, or path as-is."""
    if os.path.isdir(checkpoint_path):
        subdirs = [
            d for d in os.listdir(checkpoint_path)
            if os.path.isdir(os.path.join(checkpoint_path, d))
            and d.startswith("checkpoint-")
        ]
        if subdirs:
            subdirs.sort(key=lambda d: int(d.split("checkpoint-")[-1]))
            latest = os.path.join(checkpoint_path, subdirs[-1])
            print(f"[SwapTest] Auto-detected latest checkpoint: {latest}")
            return latest
    return checkpoint_path


def load_model(checkpoint_dir: str,
               base_model_name: str = "Qwen/Qwen3.5-0.8B",
               device: str = "cuda"):
    """
    Load the FX+Voice GRPO model from a safetensors checkpoint.
    Returns (model, tokenizer).
    """
    ckpt    = _resolve_checkpoint(checkpoint_dir)
    st_file = os.path.join(ckpt, "model.safetensors")
    if not os.path.exists(st_file):
        raise FileNotFoundError(f"[SwapTest] Checkpoint weights not found: {st_file}")

    print(f"[SwapTest] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    for tok in ["<|audio|>", "<|voice|>"]:
        if tok not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [tok]})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    fx_token_id    = tokenizer.convert_tokens_to_ids("<|audio|>")
    voice_token_id = tokenizer.convert_tokens_to_ids("<|voice|>")

    torch_dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    print(f"[SwapTest] Loading base LLM: {base_model_name} ({torch_dtype})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch_dtype, trust_remote_code=True
    )
    base_model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "out_proj", "in_proj_qkv", "in_proj_b",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    base_model = get_peft_model(base_model, lora_cfg)

    model = DualStreamConditionedQwen(
        base_model, use_audio=True, tokenizer=tokenizer,
        fx_token_id=fx_token_id, voice_token_id=voice_token_id,
        prompt_to_fx={}, prompt_to_voice={},
    ).to(device)

    print(f"[SwapTest] Loading weights from: {st_file}")
    sd       = safetensors.torch.load_file(st_file, device=device)
    fx_sd    = {k.removeprefix("fx_projector."):    v for k, v in sd.items() if k.startswith("fx_projector.")}
    voice_sd = {k.removeprefix("voice_projector."): v for k, v in sd.items() if k.startswith("voice_projector.")}
    peft_sd  = {
        k.replace(".lora_A.weight", ".lora_A.default.weight")
         .replace(".lora_B.weight", ".lora_B.default.weight"): v
        for k, v in sd.items()
        if not (k.startswith("fx_projector.") or k.startswith("voice_projector."))
    }
    r_peft  = model.qwen.load_state_dict(peft_sd,   strict=False)
    r_fx    = model.fx_adapter.load_state_dict(fx_sd,    strict=False)
    r_voice = model.voice_adapter.load_state_dict(voice_sd, strict=False)
    print(
        f"[SwapTest] Weights loaded. "
        f"(PEFT unexpected={len(r_peft.unexpected_keys)}, "
        f"FX unexpected={len(r_fx.unexpected_keys)}, "
        f"Voice unexpected={len(r_voice.unexpected_keys)})"
    )
    model.eval()
    return model, tokenizer


# ============================================================
# 3. INFERENCE HELPER
# ============================================================

def _infer(
    model,
    tokenizer,
    prompt: str,
    fx_tensor: torch.Tensor,
    voice_tensor: torch.Tensor,
    num_rollouts: int,
    temperature: float,
    top_k: int,
    device: str,
) -> list[np.ndarray | None]:
    """
    Run inference for one (prompt, fx, voice) triple.
    Returns a list of parsed coords ([x,y] in [-6,6]) or None per rollout.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DUAL_AUDIO},
        {"role": "user",   "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(device)

    clean = prompt.strip()
    model.prompt_to_fx[clean]    = fx_tensor
    model.prompt_to_voice[clean] = voice_tensor

    coords = []
    with torch.no_grad():
        for _ in range(num_rollouts):
            out = model.generate(
                input_ids=input_ids,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                top_k=top_k if temperature > 0.0 else None,
                max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen = out[0][input_ids.shape[1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()
            coords.append(parse_completion_to_coord(text))
    return coords


def _mean_coord(coords: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [c for c in coords if c is not None]
    return np.mean(valid, axis=0) if valid else None


def _l2(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def _density(reward_fn, coord: np.ndarray | None) -> float | None:
    """Evaluate NormalizedPreferenceDensity at coord ([-6,6] scale)."""
    if coord is None or reward_fn is None:
        return None
    try:
        return float(reward_fn(coord).item())
    except Exception:
        return None


# ============================================================
# 4. DATA LOADING
# ============================================================

def _load_features(path: str, label: str) -> dict:
    """Load windowed feature pickle."""
    print(f"[SwapTest] Loading {label} features: {path}")
    with open(path, "rb") as f:
        raw = pickle.load(f)
    out = {}
    for k, v in raw.items():
        key = k.strip()
        out[key] = v if isinstance(v, dict) else {"tensor": v, "track": "unknown"}
    print(f"[SwapTest]   -> {len(out)} prompts ({label})")
    return out


def _load_reward_models(path: str) -> dict:
    """Load reward-model pickle."""
    print(f"[SwapTest] Loading reward models: {path}")
    with open(path, "rb") as f:
        rm = pickle.load(f)
    print(f"[SwapTest]   -> {len(rm)} reward models")
    return rm


# ============================================================
# 5. STRATIFIED DONOR SELECTION
# ============================================================

def _select_stratified_donors(
    anchor_prompt: str,
    anchor_track: str,
    anchor_category: str,
    fx_meta: dict,
    num_donors: int,
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """
    Select up to `num_donors` matched (in-domain) and up to `num_donors` mismatched (cross-domain)
    prompt keys from fx_meta.

    Returns list of tuples: (donor_prompt, donor_category, domain_match_type)
      where domain_match_type is 'matched' or 'mismatched'.
    """
    # Group available prompt keys by track and category
    track_to_prompts = defaultdict(list)
    for p, meta in fx_meta.items():
        if p == anchor_prompt:
            continue
        t = meta.get("track", "unknown") if isinstance(meta, dict) else "unknown"
        if t != anchor_track and t != "unknown":
            track_to_prompts[t].append(p)

    matched_tracks = []
    mismatched_tracks = []

    for t in track_to_prompts.keys():
        t_cat = get_track_category(t)
        if t_cat == anchor_category:
            matched_tracks.append(t)
        else:
            if anchor_category == "pure-instrumental" and t_cat == "pure-voice":
                mismatched_tracks.append(t)
            elif anchor_category == "pure-voice" and t_cat == "pure-instrumental":
                mismatched_tracks.append(t)
            elif anchor_category == "mixed":
                # For mixed anchors, any pure-voice or pure-instrumental is a cross-domain mismatch
                mismatched_tracks.append(t)
            else:
                mismatched_tracks.append(t)

    # Sample matched tracks without replacement
    rng.shuffle(matched_tracks)
    sampled_matched_tracks = matched_tracks[:num_donors]

    # Sample mismatched tracks without replacement
    rng.shuffle(mismatched_tracks)
    sampled_mismatched_tracks = mismatched_tracks[:num_donors]

    donors = []
    # Pick 1 prompt per selected matched track
    for t in sampled_matched_tracks:
        cand_p = rng.choice(track_to_prompts[t])
        donors.append((cand_p, get_track_category(t), "matched"))

    # Pick 1 prompt per selected mismatched track
    for t in sampled_mismatched_tracks:
        cand_p = rng.choice(track_to_prompts[t])
        donors.append((cand_p, get_track_category(t), "mismatched"))

    return donors


# ============================================================
# 6. SWAP TEST FOR ONE PROMPT
# ============================================================

def _swap_test_one_prompt(
    prompt: str,
    anchor_fx: torch.Tensor,
    anchor_voice: torch.Tensor,
    anchor_track: str,
    anchor_category: str,
    donor_tuples: list[tuple[str, str, str]],
    fx_meta: dict,
    voice_meta: dict,
    reward_fn,
    model,
    tokenizer,
    num_rollouts: int,
    temperature: float,
    top_k: int,
    device: str,
) -> dict:
    """
    Run baseline + three swap conditions for one prompt across stratified donors.
    """
    # ---- Baseline: original audio ----
    baseline_coords = _infer(
        model, tokenizer, prompt,
        anchor_fx, anchor_voice,
        num_rollouts, temperature, top_k, device,
    )
    baseline_coord   = _mean_coord(baseline_coords)
    baseline_density = _density(reward_fn, baseline_coord)

    donors = []
    for donor_prompt, donor_category, domain_match in donor_tuples:
        d_fx_meta    = fx_meta.get(donor_prompt,    {})
        d_voice_meta = voice_meta.get(donor_prompt, {})

        d_fx_tensor    = d_fx_meta.get("tensor")    if isinstance(d_fx_meta,    dict) else d_fx_meta
        d_voice_tensor = d_voice_meta.get("tensor") if isinstance(d_voice_meta, dict) else d_voice_meta
        d_track        = d_fx_meta.get("track", "unknown") if isinstance(d_fx_meta, dict) else "unknown"

        if d_fx_tensor is None or d_voice_tensor is None:
            continue

        # Condition 1: FX swapped, Voice kept
        c_fx = _mean_coord(_infer(
            model, tokenizer, prompt,
            d_fx_tensor, anchor_voice,
            num_rollouts, temperature, top_k, device,
        ))

        # Condition 2: Voice swapped, FX kept
        c_voice = _mean_coord(_infer(
            model, tokenizer, prompt,
            anchor_fx, d_voice_tensor,
            num_rollouts, temperature, top_k, device,
        ))

        # Condition 3: Both swapped
        c_both = _mean_coord(_infer(
            model, tokenizer, prompt,
            d_fx_tensor, d_voice_tensor,
            num_rollouts, temperature, top_k, device,
        ))

        donors.append({
            "donor_prompt":        donor_prompt,
            "donor_track":         d_track,
            "donor_category":      donor_category,
            "domain_match":        domain_match,
            # coords
            "fx_swap_coord":       c_fx.tolist()    if c_fx    is not None else None,
            "voice_swap_coord":    c_voice.tolist() if c_voice is not None else None,
            "both_swap_coord":     c_both.tolist()  if c_both  is not None else None,
            # L2 shifts from baseline
            "l2_shift_fx_swap":    _l2(baseline_coord, c_fx),
            "l2_shift_voice_swap": _l2(baseline_coord, c_voice),
            "l2_shift_both_swap":  _l2(baseline_coord, c_both),
            # density scores
            "density_baseline":    baseline_density,
            "density_fx_swap":     _density(reward_fn, c_fx),
            "density_voice_swap":  _density(reward_fn, c_voice),
            "density_both_swap":   _density(reward_fn, c_both),
        })

    return {
        "prompt":           prompt,
        "anchor_track":     anchor_track,
        "anchor_category":  anchor_category,
        "baseline_coord":   baseline_coord.tolist() if baseline_coord is not None else None,
        "baseline_density": baseline_density,
        "donors":           donors,
    }


# ============================================================
# 7. REPORTING & STRATIFIED SUMMARIES
# ============================================================

def _print_prompt_result(result: dict, idx: int, total: int) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  [{idx+1}/{total}] {result['prompt']!r}")
    print(f"  Anchor track    : {result['anchor_track']} ({result['anchor_category']})")
    b_coord = result["baseline_coord"]
    b_dens  = result["baseline_density"]
    dens_str = f"{b_dens:.4f}" if b_dens is not None else "N/A"
    print(f"  Baseline coord  : {b_coord}  density={dens_str}")
    print(f"{sep}")

    for d in result["donors"]:
        match_tag = f"[{d['domain_match'].upper()}]"
        print(f"  Donor {match_tag:<13} [{d['donor_track']} | {d['donor_category']}] -> {d['donor_prompt'][:50]!r}")
        for label, coord_key, shift_key, dens_key in [
            ("FX-swap",    "fx_swap_coord",    "l2_shift_fx_swap",    "density_fx_swap"),
            ("Voice-swap", "voice_swap_coord", "l2_shift_voice_swap", "density_voice_swap"),
            ("Both-swap",  "both_swap_coord",  "l2_shift_both_swap",  "density_both_swap"),
        ]:
            coord = d.get(coord_key)
            shift = d.get(shift_key)
            dens  = d.get(dens_key)
            s_str = f"{shift:.4f}" if shift is not None else "N/A (parse fail)"
            d_str = f"{dens:.4f}"  if dens  is not None else "N/A"
            print(f"    {label:12s}  coord={coord}  L2-shift={s_str}  density={d_str}")
        print()


def _print_summary(all_results: list[dict]) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print("  GROUNDING SWAP TEST -- STRATIFIED AGGREGATE SUMMARY")
    print(f"{sep}")

    baseline_dens = [r["baseline_density"] for r in all_results if r["baseline_density"] is not None]
    base_mean = np.mean(baseline_dens) if baseline_dens else 0.0
    base_std  = np.std(baseline_dens) if baseline_dens else 0.0

    print(f"\n  Baseline Anchor Preference Density: {base_mean:.4f} +/- {base_std:.4f} (N={len(baseline_dens)})")

    # 1. Overall comparison: Matched (In-Domain) vs Mismatched (Cross-Domain)
    domain_groups = {"matched": defaultdict(list), "mismatched": defaultdict(list), "all": defaultdict(list)}
    dens_groups   = {"matched": defaultdict(list), "mismatched": defaultdict(list), "all": defaultdict(list)}

    # 2. Anchor category stratification
    cat_shifts = defaultdict(lambda: defaultdict(list))
    cat_dens   = defaultdict(lambda: defaultdict(list))

    for r in all_results:
        anc_cat = r["anchor_category"]
        for d in r["donors"]:
            dm = d.get("domain_match", "all")
            for cond in ["fx_swap", "voice_swap", "both_swap"]:
                s  = d.get(f"l2_shift_{cond}")
                dv = d.get(f"density_{cond}")
                if s is not None:
                    domain_groups[dm][cond].append(s)
                    domain_groups["all"][cond].append(s)
                    cat_shifts[anc_cat][f"{dm}_{cond}"].append(s)
                if dv is not None:
                    dens_groups[dm][cond].append(dv)
                    dens_groups["all"][cond].append(dv)
                    cat_dens[anc_cat][f"{dm}_{cond}"].append(dv)

    print(f"\n  ----------------------------------------------------------------------------")
    print(f"  1. DOMAIN MATCH vs MISMATCH COMPARISON")
    print(f"  ----------------------------------------------------------------------------")
    print(f"  {'Condition / Domain':<32} {'L2 Shift (Mean +/- Std)':<26} {'Density (Mean +/- Std)':<24} {'N':>4}")
    print(f"  {'-' * 88}")

    for cond, label in [
        ("fx_swap",    "FX-only swap"),
        ("voice_swap", "Voice-only swap"),
        ("both_swap",  "Both swapped"),
    ]:
        print(f"  [{label}]")
        for dm, dm_title in [("matched", "In-Domain (Matched)"), ("mismatched", "Cross-Domain (Mismatched)"), ("all", "Overall")]:
            s_vals = domain_groups[dm][cond]
            d_vals = dens_groups[dm][cond]
            s_str = f"{np.mean(s_vals):.4f} +/- {np.std(s_vals):.4f}" if s_vals else "N/A"
            d_str = f"{np.mean(d_vals):.4f} +/- {np.std(d_vals):.4f}" if d_vals else "N/A"
            n_cnt = len(s_vals)
            print(f"    {dm_title:<30} {s_str:<26} {d_str:<24} {n_cnt:>4d}")
        print()

    print(f"\n  ----------------------------------------------------------------------------")
    print(f"  2. STRATIFIED BREAKDOWN BY ANCHOR CATEGORY")
    print(f"  ----------------------------------------------------------------------------")

    for cat_name in ["pure-instrumental", "pure-voice", "mixed"]:
        print(f"\n  >> Anchor Category: {cat_name.upper()}")
        print(f"  {'Swap Condition':<30} {'L2 Shift (Mean +/- Std)':<26} {'Density (Mean +/- Std)':<24} {'N':>4}")
        print(f"  {'-' * 84}")
        for cond, label in [("fx_swap", "FX-swap"), ("voice_swap", "Voice-swap"), ("both_swap", "Both-swap")]:
            for dm, dm_label in [("matched", "Matched"), ("mismatched", "Mismatched")]:
                key = f"{dm}_{cond}"
                s_vals = cat_shifts[cat_name][key]
                d_vals = cat_dens[cat_name][key]
                s_str = f"{np.mean(s_vals):.4f} +/- {np.std(s_vals):.4f}" if s_vals else "N/A"
                d_str = f"{np.mean(d_vals):.4f} +/- {np.std(d_vals):.4f}" if d_vals else "N/A"
                print(f"    {label + ' (' + dm_label + ')':<28} {s_str:<26} {d_str:<24} {len(s_vals):>4d}")

    print(f"\n{sep}")
    print("  INTERPRETATION GUIDELINES:")
    print("    * Mismatched > Matched L2 shift -> Model exhibits strong domain-level grounding.")
    print("    * Mismatched density << Matched density -> Cross-domain audio incurs severe penalty.")
    print("    * Pure-instrumental anchor: FX-swap shift should dominate Voice-swap shift.")
    print("    * Pure-voice anchor: Voice-swap shift should dominate FX-swap shift.")
    print(f"{sep}\n")


# ============================================================
# 8. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stratified causal audio-swap grounding test for FX+Voice GRPO model."
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(script_dir, "FX_Voice_model"),
        help="Path to the FX+Voice model directory or a specific checkpoint-N subdirectory.",
    )
    parser.add_argument(
        "--fx_features", type=str,
        default=os.path.join(repo_root, "Data_dir", "prompt_to_fx_windowed.pkl"),
        help="Path to prompt_to_fx_windowed.pkl.",
    )
    parser.add_argument(
        "--voice_features", type=str,
        default=os.path.join(repo_root, "Data_dir", "prompt_to_voice_windowed.pkl"),
        help="Path to prompt_to_voice_windowed.pkl.",
    )
    parser.add_argument(
        "--val_reward_models", type=str,
        default=os.path.join(repo_root, "Data_dir", "reward_models", "reward_models_augmented_val.pkl"),
        help="Validation reward-model pickle -- defines which prompts form the val set.",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=1,
        help="Rollouts per (prompt, audio) pair. Default: 1 (greedy).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature. 0 = greedy (default).",
    )
    parser.add_argument(
        "--top_k", type=int, default=20,
        help="Top-k for stochastic sampling (ignored when temperature=0). Default: 20.",
    )
    parser.add_argument(
        "--num_donors", type=int, default=3,
        help="Number of matched donors AND number of mismatched donors per prompt. Default: 3.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducible donor selection. Default: 42.",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Path to write full results JSON (optional).",
    )
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model ID for the base LLM.",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device. Default: cuda if available, else cpu.",
    )

    args = parser.parse_args()
    rng  = random.Random(args.seed)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, tokenizer = load_model(args.checkpoint, args.base_model, args.device)

    # ------------------------------------------------------------------
    # Load feature dicts and reward models
    # ------------------------------------------------------------------
    fx_meta    = _load_features(args.fx_features,    "FX")
    voice_meta = _load_features(args.voice_features, "Voice")
    val_rm     = _load_reward_models(args.val_reward_models)

    # ------------------------------------------------------------------
    # Build val set: prompts present in all three dicts
    # ------------------------------------------------------------------
    val_prompts = sorted(
        set(val_rm.keys()) & set(fx_meta.keys()) & set(voice_meta.keys())
    )
    print(
        f"\n[SwapTest] Effective val-set: {len(val_prompts)} prompts "
        f"(reward models ∩ FX features ∩ Voice features)"
    )
    if not val_prompts:
        print("[SwapTest] ERROR: No prompts in intersection -- check your pickle paths.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run stratified swap test
    # ------------------------------------------------------------------
    all_results = []
    for idx, prompt in enumerate(val_prompts):
        meta_fx      = fx_meta[prompt]
        meta_voice   = voice_meta[prompt]
        anchor_track = meta_fx.get("track", "unknown") if isinstance(meta_fx, dict) else "unknown"
        anchor_cat   = get_track_category(anchor_track)
        anchor_fx    = meta_fx["tensor"]    if isinstance(meta_fx,    dict) else meta_fx
        anchor_voice = meta_voice["tensor"] if isinstance(meta_voice, dict) else meta_voice
        reward_fn    = val_rm[prompt].get("positive_reward")  # NormalizedPreferenceDensity

        print(f"\n[SwapTest] [{idx+1}/{len(val_prompts)}] {prompt!r}")
        print(f"           Anchor track: {anchor_track} | Category: {anchor_cat}")

        donor_tuples = _select_stratified_donors(
            anchor_prompt=prompt,
            anchor_track=anchor_track,
            anchor_category=anchor_cat,
            fx_meta=fx_meta,
            num_donors=args.num_donors,
            rng=rng,
        )
        if not donor_tuples:
            print(f"  [SwapTest] WARNING: No valid donors found -- skipping.")
            continue

        result = _swap_test_one_prompt(
            prompt=prompt,
            anchor_fx=anchor_fx,
            anchor_voice=anchor_voice,
            anchor_track=anchor_track,
            anchor_category=anchor_cat,
            donor_tuples=donor_tuples,
            fx_meta=fx_meta,
            voice_meta=voice_meta,
            reward_fn=reward_fn,
            model=model,
            tokenizer=tokenizer,
            num_rollouts=args.num_rollouts,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
        all_results.append(result)
        _print_prompt_result(result, idx, len(val_prompts))

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    _print_summary(all_results)

    # ------------------------------------------------------------------
    # Optional JSON dump
    # ------------------------------------------------------------------
    if args.output_json:
        out_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[SwapTest] Full stratified results saved to: {out_path}")


if __name__ == "__main__":
    main()
