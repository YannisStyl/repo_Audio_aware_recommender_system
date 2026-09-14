# Script to extract unpooled chronological Fx-Encoder++ features for each
# prompt-audio pair, using the same duration-adaptive-stride windowing scheme
# as extract_voice_features_windowed.py, so each paraphrase of a prompt gets
# a distinct ~5s window of the source clip.
#
# See extract_voice_features_windowed.py header for the full windowing
# rationale; the scheme is identical here, just applied to stereo 44.1kHz
# audio instead of mono 16kHz.

import json
import os
import torch
import torchaudio
import soundfile as sf
import pickle
import sys
from collections import defaultdict

fx_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Fx-Encoder_PlusPlus"))
sys.path.append(fx_repo_path)

from fxencoder_plusplus import load_model
from generate_reward_models import augmented_prompts, iid_validation_prompts, ood_test_prompts

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JSON_PATH = "Data_dir/prompts_and_audio_data.json"
AUDIO_DIR = "Data_dir/original_audio_trimmed/"
OUTPUT_PATH = "Data_dir/prompt_to_fx_windowed.pkl"
TARGET_SR = 44100
WINDOW_SEC = 5.0
EXPECTED_EMBED_DIM = 2048


def compute_window_starts_sec(duration_sec: float, num_windows: int, window_sec: float = WINDOW_SEC):
    """Duration-adaptive stride window start times, in seconds. See
    extract_voice_features_windowed.py for the full rationale."""
    if num_windows <= 1:
        return [0.0]
    if duration_sec <= window_sec:
        return [0.0] * num_windows
    stride = (duration_sec - window_sec) / (num_windows - 1)
    return [i * stride for i in range(num_windows)]


def build_variant_texts(prompt: str) -> list:
    """Ordered list of prompt-text variants for one dataset entry: original
    prompt + paraphrases from each split dict that contains it, in the same
    order the original (non-windowed) script assigned them."""
    variants = [prompt]
    for prompt_dict in [augmented_prompts, iid_validation_prompts, ood_test_prompts]:
        if prompt in prompt_dict:
            variants.extend(prompt_dict[prompt])
    return variants


print(f"Loading Fx-Encoder++ on {DEVICE}...")
model = load_model('default', device=DEVICE)
model.eval()

hook_output = {}
def conv6_hook(module, input_val, output_val):
    hook_output["conv6_out"] = output_val.detach().cpu()

conv6_module = model.fx_encoder.conv_block6
hook_handle = conv6_module.register_forward_hook(conv6_hook)
print("Programmatic hook successfully registered on model.fx_encoder.conv_block6")

with open(JSON_PATH, 'r') as f:
    mapping = json.load(f)

prompt_to_fx = {}

# --- DEBUG tracking ---
debug_key_writers = defaultdict(list)
debug_variant_count_mismatch = []
debug_degenerate_windows = []
debug_shape_warning_count = 0
debug_stale_hook_count = 0
debug_seen_fingerprints = {}

print("\nStarting Fx-Encoder++ windowed feature extraction...")
with torch.no_grad():
    for entry in mapping:
        prompt = entry["prompt"]
        track_name = entry["track"]

        audio_path = os.path.join(AUDIO_DIR, track_name)
        if not os.path.exists(audio_path):
            continue

        try:
            data, sr = sf.read(audio_path)
            waveform = torch.from_numpy(data).float()

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)

            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                waveform = resampler(waveform)
        except Exception as e:
            print(f"Error loading {track_name} via soundfile: {e}")
            continue

        # Fx-Encoder++ requires stereo input
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        duration_sec = waveform.shape[-1] / TARGET_SR
        variant_texts = build_variant_texts(prompt)
        num_windows = len(variant_texts)

        if num_windows != 7:
            debug_variant_count_mismatch.append((track_name, prompt, num_windows))
        if duration_sec <= WINDOW_SEC:
            debug_degenerate_windows.append((track_name, duration_sec))

        window_starts = compute_window_starts_sec(duration_sec, num_windows)
        window_samples = int(round(WINDOW_SEC * TARGET_SR))

        print(f"Processing: \"{prompt}\" -> {track_name}  "
              f"(duration={duration_sec:.2f}s, {num_windows} windows, "
              f"stride={(window_starts[1]-window_starts[0]) if num_windows > 1 else 0:.3f}s)")

        for i, (variant_text, start_sec) in enumerate(zip(variant_texts, window_starts)):
            start_sample = int(round(start_sec * TARGET_SR))
            end_sample = min(start_sample + window_samples, waveform.shape[-1])
            start_sample = max(0, end_sample - window_samples)
            window_waveform = waveform[:, start_sample:end_sample].contiguous()

            audio_input = window_waveform.unsqueeze(0).to(DEVICE)  # [1, 2, window_samples]

            hook_output.clear()
            try:
                _ = model.get_fx_embedding(audio_input)
            except Exception as e:
                print(f"Error during forward pass for {track_name} window {i}: {e}")
                continue

            if "conv6_out" not in hook_output:
                print(f"Warning: hook did not fire for {track_name} window {i}")
                continue

            raw_tensor = hook_output["conv6_out"]
            unpooled_seq = torch.mean(raw_tensor, dim=3)  # [Batch*Channels, 2048, Time]
            leading_dim = unpooled_seq.shape[0]
            unpooled_seq = unpooled_seq.squeeze(0).transpose(0, 1)  # -> [T, 2048]

            if leading_dim != 1:
                debug_shape_warning_count += 1
                print(f"[DEBUG] squeeze(0) no-op for {track_name} window {i}: "
                      f"leading dim was {leading_dim}, resulting shape "
                      f"{tuple(unpooled_seq.shape)}")
            elif unpooled_seq.dim() != 2 or unpooled_seq.shape[-1] != EXPECTED_EMBED_DIM:
                debug_shape_warning_count += 1
                print(f"[DEBUG] Unexpected shape for {track_name} window {i}: "
                      f"{tuple(unpooled_seq.shape)}")

            if torch.isnan(unpooled_seq).any() or torch.isinf(unpooled_seq).any():
                print(f"[DEBUG] NaN/Inf in {track_name} window {i}")

            fingerprint = (unpooled_seq.shape[0], round(unpooled_seq.sum().item(), 4),
                           round(unpooled_seq.flatten()[0].item(), 6))
            if fingerprint in debug_seen_fingerprints:
                debug_stale_hook_count += 1
                print(f"[DEBUG] Duplicate/stale fingerprint: {track_name} window {i} "
                      f"matches {debug_seen_fingerprints[fingerprint]}")
            else:
                debug_seen_fingerprints[fingerprint] = f"{track_name}[w{i}]"

            sample_id = f"{track_name}::w{i}"
            prompt_to_fx[variant_text] = {
                "tensor": unpooled_seq,
                "sample_id": sample_id,
                "track": track_name,
                "window_index": i,
                "window_start_sec": round(start_sec, 4),
                "window_duration_sec": WINDOW_SEC,
            }
            debug_key_writers[variant_text].append(sample_id)

hook_handle.remove()
print("\nExtraction complete. Hook removed.")

# --- DEBUG reports ---
print("\n" + "=" * 50)
print("[DEBUG] Prompt-key collision report")
print("=" * 50)
collisions = {p: w for p, w in debug_key_writers.items() if len(w) > 1}
if collisions:
    print(f"{len(collisions)} prompt key(s) written more than once:")
    for p, writers in collisions.items():
        print(f"   \"{p}\" <- {writers}")
else:
    print("No prompt-key collisions detected.")

print("\n" + "=" * 50)
print("[DEBUG] Variant-count / windowing / shape summary")
print("=" * 50)
print(f"Entries processed: {len(mapping)}")
print(f"Entries with variant count != 7: {len(debug_variant_count_mismatch)}")
for track_name, prompt, n in debug_variant_count_mismatch[:20]:
    print(f"   {track_name}: \"{prompt[:50]}\" -> {n} variants")
print(f"Entries at/under the {WINDOW_SEC}s floor (degenerate identical windows): "
      f"{len(debug_degenerate_windows)}")
for track_name, dur in debug_degenerate_windows:
    print(f"     {track_name}: duration={dur:.2f}s <= {WINDOW_SEC}s floor")
print(f"Shape mismatches: {debug_shape_warning_count}")
print(f"Duplicate/stale fingerprints: {debug_stale_hook_count}")

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(prompt_to_fx, f)

# --- Provenance file: sample_id -> (track, window, timing, prompt variant) ---
PROVENANCE_PATH = os.path.join(os.path.dirname(OUTPUT_PATH), "fx_sample_id_provenance.pkl")
provenance = {
    v["sample_id"]: {
        "prompt_variant": k,
        "track": v["track"],
        "window_index": v["window_index"],
        "window_start_sec": v["window_start_sec"],
        "window_duration_sec": v["window_duration_sec"],
    }
    for k, v in prompt_to_fx.items()
}
with open(PROVENANCE_PATH, "wb") as f:
    pickle.dump(provenance, f)
print(f"Saved sample-ID provenance table to: {PROVENANCE_PATH}")

print("=" * 50)
print(f"Successfully saved windowed Fx-Encoder++ features to: {OUTPUT_PATH}")
print(f"Total Unique Prompts: {len(prompt_to_fx)}")
if prompt_to_fx:
    sample = list(prompt_to_fx.values())[0]
    print(f"Sample Sequence Shape: {sample['tensor'].shape}")
    print(f"Sample entry metadata: sample_id={sample['sample_id']!r}, "
          f"track={sample['track']!r}, window_index={sample['window_index']}")
print("=" * 50)