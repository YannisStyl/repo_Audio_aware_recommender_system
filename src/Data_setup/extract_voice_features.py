# Script to extract unpooled chronological Auden-Voice features for each
# prompt-audio pair, using duration-adaptive-stride windowing so that each
# paraphrase of a prompt is matched to a distinct ~5s window of the source
# clip instead of all paraphrases sharing one identical feature tensor.
#
# WINDOWING SCHEME
# -----------------
# For a clip of duration D and N windows (N = number of prompt-text variants
# for that entry: the original prompt + its augmented/validation/test
# paraphrases) of fixed length WINDOW_SEC, window start times are:
#
#     start_i = i * stride,   stride = (D - WINDOW_SEC) / (N - 1),   i = 0..N-1
#
# Properties:
#   - stride scales with clip duration, so overlap is duration-adaptive
#     rather than a fixed ratio.
#   - the last window always ends exactly at D (windows always span the
#     full clip, never leave a trailing unused tail).
#   - at the WINDOW_SEC floor (D == WINDOW_SEC), all N windows degenerate to
#     the same span -- unavoidable at that boundary, flagged in debug output.
#   - at D == WINDOW_SEC * N - (N-1)*WINDOW_SEC ... i.e. once D is large
#     enough, stride reaches WINDOW_SEC and windows become contiguous with
#     zero overlap.
#   - WINDOW_SEC (5s) and the theoretical max duration this supports without
#     negative stride are hard bounds for the *scheme* to be well-defined,
#     not an expectation about typical clip length. The actual dataset here
#     is ~6-15s per clip, comfortably inside the range where this produces
#     small, non-degenerate strides.

import json
import os
import torch
import torchaudio
import soundfile as sf
import pickle
import sys
from collections import defaultdict

data_setup_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(data_setup_dir)

from generate_reward_models import augmented_prompts, iid_validation_prompts, ood_test_prompts

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JSON_PATH = "Data_dir/prompts_and_audio_data.json"
AUDIO_DIR = "Data_dir/original_audio_trimmed/"
OUTPUT_PATH = "Data_dir/prompt_to_voice_windowed.pkl"

TARGET_SR = 16000
WINDOW_SEC = 5.0
EXPECTED_EMBED_DIM = 768


def compute_window_starts_sec(duration_sec: float, num_windows: int, window_sec: float = WINDOW_SEC):
    """Duration-adaptive stride window start times, in seconds.

    Returns a list of `num_windows` start times spanning [0, duration_sec],
    each window of length `window_sec`. Degenerates to all-zero starts if
    duration_sec <= window_sec (clip at or under the floor) or if
    num_windows <= 1.
    """
    if num_windows <= 1:
        return [0.0]
    if duration_sec <= window_sec:
        return [0.0] * num_windows
    stride = (duration_sec - window_sec) / (num_windows - 1)
    return [i * stride for i in range(num_windows)]


def build_variant_texts(prompt: str) -> list:
    """Ordered list of prompt-text variants for one dataset entry: the
    original prompt followed by its paraphrases from each split dict that
    contains it (augmented -> iid_validation -> ood_test), matching the
    original script's assignment order. This determines how many windows
    are cut from the source clip.
    """
    variants = [prompt]
    for prompt_dict in [augmented_prompts, iid_validation_prompts, ood_test_prompts]:
        if prompt in prompt_dict:
            variants.extend(prompt_dict[prompt])
    return variants


print(f"Loading Auden-Voice encoder on {DEVICE}...")
try:
    from auden.auto import AutoModel as AudenAutoModel
except ImportError as e:
    raise ImportError(
        "The 'auden' package is required for voice feature extraction. "
        "Install it with: pip install auden"
    ) from e

voice_encoder = AudenAutoModel.from_pretrained("AudenAI/auden-encoder-voice")
voice_encoder = voice_encoder.to(DEVICE)
voice_encoder.eval()
print("Auden-Voice encoder loaded successfully.")

with open(JSON_PATH, 'r') as f:
    mapping = json.load(f)

prompt_to_voice = {}

# --- DEBUG tracking ---
debug_key_writers = defaultdict(list)      # prompt -> list of track_names that wrote to it
debug_variant_count_mismatch = []          # entries where variant count != 7
debug_degenerate_windows = []              # (track, duration) where D <= WINDOW_SEC
debug_shape_warning_count = 0

print("\nStarting Auden-Voice windowed feature extraction...")
with torch.no_grad():
    for entry in mapping:
        prompt = entry["prompt"]
        track_name = entry["track"]

        audio_path = os.path.join(AUDIO_DIR, track_name)
        if not os.path.exists(audio_path):
            print(f"Audio file not found, skipping: {audio_path}")
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

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform_1d = waveform.squeeze(0).contiguous()
        except Exception as e:
            print(f"Error loading {track_name}: {e}")
            continue

        duration_sec = waveform_1d.shape[0] / TARGET_SR
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
            end_sample = min(start_sample + window_samples, waveform_1d.shape[0])
            start_sample = max(0, end_sample - window_samples)  # clamp, guard rounding at the tail
            window_waveform = waveform_1d[start_sample:end_sample].contiguous()

            try:
                import kaldifeat
                opts = kaldifeat.FbankOptions()
                opts.frame_opts.samp_freq = TARGET_SR
                opts.frame_opts.dither = 0.0
                opts.frame_opts.snip_edges = False
                opts.mel_opts.num_bins = 80

                fbank = kaldifeat.Fbank(opts)
                features = fbank(window_waveform)  # [T_frames, 80]

                x = features.unsqueeze(0).to(DEVICE)
                x_lens = torch.tensor([features.shape[0]], dtype=torch.long, device=DEVICE)

                raw_output = voice_encoder(x, x_lens)

                if isinstance(raw_output, dict) and "encoder_out" in raw_output:
                    frame_embeddings = raw_output["encoder_out"]
                elif isinstance(raw_output, torch.Tensor):
                    frame_embeddings = raw_output
                elif hasattr(raw_output, 'last_hidden_state'):
                    frame_embeddings = raw_output.last_hidden_state
                elif isinstance(raw_output, (tuple, list)):
                    frame_embeddings = raw_output[0]
                else:
                    raise RuntimeError("Unrecognised Auden-Voice output type.")

                if frame_embeddings.dim() == 3:
                    frame_embeddings = frame_embeddings.squeeze(0)
                elif frame_embeddings.dim() == 1:
                    frame_embeddings = frame_embeddings.unsqueeze(0)

                if frame_embeddings.shape[-1] != EXPECTED_EMBED_DIM:
                    debug_shape_warning_count += 1
                    print(f"Unexpected feature dim {frame_embeddings.shape[-1]} "
                          f"for {track_name} window {i}.")

                unpooled_seq = frame_embeddings.detach().cpu()

                if torch.isnan(unpooled_seq).any() or torch.isinf(unpooled_seq).any():
                    print(f"[DEBUG] NaN/Inf in {track_name} window {i}")

                sample_id = f"{track_name}::w{i}"
                prompt_to_voice[variant_text] = {
                    "tensor": unpooled_seq,
                    "sample_id": sample_id,
                    "track": track_name,
                    "window_index": i,
                    "window_start_sec": round(start_sec, 4),
                    "window_duration_sec": WINDOW_SEC,
                }
                debug_key_writers[variant_text].append(sample_id)

            except Exception as e:
                print(f"Error extracting window {i} for {track_name}: {e}")
                continue

# --- DEBUG reports ---
print("\n" + "=" * 50)
print("[DEBUG] Prompt-key collision report")
print("=" * 50)
collisions = {p: w for p, w in debug_key_writers.items() if len(w) > 1}
if collisions:
    print(f"{len(collisions)} prompt key(s) written more than once "
          f"(only the LAST write survives):")
    for p, writers in collisions.items():
        print(f"   \"{p}\" <- {writers}")
else:
    print("No prompt-key collisions detected -- every prompt variant now maps "
          "to exactly one window, as intended.")

print("\n" + "=" * 50)
print("[DEBUG] Variant-count / windowing summary")
print("=" * 50)
print(f"Entries processed: {len(mapping)}")
print(f"Entries with variant count != 7: {len(debug_variant_count_mismatch)}")
for track_name, prompt, n in debug_variant_count_mismatch[:20]:
    print(f"   {track_name}: \"{prompt[:50]}\" -> {n} variants")
print(f"Entries at/under the {WINDOW_SEC}s floor (degenerate identical windows): "
      f"{len(debug_degenerate_windows)}")
for track_name, dur in debug_degenerate_windows:
    print(f"{track_name}: duration={dur:.2f}s <= {WINDOW_SEC}s floor")
print(f"Shape mismatches: {debug_shape_warning_count}")

print(f"\nSaving voice features to: {OUTPUT_PATH}")
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(prompt_to_voice, f)

# --- Provenance file: sample_id -> (track, window, timing, prompt variant) ---
# Independent of the live training lookup -- this is the stable reference
# table for the causal audio-swap test and for linking perceptual-study
# stimuli back to exactly which window/track produced them.
PROVENANCE_PATH = os.path.join(os.path.dirname(OUTPUT_PATH), "voice_sample_id_provenance.pkl")
provenance = {
    v["sample_id"]: {
        "prompt_variant": k,
        "track": v["track"],
        "window_index": v["window_index"],
        "window_start_sec": v["window_start_sec"],
        "window_duration_sec": v["window_duration_sec"],
    }
    for k, v in prompt_to_voice.items()
}
with open(PROVENANCE_PATH, "wb") as f:
    pickle.dump(provenance, f)
print(f"Saved sample-ID provenance table to: {PROVENANCE_PATH}")

print("=" * 50)
print(f"Successfully saved Auden-Voice windowed features to: {OUTPUT_PATH}")
print(f"Total Unique Prompts: {len(prompt_to_voice)}")
if prompt_to_voice:
    sample = list(prompt_to_voice.values())[0]
    print(f"Sample Sequence Shape: {sample['tensor'].shape}")
    print(f"Sample entry metadata: sample_id={sample['sample_id']!r}, "
          f"track={sample['track']!r}, window_index={sample['window_index']}")
print("=" * 50)