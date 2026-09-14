# Script to extract unpooled chronological Fx-Encoder++ features (first 5s) for each 
# audio-prompt pair in Listening_Exp/Perceptual_evaluation/prompt_audio_link.json.
#
# Environment requirement: Run in .venv_fx_plusplus_312 Python environment.

import json
import os
import torch
import torchaudio
import soundfile as sf
import pickle
import sys

# Script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Setup path for Fx-Encoder++ repository inside src/Data_setup
fx_repo_path = os.path.join(repo_root, "src", "Data_setup", "Fx-Encoder_PlusPlus")
if fx_repo_path not in sys.path:
    sys.path.append(fx_repo_path)

from fxencoder_plusplus import load_model

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JSON_PATH = os.path.join(script_dir, "prompt_audio_link.json")
OUTPUT_PATH = os.path.join(script_dir, "perceptual_prompt_to_fx.pkl")

TARGET_SR = 44100
WINDOW_SEC = 5.0
EXPECTED_EMBED_DIM = 2048


def main():
    print(f"=== Perceptual Evaluation FX Feature Extraction (First 5s Only) ===")
    print(f"Device: {DEVICE}")
    print(f"Reading prompt-audio link from: {JSON_PATH}")

    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Cannot find link JSON at: {JSON_PATH}")

    print(f"Loading Fx-Encoder++ model...")
    model = load_model('default', device=DEVICE)
    model.eval()

    hook_output = {}
    def conv6_hook(module, input_val, output_val):
        hook_output["conv6_out"] = output_val.detach().cpu()

    conv6_module = model.fx_encoder.conv_block6
    hook_handle = conv6_module.register_forward_hook(conv6_hook)
    print("Hook registered on model.fx_encoder.conv_block6.")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    prompt_to_fx = {}
    window_samples = int(round(WINDOW_SEC * TARGET_SR))

    print(f"Processing {len(mapping)} audio-prompt pairs (slicing first {WINDOW_SEC}s)...")
    with torch.no_grad():
        for i, entry in enumerate(mapping):
            prompt = entry["prompt"].strip()
            clip_name = entry["audio_clip"]
            category = entry["category"]
            descriptor = entry.get("descriptor", "")
            credits_text = entry.get("credits", "")

            audio_path = os.path.join(script_dir, category, clip_name)
            if not os.path.exists(audio_path):
                print(f"[{i+1}/{len(mapping)}] ERROR: Audio file missing at {audio_path}")
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
                print(f"[{i+1}/{len(mapping)}] Error loading {clip_name}: {e}")
                continue

            # Fx-Encoder++ requires stereo input
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]

            # Slice the first 5.0 seconds once (no sliding window)
            end_sample = min(window_samples, waveform.shape[-1])
            window_waveform = waveform[:, 0:end_sample].contiguous()

            audio_input = window_waveform.unsqueeze(0).to(DEVICE)  # [1, 2, window_samples]

            hook_output.clear()
            try:
                _ = model.get_fx_embedding(audio_input)
            except Exception as e:
                print(f"[{i+1}/{len(mapping)}] Error during forward pass for {clip_name}: {e}")
                continue

            if "conv6_out" not in hook_output:
                print(f"[{i+1}/{len(mapping)}] Warning: hook did not fire for {clip_name}")
                continue

            raw_tensor = hook_output["conv6_out"]
            unpooled_seq = torch.mean(raw_tensor, dim=3)  # mean over Freq_Bins -> [1, 2048, Time]
            unpooled_seq = unpooled_seq.squeeze(0).transpose(0, 1)  # -> [T, 2048]

            sample_id = f"{category}/{clip_name}"
            prompt_to_fx[prompt] = {
                "tensor": unpooled_seq,
                "sample_id": sample_id,
                "track": clip_name,
                "category": category,
                "descriptor": descriptor,
                "credits": credits_text,
                "prompt": prompt,
                "window_duration_sec": WINDOW_SEC,
            }

            print(f"[{i+1}/{len(mapping)}] Success: \"{prompt}\" -> {category}/{clip_name} (Feature Shape: {tuple(unpooled_seq.shape)})")

    hook_handle.remove()

    print(f"\nSaving FX feature dictionary ({len(prompt_to_fx)} items)...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(prompt_to_fx, f)
    print(f"Successfully saved perceptual FX features to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
