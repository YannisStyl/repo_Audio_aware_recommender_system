# Script to extract unpooled chronological Auden-Voice features (first 5s) for each 
# audio-prompt pair in Listening_Exp/Perceptual_evaluation/prompt_audio_link.json.
#
# Environment requirement: Run in .venv_wsl Python environment in WSL.

import json
import os
import torch
import torchaudio
import soundfile as sf
import pickle
import sys

# Script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JSON_PATH = os.path.join(script_dir, "prompt_audio_link.json")
OUTPUT_PATH = os.path.join(script_dir, "perceptual_prompt_to_voice.pkl")

TARGET_SR = 16000
WINDOW_SEC = 5.0
EXPECTED_EMBED_DIM = 768


def main():
    print(f"=== Perceptual Evaluation Voice Feature Extraction (First 5s Only) ===")
    print(f"Device: {DEVICE}")
    print(f"Reading prompt-audio link from: {JSON_PATH}")

    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Cannot find link JSON at: {JSON_PATH}")

    print("Loading Auden-Voice encoder...")
    try:
        from auden.auto import AutoModel as AudenAutoModel
    except ImportError as e:
        raise ImportError(
            "The 'auden' package is required for voice feature extraction. "
            "Please run this script inside the Auden environment (.venv_wsl)."
        ) from e

    voice_encoder = AudenAutoModel.from_pretrained("AudenAI/auden-encoder-voice").to(DEVICE)
    voice_encoder.eval()
    print("Auden-Voice encoder loaded successfully.")

    try:
        import kaldifeat
    except ImportError as e:
        raise ImportError("The 'kaldifeat' package is required for feature extraction.") from e

    opts = kaldifeat.FbankOptions()
    opts.frame_opts.samp_freq = TARGET_SR
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.mel_opts.num_bins = 80
    fbank = kaldifeat.Fbank(opts)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    prompt_to_voice = {}
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

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                waveform_1d = waveform.squeeze(0).contiguous()
            except Exception as e:
                print(f"[{i+1}/{len(mapping)}] Error loading {clip_name}: {e}")
                continue

            # Slice the first 5.0 seconds once (no sliding window)
            end_sample = min(window_samples, waveform_1d.shape[0])
            window_waveform = waveform_1d[0:end_sample].contiguous()

            try:
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

                unpooled_seq = frame_embeddings.detach().cpu()

                sample_id = f"{category}/{clip_name}"
                prompt_to_voice[prompt] = {
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

            except Exception as e:
                print(f"[{i+1}/{len(mapping)}] Error extracting window for {clip_name}: {e}")
                continue

    print(f"\nSaving Voice feature dictionary ({len(prompt_to_voice)} items)...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(prompt_to_voice, f)
    print(f"Successfully saved perceptual Voice features to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
