import json
import pickle
import os
import torch

# Load the JSON mapping
with open("Data_dir/prompts_and_audio_data.json", "r") as f:
    data = json.load(f)

prompt_to_dsp = {}
dsp_dir = "Data_dir/dsp_features/"

# Get list of existing DSP feature files
available_dsp_files = [f for f in os.listdir(dsp_dir) if f.endswith(".pkl")]

for entry in data:
    prompt = entry["prompt"]
    track = entry["track"]
    
    pkl_filename = track.replace(".wav", ".pkl")
    
    if pkl_filename in available_dsp_files:
        pkl_path = os.path.join(dsp_dir, pkl_filename)
        with open(pkl_path, "rb") as f:
            features = pickle.load(f)
            prompt_to_dsp[prompt] = features
    else:
        # Some prompts might not have an audio file in my trimmed set
        pass

with open("Data_dir/prompt_to_dsp.pkl", "wb") as f:
    pickle.dump(prompt_to_dsp, f)
    
print(f"Successfully created Data_dir/prompt_to_dsp.pkl with {len(prompt_to_dsp)} mappings.")
