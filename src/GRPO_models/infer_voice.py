import os
import sys
import argparse
import pickle
import torch
import numpy as np
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# Set up module resolution path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Training_Scripts.t2b_voice_GRPO import VoiceConditionedQwen
from Training_Scripts.parse_utils import parse_completion_to_coord
from Training_Scripts.prompts import SYSTEM_PROMPT_VOICE


def resolve_checkpoint_dir(checkpoint_path: str) -> str:
    """
    If checkpoint_path is a directory containing subdirectories starting with 'checkpoint-',
    returns the directory with the highest checkpoint number. Otherwise, returns checkpoint_path.
    """
    if os.path.isdir(checkpoint_path):
        subdirs = [
            d for d in os.listdir(checkpoint_path)
            if os.path.isdir(os.path.join(checkpoint_path, d)) and d.startswith("checkpoint-")
        ]
        if subdirs:
            # Sort numerically by checkpoint step
            subdirs.sort(key=lambda d: int(d.split("checkpoint-")[-1]))
            latest = os.path.join(checkpoint_path, subdirs[-1])
            print(f"[Voice Inference] Auto-detected latest checkpoint: {latest}")
            return latest
    return checkpoint_path


def load_voice_model_and_tokenizer(checkpoint_dir: str, base_model_name: str = "Qwen/Qwen3.5-0.8B", device: str = "cuda"):
    """
    Loads base LLM, applies LoRA structure, injects Voice Q-Former adapter,
    and loads weights from the specified checkpoint safetensors file.
    """
    resolved_ckpt = resolve_checkpoint_dir(checkpoint_dir)
    st_file = os.path.join(resolved_ckpt, "model.safetensors")
    
    if not os.path.exists(st_file):
        raise FileNotFoundError(f"Checkpoint weight file not found at: {st_file}")

    print(f"[Voice Inference] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_ckpt, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if "<|voice|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|voice|>"]})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    voice_token_id = tokenizer.convert_tokens_to_ids("<|voice|>")

    print(f"[Voice Inference] Loading base LLM: {base_model_name}...")
    torch_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "out_proj", "in_proj_qkv", "in_proj_b",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    base_model = get_peft_model(base_model, lora_config)

    model = VoiceConditionedQwen(
        base_model,
        use_audio=True,
        tokenizer=tokenizer,
        voice_token_id=voice_token_id,
        prompt_to_voice={},
    ).to(device)

    print(f"[Voice Inference] Loading weights from safetensors: {st_file}")
    sd = safetensors.torch.load_file(st_file, device=device)

    voice_sd = {k.removeprefix("voice_projector."): v for k, v in sd.items() if k.startswith("voice_projector.")}
    peft_sd = {
        k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): v
        for k, v in sd.items() if not k.startswith("voice_projector.")
    }

    res_peft = model.qwen.load_state_dict(peft_sd, strict=False)
    res_voice = model.voice_adapter.load_state_dict(voice_sd, strict=False)
    
    print(f"[Voice Inference] Model loaded successfully! (PEFT unmapped: {len(res_peft.unexpected_keys)}, Projector unmapped: {len(res_voice.unexpected_keys)})")
    model.eval()
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompt: str,
    voice_tensor: torch.Tensor,
    num_rollouts: int = 1,
    temperature: float = 1.0,
    top_k: int = 20,
    device: str = "cuda"
):
    """
    Executes inference for a text instruction + Voice feature tensor.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_VOICE},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)

    # Attach Voice tensor into model's prompt_to_voice cache using prompt text as key
    clean_prompt = prompt.strip()
    model.prompt_to_voice[clean_prompt] = voice_tensor

    completions = []
    parsed_coords = []

    with torch.no_grad():
        for i in range(num_rollouts):
            output_ids = model.generate(
                input_ids=input_ids,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                top_k=top_k if temperature > 0.0 else None,
                max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Remove prompt prefix
            gen_tokens = output_ids[0][input_ids.shape[1]:]
            completion_str = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            completions.append(completion_str)
            
            coord = parse_completion_to_coord(completion_str)
            parsed_coords.append(coord)

    return completions, parsed_coords


def main():
    parser = argparse.ArgumentParser(description="Inference script for Voice-conditioned GRPO model")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(script_dir, "Voice_model"), help="Path to Voice model directory or checkpoint")
    parser.add_argument("--voice_features_path", type=str, default=os.path.join(src_dir, "..", "Data_dir", "prompt_to_voice_windowed.pkl"), help="Path to pre-extracted Voice feature pickle file")
    parser.add_argument("--prompt", type=str, default=None, help="Text instruction prompt (e.g. 'Enhance vocal clarity and presence')")
    parser.add_argument("--num_rollouts", type=int, default=5, help="Number of rollouts/generations to sample (default: 5)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0, 0 for greedy)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()

    model, tokenizer = load_voice_model_and_tokenizer(args.checkpoint, device=args.device)

    # Load feature pickle
    voice_dict = {}
    feature_paths_to_try = [
        args.voice_features_path,
        os.path.join(src_dir, "..", "Listening_Exp", "Perceptual_evaluation", "perceptual_prompt_to_voice.pkl"),
        os.path.join(src_dir, "..", "Data_dir", "perceptual_prompt_to_voice.pkl"),
    ]
    for p in feature_paths_to_try:
        if p and os.path.exists(p):
            print(f"[Voice Inference] Loading feature dictionary from: {p}")
            with open(p, "rb") as f:
                raw_data = pickle.load(f)
            loaded_dict = {k.strip(): (v["tensor"] if isinstance(v, dict) and "tensor" in v else v) for k, v in raw_data.items()}
            voice_dict.update(loaded_dict)


    if args.prompt:
        prompts_to_process = [args.prompt]
    else:
        # If no prompt specified, pick a few sample prompts from the feature dict
        if voice_dict:
            prompts_to_process = list(voice_dict.keys())[:3]
            print(f"[Voice Inference] No --prompt supplied. Running sample prompts from dataset: {prompts_to_process}")
        else:
            prompts_to_process = ["Enhance the vocal clarity and reduce harsh sibilance"]

    for prompt in prompts_to_process:
        clean_p = prompt.strip()
        if clean_p in voice_dict:
            voice_tensor = voice_dict[clean_p]
        else:
            print(f"[Voice Inference] Prompt not in feature dict. Using dummy zero feature sequence [10, 768].")
            voice_tensor = torch.zeros(10, 768, dtype=torch.bfloat16)

        completions, parsed = run_inference(
            model, tokenizer, clean_p, voice_tensor,
            num_rollouts=args.num_rollouts,
            temperature=args.temperature,
            device=args.device
        )

        print("\n" + "=" * 60)
        print(f"PROMPT: \"{clean_p}\"")
        print("=" * 60)
        valid_coords = [c for c in parsed if c is not None]
        for i, (comp, coord) in enumerate(zip(completions, parsed)):
            status = f"Parsed EQ Coord [-6,6]: {coord.tolist()}" if coord is not None else "INVALID PARSE"
            print(f"  Rollout {i+1}: {comp!r:30s} -> {status}")
        
        if valid_coords:
            valid_arr = np.array(valid_coords)
            mean_coord = np.mean(valid_arr, axis=0)
            std_coord = np.std(valid_arr, axis=0)
            print("-" * 60)
            print(f"  Valid parses: {len(valid_coords)}/{len(completions)}")
            print(f"  Mean Predicted EQ Coord: [{mean_coord[0]:.2f}, {mean_coord[1]:.2f}]")
            print(f"  Spread (Std Dev):        [{std_coord[0]:.2f}, {std_coord[1]:.2f}]")
        print("=" * 60)


if __name__ == "__main__":
    main()
