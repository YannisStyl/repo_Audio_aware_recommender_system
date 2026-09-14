import os
import sys
import argparse
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

from Training_Scripts.parse_utils import parse_completion_to_coord
from Training_Scripts.prompts import SYSTEM_PROMPT_NO_AUDIO


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
            print(f"[Text Inference] Auto-detected latest checkpoint: {latest}")
            return latest
    return checkpoint_path


def load_text_model_and_tokenizer(checkpoint_dir: str, base_model_name: str = "Qwen/Qwen3.5-0.8B", device: str = "cuda"):
    """
    Loads the base LLM, applies the LoRA structure, and loads weights from
    the specified checkpoint safetensors file.

    The text-only model has no Q-Former adapter and no audio special tokens --
    it is a pure LoRA fine-tune of Qwen on the EQ recommendation task.
    """
    resolved_ckpt = resolve_checkpoint_dir(checkpoint_dir)
    st_file = os.path.join(resolved_ckpt, "model.safetensors")

    if not os.path.exists(st_file):
        raise FileNotFoundError(f"Checkpoint weight file not found at: {st_file}")

    print(f"[Text Inference] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_ckpt, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[Text Inference] Loading base LLM: {base_model_name}...")
    torch_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch_dtype,
        trust_remote_code=True,
    )

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
    model = get_peft_model(model, lora_config)
    # Resize embeddings to match the tokenizer vocabulary saved with this checkpoint.
    # The checkpoint vocab (248079) may differ from the base model default (248320).
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    print(f"[Text Inference] Loading weights from safetensors: {st_file}")
    sd = safetensors.torch.load_file(st_file, device=device)

    # Normalise lora_A/lora_B key format (saved without the 'default' group name)
    peft_sd = {
        k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): v
        for k, v in sd.items()
    }

    res = model.load_state_dict(peft_sd, strict=False)
    print(f"[Text Inference] Model loaded successfully! (Unmapped keys: {len(res.unexpected_keys)})")
    model.eval()
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompt: str,
    num_rollouts: int = 1,
    temperature: float = 1.0,
    top_k: int = 20,
    device: str = "cuda"
):
    """
    Executes inference for a text-only instruction prompt.
    No audio features are involved -- the model relies solely on the
    textual description to predict an EQ coordinate.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_NO_AUDIO},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)

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
    parser = argparse.ArgumentParser(description="Inference script for Text-only GRPO model")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(script_dir, "Text_only"), help="Path to Text-only model directory or checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Text instruction prompt (e.g. 'Make the mix warm and full')")
    parser.add_argument("--num_rollouts", type=int, default=5, help="Number of rollouts/generations to sample (default: 5)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0, 0 for greedy)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()

    model, tokenizer = load_text_model_and_tokenizer(args.checkpoint, device=args.device)

    if args.prompt:
        prompts_to_process = [args.prompt]
    else:
        # Default demo prompts covering a range of EQ intents
        prompts_to_process = [
            "Make the mix warm and full",
            "Boost the high frequencies and add brightness",
            "Reduce harsh treble and add low-end depth",
        ]
        print(f"[Text Inference] No --prompt supplied. Running default demo prompts: {prompts_to_process}")

    for prompt in prompts_to_process:
        clean_p = prompt.strip()

        completions, parsed = run_inference(
            model, tokenizer, clean_p,
            num_rollouts=args.num_rollouts,
            temperature=args.temperature,
            device=args.device
        )

        print("\n" + "=" * 60)
        print(f'PROMPT: "{clean_p}"')
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
