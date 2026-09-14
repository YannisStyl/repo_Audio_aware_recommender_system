# AGENT.md — Technical Context & Operational Guide for AI Assistants

> **Purpose:** Read this file at the start of every session before making any changes.
> It is the single, authoritative source of truth for codebase conventions, mathematical
> formulations, data formats, model architectures, checkpoints, and operational workflows.
> **Keep it updated** whenever a significant architectural change or decision is resolved.
>
> Last updated: 2026-09-14 (rev 5)

---

## 0. Quick-Start Checklist for Agents

When resuming work or performing tasks in this repository, follow these rules in order:
1. **Read this file (`AGENT.md`)**: Contains all technical specifications, mathematical definitions, data schemas, checkpoint mappings, and gotchas.
2. **Read `README.md`**: Provides the human-facing domain motivation, architectural overview, and summary of empirical findings.
3. **Check Python Environment**:
   - Main environment: **`python`** (PyTorch 2.0+, Transformers, TRL, PEFT, datasets) — used for all training, inference, reward modeling, and evaluation.
   - Feature extraction environments (if re-extracting from raw audio):
     - FX feature extraction (`extract_fx_features.py`): Requires `.venv_fx_plusplus_312` (Python 3.12, Fx-Encoder++ dependencies).
     - Voice feature extraction (`extract_voice_features.py`): Requires `.venv_wsl` (Linux/WSL environment for Auden-Voice dependencies).
4. **Never introduce broken file dependencies**: The files `GEMINI.md` and `project_overview.md` have been consolidated into `AGENT.md` and `README.md`. Do not reference or look for them.

---

## 1. Project in One Paragraph

Train a compact Large Language Model (**Qwen3.5-0.8B**, $D_{\text{LLM}} = 1024$) via **Group Relative Policy Optimization (GRPO)** to map an open-ended natural-language audio-editing instruction (e.g., *"make the sound warmer"*, *"enhance vocal clarity"*) plus a 5-second playback audio clip to a continuous 2D equalization coordinate $(x, y) \in [-6, 6] \times [-6, 6]$ on Bang & Olufsen's **Beosonic** control plane. Audio conditioning is provided by **two frozen, pretrained encoders**—**Fx-Encoder++** (music/mix features) and **Auden-Voice** (speech/paralinguistic features)—bridged into Qwen's embedding space via independent **Q-Former cross-attention adapters** (16 query tokens each, 32 audio tokens total). The GRPO policy is optimized against a composite reward: a **direct preference density** estimated via Reflective Kernel Density Estimation (KDE) from $\approx 90,000$ user interaction ratings (no separate reward model), balanced with a **pairwise geometric diversity reward** to preserve multimodal solution modes. The central empirical finding is that modular, stream-specialized routing ($\max(\text{Voice}, \text{FX})$) achieves human expert (Tonmeister) level performance and substantially outperforms both text-only baselines and $5\times$ larger LLMs (Qwen3.5-4B).

---

## 2. Complete Repository Layout

```
repo_Audio_aware_recommender_system/
├── AGENT.md                                   ← THIS FILE (Technical source of truth for agents)
├── README.md                                  ← Human-readable project overview & empirical summary
├── fx_voice_GRPO.png                          ← System architecture & training schematic
├── .env.example                               ← Template for WANDB_API_KEY
├── .gitignore                                 ← Git exclusions (checkpoints, env, data caches)
│
├── Data_dir/
│   ├── prompt_to_fx_windowed.pkl              ← FX feature dict: {prompt_str -> dict} [T, 2048]
│   ├── prompt_to_voice_windowed.pkl           ← Voice feature dict: {prompt_str -> dict} [T, 768]
│   ├── fx_sample_id_provenance.pkl            ← Provenance metadata table: {track::wN -> dict}
│   ├── voice_sample_id_provenance.pkl         ← Same for voice features
│   ├── prompts_and_audio_data.json            ← 130 entries [{prompt, responses, track, ...}]
│   ├── original_audio_trimmed/                ← 5s trimmed WAV audio clips (44.1 kHz, stereo)
│   └── reward_models/
│       ├── reward_models_augmented_train.pkl  ← Train KDE preference densities {prompt -> {"positive_reward": KDE}}
│       ├── reward_models_augmented_val.pkl    ← Validation KDE densities (held-out paraphrase)
│       └── reward_models_simulated_val.pkl    ← Small synthetic set for quick local testing
│
├── src/
│   ├── Data_setup/
│   │   ├── extract_fx_features.py             ← Fx-Encoder++ Conv6 feature extraction
│   │   ├── extract_voice_features.py          ← Auden-Voice Zipformer feature extraction
│   │   ├── generate_reward_models.py          ← Reflective KDE builder + NormalizedPreferenceDensity
│   │   ├── visualize_predictions.py           ← 2D coordinate scatter plots (Plotly / Matplotlib)
│   │   ├── visualize_attention.py             ← Eager-mode attention matrix visualization
│   │   └── visualize_reward_models.py         ← Reward density contour visualizer
│   │
│   ├── Training_Scripts/
│   │   ├── t2b_fx_and_voice_GRPO.py           ← MAIN training script (Dual-Stream FX + Voice)
│   │   ├── t2b_fx_GRPO.py                     ← Single-stream FX-only training script
│   │   ├── t2b_voice_GRPO.py                  ← Single-stream Voice-only training script
│   │   ├── parse_utils.py                     ← parse_completion_to_coord (canonical coordinate parser)
│   │   └── prompts.py                         ← System prompts & placeholder token definitions
│   │
│   └── GRPO_models/
│       ├── infer_fx_and_voice.py              ← Dual-stream inference entry point
│       ├── infer_fx.py                        ← FX-only inference entry point
│       ├── infer_voice.py                     ← Voice-only inference entry point
│       ├── infer_text.py                      ← Text-only (no-audio) baseline inference
│       ├── grounding_swap_test.py             ← Causal feature-swap grounding evaluation script
│       ├── FX_Voice_model/checkpoint-7300/    ← Hero dual-stream checkpoint (7,300 steps)
│       ├── FX_model/checkpoint-12600/         ← FX-only specialist checkpoint (12,600 steps)
│       ├── Voice_model/checkpoint-10700/      ← Voice-only specialist checkpoint (10,700 steps)
│       └── Text_only/checkpoint-9050/         ← No-audio baseline checkpoint (9,050 steps)
│
├── Listening_Exp/
│   └── Perceptual_evaluation/
│       ├── analyze_perceptual_results.py      ← Statistical evaluation pipeline (Friedman, Wilcoxon)
│       ├── generate_model_predictions.py      ← Generates full candidate prediction sets (7 points)
│       ├── generate_model_predictions_concise.py ← Generates concise greedy predictions (5 points)
│       ├── extract_perceptual_fx_features.py  ← Feature extraction for held-out perceptual tracks
│       ├── extract_perceptual_voice_features.py ← Voice feature extraction for held-out perceptual tracks
│       ├── prompt_audio_link.json             ← 22 held-out (prompt, clip, category) pairs
│       ├── model_predictions.json             ← Full version model prediction outputs
│       ├── model_predictions_concise.json     ← Concise greedy prediction outputs
│       ├── perceptual_prompt_to_fx.pkl        ← Held-out FX features pickle
│       ├── perceptual_prompt_to_voice.pkl     ← Held-out Voice features pickle
│       ├── perceptual_ratings_tidy.csv        ← Tidy ratings dataset (N=3,850 evaluations)
│       ├── perceptual_results_tables.tex      ← Publication LaTeX tables for perceptual study
│       └── Results/                           ← Raw individual listener JSON logs (30 assessors)
│
└── results/
    └── perceptual_experiment_and_conclusion.txt ← Research paper experimental text and conclusions
```

---

## 3. Mathematical Formulation & Architecture

### 3.1 Temporal Feature Extraction
Instead of pooling acoustic features into a single global vector (which discards temporal dynamics), chronological sequences are preserved:

1. **FX Stream ($X_{\text{FX}}$)**: Extracted from the 6th convolutional block of `Fx-Encoder++` (averaged over frequency bins):
   $$X_{\text{FX}} \in \mathbb{R}^{T_{\text{Fx}} \times 2048}$$
2. **Voice Stream ($X_{\text{Voice}}$)**: Extracted from the final hidden state of `Auden-Voice` (Zipformer architecture):
   $$X_{\text{Voice}} \in \mathbb{R}^{T_{\text{Voice}} \times 768}$$

### 3.2 Q-Former Cross-Attention Adapter
Each stream connects to Qwen's embedding space via an identical `QFormerAdapter` architecture parameterized by its input encoder dimension $\text{enc\_dim} \in \{2048, 768\}$ and LLM embedding dimension $D_{\text{LLM}} = 1024$:

1. **Linear Projection:**
   $$X_{\text{proj}} = W_p \cdot X + b_p \in \mathbb{R}^{T \times D_{\text{LLM}}}$$
   where $W_p \in \mathbb{R}^{D_{\text{LLM}} \times \text{enc\_dim}}$ and $b_p \in \mathbb{R}^{D_{\text{LLM}}}$.

2. **Learnable Query Embeddings:**
   Initialize $N_q = 16$ learnable query tokens:
   $$Q_{\text{tokens}} \in \mathbb{R}^{16 \times D_{\text{LLM}}} \sim \mathcal{N}(0, 0.02^2)$$

3. **Multi-Head Cross-Attention (8 Heads):**
   The query tokens $Q$ attend to the projected chronological features $X_{\text{proj}}$ acting as Keys ($K$) and Values ($V$):
   $$A = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{D_{\text{LLM}}}}\right) \cdot V \in \mathbb{R}^{16 \times D_{\text{LLM}}}$$

4. **Residual Connection, FFN, and Layer Normalization:**
   $$X_{\text{attn}} = \text{LayerNorm}(Q + A)$$
   $$X_{\text{ffn}} = W_2 \cdot \text{GELU}(W_1 \cdot X_{\text{attn}} + b_1) + b_2$$
   $$V_{\text{Audio}} = \text{LayerNorm}(X_{\text{attn}} + X_{\text{ffn}}) \in \mathbb{R}^{16 \times D_{\text{LLM}}}$$

### 3.3 Audio Token Injection
The resulting embeddings are dynamically injected into Qwen's input embedding sequence:
- 16 FX tokens $\to$ injected at the 16 `<|audio|>` special token positions.
- 16 Voice tokens $\to$ injected at the 16 `<|voice|>` special token positions.
- Total audio context prepended prior to user text tokens: $32 \times D_{\text{LLM}}$.

### 3.4 Qwen3.5-0.8B Hybrid Architecture (24 Layers)
Qwen3.5-0.8B employs a hybrid decoder architecture combining standard attention with DeltaNet linear attention:
- **Standard Scaled Dot-Product Attention (SDPA)** (5 layers: Layers 7, 11, 15, 19, 23):
  - Computes full $N \times N$ pairwise token attention.
  - Parameter modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- **DeltaNet Linear Attention** (19 layers: All other layers):
  - Compresses and routes tokens iteratively with an RNN-like linear state update.
  - Parameter modules: `in_proj_qkv`, `out_proj`, `in_proj_b` (memory write strength), `in_proj_z` (gating), `in_proj_a` (decay rate).
- **Feed-Forward Networks (MLP)** (All 24 layers):
  - Parameter modules: `gate_proj`, `up_proj`, `down_proj`.

### 3.5 LoRA Parameter Targets
LoRA ($r = 8, \alpha = 16, \text{dropout} = 0.05$) targets:
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Standard self-attention
    "out_proj", "in_proj_qkv", "in_proj_b",   # DeltaNet linear attention
    "gate_proj", "up_proj", "down_proj",       # MLP feed-forward
]
```
*(Note: `in_proj_z` and `in_proj_a` are excluded from LoRA targets).*

### 3.6 Attention Visualization Constraints
Because DeltaNet layers do not produce an explicit $N \times N$ Softmax matrix, classical attention heatmaps can only target the **5 standard self-attention layers** (7, 11, 15, 19, 23). `src/Data_setup/visualize_attention.py` must run inference with `attn_implementation='eager'`.

---

## 4. Data Formats & Provenance

### 4.1 Windowed Feature Pickles
Files: `Data_dir/prompt_to_fx_windowed.pkl` and `Data_dir/prompt_to_voice_windowed.pkl`
- **Keys**: `prompt_str.strip()` (210 unique prompt text strings: 30 concepts $\times$ 7 paraphrases).
- **Values**: Dict containing:
  ```python
  {
      "tensor": torch.Tensor,       # [T, 2048] for FX; [T, 768] for Voice
      "sample_id": str,             # e.g. "BorderlessPodcast.wav::w2"
      "track": str,                 # Source audio filename, e.g. "BorderlessPodcast.wav"
      "window_index": int,          # 0-indexed window number (0 to 6)
      "window_start_sec": float,    # Window start timestamp in seconds
      "window_duration_sec": 5.0,   # Fixed 5.0-second window duration
  }
  ```
- **Loading convention used across the codebase:**
  ```python
  prompt_to_fx = {k.strip(): v["tensor"] for k, v in raw_fx.items()}
  prompt_to_voice = {k.strip(): v["tensor"] for k, v in raw_voice.items()}
  ```

### 4.2 Windowing & Duration-Adaptive Stride Formula
Source audio clips vary in duration ($D \approx 6\text{--}15\,\text{s}$). To guarantee fixed temporal resolution (5s window) without repeating feature tensors across the 7 synonymous text variants ($N = 7$):
$$\text{stride} = \frac{D - 5.0}{N - 1} = \frac{D - 5.0}{6}$$
$$\text{start}_i = i \cdot \text{stride}, \quad i \in \{0, 1, \dots, 6\}$$
Each text paraphrase is assigned a distinct temporal window spanning the track.

### 4.3 Provenance Tables
Files: `Data_dir/fx_sample_id_provenance.pkl` and `Data_dir/voice_sample_id_provenance.pkl`
- **Keys**: `"{track}::w{window_index}"` strings.
- **Values**: Metadata dict (`prompt_variant`, `track`, `window_index`, `window_start_sec`, `window_duration_sec`).
- **Usage**: Used by external evaluation tooling (causal swap test, perceptual study linking). Not consulted in the live training forward pass.

### 4.4 Reward Model Pickles
Files: `Data_dir/reward_models/reward_models_augmented_train.pkl` and `reward_models_augmented_val.pkl`
- **Keys**: `prompt_str.strip()`.
- **Values**: `{"positive_reward": NormalizedPreferenceDensity instance}`.
- **Call convention:**
  ```python
  from Data_setup.generate_reward_models import NormalizedPreferenceDensity
  # pred is a numpy array [x, y] in [-6, 6]
  reward_val = float(val_rm[prompt]["positive_reward"](pred).item())
  ```
- **Reflective KDE Implementation**:
  - Eliminates boundary bias by reflecting each point $x_i$ into 8 adjacent mirror squares ($3 \times 3$ grid of "ghost" points).
  - Bandwidth: $h = 0.25 \cdot h_{\text{Scott}}$ to compensate for extra point mass.
  - Internally operates on an expanded reflective space, but accepts and scores coordinates directly in $[-6, 6]^2$. Returns a normalized score $R_{\text{pref}} \in [-1.0, 1.0]$. Malformed completions return $-1.0$.

### 4.5 Dataset Metadata Files
- `Data_dir/prompts_and_audio_data.json`: 130 entries with schema `{"prompt", "responses", "initial position", "track"}` across 119 unique tracks.
- `Listening_Exp/Perceptual_evaluation/prompt_audio_link.json`: 22 held-out pairs across 4 categories (7 Instrumental, 8 Audiobook, 4 Music, 3 Movie).

---

## 5. Coordinate Conventions & Parsing Pipeline

### 5.1 Coordinate Spaces
- **Model Space:** Prompt instructions state the space is $[-1, 1] \times [-1, 1]$ (e.g. `[0.2, -0.5]`).
- **Beosonic Space:** The true physical parameter plane is $[-6, 6] \times [-6, 6]$.
- **Scale Factor:** `parse_completion_to_coord()` **multiplies the model output by 6**:
  $$\text{coord}_{\text{Beosonic}} = 6 \cdot \text{coord}_{\text{model}}$$
  This scaling factor is consistent across all training, inference, and evaluation scripts. Always pass the parsed $[-6, 6]$ coordinates directly to the reward models without additional rescaling.

### 5.2 Parsing Function: `parse_completion_to_coord(text: str)`
Located in `src/Training_Scripts/parse_utils.py`:
- Strips whitespace and special tokens.
- Default (`use_regex=False`): Uses `ast.literal_eval` with a strict 20-character limit.
- Regex fallback (`use_regex=True`): Extracts coordinate patterns from longer outputs:
  ```python
  r"\[\s*(-?\d*(?:\.\d+)?)\s*,\s*(-?\d*(?:\.\d+)?)\s*\]"
  ```
- Clamps values to $[-1.0, 1.0]$, multiplies by 6, and returns `np.ndarray([x, y])` in $[-6.0, 6.0]$.
- Returns `None` on parse failure (which translates to a $-1.0$ reward penalty in GRPO).

---

## 6. Checkpoint Registry

Pre-trained model checkpoints are stored in `src/GRPO_models/`:

| Model Description | Checkpoint Path | Training Steps | Notes |
| :--- | :--- | :---: | :--- |
| **FX + Voice Dual Model** | `src/GRPO_models/FX_Voice_model/checkpoint-7300` | 7,300 | Multi-stream model (requires mode-aware decoding) |
| **FX-Only Specialist** | `src/GRPO_models/FX_model/checkpoint-12600` | 12,600 | Best for music mixtures and production |
| **Voice-Only Specialist** | `src/GRPO_models/Voice_model/checkpoint-10700` | 10,700 | Best for audiobook, speech, and solo instruments |
| **No-Audio Baseline** | `src/GRPO_models/Text_only/checkpoint-9050` | 9,050 | Text-only control baseline (0.8B) |

### Safetensors Key Remapping
Checkpoints store both PEFT LoRA weights and Q-Former projector weights in `model.safetensors`:
- `fx_projector.*` $\to$ Loads into `model.fx_adapter` state dict.
- `voice_projector.*` $\to$ Loads into `model.voice_adapter` state dict.
- All other keys $\to$ PEFT LoRA state dict.
  *(Note: HuggingFace PEFT expects `.lora_A.default.weight`; checkpoints store `.lora_A.weight`. Renaming is handled automatically in the inference loading routines).*

---

## 7. System Prompts & Formatting

Defined in `src/Training_Scripts/prompts.py`:

| Constant | Model Application | Placeholder Tokens |
| :--- | :--- | :--- |
| `SYSTEM_PROMPT_DUAL_AUDIO` | FX + Voice Model | 16 $\times$ `<|audio|>` followed by 16 $\times$ `<|voice|>` |
| `SYSTEM_PROMPT_AUDIO` | FX-Only Model | 16 $\times$ `<|audio|>` |
| `SYSTEM_PROMPT_VOICE` | Voice-Only Model | 16 $\times$ `<|voice|>` |
| `SYSTEM_PROMPT_NO_AUDIO` | Text-Only Baseline | No placeholder tokens |

### Token Injection Structure
```
<|im_start|>system
...
Acoustic Features:
<|audio|><|audio|>...<|audio|> (16 tokens)
<|voice|><|voice|>...<|voice|> (16 tokens)
...<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
```

Both `<|audio|>` and `<|voice|>` are registered as additional special tokens via:
```python
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>', '<|voice|>']})
base_model.resize_token_embeddings(len(tokenizer))  # Call exactly ONCE
```

---

## 8. Training Hyperparameters & GRPO Configuration

### 8.1 Hyperparameter Specifications
```python
GRPO_BETA_VALUE              = 0.1          # KL penalty against reference model
NUM_GEN                      = 16           # Number of rollouts per prompt group (G)
LEARNING_RATE                = 1e-4         # Base learning rate for LoRA
LR_SCHEDULER                 = "cosine"
MAX_STEPS                    = 13500        # Total training steps
LORA_WARMUP_STEPS            = 1000         # LoRA learning rate warmup
ADAPTER_WARMUP_STEPS         = 3500         # LoRA frozen; train Q-Formers only (lr = 1e-3)
ADAPTER_LR_RATIO             = 10.0         # Q-Former lr multiplier during warmup
REWARD_WEIGHTS               = [0.75, 0.25] # [R_pref, R_div]
TEXT_MASK_MAX_PROB           = 0.0          # Text modality dropout (disabled in primary run)
DIVERSITY_SCHEDULE           = "step"       # Diversity scaling schedule
DIVERSITY_STEP_WINDOW        = (0, 13500)   # Diversity reward active full run
DIVERSITY_MIN_SCALE          = 0.0
TEMPERATURE                  = 1.0          # Rollout sampling temperature
TOP_K                        = 20           # Rollout sampling top-k
WEIGHT_DECAY                 = 0.05
```

### 8.2 GRPOConfig Settings
```python
per_device_train_batch_size  = 8            # Micro-batch size
gradient_accumulation_steps  = 16           # Effective batch size = 16 * 16 = 256 (with num_generations=16)
num_generations              = 16           # Group size G
per_device_eval_batch_size   = 16
eval_steps                   = 50
save_steps                   = 50
save_total_limit             = 50
beta                         = 0.1
loss_type                    = "grpo"
use_vllm                     = False
```

### 8.3 GRPO Objective & Reward Formulation
For each prompt context $c$, the policy samples $G = 16$ rollouts $\{x_1, \dots, x_G\}$. The group advantage is:
$$A_i = \frac{r_i - \mu_r}{\sigma_r}$$
where $r_i = 0.75 \cdot R_{\text{pref}, i} + 0.25 \cdot R_{\text{div}, i}$.

1. **Preference Density Reward:**
   $$R_{\text{pref}, i} = \text{KDE}_{\text{prompt}}(x_i, y_i) \in [-1.0, 1.0]$$
2. **Pairwise Geometric Diversity Reward:**
   $$R_{\text{div}, i} = \text{scale} \cdot \left( \frac{2 \cdot d_{\text{mean}, i}}{D_{\text{max}}} - 1.0 \right)$$
   where $d_{\text{mean}, i} = \frac{1}{N_{\text{valid}} - 1} \sum_{j \neq i} \| P_i - P_j \|_2$ and $D_{\text{max}} = 12\sqrt{2}$.

---

## 9. Inference Protocols & Code Snippets

### 9.1 Canonical Inference Workflow
All inference scripts follow this exact execution pattern:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Training_Scripts.parse_utils import parse_completion_to_coord
from Training_Scripts.prompts import SYSTEM_PROMPT_DUAL_AUDIO

# 1. Inject pre-extracted features into model prompt cache
model.prompt_to_fx[clean_prompt] = fx_tensor        # Tensor shape: [T, 2048]
model.prompt_to_voice[clean_prompt] = voice_tensor  # Tensor shape: [T, 768]

# 2. Format chat input template
messages = [
    {"role": "system", "content": SYSTEM_PROMPT_DUAL_AUDIO},
    {"role": "user", "content": user_prompt}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(device)

# 3. Generate completion
output = model.generate(input_ids=input_ids, max_new_tokens=32, temperature=0.7, top_k=20)
gen_tokens = output[0][input_ids.shape[1]:]
text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

# 4. Parse to Beosonic coordinate
coord = parse_completion_to_coord(text)  # np.ndarray([x, y]) in [-6, 6] or None
```

### 9.2 Script-to-Model Mapping
| Inference Script | Model Class | Default Checkpoint | Conditioning |
| :--- | :--- | :--- | :--- |
| `src/GRPO_models/infer_fx_and_voice.py` | `DualStreamConditionedQwen` | `FX_Voice_model/checkpoint-7300` | Text + FX + Voice |
| `src/GRPO_models/infer_fx.py` | `AudioConditionedQwen` | `FX_model/checkpoint-12600` | Text + FX |
| `src/GRPO_models/infer_voice.py` | `VoiceConditionedQwen` | `Voice_model/checkpoint-10700` | Text + Voice |
| `src/GRPO_models/infer_text.py` | Standard PEFT Qwen | `Text_only/checkpoint-9050` | Text only |
| `src/GRPO_models/grounding_swap_test.py` | `DualStreamConditionedQwen` | `FX_Voice_model/checkpoint-7300` | Causal perturbation |

---

## 10. Causal Grounding Swap Test

File: `src/GRPO_models/grounding_swap_test.py`
Proves the model causally responds to physical audio features rather than relying on text shortcuts.

### 10.1 Track Taxonomy (30 Tracks)
- **`pure-instrumental` (5 tracks):** `AlexandreDesplatTheI`, `ESQUIVELOyeNegra`, `QuartangoMilongaDiab`, `TresRythme`, `VladimirAshkenazyJSB`.
- **`pure-voice` (9 tracks):** `dev228`, `dev276`, `dev1197`, `dev13101311`, `ArtnCompany`, `BorderlessPodcast`, `LinearDigressions`, `CosmosLaundromat1`, `Sintel4`.
- **`mixed` (16 tracks):** `BobbyMcFerrinATrainL`, `ClubForFiveBrothersi`, `GlennHughesYoungLust`, `KingsOfLeonSexonFire`, `LaBottineSourianteLa`, `MilkyChanceStolenDan`, `MobyPorcelain`, `OneLoveInMyLifetime`, `PassengerLetHerGo`, `Penguins`, `PentatonixHavana`, `Tearsofsteel5`, `ZhuFaded`, `ambientforestsoundsc`, `calmzenriverflowing2`, `seaandseagullwave593`.

### 10.2 Perturbation Protocol
For each prompt in the validation set:
1. Generate baseline coordinate $y_{\text{base}}$ ($R_{\text{pref}} = 0.75 \pm 0.25$).
2. Perturb audio features across 3 swap conditions: FX-only, Voice-only, Both.
3. Test against both **In-Domain** donor tracks (matched category) and **Cross-Domain** donor tracks (incompatible category, e.g., podcast audio swapped into an instrumental rock prompt).
4. Measure coordinate shift ($\Delta L_2$) and preference density change ($R_{\text{pref}}$).

### 10.3 Execution Command
```bash
python src/GRPO_models/grounding_swap_test.py \
    --checkpoint src/GRPO_models/FX_Voice_model/checkpoint-7300 \
    --num_donors 3 \
    --output_json results/grounding_swap_results.json
```

---

## 11. Perceptual Evaluation (MUSHRA) Pipeline

Located in `Listening_Exp/Perceptual_evaluation/`:
- **Held-Out Test Set:** 22 items across 4 categories (7 Instrumental, 8 Audiobook, 4 Music, 3 Movie). Audio sourced from LibriSpeech, Free Music Archive, and Blender open movies.
- **Experimental Design:** Blinded MUSHRA listening test (ITU-R BS.1534) comparing 7 conditions:
  1. `Tonmeister 1`: Settings from an expert human sound engineer.
  2. `ICL (Qwen3.5-4B)`: $5\times$ larger LLM baseline (in-context learning, no audio).
  3. `Voice-only (0.8B)`: Speech-specialist model.
  4. `FX-only (0.8B)`: Music/mix-specialist model.
  5. `Voice+FX (0.8B Dual)`: Joint multi-stream model.
  6. `No Audio (0.8B Baseline)`: Compact text-only baseline.
  7. `Hidden Reference`: Unprocessed raw audio (objective control).
- **Screening & Consistency:** 30 assessors evaluated all stimuli; 5 failed the hidden reference control check (rating unmodified audio $> 50$), leaving $N = 25$ consistent listeners ($3,850$ total evaluations).
- **Statistical Pipeline:** `analyze_perceptual_results.py` computes Friedman test, Wilcoxon signed-rank tests with Bonferroni-Holm correction, and exports `perceptual_results_tables.tex`.

---

## 12. Comprehensive Gotchas & Edge Cases

| Gotcha | Root Cause | Operational Fix |
| :--- | :--- | :--- |
| **Reward pickle is a dict** | Calling `rm[prompt]` directly causes a `TypeError` | Access `rm[prompt]["positive_reward"]`, then call it with the coordinate array. |
| **Coordinate scale discrepancy** | Model is prompted with $[-1, 1]$, but Beosonic space is $[-6, 6]$ | `parse_completion_to_coord` multiplies by 6. Pass its $[-6, 6]$ output directly to reward functions. Never rescale manually. |
| **KDE bounds show $[-18, 18]$** | Reflective KDE creates an expanded $3 \times 3$ grid of ghost points | Normal behavior. Pass coordinates in $[-6, 6]$; the class handles boundary reflection internally. |
| **Feature pickle values are dicts** | Values contain tensor + metadata | Extract the tensor using `v["tensor"]`. |
| **Key whitespace mismatches** | Trailing spaces in prompt strings lead to lookup failure | Always apply `.strip()` to prompt strings before looking up in feature or reward dicts. |
| **Special token resize mismatch** | Calling `resize_token_embeddings()` multiple times causes dimension mismatch | Register both `<|audio|>` and `<|voice|>` in one dictionary and call `resize_token_embeddings()` exactly ONCE. |
| **PEFT LoRA weight key names** | PEFT expects `.lora_A.default.weight`, but checkpoint stores `.lora_A.weight` | Rename keys during loading (`k.replace('.lora_A.weight', '.lora_A.default.weight')`). |
| **`model.generate()` inputs** | Passing `inputs_embeds` directly to `generate()` bypasses embedding injection | Pass `input_ids`; the custom model wrapper intercepts `input_ids` and injects audio embeddings at placeholder token IDs. |
| **DeltaNet attention maps** | 19 of 24 layers use DeltaNet linear attention without an $N \times N$ matrix | Classical attention visualization only applies to layers 7, 11, 15, 19, 23 with `attn_implementation='eager'`. |
| **No substring prompt matching** | Substring lookups in `_recover_features()` risk silent feature collision | Substring matching is removed. Use exact match, falling back only to user-turn prompt tail. |
| **Separate feature extract virtualenvs** | Incompatible C-libraries between audio encoders and LLM dependencies | Use `python` for training and inference; use `.venv_fx_plusplus_312` for FX and `.venv_wsl` for Voice extraction. |

---

## 13. Agent Maintenance Protocol

Whenever modifying the repository, adhere strictly to these rules:
1. **Preserve this file:** When adding new scripts, checkpoints, or resolving open issues, update the relevant sections of `AGENT.md`.
2. **Never commit `.env` or weights to git:** Always verify `.gitignore` covers model checkpoints and API credentials.
3. **Keep `README.md` human-facing:** Technical implementation details belong here in `AGENT.md`; keep `README.md` clean, accessible, and focused on high-level architecture, empirical discoveries, and reproduction guides.
