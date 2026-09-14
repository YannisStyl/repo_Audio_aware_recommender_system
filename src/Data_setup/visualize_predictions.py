import os
import sys
import argparse
import pickle
import io
import base64
import math
import numpy as np
import torch
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file

# Add repository root and Training_Scripts directory to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
training_scripts_dir = os.path.join(repo_root, "src", "Training_Scripts")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if training_scripts_dir not in sys.path:
    sys.path.insert(0, training_scripts_dir)

from parse_utils import parse_completion_to_coord
from prompts import SYSTEM_PROMPT_AUDIO, SYSTEM_PROMPT_NO_AUDIO
from generate_reward_models import NormalizedPreferenceDensity  # type: ignore
from t2b_fx_GRPO import AudioFusedConditionedQwen, QFormerAdapter  # type: ignore


def resolve_checkpoint_paths(paths: list[str]) -> list[str]:
    """
    Scans the given paths. If a path is a directory containing 'checkpoint-*'
    subdirectories, it resolves them and sorts them numerically.
    Otherwise, treats the path as a model checkpoint itself.
    """
    resolved = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Path '{path}' does not exist. Skipping.")
            continue

        # Check if the path contains 'checkpoint-*' subdirectories
        subdirs = [
            os.path.join(path, d) for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d.startswith("checkpoint-")
        ]

        if subdirs:
            # Sort numerically based on the integer after 'checkpoint-'
            def get_step(dir_path):
                basename = os.path.basename(dir_path)
                try:
                    return int(basename.split("checkpoint-")[-1])
                except ValueError:
                    return -1
            subdirs = sorted(subdirs, key=get_step)
            resolved.extend(subdirs)
        else:
            resolved.append(path)

    # Remove duplicates while maintaining order
    seen = set()
    final_resolved = []
    for p in resolved:
        norm_p = os.path.normpath(p)
        if norm_p not in seen:
            seen.add(norm_p)
            final_resolved.append(p)

    return final_resolved


def load_reward_model_for_prompt(reward_models_path: str, prompt: str):
    """
    Loads reward models dictionary and retrieves the positive_reward density function for the given prompt.
    """
    if not os.path.exists(reward_models_path):
        raise FileNotFoundError(f"Reward models file not found at: {reward_models_path}")

    with open(reward_models_path, "rb") as f:
        reward_models = pickle.load(f)

    if prompt not in reward_models:
        print(f"Warning: Prompt '{prompt}' not found in reward models dictionary.")
        print(f"Available prompts sample: {list(reward_models.keys())[:5]}")
        return reward_models, None

    return reward_models, reward_models[prompt]["positive_reward"]


def load_model_and_tokenizer(
    model_path: str,
    base_model_name: str = "Qwen/Qwen3.5-0.8B",
    fx_path: str = None,
    device: str = "cuda"
):
    """
    Loads model checkpoint according to visualize_attention.py loading logic.
    """
    print(f"Loading tokenizer from checkpoint or base: {model_path if os.path.exists(model_path) else base_model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if "<|audio|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio|>"]})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt_to_fx = {}
    if fx_path and os.path.exists(fx_path):
        print(f"Loading audio features from: {fx_path}")
        with open(fx_path, "rb") as f:
            raw_fx = pickle.load(f)
        prompt_to_fx = {k.strip(): v for k, v in raw_fx.items()}

    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    base_model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "out_proj",
            "in_proj_qkv", "in_proj_b",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_dora=False,
        task_type=TaskType.CAUSAL_LM,
    )
    base_model = get_peft_model(base_model, lora_config)

    audio_token_id = tokenizer.convert_tokens_to_ids("<|audio|>")
    model = AudioFusedConditionedQwen(
        base_model,
        use_audio=True,
        tokenizer=tokenizer,
        audio_token_id=audio_token_id,
        prompt_to_fx=prompt_to_fx,
    ).to(device)

    has_adapter_config = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    st_path = os.path.join(model_path, "model.safetensors")

    if has_adapter_config:
        print("Loading PEFT adapter weights via PeftModel.from_pretrained...")
        model.qwen = PeftModel.from_pretrained(model.qwen, model_path).to(device)
        adapter_bin = os.path.join(model_path, "audio_projector.bin")
        if os.path.exists(adapter_bin):
            model.audio_adapter.load_state_dict(torch.load(adapter_bin, map_location=device))
            print("audio_projector.bin loaded successfully \u2713")
        else:
            print("audio_projector.bin not found -- adapter weights are random init.")
    elif os.path.exists(st_path):
        print(f"Loading monolithic checkpoint from: {st_path}")
        sd = load_file(st_path)

        adapter_sd = {
            k.removeprefix("audio_projector."): v
            for k, v in sd.items() if k.startswith("audio_projector.")
        }
        peft_sd = {k: v for k, v in sd.items() if not k.startswith("audio_projector.")}

        needs_remap = any(".lora_A.weight" in k or ".lora_B.weight" in k for k in peft_sd)
        if needs_remap:
            print("Remapping PEFT key format...")
            peft_sd = {
                k.replace(".lora_A.weight", ".lora_A.default.weight")
                 .replace(".lora_B.weight", ".lora_B.default.weight"): v
                for k, v in peft_sd.items()
            }

        missing, unexpected = model.qwen.load_state_dict(peft_sd, strict=False)
        print(f"Loaded {len(peft_sd) - len(unexpected)} / {len(peft_sd)} PEFT state dict keys \u2713")

        if adapter_sd:
            model.audio_adapter.load_state_dict(adapter_sd, strict=False)
            print(f"Loaded {len(adapter_sd)} audio projector state dict keys \u2713")
    else:
        print(f"Warning: Neither adapter_config.json nor model.safetensors found in {model_path}. Using base initialized weights.")

    model.eval()
    return model, tokenizer, prompt_to_fx


def generate_rollouts(
    model,
    tokenizer,
    prompt: str,
    num_rollouts: int = 16,
    temperature: float = 1.0,
    device: str = "cuda"
) -> list[str]:
    """
    Generates rollouts for a prompt using chat formatting matching GRPO training logic.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_AUDIO},
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    completions = []
    print(f"Generating {num_rollouts} rollouts (temperature={temperature})...")

    with torch.no_grad():
        for i in range(num_rollouts):
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                do_sample=True,
                temperature=temperature,
                max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            generated_tokens = outputs[0][prompt_len:]
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            completions.append(text)

    return completions


def compute_metrics(parsed_coords: list, rewards: list):
    """
    Computes summary metrics for group rollouts:
    - valid_ratio: fraction of rollouts that produced valid coordinates
    - mean_reward: average reward across all 16 rollouts
    - mean_l2_dist: mean pairwise Euclidean distance among valid coordinates (exploration diversity)
    """
    valid_coords = [p for p in parsed_coords if p is not None]
    num_valid = len(valid_coords)
    valid_ratio = num_valid / len(parsed_coords) if len(parsed_coords) > 0 else 0.0
    mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else -1.0

    if num_valid > 1:
        coords_arr = np.array(valid_coords)
        dists = []
        for i in range(num_valid):
            for j in range(i + 1, num_valid):
                dists.append(np.linalg.norm(coords_arr[i] - coords_arr[j]))
        mean_l2_dist = float(np.mean(dists))
    else:
        mean_l2_dist = 0.0

    return {
        "num_valid": num_valid,
        "valid_ratio": valid_ratio,
        "mean_reward": mean_reward,
        "mean_l2_dist": mean_l2_dist
    }


def render_density_background(Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> str:
    """
    Rasterizes the ground-truth preference density as a static PNG and returns it as a
    base64 data URI. This is placed on the Plotly figure via layout.images (layer='below')
    instead of a go.Contour trace, because Contour traces are redrawn as SVG shapes on every
    animation frame and either flicker (redraw=True) or fail to render mid-animation
    (redraw=False). Since the density never changes across checkpoints, a static image avoids
    both problems entirely and lets the marker animation stay smooth.
    """
    fig_mpl, ax = plt.subplots(figsize=(8.5, 7.2), dpi=100)
    ax.contourf(x, y, Z, levels=100, cmap="viridis")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.axis("off")
    fig_mpl.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="png", transparent=False)
    plt.close(fig_mpl)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def create_animated_interactive_plot(
    prompt: str,
    all_checkpoints_data: list[dict],
    prompt_density,
    output_html_path: str
):
    """
    Builds and saves an animated interactive HTML visualization dashboard.
    Animates through multiple model checkpoints across steps to visualize training progression.

    The ground-truth density is rendered once as a static background image (see
    render_density_background) rather than as an animated Contour trace, so the heatmap is
    always visible immediately and never flickers or disappears during playback. Only the
    Scatter trace (model predictions) is animated across frames.
    """
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.flatten(), Y.flatten()])

    if prompt_density is not None:
        Z = prompt_density(XY).reshape(X.shape)
    else:
        Z = np.zeros(X.shape)

    bg_image_uri = render_density_background(Z, x, y)

    fig = go.Figure()

    def get_scatter_data(data):
        points_x, points_y, hover_texts, symbols, colors = [], [], [], [], []

        for idx, (comp, coord, reward) in enumerate(zip(data["completions"], data["parsed_coords"], data["rewards"])):
            escaped_comp = comp.strip().replace("\n", " ").replace("'", "&#39;").replace('"', "&quot;")
            if coord is not None:
                points_x.append(coord[0])
                points_y.append(coord[1])
                hover_texts.append(
                    f"<b>Rollout #{idx+1}</b><br>"
                    f"Completion: '{escaped_comp}'<br>"
                    f"Parsed Coord: [{coord[0]:.2f}, {coord[1]:.2f}]<br>"
                    f"Reward: {reward:.4f}"
                )
                symbols.append('circle')
                colors.append('#ff1744')  # Bright red for valid (shows nicely on Viridis)
            else:
                points_x.append(0.0)
                points_y.append(0.0)
                hover_texts.append(
                    f"<b>Rollout #{idx+1} (INVALID PARSE)</b><br>"
                    f"Completion: '{escaped_comp}'<br>"
                    f"Parsed Coord: None<br>"
                    f"Reward: -1.0000"
                )
                symbols.append('x')
                colors.append('#ff9900')  # Orange for invalid

        return points_x, points_y, hover_texts, symbols, colors

    # Plot Step 0 for the base interactive layer (Trace 0 -- now the only real trace).
    init_data = all_checkpoints_data[0]
    init_px, init_py, init_hover, init_symbols, init_colors = get_scatter_data(init_data)

    fig.add_trace(go.Scatter(
        x=init_px,
        y=init_py,
        mode='markers',
        marker=dict(
            size=7,
            color=init_colors,
            symbol=init_symbols,
            opacity=0.3,
        ),
        hovertext=init_hover,
        hoverinfo="text",
        name="Model Predictions"
    ))

    # Dummy invisible heatmap trace solely to keep a colorbar legend. It is never referenced
    # by any frame, so animation never touches it and it can't cause flicker.
    fig.add_trace(go.Heatmap(
        z=Z,
        x=x,
        y=y,
        colorscale='Viridis',
        opacity=0,
        showscale=True,
        hoverinfo='skip',
        colorbar=dict(
            title=dict(text='Preference Density', font=dict(color='white')),
            tickfont=dict(color='white')
        ),
        name='Ground Truth Density'
    ))

    # Construct frames for each checkpoint. Only trace 0 (Scatter) is animated.
    frames = []
    slider_steps = []

    for data in all_checkpoints_data:
        step_name = data["step_name"]
        px, py, phover, psymbols, pcolors = get_scatter_data(data)

        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=px,
                    y=py,
                    marker=dict(size=7, color=pcolors, symbol=psymbols, opacity=0.3),
                    hovertext=phover
                )
            ],
            traces=[0],  # Only the Scatter trace is animated; the background image and
                         # dummy colorbar trace are untouched.
            name=step_name
        ))

        slider_steps.append(dict(
            method="animate",
            args=[
                [step_name],
                dict(mode="immediate", frame=dict(duration=600, redraw=False), transition=dict(duration=0))
            ],
            label=step_name
        ))

    fig.frames = frames

    title_text = (
        f"<b>Rollouts During Training</b><br>"
        f"<sup>The model explores and tries to find the brighter areas in the preference space</sup>"
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor='center', font=dict(size=16, color='white')),
        xaxis=dict(title="X Coordinate", range=[-6.5, 6.5], zeroline=False, gridcolor='rgba(255,255,255,0.15)', tickfont=dict(color='white')),
        yaxis=dict(title="Y Coordinate", range=[-6.5, 6.5], zeroline=False, gridcolor='rgba(255,255,255,0.15)', tickfont=dict(color='white')),
        width=850,
        height=720,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=100),

        images=[dict(
            source=bg_image_uri,
            xref="x",
            yref="y",
            x=-6,
            y=6,
            sizex=12,
            sizey=12,
            sizing="stretch",
            layer="below"
        )],

        # Position the play/pause buttons securely to the left beneath the chart
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            direction="left",
            y=-0.12,
            x=0.0,
            xanchor="left",
            yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="\u25b6 Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=800, redraw=False), transition=dict(duration=0), fromcurrent=True, mode="immediate")]
                ),
                dict(
                    label="\u23f8 Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
                )
            ]
        )],

        # Position the slider cleanly to the right of the buttons
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(font=dict(size=14, color="white"), prefix="Checkpoint: ", visible=True, xanchor="right"),
            transition=dict(duration=0),
            pad=dict(b=0, t=0),
            len=0.85,
            x=0.15,
            y=-0.12,
            steps=slider_steps
        )]
    )

    plotly_div = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='plotly-div')

    # Construct the synchronized HTML Tables container
    all_tables_html = ""
    for i, data in enumerate(all_checkpoints_data):
        step_name = data["step_name"]
        display_style = "block" if i == 0 else "none"
        metrics = data["metrics"]

        table_rows = []
        for idx, (comp, coord, reward) in enumerate(zip(data["completions"], data["parsed_coords"], data["rewards"])):
            escaped_comp = comp.strip().replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")
            if coord is not None:
                coord_str = f"[{coord[0]:.2f}, {coord[1]:.2f}]"
                status_badge = '<span style="color:#00ff88; font-weight:bold;">Valid</span>'
                reward_str = f"{reward:.4f}"
            else:
                coord_str = "None"
                status_badge = '<span style="color:#ff4444; font-weight:bold;">Invalid Parse</span>'
                reward_str = "-1.0000"

            bg_color = "#1e222b" if idx % 2 == 0 else "#161920"
            table_rows.append(f"""
                <tr style="background-color: {bg_color}; transition: background 0.2s;" onmouseover="this.style.background='#2c3240'" onmouseout="this.style.background='{bg_color}'">
                    <td style="padding: 10px; border-bottom: 1px solid #2a2e3d; text-align: center; font-weight: bold; color: #64b5f6;">Rollout #{idx+1}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #2a2e3d; font-family: monospace; color: #e0e0e0;">{escaped_comp}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #2a2e3d; text-align: center; font-family: monospace; color: #ffd54f;">{coord_str}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #2a2e3d; text-align: center; font-weight: bold; color: {'#00ff88' if reward > 0 else '#ff5252'};">{reward_str}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #2a2e3d; text-align: center;">{status_badge}</td>
                </tr>
            """)

        table_body = "\n".join(table_rows)

        checkpoint_html = f"""
        <div id="data-step-{i}" class="checkpoint-data" data-step-name="{step_name}" style="display: {display_style};">
            <div class="metrics-grid" style="margin-bottom: 30px;">
                <div class="metric-card" style="border-left-color: #00e676;">
                    <div class="metric-label">Valid Completions</div>
                    <div class="metric-value">{metrics['num_valid']} / {len(data['completions'])} ({metrics['valid_ratio']*100:.1f}%)</div>
                </div>
                <div class="metric-card" style="border-left-color: #ffb74d;">
                    <div class="metric-label">Mean Group Reward</div>
                    <div class="metric-value">{metrics['mean_reward']:.4f}</div>
                </div>
                <div class="metric-card" style="border-left-color: #ab47bc;">
                    <div class="metric-label">Spatial L2 Diversity</div>
                    <div class="metric-value">{metrics['mean_l2_dist']:.4f}</div>
                </div>
            </div>

            <div class="table-card">
                <div class="table-title">Rollout Completion Details ({step_name})</div>
                <table>
                    <thead>
                        <tr>
                            <th style="width: 10%; text-align: center;">Rollout</th>
                            <th style="width: 50%;">Generated Completion</th>
                            <th style="width: 15%; text-align: center;">Parsed [X, Y]</th>
                            <th style="width: 13%; text-align: center;">Reward</th>
                            <th style="width: 12%; text-align: center;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_body}
                    </tbody>
                </table>
            </div>
        </div>
        """
        all_tables_html += checkpoint_html

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Animated Predictions Dashboard - {prompt}</title>
    <style>
        body {{
            background-color: #0e1117;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1050px;
            margin: 0 auto;
        }}
        .header-card {{
            background: linear-gradient(135deg, #1e222d 0%, #161922 100%);
            border-radius: 12px;
            padding: 20px 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            border: 1px solid #2a2e3d;
        }}
        .header-title {{
            font-size: 22px;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 8px 0;
        }}
        .header-subtitle {{
            font-size: 14px;
            color: #90a4ae;
            margin: 0 0 18px 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 15px;
        }}
        .metric-card {{
            background-color: #12151c;
            border-radius: 8px;
            padding: 12px 18px;
            border-left: 4px solid #64b5f6;
        }}
        .metric-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #78909c;
            margin-bottom: 4px;
        }}
        .metric-value {{
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
        }}
        .plot-container {{
            background-color: #161922;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            border: 1px solid #2a2e3d;
            display: flex;
            justify-content: center;
        }}
        .table-card {{
            background-color: #161922;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            border: 1px solid #2a2e3d;
        }}
        .table-title {{
            font-size: 18px;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 15px;
            color: #ffffff;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th {{
            background-color: #1f2430;
            color: #b0bec5;
            padding: 12px 10px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #37474f;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-card">
            <div class="header-title">Prompt: "{prompt}"</div>
            <div class="header-subtitle">Training Progression: <code>{len(all_checkpoints_data)} Steps</code></div>
        </div>

        <div class="plot-container">
            {plotly_div}
        </div>

        <div id="tables-container">
            {all_tables_html}
        </div>
    </div>

    <script>
        // Sync JS logic to switch metrics & table displays corresponding to the active Plotly frame slider
        function showStepData(stepName) {{
            document.querySelectorAll('.checkpoint-data').forEach(el => {{
                el.style.display = 'none';
                if (el.getAttribute('data-step-name') === stepName) {{
                    el.style.display = 'block';
                }}
            }});
        }}

        window.addEventListener('load', function() {{
            const gd = document.getElementById('plotly-div');
            if (gd) {{
                // Triggered upon manual slider manipulation
                gd.on('plotly_sliderchange', function(e) {{
                    if (e && e.step && e.step.label) {{
                        showStepData(e.step.label);
                    }}
                }});

                // Triggered per frame during auto-playback
                gd.on('plotly_animatingframe', function(e) {{
                    if (e && e.name) {{
                        showStepData(e.name);
                    }}
                }});
            }}
        }});
    </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Animated interactive dashboard saved to: {output_html_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions interactively over reward preference densities across multiple checkpoints.")
    parser.add_argument("--model_path", type=str, nargs="+", required=True, help="Path(s) to model checkpoint(s) or parent directory containing checkpoint-* subdirectories.")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3.5-0.8B", help="Base model name for LoRA checkpoints.")
    parser.add_argument("--fx_path", type=str, default=os.path.join(repo_root, "Data_dir", "prompt_to_fx_unpooled.pkl"), help="Path to prompt_to_fx_unpooled.pkl")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt to evaluate. If not specified, picks first matching prompt from reward models.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Dataset split for reward models.")
    parser.add_argument("--reward_models_path", type=str, default=None, help="Path to reward_models_augmented_{split}.pkl")
    parser.add_argument("--num_rollouts", type=int, default=16, help="Number of rollouts to generate per checkpoint (default: 16).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for rollouts (default: 1.0).")
    parser.add_argument("--output_html", type=str, default="src/Data_setup/predictions_visual.html", help="Path to output HTML file.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")

    args = parser.parse_args()

    checkpoint_paths = resolve_checkpoint_paths(args.model_path)
    if not checkpoint_paths:
        print("No valid checkpoints found. Exiting.")
        return

    if args.reward_models_path is None:
        rm_path = os.path.join(repo_root, "Data_dir", "reward_models", f"reward_models_augmented_{args.split}.pkl")
    else:
        rm_path = args.reward_models_path

    # Load audio feature dictionary base keys for default prompt selection logic
    prompt_to_fx_base = {}
    if os.path.exists(args.fx_path):
        with open(args.fx_path, "rb") as f:
            raw_fx = pickle.load(f)
        prompt_to_fx_base = {k.strip(): v for k, v in raw_fx.items()}

    # Resolve dataset density models and default prompt
    reward_models_dict, prompt_density = None, None
    if os.path.exists(rm_path):
        with open(rm_path, "rb") as f:
            reward_models_dict = pickle.load(f)

        if args.prompt is None:
            common_prompts = [p for p in reward_models_dict.keys() if p.strip() in prompt_to_fx_base]
            if common_prompts:
                args.prompt = common_prompts[0]
            else:
                args.prompt = list(reward_models_dict.keys())[0]
            print(f"No prompt provided. Selected prompt from dataset: '{args.prompt}'")

        if args.prompt in reward_models_dict:
            prompt_density = reward_models_dict[args.prompt]["positive_reward"]
        else:
            print(f"Warning: Prompt '{args.prompt}' not found in reward models.")
    else:
        if args.prompt is None:
            if prompt_to_fx_base:
                args.prompt = next(iter(prompt_to_fx_base.keys()))
            else:
                args.prompt = "Enhance the high-end details in the audio."
        print(f"Reward models file not found at {rm_path}. Visualization will proceed without ground truth heatmap.")

    # Process all resolved checkpoints chronologically
    all_checkpoints_data = []

    for cp in checkpoint_paths:
        print(f"\n{'='*50}\n--- Processing Checkpoint: {os.path.basename(os.path.normpath(cp))} ---\n{'='*50}")

        model, tokenizer, prompt_to_fx = load_model_and_tokenizer(
            model_path=cp,
            base_model_name=args.base_model_name,
            fx_path=args.fx_path,
            device=args.device
        )

        # Verify and bind audio embedding fallback mechanisms
        prompt_clean = args.prompt.strip()
        if prompt_to_fx and prompt_clean not in prompt_to_fx:
            matched_key = None
            for k in prompt_to_fx.keys():
                if k.lower() == prompt_clean.lower() or prompt_clean in k or k in prompt_clean:
                    matched_key = k
                    break
            if matched_key:
                prompt_to_fx[prompt_clean] = prompt_to_fx[matched_key]
                print(f"Matched prompt '{args.prompt}' to existing audio feature key: '{matched_key}'")
            else:
                sample_feat = next(iter(prompt_to_fx.values()))
                prompt_to_fx[prompt_clean] = torch.zeros_like(sample_feat)
                print(f"Warning: Prompt '{args.prompt}' not found in audio features. Created zero-audio feature fallback.")

            # Update the reference pointer internally
            model.prompt_to_fx = prompt_to_fx

        completions = generate_rollouts(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            num_rollouts=args.num_rollouts,
            temperature=args.temperature,
            device=args.device
        )

        parsed_coords = [parse_completion_to_coord(c) for c in completions]

        rewards = []
        for idx, (comp, coord) in enumerate(zip(completions, parsed_coords)):
            if coord is not None and prompt_density is not None:
                r = float(prompt_density(coord).item())
            elif coord is not None:
                r = 0.0
            else:
                r = -1.0
            rewards.append(r)
            print(f"Rollout {idx+1:02d}: {comp.strip()!r} -> Parsed: {coord} | Reward: {r:.4f}")

        metrics = compute_metrics(parsed_coords, rewards)

        all_checkpoints_data.append({
            "checkpoint": cp,
            "step_name": os.path.basename(os.path.normpath(cp)),
            "completions": completions,
            "parsed_coords": parsed_coords,
            "rewards": rewards,
            "metrics": metrics
        })

        # Memory flushing between heavy generations
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Compile animated plot outputs
    if all_checkpoints_data:
        create_animated_interactive_plot(
            prompt=args.prompt,
            all_checkpoints_data=all_checkpoints_data,
            prompt_density=prompt_density,
            output_html_path=args.output_html
        )


if __name__ == "__main__":
    main()