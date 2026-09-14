"""
analyze_perceptual_results.py
==============================
Statistical Analysis Pipeline for MUSHRA Perceptual Evaluation Results.

Ingests all 30 participant JSON files (4,620 total ratings across 22 prompts
and 7 evaluation conditions).

Performs:
  1. Data ingestion, validation, and tidy CSV export (perceptual_ratings_tidy.csv).
  2. Global and category-stratified descriptive statistics (Mean, SD, SEM, 95% CI, Median, IQR).
  3. Non-parametric hypothesis testing (Friedman test, pairwise Wilcoxon signed-rank tests
     with Holm-Bonferroni correction).
  4. Delta-to-reference analysis (difference vs. unprocessed hidden reference).
  5. Publication-ready LaTeX tables export (perceptual_results_tables.tex).

Usage:
  python analyze_perceptual_results.py
"""

import os
import sys
import glob
import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "Results"
PROMPT_LINK_FILE = SCRIPT_DIR / "prompt_audio_link.json"

TIDY_CSV_OUT = SCRIPT_DIR / "perceptual_ratings_tidy.csv"
TEX_OUT = SCRIPT_DIR / "perceptual_results_tables.tex"


# =============================================================================
# 1. HELPERS
# =============================================================================

def slugify(text: str) -> str:
    """Convert prompt string into slug format matching JSON filename keys."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"[\s]+", "_", text)
    return text[:80]


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Applies Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed_p):
        p_adj = p * (m - rank)
        running_max = max(running_max, p_adj)
        adjusted[orig_idx] = min(running_max, 1.0)

    return adjusted


def get_significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


# =============================================================================
# 2. DATA INGESTION
# =============================================================================

def load_data():
    if not PROMPT_LINK_FILE.exists():
        raise FileNotFoundError(f"Missing prompt_audio_link.json: {PROMPT_LINK_FILE}")

    with open(PROMPT_LINK_FILE, "r", encoding="utf-8") as f:
        pal = json.load(f)

    prompt_map = {}
    for item in pal:
        raw_p = item["prompt"].strip()
        slug = slugify(raw_p)
        prompt_map[slug] = {
            "prompt": raw_p,
            "category": item["category"],
            "audio_clip": item["audio_clip"],
            "descriptor": item.get("descriptor", ""),
        }

    json_files = sorted(glob.glob(str(RESULTS_DIR / "RESULTS_*.json")))
    if not json_files:
        raise FileNotFoundError(f"No result JSON files found in: {RESULTS_DIR}")

    print(f"[1/5] Ingesting {len(json_files)} participant result files...")

    rows = []
    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        aid = data.get("assessorDetails", {}).get("assessorId")
        inner_results = data.get("results", [{}])[0].get("results", [])

        for trial in inner_results:
            score = trial.get("score")
            if score is None:
                continue

            model_raw = trial.get("Model")
            is_ref = trial.get("isHiddenReference", False)
            prompt_slug = trial.get("Prompt")

            if is_ref:
                model_name = "HiddenRef"
                fn = trial.get("filename", "")
                prompt_slug = fn.replace("Original_", "").replace(".wav", "")
            elif model_raw == "Tonmeister_1":
                model_name = "Tonmeister_1"
            elif model_raw == "NoAudio":
                model_name = "NoAudio"
            elif model_raw == "Voice+FX":
                model_name = "Voice+FX"
            elif model_raw == "Voice":
                model_name = "Voice"
            elif model_raw == "FX":
                model_name = "FX"
            elif model_raw == "ICL":
                model_name = "ICL"
            else:
                model_name = model_raw

            meta = prompt_map.get(prompt_slug, {
                "prompt": prompt_slug,
                "category": "Unknown",
                "audio_clip": "Unknown",
                "descriptor": "",
            })

            rows.append({
                "assessor_id": aid,
                "prompt_slug": prompt_slug,
                "prompt": meta["prompt"],
                "category": meta["category"],
                "audio_clip": meta["audio_clip"],
                "descriptor": meta["descriptor"],
                "model": model_name,
                "is_hidden_ref": is_ref,
                "score": float(score),
                "time_spent_ms": trial.get("timeSpentListening", 0),
            })

    df = pd.DataFrame(rows)
    print(f"      Total ratings ingested: {len(df)} ({len(df['assessor_id'].unique())} assessors, {len(df['prompt'].unique())} prompts)")
    df.to_csv(TIDY_CSV_OUT, index=False)
    print(f"      Tidy dataset written to: {TIDY_CSV_OUT}")
    return df


# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================

MODEL_DISPLAY_NAMES = {
    "Tonmeister_1": "Tonmeister 1 (Expert)",
    "ICL":          "ICL (Qwen3.5-4B)",
    "Voice":        "Voice-only (0.8B)",
    "FX":           "FX-only (0.8B)",
    "Voice+FX":     "Voice+FX (0.8B Dual)",
    "NoAudio":      "No Audio (0.8B Baseline)",
    "HiddenRef":    "Hidden Reference (Original)",
}

MODEL_ORDER = ["Tonmeister_1", "ICL", "Voice", "FX", "Voice+FX", "NoAudio", "HiddenRef"]


def compute_summary_table(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    records = []

    if group_col is None:
        groups = [("All", df)]
    else:
        groups = [(cat, sub_df) for cat, sub_df in df.groupby(group_col)]

    for grp_name, sub_df in groups:
        for m in MODEL_ORDER:
            scores = sub_df[sub_df["model"] == m]["score"].values
            if len(scores) == 0:
                continue
            n = len(scores)
            mean_val = np.mean(scores)
            std_val = np.std(scores, ddof=1)
            sem_val = std_val / np.sqrt(n)
            ci95 = 1.96 * sem_val
            med_val = np.median(scores)
            q25, q75 = np.percentile(scores, [25, 75])
            iqr_val = q75 - q25

            records.append({
                "Group": grp_name,
                "Model": m,
                "DisplayName": MODEL_DISPLAY_NAMES.get(m, m),
                "N": n,
                "Mean": mean_val,
                "SD": std_val,
                "SEM": sem_val,
                "CI95_Lower": mean_val - ci95,
                "CI95_Upper": mean_val + ci95,
                "Median": med_val,
                "IQR": iqr_val,
            })

    return pd.DataFrame(records)


# =============================================================================
# 4. HYPOTHESIS & SIGNIFICANCE TESTING
# =============================================================================

def run_significance_tests(df: pd.DataFrame):
    print("\n[3/5] Running Non-Parametric & Repeated-Measures Significance Tests...")

    # Pivot to wide format: Index=(assessor_id, prompt_slug), Columns=model
    pivot_df = df.pivot(index=["assessor_id", "prompt_slug"], columns="model", values="score").dropna()
    N_pairs = len(pivot_df)

    # 1. Friedman Test (Repeated Measures across all 7 conditions)
    friedman_data = [pivot_df[m].values for m in MODEL_ORDER]
    stat_f, p_val_f = stats.friedmanchisquare(*friedman_data)
    print(f"      Global Friedman Test: Chi-Square({len(MODEL_ORDER)-1}) = {stat_f:.3f}, p = {p_val_f:.3e}")

    # 2. Key Pairwise Wilcoxon Signed-Rank Tests
    comparisons = [
        ("Voice+FX", "NoAudio",      "Voice+FX vs. No-Audio Baseline"),
        ("Voice",    "NoAudio",      "Voice-only vs. No-Audio Baseline"),
        ("FX",       "NoAudio",      "FX-only vs. No-Audio Baseline"),
        ("Voice+FX", "Tonmeister_1", "Voice+FX vs. Tonmeister 1"),
        ("Voice",    "Tonmeister_1", "Voice-only vs. Tonmeister 1"),
        ("FX",       "Tonmeister_1", "FX-only vs. Tonmeister 1"),
        ("ICL",      "Voice+FX",     "ICL (4B) vs. Voice+FX (0.8B)"),
        ("ICL",      "Tonmeister_1", "ICL (4B) vs. Tonmeister 1"),
        ("Voice",    "FX",           "Voice-only vs. FX-only"),
        ("Tonmeister_1", "HiddenRef", "Tonmeister 1 vs. Hidden Reference"),
        ("ICL",          "HiddenRef", "ICL (4B) vs. Hidden Reference"),
        ("Voice",        "HiddenRef", "Voice-only vs. Hidden Reference"),
        ("FX",           "HiddenRef", "FX-only vs. Hidden Reference"),
        ("Voice+FX",     "HiddenRef", "Voice+FX vs. Hidden Reference"),
        ("NoAudio",      "HiddenRef", "No-Audio vs. Hidden Reference"),
    ]

    test_records = []
    raw_p_values = []

    for m1, m2, label in comparisons:
        w_stat, p_val = stats.wilcoxon(pivot_df[m1], pivot_df[m2])
        mean_diff = np.mean(pivot_df[m1] - pivot_df[m2])
        raw_p_values.append(p_val)
        test_records.append({
            "Comparison": label,
            "Model_A": m1,
            "Model_B": m2,
            "Mean_Diff": mean_diff,
            "W_Stat": w_stat,
            "p_raw": p_val,
        })

    p_adjusted = holm_bonferroni(raw_p_values)
    for rec, p_adj in zip(test_records, p_adjusted):
        rec["p_adj"] = p_adj
        rec["stars"] = get_significance_stars(p_adj)

    pairwise_df = pd.DataFrame(test_records)
    return stat_f, p_val_f, pairwise_df


# =============================================================================
# 5. LATEX TABLES EXPORT
# =============================================================================

def export_latex_tables(overall_summary: pd.DataFrame,
                        cat_summary: pd.DataFrame,
                        pairwise_df: pd.DataFrame,
                        stat_f: float, p_val_f: float):
    print("\n[4/5] Generating Publication LaTeX Tables...")

    tex_content = []
    tex_content.append("% =============================================================================")
    tex_content.append("% MUSHRA PERCEPTUAL EVALUATION RESULTS (N = 30 Participants, 4620 Ratings)")
    tex_content.append("% Auto-generated by analyze_perceptual_results.py")
    tex_content.append("% =============================================================================\n")

    # Table 1: Overall Summary
    tex_content.append("% --- TABLE 1: Overall MUSHRA Evaluation Summary ---")
    tex_content.append("\\begin{table}[t]")
    tex_content.append("\\centering")
    tex_content.append("\\caption{\\textbf{Perceptual MUSHRA Listening Evaluation Results.} Aggregate ratings ($N=660$ per condition across 30 listeners and 22 prompts). Scale is $0$--$100$. Repeated-measures Friedman test: $\\chi^2(6) = " + f"{stat_f:.2f}" + ", p < 0.001$.}")
    tex_content.append("\\label{tab:mushra_overall}")
    tex_content.append("\\vspace{0.15cm}")
    tex_content.append("\\begin{tabular}{lccccc}")
    tex_content.append("\\hline")
    tex_content.append("\\textbf{Model / Condition} & \\textbf{Mean $\\pm$ SD} & \\textbf{95\\% CI} & \\textbf{Median} & \\textbf{IQR} & \\textbf{$\\Delta$ vs. Ref} \\\\")
    tex_content.append("\\hline")

    ref_row = overall_summary[overall_summary["Model"] == "HiddenRef"].iloc[0]
    ref_mean = ref_row["Mean"]

    for _, row in overall_summary.sort_values("Mean", ascending=False).iterrows():
        name = row["DisplayName"]
        mean_sd = f"{row['Mean']:.2f} \\pm {row['SD']:.2f}"
        ci = f"[{row['CI95_Lower']:.1f}, {row['CI95_Upper']:.1f}]"
        med = f"{row['Median']:.1f}"
        iqr = f"{row['IQR']:.1f}"
        delta = f"{row['Mean'] - ref_mean:+.2f}"
        tex_content.append(f"{name:<28} & ${mean_sd}$ & ${ci}$ & ${med}$ & ${iqr}$ & ${delta}$ \\\\")

    tex_content.append("\\hline")
    tex_content.append("\\end{tabular}")
    tex_content.append("\\end{table}\n\n")

    # Table 2: Category-Stratified Breakdown
    tex_content.append("% --- TABLE 2: Category-Stratified Breakdown ---")
    tex_content.append("\\begin{table*}[t]")
    tex_content.append("\\centering")
    tex_content.append("\\caption{\\textbf{Category-Stratified Perceptual Ratings (Mean $\\pm$ SD).} Evaluated across Instrumental ($N=7$ prompts), Audiobook ($N=8$ prompts), Music ($N=4$ prompts), and Movie ($N=3$ prompts). Scale $0$--$100$.}")
    tex_content.append("\\label{tab:mushra_categories}")
    tex_content.append("\\vspace{0.15cm}")
    tex_content.append("\\begin{tabular}{lcccc}")
    tex_content.append("\\hline")
    tex_content.append("\\textbf{Model / Condition} & \\textbf{Instrumental} ($N=210$) & \\textbf{Audiobook} ($N=240$) & \\textbf{Music} ($N=120$) & \\textbf{Movie} ($N=90$) \\\\")
    tex_content.append("\\hline")

    categories = ["Instrumental", "Audiobook", "Music", "Movie"]
    for m in MODEL_ORDER:
        dname = MODEL_DISPLAY_NAMES.get(m, m)
        cells = []
        for cat in categories:
            match = cat_summary[(cat_summary["Group"] == cat) & (cat_summary["Model"] == m)]
            if len(match) > 0:
                r = match.iloc[0]
                cells.append(f"${r['Mean']:.2f} \\pm {r['SD']:.2f}$")
            else:
                cells.append("N/A")
        tex_content.append(f"{dname:<28} & " + " & ".join(cells) + " \\\\")

    tex_content.append("\\hline")
    tex_content.append("\\end{tabular*}")
    tex_content.append("\\end{table*}\n\n")

    # Table 3: Pairwise Wilcoxon Comparisons
    tex_content.append("% --- TABLE 3: Pairwise Wilcoxon Significance Tests ---")
    tex_content.append("\\begin{table}[t]")
    tex_content.append("\\centering")
    tex_content.append("\\caption{\\textbf{Pairwise Hypothesis Significance Tests.} Paired Wilcoxon signed-rank tests with Holm-Bonferroni Family-Wise Error Rate (FWER) correction across all $N=660$ matched trials ($^*p<0.05, ^{**}p<0.01, ^{***}p<0.001$).}")
    tex_content.append("\\label{tab:mushra_significance}")
    tex_content.append("\\vspace{0.15cm}")
    tex_content.append("\\begin{tabular}{lcccc}")
    tex_content.append("\\hline")
    tex_content.append("\\textbf{Comparison} & \\textbf{Mean $\\Delta$} & \\textbf{$W$-Statistic} & \\textbf{$p_{\\text{adj}}$ (Holm)} & \\textbf{Sig.} \\\\")
    tex_content.append("\\hline")

    for _, row in pairwise_df.iterrows():
        comp = row["Comparison"]
        diff = f"{row['Mean_Diff']:+.2f}"
        w_stat = f"{row['W_Stat']:.1f}"
        p_val = f"{row['p_adj']:.3e}" if row['p_adj'] < 0.001 else f"{row['p_adj']:.3f}"
        stars = row["stars"]
        tex_content.append(f"{comp:<36} & ${diff}$ & ${w_stat}$ & ${p_val}$ & {stars} \\\\")

    tex_content.append("\\hline")
    tex_content.append("\\end{tabular}")
    tex_content.append("\\end{table}\n")

    full_tex = "\n".join(tex_content)
    with open(TEX_OUT, "w", encoding="utf-8") as f:
        f.write(full_tex)

    print(f"      LaTeX tables successfully exported to: {TEX_OUT}")


# =============================================================================
# 6. MAIN & CONSOLE SUMMARY
# =============================================================================

def main():
    print("=============================================================================")
    print("  PERCEPTUAL MUSHRA EVALUATION ANALYSIS (FULL N=30 PARTICIPANTS)")
    print("=============================================================================\n")

    df = load_data()

    # 1. Overall summary
    overall_summary = compute_summary_table(df)

    # 2. Category summary
    cat_summary = compute_summary_table(df, group_col="category")

    # 3. Significance tests
    stat_f, p_val_f, pairwise_df = run_significance_tests(df)

    # 4. LaTeX Export
    export_latex_tables(overall_summary, cat_summary, pairwise_df, stat_f, p_val_f)

    # 5. Print Console Report
    print("\n[5/5] Analysis Complete! Summary of Key Results:")
    print("-----------------------------------------------------------------------------")
    print(f"{'Model / Condition':<28} {'Mean +/- SD':<20} {'95% CI':<18} {'Median [IQR]':<14}")
    print("-----------------------------------------------------------------------------")
    for _, row in overall_summary.sort_values("Mean", ascending=False).iterrows():
        mean_sd = f"{row['Mean']:.2f} +/- {row['SD']:.2f}"
        ci = f"[{row['CI95_Lower']:.1f}, {row['CI95_Upper']:.1f}]"
        med_iqr = f"{row['Median']:.1f} [{row['IQR']:.1f}]"
        print(f"{row['DisplayName']:<28} {mean_sd:<20} {ci:<18} {med_iqr:<14}")

    print("\nKey Pairwise Significance (Holm-Bonferroni Corrected):")
    print("-----------------------------------------------------------------------------")
    for _, row in pairwise_df.iterrows():
        print(f"  * {row['Comparison']:<36} : Delta={row['Mean_Diff']:+.2f}, p_adj={row['p_adj']:.3e} ({row['stars']})")
    print("=============================================================================\n")


if __name__ == "__main__":
    main()
