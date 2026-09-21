import pandas as pd
from scipy import stats
import numpy as np

# Load data
df = pd.read_csv('perceptual_ratings_tidy.csv')

# FILTER: out participants with average hidden reference score > 50
hidden_ref_scores = df[df['is_hidden_ref'] == True]
avg_hidden_ref = hidden_ref_scores.groupby('assessor_id')['score'].mean()
valid_assessors = avg_hidden_ref[avg_hidden_ref <= 50].index
df = df[df['assessor_id'].isin(valid_assessors)].copy()

def holm_bonferroni(p_values):
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        p_adj = p * (m - rank)
        running_max = max(running_max, p_adj)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted

def get_significance_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "n.s."

model_display = {
    'Tonmeister_1': 'Tonmeister (Expert)',
    'ICL': 'ICL (4B)',
    'Voice': 'Voice-only',
    'NoAudio': 'No-Audio Baseline',
    'FX': 'FX-only',
    'Voice+FX': 'Voice+FX',
    'HiddenRef': 'Hidden Ref'
}

# Pre-specified comparisons of interest (matches analyze_perceptual_results.py),
# rather than every pairwise combination, so Holm-Bonferroni is not spent on
# comparisons the paper doesn't report.
key_pairs = [
    ("Voice+FX", "NoAudio"),
    ("Voice", "NoAudio"),
    ("FX", "NoAudio"),
    ("Voice+FX", "Tonmeister_1"),
    ("Voice", "Tonmeister_1"),
    ("FX", "Tonmeister_1"),
    ("ICL", "Voice+FX"),
    ("ICL", "Tonmeister_1"),
    ("Voice", "FX"),
]

comparisons = [(m1, m2, f"{model_display[m1]} vs. {model_display[m2]}") for m1, m2 in key_pairs]

def run_tests_for_subset(subset_df, label):
    pivot_df = subset_df.pivot(index=["assessor_id", "prompt_slug"], columns="model", values="score").dropna()
    if len(pivot_df) == 0:
        return []
    
    test_records = []
    raw_p_values = []
    for m1, m2, comp_label in comparisons:
        try:
            w_stat, p_val = stats.wilcoxon(pivot_df[m1], pivot_df[m2])
            mean_diff = np.mean(pivot_df[m1] - pivot_df[m2])
            raw_p_values.append(p_val)
            test_records.append({
                "Comparison": comp_label,
                "Mean_Diff": mean_diff,
                "W_Stat": w_stat,
                "p_raw": p_val,
            })
        except Exception as e:
            pass # e.g. all zero differences

    if not raw_p_values: return []
    
    p_adjusted = holm_bonferroni(raw_p_values)
    for rec, p_adj in zip(test_records, p_adjusted):
        rec["p_adj"] = p_adj
        rec["stars"] = get_significance_stars(p_adj)
    
    return test_records

categories = ["Overall", "Instrumental", "Audiobook", "Music", "Movie"]

tex_content = []

for cat in categories:
    if cat == "Overall":
        sub_df = df
    else:
        sub_df = df[df['category'] == cat]
        
    records = run_tests_for_subset(sub_df, cat)
    if not records: continue
    
    tex_content.append(f"% --- Pairwise Wilcoxon Significance Tests: {cat} ---")
    tex_content.append("\\begin{table}[h!]")
    tex_content.append("\\centering")
    tex_content.append(f"\\caption{{\\textbf{{Pairwise Hypothesis Significance Tests ({cat}).}} Paired Wilcoxon signed-rank tests on {len(comparisons)} pre-specified comparisons of interest, with Holm-Bonferroni correction (filtered $N=25$ participants). ($^*p<0.05, ^{{**}}p<0.01, ^{{***}}p<0.001$).}}")
    tex_content.append(f"\\label{{tab:wilcoxon_{cat.lower()}}}")
    tex_content.append("\\vspace{0.15cm}")
    tex_content.append("\\begin{tabular}{lcccc}")
    tex_content.append("\\hline")
    tex_content.append("\\textbf{Comparison} & \\textbf{Mean $\\Delta$} & \\textbf{$W$-Statistic} & \\textbf{$p_{\\text{adj}}$} & \\textbf{Sig.} \\\\")
    tex_content.append("\\hline")

    for row in records:
        comp = row["Comparison"]
        diff = f"{row['Mean_Diff']:+.2f}"
        w_stat = f"{row['W_Stat']:.1f}"
        p_val = f"{row['p_adj']:.3e}" if row['p_adj'] < 0.001 else f"{row['p_adj']:.3f}"
        stars = row["stars"]
        tex_content.append(f"{comp:<40} & ${diff}$ & ${w_stat}$ & ${p_val}$ & {stars} \\\\")

    tex_content.append("\\hline")
    tex_content.append("\\end{tabular}")
    tex_content.append("\\end{table}\n")

with open('wilcoxon_per_category.tex', 'w') as f:
    f.write("\n".join(tex_content))

print("Saved to wilcoxon_per_category.tex")
