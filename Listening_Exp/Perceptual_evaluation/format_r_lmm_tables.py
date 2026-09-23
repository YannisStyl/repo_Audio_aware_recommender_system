"""
Formats run_lmm_glmmTMB.R's output (lmm_emm_r.csv, lmm_pairwise_contrasts_r.csv) into
lmm_table.tex / lmm_contrasts.tex, which perceptual_results_tables.tex is assembled
from. Run run_lmm_glmmTMB.R in R first, then this script.
"""
import pandas as pd

model_display = {
    'Tonmeister_1': 'Tonmeister (Expert)',
    'ICL': 'ICL (Qwen3.5-4B)',
    'Voice': 'Voice-only (0.8B)',
    'NoAudio': 'No Audio (0.8B Baseline)',
    'FX': 'FX-only (0.8B)',
    'Voice+FX': 'Voice+FX (0.8B Dual)',
    'HiddenRef': 'Hidden Reference (Original)'
}
ordered_models = ['Tonmeister_1', 'ICL', 'Voice', 'NoAudio', 'FX', 'Voice+FX', 'HiddenRef']
cat_order = ['Instrumental', 'Audiobook', 'Mixed', 'Overall']

# =============================================================================
# Table 1: EMMs, from glmmTMB (heteroscedastic residual variance per condition).
# Grouped by row (Baselines / Audio-Conditioned) via \multirow. Display names here
# are table-specific (shorter than model_display, used below for the contrasts
# table's "A vs. B" labels, which are left unchanged).
# =============================================================================
emm_display = {
    'Tonmeister_1': 'Tonmeister (Expert)',
    'ICL': 'ICL (4B)',
    'HiddenRef': 'Hidden Reference',
    'NoAudio': 'No Audio (0.8B)',
    'Voice': 'Voice-only (0.8B)',
    'FX': 'FX-only (0.8B)',
    'Voice+FX': 'Voice+FX (0.8B Dual)',
}
row_groups = [
    ("Baselines", ['Tonmeister_1', 'ICL', 'HiddenRef', 'NoAudio']),
    ("Audio-Conditioned", ['Voice', 'FX', 'Voice+FX']),
]

emm = pd.read_csv('lmm_emm_r.csv')
pivot = emm.pivot(index='Model', columns='Category', values=['EMM', 'SE'])

max_emms = {}
for cat in cat_order:
    emms = [pivot.loc[m, ('EMM', cat)] for m in ordered_models if m != 'HiddenRef']
    max_emms[cat] = max(emms)

latex_code = "\\begin{table*}[t]\n\\centering\n"
latex_code += ("\\caption{\\textbf{Perceptual Listening Test Results ($N = 25$ Listeners).} "
               "Estimated Marginal Means $\\pm$ SE across 22 prompts (scale: 0-100 $\\uparrow$).}\n")
latex_code += "\\label{tab:perceptual_results}\n\\setlength{\\tabcolsep}{6pt}\n\\renewcommand{\\arraystretch}{1}\n\\scriptsize\n"
latex_code += "\\begin{tabular}{llcccc}\n\\hline\n"
latex_code += "\\textbf{Category} & \\textbf{Condition} & \\textbf{Instrumental} ($N=175$) & \\textbf{Audiobook} ($N=200$) & \\textbf{Mixed} ($N=175$) & \\textbf{Overall} ($N=550$)  \\\\\n\\hline\n"

for group_name, group_models in row_groups:
    latex_code += f"\\multirow{{{len(group_models)}}}{{*}}{{{{{group_name}}}}} \n"
    for model_name in group_models:
        latex_code += f"& {emm_display[model_name]}"
        for cat in cat_order:
            e = pivot.loc[model_name, ('EMM', cat)]
            se = pivot.loc[model_name, ('SE', cat)]
            if abs(e - max_emms[cat]) < 1e-4 and model_name != 'HiddenRef':
                latex_code += f" & $\\mathbf{{{e:.2f} \\pm {se:.2f}}}$"
            else:
                latex_code += f" & ${e:.2f} \\pm {se:.2f}$"
        latex_code += " \\\\\n"
    latex_code += "\\hline\n"

latex_code += "\\end{tabular}\n\\end{table*}\n"
with open('lmm_table.tex', 'w') as f:
    f.write(latex_code)
print("Wrote lmm_table.tex (from R/glmmTMB EMMs)")

# =============================================================================
# Table 2: planned contrasts, multivariate-t (mvt) corrected ONCE across the whole
# family (not per category table) -- see run_lmm_glmmTMB.R. mvt uses the actual
# covariance between these test statistics (several share a reference condition,
# e.g. Voice-NoAudio and Voice-ICL within Instrumental, so they're correlated, not
# independent), giving the same family-wise error guarantee as Holm-Bonferroni
# without paying Holm's independence-worst-case penalty. p_holm is also in
# lmm_pairwise_contrasts_r.csv for comparison, but not used here.
# =============================================================================
contrasts = pd.read_csv('lmm_pairwise_contrasts_r.csv')
n_pairs = contrasts['contrast'].nunique()
n_families = contrasts['category'].nunique()
n_tests = len(contrasts)


def relabel(contrast_str):
    m1, m2 = [s.strip() for s in contrast_str.split(' - ')]
    return f"{model_display[m1]} vs. {model_display[m2]}"


tex_content = []
tex_content.append("% --- Planned Pairwise Contrasts (glmmTMB joint model, heteroscedastic residual) ---")
tex_content.append(f"% Multivariate-t (mvt) adjustment applied ONCE across all {n_tests} tests -- a fixed,")
tex_content.append(f"% specific set of (comparison, category) cells, not a full pairs x categories cross")
tex_content.append(f"% product ({n_pairs} distinct pairs appear across {n_families} categories, but only {n_tests}")
tex_content.append("% of the possible combinations are tested).")
tex_content.append("\\begin{table*}[h!]")
tex_content.append("\\centering")
tex_content.append("\\caption{\\textbf{Pairwise Hypothesis Significance Tests.} Wald contrasts from the joint "
                    "glmmTMB model (model $\\times$ category, heteroscedastic residual variance), for a fixed set "
                    f"of {n_tests} planned (comparison, category) combinations, "
                    "corrected via the multivariate-t adjustment (Hothorn et al., 2008) as a single family, "
                    "accounting for the correlation among planned contrasts. "
                    "($^*p<0.05, ^{**}p<0.01, ^{***}p<0.001$).}")
tex_content.append("\\label{tab:lmm_contrasts_r}")
tex_content.append("\\vspace{0.15cm}")
tex_content.append("\\begin{tabular}{llccc}")
tex_content.append("\\hline")
tex_content.append("\\textbf{Comparison} & \\textbf{Category} & \\textbf{Estimate} & \\textbf{$p_{\\text{adj}}$} & \\textbf{Sig.} \\\\")
tex_content.append("\\hline")

for cat in cat_order:
    sub = contrasts[contrasts['category'] == cat]
    for _, row in sub.iterrows():
        comp = relabel(row['contrast'])
        est = f"{row['estimate']:+.2f}"
        p_val = f"{row['p_mvt']:.3e}" if row['p_mvt'] < 0.001 else f"{row['p_mvt']:.3f}"
        tex_content.append(f"{comp:<40} & {cat:<12} & ${est}$ & ${p_val}$ & {row['stars']} \\\\")

tex_content.append("\\hline")
tex_content.append("\\end{tabular}")
tex_content.append("\\end{table*}\n")

with open('lmm_contrasts.tex', 'w') as f:
    f.write("\n".join(tex_content))
print(f"Wrote lmm_contrasts.tex (from R/glmmTMB contrasts, single {n_tests}-test mvt family)")
