import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
df = pd.read_csv('perceptual_ratings_tidy.csv')

# Filter out participants with average hidden reference score > 50
hidden_ref_scores = df[df['is_hidden_ref'] == True]
avg_hidden_ref = hidden_ref_scores.groupby('assessor_id')['score'].mean()
valid_assessors = avg_hidden_ref[avg_hidden_ref <= 50].index
df_filtered = df[df['assessor_id'].isin(valid_assessors)].copy()

# The user wants models across categories
categories = ['Instrumental', 'Audiobook', 'Music', 'Movie']
models_to_include = ['Tonmeister_1', 'ICL', 'Voice', 'FX', 'Voice+FX', 'NoAudio', 'HiddenRef']

# Clean up model names to be valid python identifiers for formula
# Actually, if we use C(model), statsmodels handles it, but let's be safe
df_filtered['model_clean'] = df_filtered['model'].replace({'Voice+FX': 'Voice_FX', 'Tonmeister_1': 'Tonmeister', 'NoAudio': 'No_Audio'})

models_clean = ['Tonmeister', 'ICL', 'Voice', 'FX', 'Voice_FX', 'No_Audio', 'HiddenRef']

results = []

for category in categories:
    print(f"\nFitting LMM for Category: {category}")
    df_cat = df_filtered[df_filtered['category'] == category].copy()
    
    # Check if we have enough data
    if len(df_cat) == 0:
        continue
        
    # Fit Linear Mixed-Effects Model
    # Fixed effect: model (without intercept, so we get the mean for each model directly)
    # Random effect: assessor_id (group)
    # Variance Component: prompt_slug
    # By omitting the intercept (-1), the coefficients ARE the Estimated Marginal Means (EMMs).
    present_models = df_cat['model_clean'].unique().tolist()
    formula = "score ~ C(model_clean, levels=present_models) - 1"
    vc = {'prompt': '0 + C(prompt_slug)'}
    
    try:
        model = smf.mixedlm(formula, df_cat, groups=df_cat["assessor_id"], vc_formula=vc)
        result = model.fit()
        
        # Extract fixed effects (EMMs) and their standard errors
        for model_name in present_models:
            exact_coef_name = f"C(model_clean, levels=present_models)[{model_name}]"
            
            if exact_coef_name in result.fe_params.index:
                emm = result.fe_params[exact_coef_name]
                se = result.bse[exact_coef_name]
                results.append({
                    'Category': category,
                    'Model': model_name.replace('Voice_FX', 'Voice+FX').replace('No_Audio', 'NoAudio').replace('Tonmeister', 'Tonmeister_1'),
                    'EMM': emm,
                    'SE': se
                })
    except Exception as e:
        print(f"Error fitting LMM for {category}: {e}")

# Also calculate Overall (All categories)
print("\nFitting LMM for Overall")
present_models = df_filtered['model_clean'].unique().tolist()
formula = "score ~ C(model_clean, levels=present_models) - 1"
vc = {'prompt': '0 + C(prompt_slug)'}
try:
    model_overall = smf.mixedlm(formula, df_filtered, groups=df_filtered["assessor_id"], vc_formula=vc)
    result_overall = model_overall.fit()
    
    for model_name in present_models:
        # Patsy format: C(model_clean, levels=present_models)[Tonmeister]
        exact_coef_name = f"C(model_clean, levels=present_models)[{model_name}]"
        
        if exact_coef_name in result_overall.fe_params.index:
            emm = result_overall.fe_params[exact_coef_name]
            se = result_overall.bse[exact_coef_name]
            results.append({
                'Category': 'Overall',
                'Model': model_name.replace('Voice_FX', 'Voice+FX').replace('No_Audio', 'NoAudio').replace('Tonmeister', 'Tonmeister_1'),
                'EMM': emm,
                'SE': se
            })
except Exception as e:
    print(f"Error fitting overall LMM: {e}")

# Format as DataFrame
results_df = pd.DataFrame(results)

# Create a pivot table
pivot_table = results_df.pivot(index='Model', columns='Category', values=['EMM', 'SE'])

# Reorder models
model_order = ['Tonmeister_1', 'FX', 'Voice', 'Voice+FX', 'ICL', 'NoAudio', 'HiddenRef']
pivot_table = pivot_table.reindex(model_order)

# Reorder categories
cat_order = ['Instrumental', 'Audiobook', 'Music', 'Movie', 'Overall']
# Flatten columns and format
print("\n--- LMM Results (Estimated Marginal Means ± SE) ---")

output_str = "Model"
for cat in cat_order:
    output_str += f" | {cat}"
print(output_str)
print("-" * len(output_str))

for model_name in model_order:
    row_str = f"{model_name:15}"
    for cat in cat_order:
        if (model_name, cat) in pivot_table.index and cat in pivot_table.columns.levels[1]:
            pass # this is getting complicated.
        try:
            emm = pivot_table.loc[model_name, ('EMM', cat)]
            se = pivot_table.loc[model_name, ('SE', cat)]
            if pd.isna(emm):
                row_str += " |      -      "
            else:
                row_str += f" | {emm:5.1f} ± {se:3.1f}"
        except KeyError:
            row_str += " |      -      "
    print(row_str)

# Define the display names and order
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

# Find max EMM in each column to bold
max_emms = {}
for cat in cat_order:
    # Get all valid EMMs for this category
    emms = [pivot_table.loc[m, ('EMM', cat)] for m in ordered_models if not pd.isna(pivot_table.loc[m, ('EMM', cat)]) and m != 'HiddenRef']
    max_emms[cat] = max(emms)

latex_code = "\\begin{table*}\n\\centering\n"
latex_code += "\\caption{\\textbf{Perceptual Listening Test Results ($N = 25$ Listeners).} Estimated Marginal Means $\\pm$ Standard Errors (SE) across 22 prompts (scale: 0-100).}\n"
latex_code += "\\label{tab:perceptual_results}\n\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1}\n"
latex_code += "\\begin{tabular}{lccccc}\n\\hline\n"
latex_code += "\\textbf{Model / Condition} & \\textbf{Instrumental} ($N=175$) & \\textbf{Audiobook} ($N=200$) & \\textbf{Music} ($N=100$) & \\textbf{Movie} ($N=75$) & \\textbf{Overall} ($N=550$)  \\\\\n\\hline\n"

for model_name in ordered_models:
    latex_code += model_display[model_name]
    for cat in cat_order:
        try:
            emm = pivot_table.loc[model_name, ('EMM', cat)]
            se = pivot_table.loc[model_name, ('SE', cat)]
            if pd.isna(emm):
                latex_code += " & - "
            else:
                # Bold if it matches the maximum (with some tolerance for floating point)
                if abs(emm - max_emms[cat]) < 1e-4:
                    latex_code += f" & $\\mathbf{{{emm:.2f} \\pm {se:.2f}}}$"
                else:
                    latex_code += f" & ${emm:.2f} \\pm {se:.2f}$"
        except KeyError:
            latex_code += " & - "
    latex_code += " \\\\\n"

latex_code += "\\hline\n\\end{tabular}\n\\end{table*}\n"

with open('lmm_table.tex', 'w') as f:
    f.write(latex_code)
    
print("\nLaTeX table saved to lmm_table.tex")
