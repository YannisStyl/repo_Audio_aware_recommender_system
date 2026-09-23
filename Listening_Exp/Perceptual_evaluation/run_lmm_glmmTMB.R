# Heteroscedastic-residual pairwise significance + EMM pipeline. Fits one joint
# mixed model over all screened listener ratings, then reports estimated marginal
# means per (condition, category) cell and a fixed set of planned pairwise
# contrasts.
#
# Input: perceptual_ratings_tidy.csv (from analyze_perceptual_results.py's raw-JSON
# ingestion) -- this script does its own screening/filtering (below).
# Output: lmm_emm_r.csv, lmm_pairwise_contrasts_r.csv -- read by
# format_r_lmm_tables.py to produce lmm_table.tex / lmm_contrasts.tex, which
# perceptual_results_tables.tex is assembled from.
#
# Requires: install.packages(c("glmmTMB", "emmeans"))

library(glmmTMB)
library(emmeans)

df <- read.csv("perceptual_ratings_tidy.csv")

# Screen out assessors who failed the hidden-reference control check (average
# hidden-reference score > 50) -- see AGENT.md section 11.
hidden_ref_scores <- df[df$is_hidden_ref == "True" | df$is_hidden_ref == TRUE, ]
avg_hidden_ref <- aggregate(score ~ assessor_id, data = hidden_ref_scores, FUN = mean)
valid_assessors <- avg_hidden_ref$assessor_id[avg_hidden_ref$score <= 50]
df <- df[df$assessor_id %in% valid_assessors, ]

# Songs (CSV label "Music") and Movie are merged into a single "Mixed" category.
df$category[df$category %in% c("Music", "Movie")] <- "Mixed"

model_levels <- c("Tonmeister_1", "ICL", "Voice", "NoAudio", "FX", "Voice+FX", "HiddenRef")
category_levels <- c("Instrumental", "Audiobook", "Mixed")
df$model <- factor(df$model, levels = model_levels)
df$category <- factor(df$category, levels = category_levels)
df$assessor_id <- factor(df$assessor_id)
df$prompt_slug <- factor(df$prompt_slug)

stopifnot(length(valid_assessors) == 25, nrow(df) == 3850)

m <- glmmTMB(
  score ~ model * category + (1 | assessor_id) + (1 | prompt_slug),
  dispformula = ~ model,     # separate residual variance per condition
  data = df,
  REML = TRUE
)

cat("=== Convergence check ===\n")
cat("pdHess (should be TRUE):", m$sdr$pdHess, "\n")
if (!is.null(m$fit$convergence) && m$fit$convergence != 0) {
  cat("WARNING: optimizer convergence code:", m$fit$convergence, "\n")
}
print(summary(m))

# If pdHess is FALSE or there are convergence warnings above, the model is unstable
# with a per-condition dispformula. Fallback: refit with dispformula = ~1 (shared
# residual variance) and disclose that limitation when reporting results -- do not
# silently trust an unconverged heteroscedastic fit.

# weights = "cells": average over category weighted by each cell's observation count
# (175/200/175 -> Audiobook has 8 prompts vs. 7 for Instrumental/Mixed), so "Overall"
# matches the raw pooled mean. emmeans' default weights = "equal" would average the
# 3 category cells unweighted, which is NOT what "Overall" should mean here (the
# default gives FX Overall = 35.32, the unweighted mean of 38.74/26.00/41.22; the
# raw pooled mean, and what weights="cells" gives, is 34.90).
cat_weights <- table(df$category[df$model == model_levels[1]])
cat_weights <- as.list(as.numeric(cat_weights[category_levels]) / sum(cat_weights))
names(cat_weights) <- category_levels

emm_overall <- emmeans(m, ~ model, weights = "cells")
emm_by_cat <- emmeans(m, ~ model | category)

emm_overall_df <- as.data.frame(emm_overall)
emm_overall_df$category <- "Overall"
emm_cat_df <- as.data.frame(emm_by_cat)
emm_all <- rbind(
  emm_overall_df[, c("model", "category", "emmean", "SE")],
  emm_cat_df[, c("model", "category", "emmean", "SE")]
)
names(emm_all) <- c("Model", "Category", "EMM", "SE")
write.csv(emm_all, "lmm_emm_r.csv", row.names = FALSE)
cat("\nWrote lmm_emm_r.csv\n")

# Planned contrasts: exactly the (comparison, category) cells this pipeline reports
# significance for -- not a full pairs x categories cross product. Restricting to a
# small, specific set of planned comparisons keeps the multiple-comparisons
# correction from paying a penalty for combinations that are never reported. No
# "Overall" row is included here (mk_overall / cat_weights below are unused by this
# contrast set, kept only because the EMM table above still needs weights="cells").
# Built as explicit named contrast vectors rather than filtering pairs()'s
# auto-generated contrast labels, because "Voice+FX" (the "+") is not guaranteed to
# round-trip through emmeans' default label text.
#
# All contrasts are built as vectors over ONE flat 21-cell (model x category)
# emmGrid so adjust = "mvt" sees their TRUE joint covariance (contrasts sharing a
# reference condition, e.g. Voice-NoAudio and Voice-ICL both @ Instrumental, are
# correlated, not independent -- mvt accounts for that instead of assuming
# worst-case independence like Holm-Bonferroni does).
emm_full <- emmeans(m, ~ model * category)
grid_df <- as.data.frame(emm_full)  # 21 rows, one per (model, category) cell

cell_index <- function(mod, cat) which(grid_df$model == mod & grid_df$category == cat)

# Same weights = "cells" logic as emm_overall above (N-weighted, not equal, average
# across categories), expressed as a contrast vector over the flat 21-cell grid.
# Unused by the contrast set below (no "Overall" row is included) but kept as a
# reusable helper for adding an Overall contrast.
mk_overall <- function(pos, neg) {
  v <- rep(0, nrow(grid_df))
  for (cat in category_levels) {
    v[cell_index(pos, cat)] <- v[cell_index(pos, cat)] + cat_weights[[cat]]
    v[cell_index(neg, cat)] <- v[cell_index(neg, cat)] - cat_weights[[cat]]
  }
  v
}
mk_category <- function(pos, neg, cat) {
  v <- rep(0, nrow(grid_df))
  v[cell_index(pos, cat)] <- 1
  v[cell_index(neg, cat)] <- -1
  v
}

# (positive, negative, category): the fixed set of pairwise comparisons this
# pipeline reports significance for.
planned_contrasts <- list(
  c("Voice", "NoAudio",      "Audiobook"),
  c("FX",    "NoAudio",      "Mixed"),
  c("Voice", "ICL",          "Instrumental"),
  c("Voice", "NoAudio",      "Instrumental"),
  c("Voice", "Tonmeister_1", "Instrumental"),
  c("FX",    "ICL",          "Mixed"),
  c("FX",    "Tonmeister_1", "Mixed")
)

planned_full <- list()
for (t in planned_contrasts) {
  planned_full[[paste(t[1], "-", t[2], "@", t[3])]] <- mk_category(t[1], t[2], t[3])
}
stopifnot(length(planned_full) == length(planned_contrasts))

contrast_obj <- contrast(emm_full, method = planned_full)

# adjust = "mvt": corrects using the actual covariance between these test
# statistics (computed from the fitted model, via mvtnorm) instead of Holm's
# worst-case assumption that they could be independent. Same family-wise error
# guarantee, but doesn't pay a correlation penalty that isn't actually there.
# p_holm is kept alongside for comparison/transparency, not used for the stars.
mvt_summary <- as.data.frame(summary(contrast_obj, adjust = "mvt"))
raw_p <- as.data.frame(summary(contrast_obj, adjust = "none"))$p.value
holm_p <- p.adjust(raw_p, method = "holm")

all_planned <- data.frame(
  contrast = sub(" @ .*", "", mvt_summary$contrast),
  category = sub(".* @ ", "", mvt_summary$contrast),
  estimate = mvt_summary$estimate,
  SE = mvt_summary$SE,
  z.ratio = mvt_summary$z.ratio,
  p.value = raw_p,
  p_mvt = mvt_summary$p.value,
  p_holm = holm_p
)
all_planned$stars <- ifelse(all_planned$p_mvt < 0.001, "***",
                      ifelse(all_planned$p_mvt < 0.01, "**",
                      ifelse(all_planned$p_mvt < 0.05, "*", "n.s.")))

write.csv(all_planned, "lmm_pairwise_contrasts_r.csv", row.names = FALSE)
cat("Wrote lmm_pairwise_contrasts_r.csv (mvt-adjusted, Holm kept for comparison)\n")
print(all_planned)
