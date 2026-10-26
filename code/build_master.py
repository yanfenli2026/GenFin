"""Assemble main tables, supplementary tables, the workflow figure and the revised
figure set from the finished experiment outputs (E1-E8, foundation).

Run: python code/build_master.py

No model is fitted here. Every number is read from an existing CSV in
output/ and only reshaped or rounded. Output:
  output/MAIN_TABLES/   table1..3 (CSV + Markdown)
  output/SUPP_TABLES/   supplementary tables (CSV) + INDEX.csv
  figures/ (config.FIG_DIR)       fig1..fig5 (main) and figS* (supplementary), PNG + PDF
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

ROOT = Path(config.ROOT)
OUT = Path(config.OUT_MAIN)
MAIN = OUT / "MAIN_TABLES"
SUPP = OUT / "SUPP_TABLES"
FIG = Path(config.FIG_DIR)
for d in (MAIN, SUPP, FIG):
    d.mkdir(parents=True, exist_ok=True)


def f3(x):
    return "" if pd.isna(x) else f"{x:.3f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- Table 1
r2 = pd.read_csv(OUT / "e1_benchmark/bench_r2_wide.csv")
unc = pd.read_csv(OUT / "e1_benchmark/uncertainty_strict_grouped5x5.csv").set_index("model")
orig = r2[r2.spec == "original"].set_index("model")
strict = r2[r2.spec == "strict"].set_index("model")
order = unc.sort_values("rank").index.tolist()
rows = []
for m in order:
    u = unc.loc[m]
    rows.append({
        "Model": m,
        "Original features, random 80/20": f3(orig.loc[m, "legacy_random_80_20"]),
        "Original features, grouped 5x5": f3(orig.loc[m, "grouped_5x5"]),
        "Strict, random 80/20": f3(strict.loc[m, "legacy_random_80_20"]),
        "Strict, grouped 5x5 mean (s.d.)": f"{u.R2_mean:.3f} ({u.R2_sd:.3f})",
        "Strict, grouped 5x5 corrected 95% CI": f"[{u.R2_ci95_corrected_lo:.3f}, {u.R2_ci95_corrected_hi:.3f}]",
        "Strict, temporal": f3(strict.loc[m, "temporal"]),
        "Strict, unseen firms future years": f3(strict.loc[m, "unseen_firm_future"]),
    })
t1 = pd.DataFrame(rows)
t1.to_csv(MAIN / "table1_benchmark_r2.csv", index=False)

# ---------------------------------------------------------------- Table 2
gs = pd.read_csv(OUT / "e4_generative/generative_summary.csv")
label = {
    ("B1_noise", 0): "B1 noise baseline (original mechanism)",
    ("B1_noise_log", 0): "B1-log noise baseline",
    ("B2_gauss", 0): "B2 Gaussian baseline",
    ("CTGAN", 300): "CTGAN, 300 epochs",
    ("CTGAN_proj", 300): "CTGAN + projection, 300 epochs",
    ("TVAE", 300): "TVAE, 300 epochs",
    ("TVAE_proj", 300): "TVAE + projection, 300 epochs",
    ("TVAE", 600): "TVAE, 600 epochs",
}
gs["key"] = list(zip(gs.generator, gs.epochs))
missing = [k for k in label if k not in set(gs.key)]
if missing:
    print("generator labels not found:", missing, "available:", sorted(set(gs.key)))


def pm(r, c, d=3):
    return f"{r[c + '_mean']:.{d}f} ± {r[c + '_std']:.{d}f}"


rows = []
for k, lab in label.items():
    sub = gs[gs.key == k]
    if sub.empty:
        continue
    r = sub.iloc[0]
    rows.append({
        "Generator": lab,
        "KS mean": pm(r, "KS_mean"),
        "W (ref. s.d.)": pm(r, "W_std_mean"),
        "JSD (base 2)": pm(r, "JSD_mean"),
        "Pearson MAD": pm(r, "corr_MAD"),
        "CVR (any)": pm(r, "CVR"),
        "Logistic disc. AUC": pm(r, "disc_AUC_logit"),
        "DCR median": pm(r, "DCR_median") if "DCR_median_mean" in gs.columns else "",
    })
t2 = pd.DataFrame(rows)
t2.to_csv(MAIN / "table2_generative.csv", index=False)

# ---------------------------------------------------------------- Table 3
roa = pd.read_csv(OUT / "e6_econometrics/roa_coefficient_across_specifications.csv")
real = pd.read_csv(OUT / "e6_econometrics/real_panel_roa_coefficient.csv")
def pfmt(p):
    return "<1e-15" if p < 1e-15 else f"{p:.2g}"


rows = []
for _, r in roa.iterrows():
    rows.append({"Sample": "Reference synthetic panel (N = 4,440; 296 firms)",
                 "Specification": f"{r.model}: {r.description}",
                 "ROA coef.": f"{r.coef:.4f}", "SE": f"{r.se:.4f}",
                 "p": pfmt(r.p), "95% CI": f"[{r.ci_low:.3f}, {r.ci_high:.3f}]",
                 "SE type": r.se_type})
for _, r in real.iterrows():
    rows.append({"Sample": f"Real 46-firm sample ({r['sample']})",
                 "Specification": ("Pooled OLS, year + sector dummies" if "_M2_" in r.model else "Firm + year FE") + f", spec {r.spec}",
                 "ROA coef.": f"{r.coef:.4f}", "SE": f"{r.se:.4f}",
                 "p": pfmt(r.p), "95% CI": f"[{r.ci_low:.3f}, {r.ci_high:.3f}]",
                 "SE type": "cluster(firm)"})
t3 = pd.DataFrame(rows)
t3.to_csv(MAIN / "table3_roa_coefficient.csv", index=False)

with open(MAIN / "main_tables.md", "w", encoding="utf-8") as fh:
    for name, t in (("Table 1", t1), ("Table 2", t2), ("Table 3", t3)):
        fh.write(f"## {name}\n\n{md_table(t)}\n\n")

# ---------------------------------------------------------------- Supplementary tables
supp = [
    ("S01_feature_leakage_audit", "foundation/feature_audit.csv"),
    ("S02_feature_sets", "foundation/feature_sets.csv"),
    ("S03_dgp_parameters", "foundation/dgp_parameters.csv"),
    ("S04_leakage_perturbation_check", "foundation/leakage_perturbation_check.csv"),
    ("S05_split_sizes", "foundation/split_sizes.csv"),
    ("S06_synthetic_variance_decomposition", "foundation/synthetic_debt_ratio_variance_decomposition.csv"),
    ("S07_benchmark_all_models_protocols", "e1_benchmark/bench_summary_long.csv"),
    ("S08_benchmark_inflation_lightgbm_ridge", "e1_benchmark/inflation_table.csv"),
    ("S09_uncertainty_strict_grouped5x5", "e1_benchmark/uncertainty_strict_grouped5x5.csv"),
    ("S10_pairwise_nadeau_bengio_top3", "e1_benchmark/pairwise_nadeau_bengio_top3.csv"),
    ("S11_across_panels_lightgbm", "e1_benchmark/across_panels_lightgbm_summary.csv"),
    ("S12_dgp_sensitivity_lambda_sigma", "e1_benchmark/sensitivity_lambda_sigma.csv"),
    ("S13_secondary_targets", "e1_benchmark/secondary_targets_strict_grouped.csv"),
    ("S14_wacc_leakage", "e1_benchmark/wacc_leakage_lightgbm_grouped.csv"),
    ("S15_learning_curve", "e1_benchmark/learning_curve_lightgbm_grouped.csv"),
    ("S16_legacy_reproduction_benchmark", "e1_benchmark/legacy_reproduction_check.csv"),
    ("S17_deep_tabular_summary", "e2_deep_tabular/summary.csv"),
    ("S18_deep_tabular_pairwise_tests", "e2_deep_tabular/pairwise_tests_grouped5.csv"),
    ("S19_shap_spec_performance", "e3_shap/spec_performance.csv"),
    ("S20_shap_spec_difference_tests", "e3_shap/spec_r2_difference_tests.csv"),
    ("S21_shap_ranking_S1", "e3_shap/shap_ranking_S1.csv"),
    ("S22_shap_ranking_S2", "e3_shap/shap_ranking_S2.csv"),
    ("S23_shap_ranking_S4", "e3_shap/shap_ranking_S4.csv"),
    ("S24_shap_block_importance", "e3_shap/shap_block_importance.csv"),
    ("S25_shap_S4_stability", "e3_shap/stability_S4_summary.csv"),
    ("S26_shap_S4_rank_ranges", "e3_shap/stability_S4_ranks.csv"),
    ("S27_rf_surrogate_vs_lgbm", "e3_shap/rf_surrogate_rank_agreement.csv"),
    ("S28_tableXV_provenance", "e3_shap/table_XV_provenance_reproduction.csv"),
    ("S29_generative_tableX_provenance", "e4_generative/diag_table_x_provenance.csv"),
    ("S30_generative_original_reproduction", "e4_generative/diag_original_reproduction.csv"),
    ("S31_generative_pipeline_baseline_20seeds", "e4_generative/diag_pipeline_noise_generator_rerun.csv"),
    ("S32_generative_summary_all", "e4_generative/generative_summary.csv"),
    ("S33_generative_per_feature_ks", "e4_generative/per_feature_ks_table.csv"),
    ("S34_generative_cvr_by_column", "e4_generative/cvr_by_column_summary.csv"),
    ("S35_generative_loss_diagnostics", "e4_generative/loss_diagnostics.csv"),
    ("S36_generative_sensitivity_dgp", "e4_generative/sensitivity_dgp_grid.csv"),
    ("S37_generative_sensitivity_noise", "e4_generative/sensitivity_noise_scale.csv"),
    ("S38_generative_sensitivity_panel_seed", "e4_generative/sensitivity_panel_seed.csv"),
    ("S39_opt_legacy_optima", "e5_optimization/legacy_optima.csv"),
    ("S40_opt_legacy_pareto_count_vs_grid", "e5_optimization/legacy_pareto_count_vs_grid.csv"),
    ("S41_opt_model_parameters", "e5_optimization/model_parameters.csv"),
    ("S42_opt_sector_calibration", "e5_optimization/calibration_sector_parameters.csv"),
    ("S43_opt_scenarios", "e5_optimization/scenarios.csv"),
    ("S44_opt_base_results", "e5_optimization/base_results_sector_scenario.csv"),
    ("S45_opt_mc_summary", "e5_optimization/mc_summary_sector_scenario.csv"),
    ("S46_opt_mc_scenario_differences", "e5_optimization/mc_scenario_differences.csv"),
    ("S47_opt_tornado", "e5_optimization/tornado_oat.csv"),
    ("S48_opt_channel_decomposition", "e5_optimization/scenario_channel_decomposition.csv"),
    ("S49_opt_seed_sensitivity", "e5_optimization/seed_sensitivity_summary.csv"),
    ("S50_opt_sector_actual_vs_model", "e5_optimization/sector_table_actual_vs_model.csv"),
    ("S51_opt_dgp_grid", "e5_optimization/dgp_grid_sensitivity.csv"),
    ("S52_econ_design_matrix", "e6_econometrics/design_matrix_diagnostics.csv"),
    ("S53_econ_synthetic_M1_M6", "e6_econometrics/table_synthetic_panel_M1_M6.csv"),
    ("S54_econ_synthetic_strict", "e6_econometrics/table_synthetic_panel_strict_M7_M6_M8.csv"),
    ("S55_econ_specification_tests", "e6_econometrics/synthetic_panel_specification_tests.csv"),
    ("S56_econ_real_panel", "e6_econometrics/table_real_panel.csv"),
    ("S57_real_descriptives", "e7_real_data/a_descriptives_real_vs_synthetic.csv"),
    ("S58_real_sector_mix", "e7_real_data/a_sector_mix_and_mean_dr.csv"),
    ("S59_real_variance_decomposition", "e7_real_data/a_variance_decomposition.csv"),
    ("S60_real_prediction_all", "e7_real_data/bc_results_all.csv"),
    ("S61_real_grouped5_repeats", "e7_real_data/b_grouped5_repeat_summary.csv"),
    ("S62_real_robustness_subsample", "e7_real_data/b_robustness_grouped5_repeat_summary.csv"),
    ("S63_real_covariate_overlap", "e7_real_data/c_covariate_overlap.csv"),
    ("S64_real_shap_grouped", "e7_real_data/d_shap_grouped_strict.csv"),
    ("S65_real_sector_median", "e7_real_data/e_sector_median_summary.csv"),
    ("S66_real_sector_map", "foundation/real_sector_map.csv"),
    ("S67_sec_predictive", "e8_sec/c_predictive_main.csv"),
    ("S68_sec_regressions_full", "e8_sec/b_table_debt_ratio_full.csv"),
    ("S69_sec_wild_bootstrap", "e8_sec/b_wild_cluster_bootstrap.csv"),
    ("S70_sec_transfer", "e8_sec/d_transfer_vs_within.csv"),
]
index = []
for name, src in supp:
    p = OUT / src
    df = pd.read_csv(p)
    df.to_csv(SUPP / f"{name}.csv", index=False)
    index.append({"table": name, "source": f"output/{src}", "rows": len(df), "cols": df.shape[1]})
pd.DataFrame(index).to_csv(SUPP / "INDEX.csv", index=False)

# ---------------------------------------------------------------- Figure 1 (workflow schematic)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
OK = {"grey": "#E6E6E6", "blue": "#CCE3F2", "orange": "#F9DEC2", "green": "#C7EBDD",
      "edge": "#333333"}
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 60)
ax.axis("off")


def box(x, y, w, h, text, fc, bold_first=True, body_fs=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=OK["edge"], lw=0.7))
    lines = text.split("\n")
    head, rest = lines[0], "\n".join(lines[1:])
    ax.text(x + w / 2, y + h - 1.6, head, ha="center", va="top", fontsize=8,
            fontweight="bold" if bold_first else "normal")
    if rest:
        ax.text(x + w / 2, y + h - 5.0, rest, ha="center", va="top", fontsize=body_fs,
                linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=OK["edge"], shrinkA=0, shrinkB=0))


box(1, 45, 30, 13.5, "Reference synthetic panel\n296 firms, 2010-2024\n4,440 firm-years, 8 sectors\npartial-adjustment DGP", OK["grey"])
box(69, 45, 30, 13.5, "Real sample\n46 S&P-100 firms, Yahoo Finance\n184 firm-years, FY2021-2025\nno macro or market data", OK["grey"])
box(35, 45, 30, 13.5, "Leakage audit\nclasses: direct, indirect, clean\nstrict set: 16 debt-free features\n(12 computable on real data)", OK["orange"])
arrow(31.6, 51.5, 34.4, 51.5)
arrow(68.4, 51.5, 65.6, 51.5)

mods = [
    ["Prediction", "12 baseline models,", "FT-Transformer,", "TabNet, tuned MLP", "random, grouped,", "temporal and", "unseen-firm splits"],
    ["Attribution", "TreeSHAP on", "LightGBM,", "out-of-fold,", "specs S1-S4,", "rank stability"],
    ["Generation", "noise, Gaussian,", "CTGAN, TVAE", "fidelity, constraint", "and privacy", "metrics"],
    ["Optimisation", "Merton default risk,", "U-shaped WACC,", "firm value,", "5 scenarios,", "Monte Carlo"],
    ["Econometrics", "pooled OLS,", "firm FE,", "two-way FE,", "clustered and", "Driscoll-Kraay SEs"],
]
w, gap, x0 = 18.4, 1.6, 1.0
for i, t in enumerate(mods):
    x = x0 + i * (w + gap)
    box(x, 19.5, w, 19, "\n".join(t), OK["blue"], body_fs=7.0)
    arrow(50, 44.3, x + w / 2, 39.1)

box(1, 1, 98, 13.5, "External validation on the real sample\nwithin-real prediction: leave-firms-out, leave-one-firm-out, out-of-time (2025)\ntransfer: models trained on the synthetic panel and tested on real firms\nSHAP comparison, real-panel regressions with firm and year effects", OK["green"])
for i in (0, 1, 3, 4):
    x = x0 + i * (w + gap) + w / 2
    arrow(x, 18.9, x, 15.1)
fig.savefig(FIG / "fig1_workflow.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig1_workflow.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- copy figures
copies = {
    "fig2_deep_tabular_r2": "e2_deep_tabular/fig_e2_r2_comparison",
    "fig3_shap_leakage_ablation": "e3_shap/figures/fig_e3_top8_S1_S4",
    "fig4_optimisation_mc": "e5_optimization/fig_e5_mc_optimum_distributions",
    "fig5_real_validation_r2": "e7_real_data/fig_e7_r2_by_protocol",
    "figS01_r2_by_model_grouped5x5": "e1_benchmark/figures/fig_r2_by_model_grouped5x5",
    "figS02_inflation_lightgbm": "e1_benchmark/figures/fig_inflation_lightgbm",
    "figS03_actual_vs_pred_strict_temporal": "e1_benchmark/figures/fig_actual_vs_pred_lightgbm_strict_temporal",
    "figS04_learning_curve_grouped": "e1_benchmark/figures/fig_learning_curve_grouped",
    "figS05_across_panels_lightgbm": "e1_benchmark/figures/fig_across_panels_lightgbm",
    "figS06_dgp_sensitivity_heatmap": "e1_benchmark/figures/fig_sensitivity_heatmap",
    "figS07_deep_tabular_with_paper_configs": "e2_deep_tabular/fig_e2_r2_comparison_with_paper_configs",
    "figS08_shap_S4_beeswarm": "e3_shap/figures/fig_e3_S4_beeswarm",
    "figS09_shap_S4_bar": "e3_shap/figures/fig_e3_S4_bar",
    "figS10_shap_S4_rank_stability": "e3_shap/figures/fig_e3_S4_rank_stability",
    "figS11_shap_S4_dependence_sector": "e3_shap/figures/fig_e3_S4_dependence_top2",
    "figS12_shap_S4_dependence_roa_size": "e3_shap/figures/fig_e3_S4_dependence_top2_nonsector",
    "figS13_generative_distributions": "e4_generative/figures/fig_e4_distributions",
    "figS14_generative_pca": "e4_generative/figures/fig_e4_pca",
    "figS15_generative_corr_diff": "e4_generative/figures/fig_e4_corr_diff",
    "figS16_generative_loss_curves": "e4_generative/figures/fig_e4_loss_curves",
    "figS17_generative_sensitivity_dgp": "e4_generative/figures/fig_e4_sensitivity_dgp",
    "figS18_generative_sensitivity_noise": "e4_generative/figures/fig_e4_sensitivity_noise",
    "figS19_opt_wacc_value_curves": "e5_optimization/fig_e5_wacc_value_curves",
    "figS20_opt_pareto_wacc_vs_pd": "e5_optimization/fig_e5_pareto_wacc_vs_pd",
    "figS21_opt_tornado": "e5_optimization/fig_e5_tornado",
    "figS22_opt_sector_actual_vs_model": "e5_optimization/fig_e5_sector_actual_vs_model",
    "figS23_econ_roa_coefficient": "e6_econometrics/fig_roa_coefficient_synthetic",
    "figS24_real_distributions": "e7_real_data/fig_e7_distributions_real_vs_synthetic",
    "figS25_real_sector_mix": "e7_real_data/fig_e7_sector_mix_and_debt_ratio",
    "figS26_real_pred_vs_actual": "e7_real_data/fig_e7_pred_vs_actual",
    "figS27_real_shap_vs_synthetic": "e7_real_data/fig_e7_shap_real_vs_synthetic",
    "figS28_real_firm_vs_sector_median": "e7_real_data/fig_e7_firm_vs_sector_median",
    "figS29_sec_r2_by_protocol": "e8_sec/figures/fig_e8_r2_by_protocol",
    "figS30_sec_coefficients": "e8_sec/figures/fig_e8_coefficients",
}
rows = []
for new, src in copies.items():
    for ext in (".png", ".pdf"):
        shutil.copy2(OUT / (src + ext), FIG / (new + ext))
    rows.append({"figure": new, "source": f"output/{src}.png/.pdf"})
rows.insert(0, {"figure": "fig1_workflow", "source": "code/build_master.py (schematic, no data)"})
pd.DataFrame(rows).to_csv(FIG / "figure_sources.csv", index=False)

import sys
sys.stdout.reconfigure(encoding="utf-8")
print(md_table(t1))
print()
print(md_table(t2))
print()
print(md_table(t3))
print("supp tables:", len(supp), "figures:", len(rows))
