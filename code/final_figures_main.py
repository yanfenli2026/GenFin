"""Main figures (Figs. 1-5).

Re-plots every main figure from the saved output CSVs; no value is recomputed or changed.
Style: figstyle.py (Arial 7 pt, bold lower-case panel letters, Okabe-Ito, lines >= 1 pt,
double-column width 180 mm, PDF + 300 dpi PNG).

Run: python code/final_figures_main.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

import figstyle as fs

import config
ROOT = config.ROOT
OUT = config.OUT_MAIN
FIGDIR = config.FIG_DIR
plt = fs.apply()
OI = fs.OI
W = fs.W_DOUBLE


# ----------------------------------------------------------------------------------------------
# Fig. 1  workflow schematic (no data)
# ----------------------------------------------------------------------------------------------
def fig1():
    Wmm, Hmm = 180, 96
    fig = plt.figure(figsize=(W, Hmm * fs.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, Wmm); ax.set_ylim(0, Hmm); ax.axis("off")

    F_IN, F_AUD, F_MOD, F_VAL = "#EBEBEB", "#FBE3C0", "#D4E8F5", "#CDEBE0"   # tints of grey / orange / blue / green

    def box(x0, y0, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc=fc, ec="#333333", lw=1.0))
        cx = x0 + w / 2
        ax.text(cx, y0 + h - 2.2, title, ha="center", va="top", fontsize=fs.FS, fontweight="bold")
        ax.text(cx, y0 + h - 6.6, body, ha="center", va="top", fontsize=fs.FS, linespacing=1.25)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=1.0, color="#333333",
                                     shrinkA=0, shrinkB=0))

    # inputs row
    y1, h1 = 64, 30
    box(2, y1, 54, h1, "Reference synthetic panel",
        "simulated, not real firms\n296 firms, 8 sectors\n2010–2024, 4,440 firm-years\npartial-adjustment generator", F_IN)
    box(63, y1, 54, h1, "Leakage audit",
        "features classed as direct,\nindirect or clean\nstrict set: 16 debt-free features\n(12 computable on real sample 1)",
        F_AUD)
    hr = 14.2
    box(124, y1 + h1 - hr, 54, hr, "Real sample 1: Yahoo Finance",
        "46 S&P 100 firms, 184 firm-years\nFY2021–2025, no market data", F_IN)
    box(124, y1, 54, hr, "Real sample 2: SEC XBRL 10-K",
        "21 non-financial firms, 283 firm-years\nFY2010–2025, 11 debt-free determinants", F_IN)
    arrow((56, y1 + h1 / 2), (63, y1 + h1 / 2))
    arrow((124, y1 + h1 - hr / 2), (117, y1 + h1 / 2 + 3))
    arrow((124, y1 + hr / 2), (117, y1 + h1 / 2 - 3))

    # module row
    y2, h2 = 28.5, 27
    n, gap = 5, 2.5
    wmod = (176 - gap * (n - 1)) / n
    mods = [("Prediction", "12 baseline models,\nFT-Transformer, TabNet,\ntuned MLP; random,\nfirm-grouped CV,\nout-of-time and\nunseen-firm tests"),
            ("Attribution", "TreeSHAP on LightGBM,\nout-of-fold,\nfeature sets S1–S4,\nrank stability"),
            ("Generation", "constrained noise-based\ngenerator, Gaussian\nbaseline, CTGAN, TVAE;\nfidelity, constraint and\nprivacy metrics"),
            ("Trade-off model", "Merton default risk,\nU-shaped WACC,\nfirm value, 5 scenarios,\nMonte Carlo;\nmodel-implied D/V"),
            ("Econometrics", "pooled OLS, firm FE,\ntwo-way FE,\nfirm-clustered and\nDriscoll–Kraay SEs")]
    xs = []
    for i, (t, b) in enumerate(mods):
        x0 = 2 + i * (wmod + gap)
        xs.append(x0 + wmod / 2)
        box(x0, y2, wmod, h2, t, b, F_MOD)
        arrow((90, y1), (x0 + wmod / 2, y2 + h2))

    # validation row
    y3, h3 = 2, 19.5
    box(2, y3, 176, h3, "External validation on the real samples",
        "within-sample prediction: leave-firms-out, leave-one-firm-out, out-of-time, unseen firms in future years\n"
        "transfer: models trained on the reference synthetic panel and tested on real firms\n"
        "SHAP comparison; panel regressions with firm and year effects and firm-clustered SEs\n"
        "sector debt ratios compared with model-implied D/V", F_VAL)
    for i in (0, 1, 3, 4):
        arrow((xs[i], y2), (xs[i], y3 + h3))
    return fs.save(fig, FIGDIR, "fig1_workflow")


# ----------------------------------------------------------------------------------------------
# Fig. 2  deep tabular models  (e2_deep_tabular/summary.csv)
# ----------------------------------------------------------------------------------------------
def fig2():
    s = pd.read_csv(os.path.join(OUT, "e2_deep_tabular", "summary.csv"))
    models = ["LightGBM", "FT-Transformer", "TabNet", "MLP"]
    label = {"LightGBM": "LightGBM (tuned)", "FT-Transformer": "FT-Transformer", "TabNet": "TabNet",
             "MLP": "MLP (tuned)"}
    col = {"LightGBM": OI["blue"], "FT-Transformer": OI["vermillion"], "TabNet": OI["green"], "MLP": OI["orange"]}
    mk = {"LightGBM": "o", "FT-Transformer": "s", "TabNet": "^", "MLP": "D"}
    protos = ["grouped5", "temporal", "unseen"]
    plab = {"grouped5": "Firm-grouped\n5-fold CV", "temporal": "Out-of-time\ntest (2021-2024)",
            "unseen": "Unseen-firm test\n(future years)"}
    fig, axes = plt.subplots(1, 2, figsize=(W, 62 * fs.MM))
    off = np.linspace(-0.27, 0.27, len(models))
    for ax, fset, letter, ylab in zip(axes, ["strict", "original"], "ab",
                                      ["Test R² (debt ratio)", "Test R² (debt ratio)"]):
        d = s[s.feature_set == fset]
        for j, m in enumerate(models):
            xx, yy, ee = [], [], []
            for i, p in enumerate(protos):
                r = d[(d.protocol == p) & (d.model == m)]
                if len(r):
                    xx.append(i + off[j]); yy.append(r.r2_mean.iloc[0]); ee.append(r.r2_sd.iloc[0])
            ax.errorbar(xx, yy, yerr=ee, fmt=mk[m], ms=4, lw=1.0, capsize=2, capthick=1.0, color=col[m],
                        label=label[m])
        ax.axhline(0, color=OI["grey"], lw=1.0, ls=":")
        ax.set_xticks(range(3)); ax.set_xticklabels([plab[p] for p in protos])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel(ylab)
        fs.panel_letter(ax, letter, x=-0.16, y=1.02)
        ax.text(0.0, 1.03, "Strict (leakage-free) features" if fset == "strict" else "Original 21 features (leaked information)",
                transform=ax.transAxes, ha="left", va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.5)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.30, wspace=0.25)
    return fs.save(fig, FIGDIR, "fig2_deep_tabular_r2")


# ----------------------------------------------------------------------------------------------
# Fig. 3  SHAP leakage ablation  (e3_shap/shap_ranking_S*.csv, spec_performance.csv)
# ----------------------------------------------------------------------------------------------
def fig3():
    perf = pd.read_csv(os.path.join(OUT, "e3_shap", "spec_performance.csv")).set_index("spec")
    ccol = {"direct": OI["vermillion"], "indirect": OI["orange"], "clean": OI["blue"]}
    cname = {"direct": "Direct leakage (debt-dependent)", "indirect": "Indirect leakage", "clean": "Clean (debt-free)"}
    slab = {"S1": "S1: original features", "S2": "S2: minus Altman Z", "S3": "S3: minus Altman Z and\ndefault risk score",
            "S4": "S4: strict features"}
    fig, axes = plt.subplots(2, 2, figsize=(W, 118 * fs.MM))
    for ax, key, letter in zip(axes.ravel(), ["S1", "S2", "S3", "S4"], "abcd"):
        rk = pd.read_csv(os.path.join(OUT, "e3_shap", f"shap_ranking_{key}.csv")).head(8).iloc[::-1]
        pr = perf.loc[key]
        ax.barh(rk.display_name, rk.mean_abs_shap_oof, color=[ccol[c] for c in rk.audit_class], height=0.7,
                edgecolor="none")
        ax.set_xlabel("Mean |SHAP value| (debt-ratio units)")
        ax.tick_params(axis="y", length=0)
        ax.text(0.98, 0.04, f"{slab[key]}\ngrouped-CV R² = {pr.grouped_R2_mean:.3f}\n"
                f"random-split R² = {pr.legacy_R2:.3f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=fs.FS, linespacing=1.3)
        fs.panel_letter(ax, letter, x=-0.50, y=1.0)
    handles = [Patch(color=ccol[c], label=cname[c]) for c in ["direct", "indirect", "clean"]]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=2.0)
    fig.subplots_adjust(left=0.205, right=0.985, top=0.96, bottom=0.145, wspace=0.62, hspace=0.42)
    return fs.save(fig, FIGDIR, "fig3_shap_leakage_ablation")


# ----------------------------------------------------------------------------------------------
# Fig. 4  Monte Carlo optima  (e5_optimization/mc_summary_sector_scenario.csv; nested, zero debt beta)
# ----------------------------------------------------------------------------------------------
def fig4():
    m = pd.read_csv(os.path.join(OUT, "e5_optimization", "mc_summary_sector_scenario.csv"))
    m = m[(m.mc_type == "nested") & (m.convention == "zero")]
    sectors = sorted(m.sector.unique())
    slab = {"ConsDisc": "Consumer\ndiscretionary", "Energy": "Energy", "Financials": "Financials",
            "Healthcare": "Healthcare", "Industrials": "Industrials", "Materials": "Materials",
            "Tech": "Technology", "Utilities": "Utilities"}
    scen = ["Base", "Recession", "High_Rate", "Credit_Crunch", "Strong_Growth"]
    sclab = {"Base": "Base", "Recession": "Recession", "High_Rate": "High Rate", "Credit_Crunch": "Credit Crunch",
             "Strong_Growth": "Strong Growth"}
    scol = {"Base": OI["black"], "Recession": OI["blue"], "High_Rate": OI["vermillion"],
            "Credit_Crunch": OI["purple"], "Strong_Growth": OI["green"]}
    X_LO, X_HI = 0.05, 0.85
    fig, axes = plt.subplots(2, 1, figsize=(W, 112 * fs.MM), sharex=True)
    w = 0.155
    for ax, pre, ylab, letter in zip(axes, ["xW", "xV"],
                                     ["WACC-minimizing D/V\n(model-implied)", "Value-maximizing D/V\n(model-implied)"],
                                     "ab"):
        for i, sec in enumerate(sectors):
            for j, sc in enumerate(scen):
                r = m[(m.sector == sec) & (m.scenario == sc)].iloc[0]
                st = dict(med=r[f"{pre}_median"], q1=r[f"{pre}_p25"], q3=r[f"{pre}_p75"],
                          whislo=r[f"{pre}_p05"], whishi=r[f"{pre}_p95"], fliers=[])
                bp = ax.bxp([st], positions=[i + (j - 2) * w], widths=w * 0.82, showfliers=False,
                            patch_artist=True, manage_ticks=False,
                            medianprops=dict(color="white" if sc in ("Base", "Recession") else "black", lw=1.0),
                            boxprops=dict(lw=1.0, edgecolor="black"), whiskerprops=dict(lw=1.0),
                            capprops=dict(lw=1.0))
                bp["boxes"][0].set_facecolor(scol[sc])
        ax.axhline(X_LO, color=OI["grey"], lw=1.0, ls=":"); ax.axhline(X_HI, color=OI["grey"], lw=1.0, ls=":")
        ax.set_ylim(0, 0.9)
        ax.set_ylabel(ylab)
        fs.panel_letter(ax, letter, x=-0.075, y=1.0)
    axes[1].set_xticks(range(len(sectors)))
    axes[1].set_xticklabels([slab[s] for s in sectors])
    axes[1].set_xlim(-0.55, len(sectors) - 0.45)
    handles = [Patch(facecolor=scol[s], edgecolor="black", lw=1.0, label=sclab[s]) for s in scen]
    handles.append(Line2D([], [], color=OI["grey"], lw=1.0, ls=":", label="Grid bounds (0.05, 0.85)"))
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), columnspacing=1.4,
               handlelength=1.6)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.10, hspace=0.18)
    return fs.save(fig, FIGDIR, "fig4_optimisation_mc")


# ----------------------------------------------------------------------------------------------
# Fig. 5  real-sample validation  (e7_real_data/bc_results_all.csv; e8_sec/fig_e8_r2_by_protocol_data.csv)
# ----------------------------------------------------------------------------------------------
def _bars(ax, groups, models, getter, col, lab, XMIN, w):
    clipped = []
    for i, g in enumerate(groups):
        for j, mname in enumerate(models):
            r2, lo, hi = getter(g, mname)
            x = i + (j - (len(models) - 1) / 2) * w
            ax.bar(x, max(r2, XMIN), width=w, color=col[mname], edgecolor="white", lw=0.5,
                   label=lab[mname] if i == 0 else None)
            ax.plot([x, x], [max(lo, XMIN), max(hi, XMIN)], color="#222222", lw=1.0, solid_capstyle="butt")
            if lo < XMIN:
                ax.plot(x, XMIN + 0.035, marker="v", color="#222222", ms=3.2, clip_on=False)
                clipped.append((g, mname, lo))
    ax.axhline(0, color="#222222", lw=1.0)
    ax.set_ylim(XMIN, 1.0)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_xticks(range(len(groups)))
    return clipped


def fig5():
    XMIN = -1.0
    res = pd.read_csv(os.path.join(OUT, "e7_real_data", "bc_results_all.csv"))
    res = res[res.feature_set == "strict"].set_index(["protocol", "model"])
    protos = [("grouped5 (repeat 0)", "Firm-grouped\n5-fold CV"),
              ("LOFO", "Leave-one-\nfirm-out"),
              ("out-of-time (2025)", "Out-of-time\n(2025)"),
              ("transfer: synthetic -> real (all firm-years)", "Transfer: synthetic\n→ real (all years)"),
              ("transfer: synthetic -> real (2025)", "Transfer: synthetic\n→ real (2025)")]
    models = ["SectorMean", "IndustryMedian", "LinearReg", "Ridge", "RandomForest", "LightGBM"]
    lab = {"SectorMean": "Sector mean", "IndustryMedian": "Industry median", "LinearReg": "Linear regression",
           "Ridge": "Ridge", "RandomForest": "Random forest", "LightGBM": "LightGBM",
           "SectorMedian (naive)": "Sector median"}
    col = {"SectorMean": OI["black"], "IndustryMedian": OI["grey"], "LinearReg": OI["orange"],
           "Ridge": OI["skyblue"], "RandomForest": OI["green"], "LightGBM": OI["vermillion"],
           "SectorMedian (naive)": OI["grey"]}

    sec = pd.read_csv(os.path.join(OUT, "e8_sec", "fig_e8_r2_by_protocol_data.csv"))
    sec_groups = list(dict.fromkeys(sec.group))
    sec_glab = {sec_groups[0]: "Firm-grouped\n5-fold CV", sec_groups[1]: "Leave-one-\nfirm-out",
                sec_groups[2]: "Out-of-time\n(2020-2025)", sec_groups[3]: "Unseen firms,\nfuture years",
                sec_groups[4]: "Transfer: synthetic\n→ SEC (all years)"}
    sec_models = ["SectorMedian (naive)", "Ridge", "LightGBM"]
    secd = sec.set_index(["group", "model"])

    fig, axes = plt.subplots(2, 1, figsize=(W, 128 * fs.MM))
    ax = axes[0]
    c1 = _bars(ax, [p for p, _ in protos], models,
               lambda g, mm: (res.loc[(g, mm)].r2, res.loc[(g, mm)].r2_lo, res.loc[(g, mm)].r2_hi),
               col, lab, XMIN, 0.13)
    ax.set_xticklabels([pl for _, pl in protos])
    ax.set_ylabel("Out-of-sample R²\n(debt ratio)")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.2)
    ax.text(-0.05, 1.17, "Real sample 1: 46 S&P 100 firms (Yahoo Finance), 184 firm-years, 12 strict features",
            transform=ax.transAxes, ha="left", va="bottom")
    fs.panel_letter(ax, "a", x=-0.085, y=1.17)

    ax = axes[1]
    c2 = _bars(ax, sec_groups, sec_models,
               lambda g, mm: (secd.loc[(g, mm)].R2, secd.loc[(g, mm)].lo, secd.loc[(g, mm)].hi),
               col, lab, XMIN, 0.2)
    ax.set_xticklabels([sec_glab[g] for g in sec_groups])
    ax.set_ylabel("Out-of-sample R²\n(debt ratio)")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.02), handlelength=1.2)
    ax.text(-0.05, 1.04, "Real sample 2: 21 non-financial firms (SEC XBRL 10-K filings), 283 firm-years, FY2010-2025",
            transform=ax.transAxes, ha="left", va="bottom")
    fs.panel_letter(ax, "b", x=-0.085, y=1.04)
    fig.text(0.99, 0.005, "Axis truncated at R² = −1; ▼ marks a 95% CI lower bound below −1.",
             ha="right", va="bottom")
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.11, hspace=0.55)
    fs.save(fig, FIGDIR, "fig5_real_validation_r2")
    return c1, c2


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    c1, c2 = fig5()
    print(ascii(("panel a clipped", c1)))
    print(ascii(("panel b clipped", c2)))
