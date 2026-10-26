"""
final_figures_si_b.py -- Supplementary Figs. S16-S30 in the shared figure style (figstyle.py).

Re-plots each figure from the saved experiment outputs only; no model is refitted and no value is changed.
  S16-S18  output/e4_generative/loss_curves/*.csv, sensitivity_dgp_grid.csv, sensitivity_noise_scale.csv
  S19-S22  output/e5_optimization/curves_sector_scenario.csv.gz, base_results_sector_scenario.csv,
           tornado_oat.csv, sector_table_actual_vs_model.csv, fig_pareto_truncation.csv
  S23      output/e6_econometrics/roa_coefficient_across_specifications.csv
  S24-S28  output/e7_real_data/*.csv and output/foundation/real_panel.csv
           (S24 also needs the reference synthetic panel; it is the deterministic common.generate_panel(42),
           and the KS statistics recomputed from it are asserted equal to e7 a_descriptives_real_vs_synthetic.csv)
  S29-S30  output/e8_sec/fig_e8_r2_by_protocol_data.csv, b_panel_regressions_all.csv

Style: sans-serif 7 pt, bold lower-case 8 pt panel letters, no titles, Okabe-Ito (viridis only for the
continuous debt-ratio scale in S20), PDF + 300 dpi PNG at a fixed physical width (88 / 180 mm).
Previous versions are copied to figures/_backup/ before overwriting (once; existing backups kept).

Run: python code/final_figures_si_b.py
"""
from __future__ import annotations

import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
try:
    import figstyle as fsty  # noqa: E402
except ImportError:  # fallback with the same rules
    import figstyle_si as fsty  # noqa: E402

plt = fsty.apply()
import matplotlib as mpl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

import config  # noqa: E402
ROOT = config.ROOT
OUTR = config.OUT_MAIN
E4, E5, E6, E7, E8 = (os.path.join(OUTR, d) for d in
                      ("e4_generative", "e5_optimization", "e6_econometrics", "e7_real_data", "e8_sec"))
FOUND = os.path.join(OUTR, "foundation")
FIGDIR = config.FIG_DIR
BACKUP = os.path.join(FIGDIR, "_backup")
OI = fsty.OI
W1, W2, FS = fsty.W_SINGLE, fsty.W_DOUBLE, fsty.FS
R2 = "R²"

# sector labels and colours as in the E5 source figures (Okabe-Ito + grey; no yellow on white)
SECTORS = ["ConsDisc", "Energy", "Financials", "Healthcare", "Industrials", "Materials", "Tech", "Utilities"]
SECTOR_LABEL = {"ConsDisc": "Consumer Disc.", "Energy": "Energy", "Financials": "Financials",
                "Healthcare": "Healthcare", "Industrials": "Industrials", "Materials": "Materials",
                "Tech": "Technology", "Utilities": "Utilities"}
SEC_COLOR = {"ConsDisc": OI["orange"], "Energy": OI["skyblue"], "Financials": OI["green"],
             "Healthcare": OI["black"], "Industrials": OI["blue"], "Materials": OI["vermillion"],
             "Tech": OI["purple"], "Utilities": OI["grey"]}
SEC_LS = {"ConsDisc": "-", "Energy": "--", "Financials": "-.", "Healthcare": "-",
          "Industrials": "--", "Materials": "-.", "Tech": "-", "Utilities": "--"}
SEC_MARK = dict(zip(SECTORS, ["o", "s", "^", "D", "v", "P", "X", "*"]))


def letter(ax, s, x=-0.12):
    """Bold lower-case panel letter, top left; a left title so the layout engine reserves room."""
    ax.set_title(s, loc="left", fontsize=fsty.FS_LETTER, fontweight="bold", x=x, pad=3)


def newfig(w, h, **kw):
    return plt.subplots(figsize=(w, h), layout="constrained", **kw)


def save(fig, name):
    for ext in (".pdf", ".png"):
        src = os.path.join(FIGDIR, name + ext)
        dst = os.path.join(BACKUP, name + ext)
        if os.path.exists(src) and not os.path.exists(dst):
            os.makedirs(BACKUP, exist_ok=True)
            shutil.copy2(src, dst)
    fsty.save(fig, FIGDIR, name)
    print("  wrote", name, flush=True)


# ---------------------------------------------------------------------------
# S16-S18 (E4)
# ---------------------------------------------------------------------------
GEN_LABEL = {"B1_noise": "Noise baseline (B1)", "B2_gauss": "Gaussian baseline (B2)",
             "B1_noise_log": "Noise baseline, log levels (B1-log)", "CTGAN": "CTGAN", "TVAE": "TVAE"}
GEN_COLOR = {"B1_noise": OI["orange"], "B2_gauss": OI["skyblue"], "B1_noise_log": OI["purple"],
             "CTGAN": OI["green"], "TVAE": OI["blue"]}
LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.6]
SIGMAS = [0.005, 0.01, 0.02, 0.03, 0.05]
SEED_LS = {42: "-", 43: "--", 44: ":"}


def s16():
    """Loss trajectories of the longest saved run per model (CTGAN 300 epochs, TVAE 600 epochs), 3 seeds."""
    d = os.path.join(E4, "loss_curves")
    files = [f for f in os.listdir(d) if f.endswith(".csv")]
    longest = {}
    for f in files:
        kind, ep, _ = f[:-4].split("_")
        longest[kind] = max(longest.get(kind, 0), int(ep[2:]))
    fig, axes = newfig(W2, 2.4, ncols=3)
    for seed in (42, 43, 44):
        c = pd.read_csv(os.path.join(d, f"CTGAN_ep{longest['CTGAN']}_seed{seed}.csv"))
        t = pd.read_csv(os.path.join(d, f"TVAE_ep{longest['TVAE']}_seed{seed}.csv"))
        axes[0].plot(c.epoch + 1, c.generator_loss, SEED_LS[seed], color=GEN_COLOR["CTGAN"], label=f"Seed {seed}")
        axes[1].plot(c.epoch + 1, c.discriminator_loss, SEED_LS[seed], color=GEN_COLOR["CTGAN"],
                     label=f"Seed {seed}")
        axes[2].plot(t.epoch + 1, t.loss, SEED_LS[seed], color=GEN_COLOR["TVAE"], label=f"Seed {seed}")
    for ax, yl, lt in zip(axes, ["CTGAN generator loss", "CTGAN critic loss", "TVAE loss (reconstruction + KL)"],
                          "abc"):
        for v in (50, 300):
            ax.axvline(v, color=OI["grey"], lw=0.8, ls=":")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(yl)
        letter(ax, lt, -0.2)
    hs = [Line2D([], [], ls=SEED_LS[sd_], color="0.3") for sd_ in (42, 43, 44)]
    hs += [Line2D([], [], color=GEN_COLOR["CTGAN"]), Line2D([], [], color=GEN_COLOR["TVAE"]),
           Line2D([], [], color=OI["grey"], lw=0.8, ls=":")]
    fig.legend(hs, ["Seed 42", "Seed 43", "Seed 44", "CTGAN", "TVAE", "50 and 300 epochs"],
               loc="outside upper center", ncol=6, handlelength=2.2, columnspacing=1.5)
    save(fig, "figS16_generative_loss_curves")


def s17():
    grid = pd.read_csv(os.path.join(E4, "sensitivity_dgp_grid.csv"))
    lam_col = dict(zip(LAMBDAS, [OI["orange"], OI["skyblue"], OI["green"], OI["blue"], OI["vermillion"]]))
    rows = [("KS_mean", "Mean KS"), ("W_std_mean", "Mean Wasserstein\n(z units)"), ("CVR", "CVR (share of rows)")]
    gens = ["B1_noise", "B2_gauss"]
    fig, axes = newfig(W2, 5.2, nrows=3, ncols=2, sharey="row", sharex=True)
    for i, (met, lab) in enumerate(rows):
        for j, g in enumerate(gens):
            ax = axes[i, j]
            sub = grid[grid.generator == g]
            for lam in sorted(sub.lam.unique()):
                t = sub[sub.lam == lam].sort_values("sigma_zeta")
                ax.plot(t.sigma_zeta, t[met], marker="o", ms=3, color=lam_col.get(lam, OI["black"]),
                        label=rf"$\lambda$ = {lam:g}")
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(SIGMAS))
            ax.set_xticklabels([f"{v:g}" for v in SIGMAS])
            ax.xaxis.set_minor_locator(NullLocator())
            if j == 0:
                ax.set_ylabel(lab)
            if i == 2:
                ax.set_xlabel(r"$\sigma_\zeta$ (s.d. of adjustment shock)")
            if i == 0:
                ax.text(0.03, 0.96, GEN_LABEL[g], transform=ax.transAxes, va="top", ha="left")
            letter(ax, "abcdef"[2 * i + j], -0.14 if j == 0 else -0.06)
    lo = min(0.0, grid["KS_mean"].min())
    axes[0, 0].set_ylim(lo, grid["KS_mean"].max() * 1.25)
    axes[1, 0].set_ylim(0, grid["W_std_mean"].max() * 1.25)
    axes[2, 0].set_ylim(0, max(0.05, grid["CVR"].max() * 1.15))
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=5, columnspacing=1.5)
    save(fig, "figS17_generative_sensitivity_dgp")


def s18():
    ns = pd.read_csv(os.path.join(E4, "sensitivity_noise_scale.csv"))
    fig, axes = newfig(W2, 2.2, ncols=4)
    mets = [("KS_mean", "Mean KS"), ("W_std_mean", "Mean Wasserstein (z units)"),
            ("CVR", "CVR (share of rows)"), ("DCR_median", "Median DCR (quantile units)")]
    for ax, (met, lab), lt in zip(axes, mets, "abcd"):
        for gname in ["B1_noise", "B1_noise_log"]:
            agg = ns[ns.generator == gname].groupby("noise_scale")[met].agg(["mean", "std"])
            ax.errorbar(agg.index.values, agg["mean"], yerr=agg["std"], color=GEN_COLOR[gname], marker="o",
                        ms=3, capsize=2, label=GEN_LABEL[gname])
        if met == "DCR_median":
            ax.axhline(ns["DCR_ref_median"].iloc[0], color=OI["black"], ls="--", lw=1,
                       label="Reference panel (within-panel DCR)")
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xlim(-0.05, 10)
        ax.xaxis.set_major_locator(FixedLocator([0, 0.25, 1, 4, 8]))
        ax.set_xticklabels(["0", "0.25", "1", "4", "8"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Noise scale k (× original)")
        ax.set_ylabel(lab)
        letter(ax, lt, -0.3)
    h, lab = axes[0].get_legend_handles_labels()
    h2, lab2 = axes[3].get_legend_handles_labels()
    extra = [(hh, ll) for hh, ll in zip(h2, lab2) if ll.startswith("Reference")]
    h += [e[0] for e in extra]
    lab += [e[1] for e in extra]
    fig.legend(h, lab, loc="outside upper center", ncol=3, columnspacing=1.5)
    save(fig, "figS18_generative_sensitivity_noise")


# ---------------------------------------------------------------------------
# S19-S22 (E5)
# ---------------------------------------------------------------------------
X_LO, X_HI = 0.05, 0.85
SCEN_NAMES = ["Base", "Recession", "High_Rate", "Credit_Crunch", "Strong_Growth"]
SCEN_LABEL = {"Base": "Base", "Recession": "Recession", "High_Rate": "High rate",
              "Credit_Crunch": "Credit crunch", "Strong_Growth": "Strong growth"}
_E5: dict = {}


def e5_data():
    if not _E5:
        curves = pd.read_csv(os.path.join(E5, "curves_sector_scenario.csv.gz"))
        det = pd.read_csv(os.path.join(E5, "base_results_sector_scenario.csv"))
        det = det[det.calibration == "seed_average"]
        _E5["base_c"] = curves[(curves.scenario == "Base") & (curves.convention == "zero")]
        _E5["bdet"] = det[(det.scenario == "Base") & (det.convention == "zero")].set_index("sector")
    return _E5


def nondominated_2d(f1, f2):
    """Boolean mask of non-dominated points (minimise f1 and f2); same rule as e5_optimization.py."""
    order = np.lexsort((f1, f2))
    keep = np.zeros(len(f1), dtype=bool)
    best = np.inf
    for i in order:
        if f1[i] < best:
            keep[i] = True
            best = f1[i]
    return keep


def s19():
    d = e5_data()
    base_c, bdet = d["base_c"], d["bdet"]
    fig, axes = newfig(W2, 3.0, ncols=2)
    for sec in SECTORS:
        c = base_c[base_c.sector == sec]
        lw = 1.4 if sec == "Utilities" else 1.2
        axes[0].plot(c.x, c.wacc * 100, color=SEC_COLOR[sec], ls=SEC_LS[sec], lw=lw, label=SECTOR_LABEL[sec])
        axes[0].plot(bdet.loc[sec, "x_wacc_min"], bdet.loc[sec, "wacc_min"] * 100, "o", color=SEC_COLOR[sec],
                     ms=4, mec="black", mew=0.5)
        axes[1].plot(c.x, (c.value_rel - 1) * 100, color=SEC_COLOR[sec], ls=SEC_LS[sec], lw=lw)
        axes[1].plot(bdet.loc[sec, "x_value_max"], (bdet.loc[sec, "value_max"] - 1) * 100, "o",
                     color=SEC_COLOR[sec], ms=4, mec="black", mew=0.5)
    axes[0].set_xlabel("Debt ratio, D/V")
    axes[0].set_ylabel("WACC (%)")
    axes[1].set_xlabel("Debt ratio, D/V")
    axes[1].set_ylabel(r"Change in firm value, $V_L/V_U - 1$ (%)")
    axes[1].axhline(0, color=OI["grey"], lw=0.8, ls=":")
    lo = min(base_c.wacc.min() * 100 - 0.1, 7.0)
    axes[0].set_ylim(lo, min(base_c.wacc.max() * 100 + 0.1, 10.0))
    axes[1].set_ylim(max((base_c.value_rel.min() - 1) * 100, -6), (base_c.value_rel.max() - 1) * 100 + 0.5)
    for ax, lt in zip(axes, "ab"):
        ax.set_xlim(X_LO, X_HI)
        letter(ax, lt, -0.12)
    h, lab = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], ls="none", marker="o", color="0.6", mec="black", mew=0.5, ms=4))
    lab.append("Model-implied optimum")
    fig.legend(h, lab, loc="outside lower center", ncol=5, handlelength=2.6, columnspacing=1.4)
    save(fig, "figS19_opt_wacc_value_curves")


def s20():
    """WACC vs 5-year default probability; the PD axis starts at 1e-4 % (PD_MIN = 1e-6 as a fraction)."""
    d = e5_data()
    base_c, bdet = d["base_c"], d["bdet"]
    trunc_saved = pd.read_csv(os.path.join(E5, "fig_pareto_truncation.csv")).set_index("sector")
    PD_MIN, PD_MAX = float(trunc_saved.pd_min_plotted.iloc[0]), 0.5
    assert PD_MIN == 1e-6
    xs_hi = float(np.ceil(bdet["pareto_x_high"].max() * 20) / 20)
    norm = mpl.colors.Normalize(X_LO, xs_hi)
    cmap = mpl.colormaps["viridis"]
    fig, axes = newfig(W2, 3.9, nrows=2, ncols=4, sharex=True, sharey=True)
    for k, (ax, sec) in enumerate(zip(axes.ravel(), SECTORS)):
        call = base_c[base_c.sector == sec]
        nd_all = nondominated_2d(call.wacc.values, call.pd_T.values)
        call = call.assign(nd=nd_all)
        c = call[call.pd_T >= PD_MIN]
        x_first = float(c.x.min())
        assert abs(x_first - trunc_saved.loc[sec, "first_plotted_x"]) < 1e-9, sec
        ax.plot(c.pd_T[~c.nd] * 100, c.wacc[~c.nd] * 100, color="#BBBBBB", lw=1.2)
        ax.scatter(c.pd_T[c.nd] * 100, c.wacc[c.nd] * 100, c=c.x[c.nd], cmap=cmap, norm=norm, s=5, lw=0)
        ax.plot(bdet.loc[sec, "pd_at_x_wacc"] * 100, bdet.loc[sec, "wacc_min"] * 100, marker="*",
                color=OI["vermillion"], ms=8, mec="black", mew=0.4, ls="none")
        ax.annotate(rf"$\mathit{{x}}$ = {bdet.loc[sec, 'x_wacc_min']:.2f}",
                    (bdet.loc[sec, "pd_at_x_wacc"] * 100, bdet.loc[sec, "wacc_min"] * 100),
                    xytext=(0, -10), textcoords="offset points", ha="center")
        first = c.iloc[0]
        ax.annotate(rf"$\mathit{{x}}$ = {max(x_first, X_LO):.3f}", (first.pd_T * 100, first.wacc * 100),
                    xytext=(3, 4), textcoords="offset points", ha="left")
        ax.text(0.97, 0.96, SECTOR_LABEL[sec], transform=ax.transAxes, ha="right", va="top")
        ax.set_xscale("log")
        ax.set_xlim(PD_MIN * 100, PD_MAX * 100)
        ax.set_ylim(7.5, 8.8)
        ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1, 10]))
        ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(
            [r"$10^{-4}$", "", r"$10^{-2}$", "", "1", ""]))
        ax.xaxis.set_minor_locator(NullLocator())
        letter(ax, "abcdefgh"[k], -0.26 if k % 4 == 0 else -0.08)
    for ax in axes[1]:
        ax.set_xlabel("5-year default probability (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("WACC (%)")
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.8, aspect=30, pad=0.02)
    cb.set_label("Debt ratio on Pareto set, D/V")
    cb.outline.set_linewidth(0.8)
    save(fig, "figS20_opt_pareto_wacc_vs_pd")


def s21():
    tor = pd.read_csv(os.path.join(E5, "tornado_oat.csv"))
    pretty = {"tau": r"Tax rate $\tau$", "erp": "Equity risk premium", "beta_u": "Unlevered beta",
              "rf_shift": "Risk-free rate shift", "sigma_bar": r"Asset volatility $\bar{\sigma}_A$",
              "s0": r"Base spread $s_0$", "lam_cs": "Spread/expected-loss ratio",
              "alpha_scale": "Distress-cost scale", "recov_shift": "Recovery-rate shift",
              "T": r"Debt horizon $T$", "mu_shift": "Asset drift shift", "g_to_mu": "GDP-to-drift loading"}
    fig, axes = newfig(W2, 5.0, nrows=2, ncols=2)
    for r_i, sname in enumerate(["Base", "Recession"]):
        t = tor[(tor.target == "Average firm") & (tor.scenario == sname)]
        for c_i, (pre, xlab) in enumerate(zip(["xW", "xV"], ["WACC-minimizing D/V", "Value-maximizing D/V"])):
            ax = axes[r_i, c_i]
            tt = t.sort_values(f"{pre}_swing", ascending=True)
            base = tt[f"{pre}_base"].iloc[0]
            y = np.arange(len(tt))
            lo_v, hi_v = tt[f"{pre}_at_low"].values, tt[f"{pre}_at_high"].values
            ax.barh(y, lo_v - base, left=base, color=OI["blue"], height=0.6, lw=0, label="Parameter at low end")
            ax.barh(y, hi_v - base, left=base, color=OI["vermillion"], height=0.6, lw=0, alpha=0.9,
                    label="Parameter at high end")
            ax.axvline(base, color="black", lw=1)
            ax.set_yticks(y, [pretty[p] for p in tt.parameter])
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel(f"{xlab} ({SCEN_LABEL[sname]} scenario)")
            ax.set_xlim(max(0, min(lo_v.min(), hi_v.min()) - 0.03), min(0.9, max(lo_v.max(), hi_v.max()) + 0.03))
            letter(ax, "abcd"[2 * r_i + c_i], -0.55)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=2, columnspacing=2.0)
    save(fig, "figS21_opt_tornado")


def s22():
    st = pd.read_csv(os.path.join(E5, "sector_table_actual_vs_model.csv")).set_index("sector").loc[SECTORS]
    fig, ax = newfig(W1, 3.9)
    y = np.arange(len(SECTORS))[::-1].astype(float)
    ax.hlines(y + 0.24, st.mc_nested_xW_p05, st.mc_nested_xW_p95, color=OI["blue"], lw=2.2)
    ax.plot(st.x_wacc_min, y + 0.24, "o", color=OI["blue"], mec="black", mew=0.4, ms=4,
            label="WACC-minimizing D/V: base value and MC 5-95%")
    ax.hlines(y + 0.08, st.mc_nested_xV_p05, st.mc_nested_xV_p95, color=OI["purple"], lw=2.2)
    ax.plot(st.x_value_max, y + 0.08, "^", color=OI["purple"], mec="black", mew=0.4, ms=4,
            label="Value-maximizing D/V: base value and MC 5-95%")
    ax.plot(st.actual_mean_dr_synthetic_ref, y - 0.10, "D", color=OI["orange"], mec="black", mew=0.4, ms=3.8,
            label="Mean D/A, reference synthetic panel")
    ax.plot(st.actual_mean_dr_real, y - 0.26, "s", color=OI["green"], mec="black", mew=0.4, ms=3.6,
            label="Mean D/A, 46-firm real sample")
    ax.set_yticks(y, [SECTOR_LABEL[s] for s in SECTORS])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Debt ratio (model: D/V; observed: book D/A)")
    ax.set_xlim(0, X_HI + 0.02)
    ax.set_ylim(-0.6, len(SECTORS) - 0.4)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=1, handlelength=1.5)
    save(fig, "figS22_opt_sector_actual_vs_model")


# ---------------------------------------------------------------------------
# S23 (E6)
# ---------------------------------------------------------------------------
def s23():
    roa = pd.read_csv(os.path.join(E6, "roa_coefficient_across_specifications.csv"))
    keep = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
    s = roa[roa.model.isin(keep)].set_index("model").loc[keep].reset_index()
    labels = {"M1": "M1 Pooled, HC1", "M2": "M2 Pooled, firm-clustered", "M3": "M3 Firm FE",
              "M4": "M4 Firm + year FE", "M5": "M5 Firm FE, Driscoll-Kraay", "M6": "M6 Firm FE, strict",
              "M7": "M7 Pooled, strict", "M8": "M8 Firm + year FE, strict"}
    fig, ax = newfig(W1, 2.7)
    yy = np.arange(len(s))[::-1]
    for yi, (_, r) in zip(yy, s.iterrows()):
        orig = r.model in ("M1", "M2", "M3", "M4", "M5")
        c, mk = (OI["blue"], "o") if orig else (OI["vermillion"], "s")
        ax.errorbar(r.coef, yi, xerr=[[r.coef - r.ci_low], [r.ci_high - r.coef]], fmt=mk, color=c, ms=3.5,
                    capsize=2, lw=1)
    ax.axvline(0, color=OI["grey"], lw=0.8, ls="--")
    ax.set_yticks(yy, [labels[m] for m in s.model])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ROA coefficient in debt-ratio regression (95% CI)")
    hs = [Line2D([], [], ls="none", marker="o", color=OI["blue"], ms=3.5, label="Original regressors"),
          Line2D([], [], ls="none", marker="s", color=OI["vermillion"], ms=3.5, label="Strict regressors")]
    ax.legend(handles=hs, loc="lower right", handletextpad=0.3)
    save(fig, "figS23_econ_roa_coefficient")


# ---------------------------------------------------------------------------
# S24-S28 (E7)
# ---------------------------------------------------------------------------
C_REAL, C_SYN = OI["vermillion"], OI["blue"]


def s24():
    import common  # deterministic reference synthetic panel (seed 42)
    real = pd.read_csv(os.path.join(FOUND, "real_panel.csv"))
    syn = common.generate_panel(seed=42)
    ks_saved = pd.read_csv(os.path.join(E7, "a_descriptives_real_vs_synthetic.csv")).set_index("variable")
    panels = [("debt_ratio", "Debt ratio (total debt / total assets)", None),
              ("roa", "ROA (net income / total assets)", None),
              ("cash_ratio", "Cash ratio (cash / total assets)", None),
              ("interest_coverage", r"log$_{10}$ interest coverage (EBIT / interest)", "log"),
              ("log_assets", "ln(total assets, USD million)", None),
              ("asset_turnover", r"log$_{10}$ asset turnover (revenue / assets)", "log")]
    fig, axes = newfig(W2, 4.1, nrows=2, ncols=3)
    for ax, (v, lab, scale), lt in zip(axes.ravel(), panels, "abcdef"):
        a = real[v].replace([np.inf, -np.inf], np.nan).dropna()
        b = syn[v].replace([np.inf, -np.inf], np.nan).dropna()
        ks = stats.ks_2samp(a, b).statistic
        assert abs(ks - ks_saved.loc[v, "ks_stat"]) < 1e-12, (v, ks, ks_saved.loc[v, "ks_stat"])
        if scale == "log":   # zeros (synthetic EBIT floored at 0) cannot be shown on a log axis
            a, b = np.log10(a[a > 0]), np.log10(b[b > 0])
        lo = min(a.min(), np.percentile(b, 0.5))
        hi = max(a.max(), np.percentile(b, 99.5))
        bins = np.linspace(lo, hi, 40)
        ax.hist(b.clip(bins[0], bins[-1]), bins=bins, density=True, color=C_SYN, alpha=0.45, lw=0,
                label=f"Reference synthetic panel (n = {len(syn):,})")
        ax.hist(a.clip(bins[0], bins[-1]), bins=bins, density=True, histtype="step", lw=1.2, color=C_REAL,
                label=f"Real sample, 46 firms (n = {len(real)})")
        left = v == "log_assets"
        ax.text(0.03 if left else 0.97, 0.96, f"KS = {ks:.2f}", transform=ax.transAxes,
                ha="left" if left else "right", va="top")
        ax.set_xlabel(lab)
        ax.set_ylabel("Density")
        letter(ax, lt, -0.2)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=2, columnspacing=2.0)
    save(fig, "figS24_real_distributions")


def s25():
    sd = pd.read_csv(os.path.join(E7, "a_sector_mix_and_mean_dr.csv")).sort_values("synthetic_mean_dr")
    fm = pd.read_csv(os.path.join(E7, "e_firm_vs_sector_median.csv"))
    y = np.arange(len(sd))
    fig, axes = newfig(W2, 3.1, ncols=2)
    axes[0].barh(y - 0.2, sd.synthetic_share_firm_years * 100, height=0.4, color=C_SYN, lw=0,
                 label="Reference synthetic panel")
    axes[0].barh(y + 0.2, sd.real_share_firm_years * 100, height=0.4, color=C_REAL, lw=0, label="Real sample")
    for yi, nf, sh in zip(y, sd.real_firms, sd.real_share_firm_years):
        axes[0].text(sh * 100 + 0.5, yi + 0.2, f"{nf}", va="center")
    axes[0].set_yticks(y, [SECTOR_LABEL[s] for s in sd.sector])
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_xlabel("Share of firm-years (%)")
    axes[0].set_xlim(0, sd.real_share_firm_years.max() * 100 + 4)
    for yi, s in enumerate(sd.sector):
        pts = fm[fm.sector == s].mean_dr.values
        yv = np.full(len(pts), yi) + (np.linspace(-0.12, 0.12, len(pts)) if len(pts) > 1 else 0)
        axes[1].scatter(pts, yv, s=9, color=C_REAL, alpha=0.45, lw=0)
    axes[1].errorbar(sd.synthetic_mean_dr, y - 0.28, xerr=sd.synthetic_sd_dr, fmt="s", ms=4, color=C_SYN,
                     capsize=2, lw=1, label="Reference synthetic panel: mean ± s.d.")
    axes[1].errorbar(sd.real_mean_dr, y + 0.28, xerr=sd.real_sd_dr.fillna(0), fmt="o", ms=4, color=C_REAL,
                     capsize=2, lw=1, label="Real sample: mean ± s.d.")
    axes[1].scatter([], [], s=9, color=C_REAL, alpha=0.45, label="Real firm (4-year mean)")
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Debt ratio (total debt / total assets)")
    letter(axes[0], "a", -0.3)
    letter(axes[1], "b", -0.05)
    axes[0].legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.06), columnspacing=1.0)
    h1, l1 = axes[1].get_legend_handles_labels()
    order = [l1.index(x) for x in ("Reference synthetic panel: mean ± s.d.", "Real sample: mean ± s.d.",
                                   "Real firm (4-year mean)")]
    axes[1].legend([h1[i] for i in order], [l1[i] for i in order], loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, 1.06), columnspacing=1.0)
    save(fig, "figS25_real_sector_mix")


def s26():
    g5 = pd.read_csv(os.path.join(E7, "b_oof_grouped5_rep0_strict.csv"))
    tr = pd.read_csv(os.path.join(E7, "c_transfer_pred_strict.csv"))
    res = pd.read_csv(os.path.join(E7, "bc_results_all.csv")).set_index(["protocol", "feature_set", "model"])
    pan = [("SectorMean", g5, "grouped5 (repeat 0)", "Predicted debt ratio\n(sector mean, leave-firms-out)"),
           ("Ridge", g5, "grouped5 (repeat 0)", "Predicted debt ratio\n(Ridge, leave-firms-out)"),
           ("LightGBM", g5, "grouped5 (repeat 0)", "Predicted debt ratio\n(LightGBM, leave-firms-out)"),
           ("LightGBM", tr, "transfer: synthetic -> real (all firm-years)",
            "Predicted debt ratio (LightGBM\ntrained on synthetic panel)")]
    secs = sorted(g5.sector.unique())
    fig, axes = newfig(W2, 2.35, ncols=4, sharey=True)
    lim_ = (0, 1.0)
    for ax, (m, d, proto, lab), lt in zip(axes, pan, "abcd"):
        for s in secs:
            dd = d[d.sector == s]
            ax.scatter(dd[m], dd.y, s=22 if SEC_MARK[s] == "*" else 11, marker=SEC_MARK[s], color=SEC_COLOR[s], alpha=0.85, lw=0.3,
                       edgecolor="white", label=SECTOR_LABEL[s])
        ax.plot(lim_, lim_, color=OI["grey"], lw=0.8, ls="--")
        ax.set_xlim(lim_)
        ax.set_ylim(lim_)
        ax.set_aspect("equal")
        r2 = res.loc[(proto, "strict", m), "r2"]
        ax.text(0.97, 0.04, f"{R2} = {r2:.2f}".replace("-", "−"), transform=ax.transAxes, ha="right",
                va="bottom")
        ax.set_xlabel(lab)
        letter(ax, lt, -0.22 if lt == "a" else -0.08)
    axes[0].set_ylabel("Actual debt ratio")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=8, handletextpad=0.2, columnspacing=0.9)
    save(fig, "figS26_real_pred_vs_actual")


def s27():
    g = pd.read_csv(os.path.join(E7, "d_shap_grouped_strict.csv"))
    g = g.dropna(subset=["real12_share"]).sort_values("real12_share")
    pretty = {"size_code": "Size code", "cash_ratio": "Cash ratio", "sector (code + dummies)":
              "Sector (code + dummies)", "roa": "ROA"}
    fig, ax = newfig(W1, 2.2)
    y = np.arange(len(g))
    ax.barh(y - 0.2, g.synthetic12_share * 100, height=0.4, color=C_SYN, lw=0, label="Reference synthetic panel")
    ax.barh(y + 0.2, g.real12_share * 100, height=0.4, color=C_REAL, lw=0, label="Real sample")
    ax.set_yticks(y, [pretty.get(f, f.replace("_", " ")) for f in g.feature_group])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of mean |SHAP| (%)")
    ax.legend(loc="lower right")
    save(fig, "figS27_real_shap_vs_synthetic")


def s28():
    f6 = pd.read_csv(os.path.join(E7, "e_firm_vs_sector_median.csv")).sort_values("dev_from_sector_median")
    secs = sorted(f6.sector.unique())
    fig, ax = newfig(W1, 5.4)
    y = np.arange(len(f6))
    ax.axvspan(-5, 5, color="0.92", zorder=0, lw=0)
    ax.hlines(y, 0, f6.dev_from_sector_median * 100, color="0.7", lw=1)
    for s in secs:
        msk = (f6.sector == s).values
        ax.scatter(f6.dev_from_sector_median.values[msk] * 100, y[msk], s=26 if SEC_MARK[s] == "*" else 14, marker=SEC_MARK[s],
                   color=SEC_COLOR[s], lw=0, zorder=3,
                   label=f"{SECTOR_LABEL[s]} ({int(f6[f6.sector == s].sector_n_firms.iloc[0])})")
    ax.axvline(0, color="0.3", lw=1)
    ax.set_yticks(y, f6.ticker)
    ax.tick_params(axis="y", length=0, labelsize=6)
    ax.set_xlabel("Firm mean debt ratio minus sector median\n(percentage points)")
    ax.set_ylim(-0.8, len(f6) - 0.2)
    ax.legend(loc="lower right", title="Sector (firms)", title_fontsize=FS, handletextpad=0.2)
    save(fig, "figS28_real_firm_vs_sector_median")


# ---------------------------------------------------------------------------
# S29-S30 (E8)
# ---------------------------------------------------------------------------
E8_PRED = ["size_assets", "size_revenue", "tangibility", "profitability", "cash_ratio",
           "capex_assets", "da_assets", "sales_growth", "div_payer", "ocf_assets", "ca_assets"]
E8_LABEL = {"size_assets": "Size: ln total assets", "size_revenue": "Size: ln revenue",
            "tangibility": "Tangibility (net PPE/TA)", "profitability": "Profitability (EBIT/TA)",
            "cash_ratio": "Cash/TA", "capex_assets": "Capex/TA", "da_assets": "D&A/TA",
            "sales_growth": "Sales growth", "div_payer": "Dividend payer", "ocf_assets": "OCF/TA",
            "ca_assets": "Current assets/TA"}
E8_EXPECTED = {"tangibility": "+", "size_assets": "+", "size_revenue": "+", "profitability": "−",
               "ocf_assets": "−", "sales_growth": "−", "div_payer": "−", "da_assets": "−", "cash_ratio": "−",
               "capex_assets": "?", "ca_assets": "?"}


def s29():
    fr = pd.read_csv(os.path.join(E8, "fig_e8_r2_by_protocol_data.csv"))
    groups = list(dict.fromkeys(fr.group))
    LO, HI = -1.0, 1.0
    fig, ax = newfig(W2, 3.0)
    mods = [("SectorMedian (naive)", OI["black"], "o", "Sector median (naive baseline)"),
            ("Ridge", OI["blue"], "s", "Ridge"), ("LightGBM", OI["vermillion"], "D", "LightGBM")]
    for k, (mname, c, mk, lab) in enumerate(mods):
        s = fr[fr.model == mname].set_index("group").reindex(groups)
        x = np.arange(len(groups)) + (k - 1) * 0.22
        yv = s.R2.clip(LO, HI)
        lo_c = s.lo.clip(lower=LO)
        ax.errorbar(x, yv, yerr=[(yv - lo_c).clip(lower=0), (s.hi - yv).clip(lower=0)], fmt=mk, ms=4, color=c,
                    capsize=2, lw=1, label=lab)
        for xi, rv, lo_raw in zip(x, s.R2, s.lo):
            if rv < LO:
                ax.annotate(f"{rv:.1f}", (xi, LO), xytext=(0, 6), textcoords="offset points", ha="center",
                            color=c)
                ax.plot(xi, LO, marker="v", color=c, ms=4)
            elif lo_raw < LO:   # CI truncated at the axis floor
                ax.plot(xi, LO, marker="v", color=c, ms=3.5, mfc="white", mew=1)
    ax.axhline(0, color=OI["grey"], lw=0.8, ls="--")
    ax.set_ylim(LO - 0.05, HI)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Out-of-sample " + R2 + " (95% firm-bootstrap CI)")
    hs, labs = ax.get_legend_handles_labels()
    hs.append(Line2D([], [], ls="none", marker="v", color=OI["vermillion"], mfc="white", mew=1, ms=3.5))
    labs.append("CI extends below −1")
    ax.legend(hs, labs, loc="upper right", ncol=4, columnspacing=1.2)
    save(fig, "figS29_sec_r2_by_protocol")


def s30():
    reg = pd.read_csv(os.path.join(E8, "b_panel_regressions_all.csv"))
    cp = reg[(reg["sample"] == "main") & (reg.target == "debt_ratio") & (reg.standardised_X == True)  # noqa: E712
             & (reg.spec == "full")]
    ests = [("Pooled OLS (firm-clustered)", OI["black"], "o"), ("Firm FE (firm-clustered)", OI["blue"], "s"),
            ("Two-way FE (firm-clustered)", OI["vermillion"], "D"), ("Firm FE (Driscoll-Kraay)", OI["green"], "^")]
    order = E8_PRED[::-1]
    fig, ax = newfig(W2, 3.6)
    for k, (e, c, mk) in enumerate(ests):
        s = cp[cp.estimator == e].set_index("variable").reindex(order)
        assert s.coef.notna().all(), e
        yy = np.arange(len(order)) + (k - 1.5) * 0.18
        ax.errorbar(s.coef, yy, xerr=[s.coef - s.ci_lo, s.ci_hi - s.coef], fmt=mk, ms=3.5, color=c, lw=1,
                    capsize=1.5, label=e)
    ax.axvline(0, color=OI["grey"], lw=0.8, ls="--")
    ax.set_yticks(np.arange(len(order)), [f"{E8_LABEL[f]} [{E8_EXPECTED[f]}]" for f in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Change in book debt ratio per 1 s.d. of the determinant (95% CI)")
    ax.legend(loc="lower right")
    save(fig, "figS30_sec_coefficients")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    for fn in [s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30]:
        if not only or fn.__name__ in only:
            try:
                fn()
            except FileNotFoundError as e:  # firm-level third-party data are not distributed
                print(f"  skipped {fn.__name__}: input not found ({os.path.basename(e.filename or '')})")
