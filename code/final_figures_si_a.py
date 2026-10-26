"""
final_figures_si_a.py -- Supplementary Figs. S1-S15 in the shared figure style (figstyle.py).

Re-plots each figure from the saved experiment outputs only; no model is refitted and no value is changed.
  S01-S06  output/e1_benchmark/*.csv
  S07      output/e2_deep_tabular/summary.csv
  S08-S12  output/e3_shap/shap_values_oof_S4.csv.gz, shap_ranking_S4.csv, stability_S4_ranks.csv
           (feature values for the beeswarm/dependence colouring come from the deterministic reference
           synthetic panel, common.generate_panel(42); identity with the SHAP file is asserted row by row)
  S13-S15  output/e4_generative/samples/*.csv (seed 42; CTGAN/TVAE 300 epochs) and the same panel

Style: sans-serif 7 pt, bold lower-case 8 pt panel letters, no titles, Okabe-Ito, PDF + 300 dpi PNG at a
fixed physical width (88 mm single, 180 mm double column).
Previous versions are copied to figures/_backup/ before overwriting (once; existing backups kept).

Run: python code/final_figures_si_a.py
"""
from __future__ import annotations

import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
try:
    import figstyle as fsty  # noqa: E402
except ImportError:  # fallback with the same rules
    import figstyle_si as fsty  # noqa: E402
import common  # noqa: E402

plt = fsty.apply()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

import config  # noqa: E402
ROOT = config.ROOT
OUTR = config.OUT_MAIN
E1, E2, E3, E4 = (os.path.join(OUTR, d) for d in ("e1_benchmark", "e2_deep_tabular", "e3_shap", "e4_generative"))
FIGDIR = config.FIG_DIR
BACKUP = os.path.join(FIGDIR, "_backup")
OI = fsty.OI
W1, W2, FS = fsty.W_SINGLE, fsty.W_DOUBLE, fsty.FS
SEC8 = [OI["orange"], OI["skyblue"], OI["green"], OI["yellow"], OI["blue"], OI["vermillion"], OI["purple"],
        OI["black"]]   # = common.OKABE_ITO order, used for the 8 sectors as in the source figures
C_ORIG, C_STRICT = OI["vermillion"], OI["blue"]
R2 = "R²"


def letter(ax, s, x=-0.12):
    """Bold lower-case panel letter, top left; implemented as a left title so the layout engine reserves room."""
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
# S01-S06 (E1)
# ---------------------------------------------------------------------------
def s01():
    folds = pd.read_csv(os.path.join(E1, "bench_fold_metrics.csv"))
    g = folds[folds.protocol == "grouped_5x5"].groupby(["spec", "model"])["R2"].agg(["mean", "std"]).reset_index()
    order = g[g.spec == "strict"].sort_values("mean")["model"].tolist()
    fig, ax = newfig(W1, 3.0)
    yp = np.arange(len(order))
    for off, spec, col, mk, lab in [(-0.17, "original", C_ORIG, "s", "Original features"),
                                    (0.17, "strict", C_STRICT, "o", "Strict (leakage-free) features")]:
        d = g[g.spec == spec].set_index("model").loc[order]
        ax.errorbar(d["mean"], yp + off, xerr=d["std"], fmt=mk, color=col, ms=3.5, capsize=2, lw=1, label=lab)
    ax.axvline(0, color=OI["grey"], lw=0.8, ls=":")
    ax.set_yticks(yp, order)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Out-of-sample " + R2 + " (mean ± s.d., 25 grouped folds)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.3, 1.0), ncol=2, borderaxespad=0.2, handletextpad=0.3,
              columnspacing=1.0)
    save(fig, "figS01_r2_by_model_grouped5x5")


def s02():
    summ = pd.read_csv(os.path.join(E1, "bench_summary_long.csv"))
    prots = ["legacy_random_80_20", "random_5fold", "grouped_5x5", "temporal", "unseen_firm_future"]
    labs = ["Random\n80/20", "Random\n5-fold", "Grouped\n5×5", "Temporal", "Unseen\nfirms,\nfuture"]
    fig, ax = newfig(W1, 2.6)
    x = np.arange(len(prots))
    w = 0.38
    for i, (spec, col, hatch, lab) in enumerate([("original", C_ORIG, "", "Original features"),
                                                 ("strict", C_STRICT, "////", "Strict features")]):
        d = summ[(summ.spec == spec) & (summ.model == "LightGBM")].set_index("protocol").loc[prots]
        err = d["R2_sd"].values
        xs = x + (i - 0.5) * w
        bars = ax.bar(xs, d["R2"], w, color=col, edgecolor="white" if hatch else col, lw=0, hatch=hatch,
                      label=lab)
        has = ~np.isnan(err)
        ax.errorbar(xs[has], d["R2"].values[has], yerr=err[has], fmt="none", ecolor="black", elinewidth=1,
                    capsize=2)
        for b, v, e in zip(bars, d["R2"], np.nan_to_num(err)):
            ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + e + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=FS)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, labs)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("LightGBM out-of-sample " + R2)
    lg = summ[(summ.model == "LightGBM") & summ.protocol.isin(prots)]
    ax.set_ylim(min(0, (lg["R2"] - lg["R2_sd"].fillna(0)).min() - 0.03), 1.1)
    ax.legend(loc="upper right", handlelength=1.4)
    save(fig, "figS02_inflation_lightgbm")


def s03():
    pr = pd.read_csv(os.path.join(E1, "lightgbm_test_predictions.csv"))
    d = pr[(pr.spec == "strict") & (pr.protocol == "temporal")]
    fig, axes = newfig(W2, 2.7, ncols=2)
    secs = sorted(d.sector.unique())
    for k, s in enumerate(secs):
        dd = d[d.sector == s]
        axes[0].scatter(dd.y_true, dd.y_pred, s=4, alpha=0.7, color=SEC8[k % 8], lw=0, label=s, rasterized=True)
        axes[1].scatter(dd.y_pred, dd.y_true - dd.y_pred, s=4, alpha=0.7, color=SEC8[k % 8], lw=0,
                        rasterized=True)
    lim = [0, max(d.y_true.max(), d.y_pred.max()) * 1.05]
    axes[0].plot(lim, lim, color="black", lw=1, ls="--")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Actual debt ratio (debt/assets)")
    axes[0].set_ylabel("Predicted debt ratio (debt/assets)")
    r2 = r2_score(d.y_true, d.y_pred)
    axes[0].text(0.03, 0.97, f"{R2} = {r2:.3f}, n = {len(d):,}", transform=axes[0].transAxes, va="top")
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set_xlabel("Predicted debt ratio (debt/assets)")
    axes[1].set_ylabel("Residual (actual − predicted)")
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(1.0, 0.5),
                   markerscale=2.5, handletextpad=0.1, title="Sector")
    letter(axes[0], "a", -0.2)
    letter(axes[1], "b", -0.2)
    save(fig, "figS03_actual_vs_pred_strict_temporal")


def s04():
    lc = pd.read_csv(os.path.join(E1, "learning_curve_lightgbm_grouped.csv"))
    fig, ax = newfig(W1, 2.8)
    for spec, col, mk, lab in [("original", C_ORIG, "s", "Original"), ("strict", C_STRICT, "o", "Strict")]:
        d = lc[lc.spec == spec]
        ax.errorbar(d.n_train_firms, d.R2_test, yerr=d.R2_test_sd, color=col, marker=mk, ms=3, lw=1, capsize=2,
                    label=f"{lab}, held-out firms")
        ax.plot(d.n_train_firms, d.R2_train, color=col, marker=mk, ms=3, lw=1, ls="--", mfc="white",
                label=f"{lab}, training")
    ax.set_xlabel("Training firms (count; 15 years each)")
    ax.set_ylabel("LightGBM " + R2)
    ax.set_ylim(min(0, (lc.R2_test - lc.R2_test_sd).min() - 0.05), 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0, handletextpad=0.4)
    save(fig, "figS04_learning_curve_grouped")


def s05():
    ap = pd.read_csv(os.path.join(E1, "across_panels_lightgbm.csv"))
    fig, ax = newfig(W1, 2.2)
    cats = [("original", "legacy_random_80_20", "Original,\nrandom 80/20", C_ORIG),
            ("original", "grouped_5fold", "Original,\ngrouped 5-fold", OI["orange"]),
            ("strict", "grouped_5fold", "Strict,\ngrouped 5-fold", C_STRICT)]
    for i, (sp, pt, lab, col) in enumerate(cats):
        v = ap[(ap.spec == sp) & (ap.protocol == pt)]["R2"].values
        jit = np.linspace(-0.12, 0.12, len(v))
        ax.scatter(np.full(len(v), i) + jit, v, s=10, color=col, edgecolor="black", lw=0.3, zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color="black", lw=1)
    ax.set_xticks(range(len(cats)), [c[2] for c in cats])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_ylabel("LightGBM " + R2 + " (10 panels)")
    ax.set_ylim(min(0, ap.R2.min() - 0.05), 1.0)
    save(fig, "figS05_across_panels_lightgbm")


def s06():
    sens = pd.read_csv(os.path.join(E1, "sensitivity_lambda_sigma.csv"))
    fig, axes = newfig(W2, 2.6, ncols=2)
    for ax, col, lab, lt in [(axes[0], "R2_strict_grouped", "strict features, grouped 5-fold", "a"),
                             (axes[1], "R2_original_legacy_random", "original features, random 80/20", "b")]:
        piv = sens.pivot(index="lam", columns="sigma_zeta2", values=col)
        im = ax.imshow(piv.values, cmap="cividis", aspect="auto", origin="lower",
                       vmin=min(0, sens[col].min()), vmax=1)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.55 else "black")
        ax.set_xticks(range(piv.shape[1]), [f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{r:g}" for r in piv.index])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(r"Shock variance $\sigma_\zeta^2$")
        ax.set_ylabel(r"Adjustment speed $\lambda$")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label("LightGBM " + R2 + "\n(" + lab + ")")
        cb.outline.set_linewidth(0.6)
        letter(ax, lt, -0.2)
    save(fig, "figS06_dgp_sensitivity_heatmap")


# ---------------------------------------------------------------------------
# S07 (E2)
# ---------------------------------------------------------------------------
E2_LABEL = {"LightGBM": "LightGBM (tuned)", "LightGBM_paper": "LightGBM (paper config.)",
            "FT-Transformer": "FT-Transformer", "TabNet": "TabNet", "MLP": "MLP (tuned)",
            "MLP_paper": "MLP (paper config.)"}
E2_PROTO = {"grouped5": "Grouped 5-fold\n(unseen firms)", "temporal": "Temporal\n(2021-2024)",
            "unseen": "Unseen firms,\nfuture years"}


def s07():
    summary = pd.read_csv(os.path.join(E2, "summary.csv"))
    models = ["LightGBM", "FT-Transformer", "TabNet", "MLP", "LightGBM_paper", "MLP_paper"]
    colors = {"LightGBM": OI["blue"], "FT-Transformer": OI["vermillion"], "TabNet": OI["green"],
              "MLP": OI["orange"], "LightGBM_paper": OI["blue"], "MLP_paper": OI["orange"]}
    markers = {"LightGBM": "o", "FT-Transformer": "s", "TabNet": "^", "MLP": "D", "LightGBM_paper": "o",
               "MLP_paper": "D"}
    protos = ["grouped5", "temporal", "unseen"]
    fig, axes = newfig(W2, 2.8, ncols=2)
    off = np.linspace(-0.27, 0.27, len(models))
    for ax, (fs, ylab), lt in zip(axes, [("strict", "strict (leakage-free) features"),
                                        ("original", "original 21 features")], "ab"):
        s = summary[summary.feature_set == fs]
        for j, m in enumerate(models):
            xs, ys, es = [], [], []
            for i, p in enumerate(protos):
                r = s[(s.protocol == p) & (s.model == m)]
                if len(r):
                    xs.append(i + off[j]); ys.append(r["r2_mean"].iloc[0]); es.append(r["r2_sd"].iloc[0])
            hollow = m.endswith("_paper")
            ax.errorbar(xs, ys, yerr=es, fmt=markers[m], ms=4, capsize=2, lw=1, color=colors[m],
                        mfc="white" if hollow else colors[m], mec=colors[m], label=E2_LABEL[m])
        ax.set_xticks(range(len(protos)), [E2_PROTO[p] for p in protos])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("Test " + R2 + ", " + ylab)
        ax.axhline(0, color=OI["grey"], lw=0.8, ls=":")
        ax.set_xlim(-0.6, len(protos) - 0.4)
        letter(ax, lt, -0.15)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, columnspacing=1.5)
    save(fig, "figS07_deep_tabular_with_paper_configs")


# ---------------------------------------------------------------------------
# S08-S12 (E3)
# ---------------------------------------------------------------------------
DISPLAY = {"roa": "ROA", "cash_ratio": "Cash ratio", "sector_code": "Sector code", "size_code": "Size tertile",
           "gdp_growth": "GDP growth", "inflation": "Inflation", "interest_rate": "Interest rate",
           "credit_spread": "Credit spread"}
for _s in common.SECTORS:
    DISPLAY[f"sector_{_s}"] = f"Sector: {_s}"
MACRO = ["gdp_growth", "inflation", "interest_rate", "credit_spread"]
_E3 = {}


def e3_data():
    if _E3:
        return _E3
    sv = pd.read_csv(os.path.join(E3, "shap_values_oof_S4.csv.gz"))
    feats = list(common.FEATURES_STRICT)
    df = common.generate_panel(seed=42)
    assert len(df) == len(sv)
    assert np.array_equal(df["firm_id"].values, sv["firm_id"].values)
    assert np.array_equal(df["year"].values, sv["year"].values)
    assert np.allclose(df[common.TARGET].values, sv["debt_ratio"].values, rtol=0, atol=1e-12), "panel mismatch"
    _E3.update(sv=sv[feats].values, X=df[feats].astype(float), feats=feats, sector=df["sector"].values,
               rk=pd.read_csv(os.path.join(E3, "shap_ranking_S4.csv")),
               stab=pd.read_csv(os.path.join(E3, "stability_S4_ranks.csv")))
    # check the stored ranking against the SHAP file
    imp = pd.Series(np.abs(_E3["sv"]).mean(axis=0), index=feats)
    assert np.allclose(imp.loc[_E3["rk"].feature].values, _E3["rk"].mean_abs_shap_oof.values, atol=1e-12)
    return _E3


def s08():
    import shap
    d = e3_data()
    plt.figure()
    shap.summary_plot(d["sv"], d["X"].values, feature_names=[DISPLAY[f] for f in d["feats"]], show=False,
                      max_display=len(d["feats"]), plot_size=(W1, 3.9), cmap=plt.get_cmap("viridis"),
                      color_bar_label="Feature value (low to high)", alpha=0.6)
    fig = plt.gcf()
    fsty.apply()
    for ax in fig.axes:
        ax.tick_params(labelsize=FS, width=1)
        ax.xaxis.label.set_size(FS)
        ax.yaxis.label.set_size(FS)
        for t in ax.get_yticklabels() + ax.get_xticklabels():
            t.set_fontfamily("sans-serif")
            t.set_fontsize(FS)
        for t in ax.texts:
            t.set_fontsize(FS)
    main_ax = fig.axes[0]
    main_ax.set_xlabel("SHAP value (effect on predicted debt ratio)", fontsize=FS)
    main_ax.spines["bottom"].set_linewidth(1)
    for c in main_ax.collections:
        c.set_rasterized(True)
        c.set_sizes([4])
        c.set_alpha(0.7)
    if len(fig.axes) > 1:
        cb = fig.axes[-1]
        cb.set_ylabel("Feature value (low to high)", fontsize=FS, labelpad=0)
        cb.tick_params(labelsize=FS)
    fig.set_size_inches(W1, 3.9)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.33, right=0.86, top=0.99, bottom=0.11)
    save(fig, "figS08_shap_S4_beeswarm")


def s09():
    d = e3_data()
    rk = d["rk"].iloc[::-1]
    fig, ax = newfig(W1, 3.3)
    ax.barh([DISPLAY[f] for f in rk["feature"]], rk["mean_abs_shap_oof"], xerr=rk["mean_abs_shap_oof_fold_sd"],
            color=OI["blue"], height=0.7, error_kw=dict(lw=1, capsize=2, ecolor="black"))
    ax.set_xlabel("Mean |SHAP value| (debt ratio units)")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rk) - 0.4)
    save(fig, "figS09_shap_S4_bar")


def s10():
    """Rank stability of the S4 importance ordering. Per feature: grey bar = range of ranks over the 25
    panel-seed x fold fits (rank_min/max_seedfold); filled circles = 5 fold ranks for panel seed 42;
    open diamonds = out-of-fold ranks for the 5 panel seeds; right-hand number = times in the top 5 of 25.
    (The 25 individual seed x fold ranks were not stored; their min-max range and top-5 count were.)"""
    st = e3_data()["stab"].iloc[::-1].reset_index(drop=True)
    fig, ax = newfig(W1, 3.4)
    y = np.arange(len(st))
    ax.hlines(y, st.rank_min_seedfold, st.rank_max_seedfold, color="#CCCCCC", lw=4, capstyle="round", zorder=1)
    fcols = [f"rank_fold{k}" for k in range(5)]
    scols = [f"rank_seed{s}" for s in (42, 43, 44, 45, 46)]
    rng = np.random.RandomState(0)
    for c in fcols:
        ax.scatter(st[c], y + rng.uniform(-0.12, 0.12, len(y)), s=7, color=OI["skyblue"], lw=0, zorder=2)
    for c in scols:
        ax.scatter(st[c], y, s=16, marker="D", facecolor="none", edgecolor=OI["vermillion"], lw=0.8, zorder=3)
    for yi, n in zip(y, st.top5_count_seedfold_of25):
        ax.text(-0.3, yi, f"{n}", va="center", ha="center", fontsize=FS)
    ax.text(-0.3, len(y) - 0.45, "Top 5\n(of 25)", va="bottom", ha="center", fontsize=FS)
    ax.set_yticks(y, st.display_name)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(16.6, -1.0)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.spines["bottom"].set_bounds(16, 1)
    ax.set_ylim(-0.6, len(y) + 0.6)
    ax.set_xlabel("Rank by mean |SHAP value| (1 = most important)")
    hs = [Line2D([], [], color="#CCCCCC", lw=3, label="Range, 25 seed × fold fits"),
          Line2D([], [], ls="", marker="o", ms=3, color=OI["skyblue"], label="Fold, panel seed 42 (5)"),
          Line2D([], [], ls="", marker="D", ms=3.5, mfc="none", mec=OI["vermillion"], label="Panel seed, out-of-fold (5)")]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(-0.45, 1.0), ncol=2, columnspacing=1.0,
              handlelength=1.5, handletextpad=0.4, borderaxespad=0.1)
    save(fig, "figS10_shap_S4_rank_stability")


def dependence(feat_list, name, rng):
    d = e3_data()
    feats, X, sv = d["feats"], d["X"], d["sv"]
    secs = sorted(common.SECTORS)
    fig, axes = newfig(W2, 2.7, ncols=2)
    for ax, feat, lt in zip(axes, feat_list, "ab"):
        j = feats.index(feat)
        xv = X[feat].values.copy()
        nun = len(np.unique(xv))
        if nun <= 10:  # same jitter sequence as e3_shap.py (RandomState(42), S11 then S12)
            xv = xv + rng.uniform(-0.15, 0.15, len(xv)) * (np.ptp(X[feat].values) / max(nun - 1, 1))
        for si, s_ in enumerate(secs):
            m = d["sector"] == s_
            ax.scatter(xv[m], sv[m, j], s=2.5, alpha=0.5, color=SEC8[si], lw=0, label=s_, rasterized=True)
        ax.axhline(0, color=OI["grey"], lw=0.8)
        ax.set_ylabel(f"SHAP value for {DISPLAY[feat]}\n(debt ratio units)")
        if feat == "sector_code":
            ax.set_xticks(range(8), secs, rotation=45, ha="right")
            ax.set_xlabel("Sector code (alphabetical)")
        elif feat.startswith("sector_"):
            ax.set_xticks([0, 1], ["0 (other sectors)", "1 (" + feat.split("_", 1)[1] + ")"])
            ax.set_xlabel(DISPLAY[feat] + " dummy")
        elif feat == "size_code":
            ax.set_xticks(sorted(np.unique(X[feat].values)))
            ax.set_xlabel("Size tertile (sample tertile of total assets)")
        elif feat in ("roa", "cash_ratio"):
            ax.set_xlabel(DISPLAY[feat] + " (ratio to total assets)")
        elif feat in MACRO:
            ax.set_xlabel(DISPLAY[feat] + " (%)")
        else:
            ax.set_xlabel(DISPLAY[feat])
        letter(ax, lt, -0.2)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=3, handletextpad=0.1,
                   title="Sector")
    save(fig, name)


def s11_s12():
    d = e3_data()
    rng = np.random.RandomState(42)
    top2 = d["rk"]["feature"].head(2).tolist()
    dependence(top2, "figS11_shap_S4_dependence_sector", rng)
    block = ["sector_code"] + common.SECTOR_DUMMIES
    top2ns = [f for f in d["rk"]["feature"] if f not in block][:2]
    dependence(top2ns, "figS12_shap_S4_dependence_roa_size", rng)


# ---------------------------------------------------------------------------
# S13-S15 (E4)
# ---------------------------------------------------------------------------
NUM = ["total_assets", "total_debt", "total_equity", "revenue", "net_income", "cash_holdings",
       "interest_expense", "market_cap", "debt_ratio", "roa", "roe", "cash_ratio", "interest_coverage",
       "tobin_q", "wacc", "default_risk_score", "altman_zscore"]
LEVELS_POS = ["total_assets", "total_debt", "total_equity", "revenue", "cash_holdings", "interest_expense",
              "market_cap"]
LEVELS_SIGNED = ["net_income"]
GEN_LABEL = {"B1_noise": "Noise baseline (B1)", "B2_gauss": "Gaussian baseline (B2)", "CTGAN": "CTGAN",
             "TVAE": "TVAE"}
GEN_FILE = {"B1_noise": "B1_noise_seed42.csv", "B2_gauss": "B2_gauss_seed42.csv",
            "CTGAN": "CTGAN_ep300_seed42.csv", "TVAE": "TVAE_ep300_seed42.csv"}
GEN_COLOR = {"B1_noise": OI["orange"], "B2_gauss": OI["skyblue"], "CTGAN": OI["green"], "TVAE": OI["blue"]}
REF_COLOR = OI["grey"]
FEAT_LABEL = {"debt_ratio": "Debt ratio (D/A)", "roa": "ROA (NI/A)", "cash_ratio": "Cash ratio (cash/A)",
              "wacc": "WACC (fraction)", "interest_coverage": "Interest coverage (EBIT/interest)",
              "tobin_q": "Tobin's Q", "total_assets": "log$_{10}$ total assets (model units)",
              "default_risk_score": "Default-risk score (0-1)"}
SHORT = {"total_assets": "A", "total_debt": "D", "total_equity": "E", "revenue": "Rev", "net_income": "NI",
         "cash_holdings": "Cash", "interest_expense": "IntExp", "market_cap": "MCap", "debt_ratio": "DR",
         "roa": "ROA", "roe": "ROE", "cash_ratio": "CashR", "interest_coverage": "IntCov", "tobin_q": "Q",
         "wacc": "WACC", "default_risk_score": "DRisk", "altman_zscore": "Z"}
_E4 = {}


def e4_data():
    if _E4:
        return _E4
    ref = common.generate_panel(42)[NUM + ["sector"]].reset_index(drop=True)
    samples = {k: pd.read_csv(os.path.join(E4, "samples", f)) for k, f in GEN_FILE.items()}
    runs = pd.read_csv(os.path.join(E4, "generative_runs.csv"))
    _E4.update(ref=ref, samples=samples, runs=runs)
    return _E4


def s13():
    d = e4_data()
    ref, samples = d["ref"], d["samples"]
    feats = ["debt_ratio", "roa", "cash_ratio", "wacc", "interest_coverage", "tobin_q", "total_assets",
             "default_risk_score"]
    fig, axes = newfig(W2, 3.4, nrows=2, ncols=4)
    for ax, f, lt in zip(axes.ravel(), feats, "abcdefgh"):
        tf = (lambda v: np.log10(v[v > 0])) if f == "total_assets" else (lambda v: v)
        r = tf(ref[f].values.astype(float))
        lo, hi = np.percentile(r, [0.5, 99.5])
        pad = 0.15 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, 41)
        ax.hist(r, bins=edges, density=True, color=REF_COLOR, alpha=0.5, lw=0, label="Reference synthetic panel")
        for k, s in samples.items():
            v = tf(s[f].values.astype(float))
            v = v[np.isfinite(v)]
            ax.hist(np.clip(v, edges[0], edges[-1]), bins=edges, density=True, histtype="step", lw=0.9,
                    color=GEN_COLOR[k], label=GEN_LABEL[k])
        ax.set_xlabel(FEAT_LABEL[f])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        letter(ax, lt, -0.04)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=5, handlelength=1.6, columnspacing=1.2)
    save(fig, "figS13_generative_distributions")


def _pca_tr(df, mu=None, sd=None):
    X = df[NUM].astype(float).copy()
    for c in LEVELS_POS + LEVELS_SIGNED:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    if mu is None:
        return X.mean(), X.std()
    return ((X - mu) / sd).values


def s14():
    from sklearn.decomposition import PCA
    d = e4_data()
    ref, samples = d["ref"], d["samples"]
    mu, sd = _pca_tr(ref)
    Zr = _pca_tr(ref, mu, sd)
    pca = PCA(2, random_state=0).fit(Zr)
    Pr = pca.transform(Zr)
    ev = pca.explained_variance_ratio_
    print(f"    PCA explained variance {ev}", flush=True)
    fig, axes = newfig(W2, 2.0, ncols=4, sharex=True, sharey=True)
    xl = np.percentile(Pr[:, 0], [0.2, 99.8]); yl = np.percentile(Pr[:, 1], [0.2, 99.8])
    padx, pady = 0.4 * (xl[1] - xl[0]), 0.4 * (yl[1] - yl[0])
    for ax, (k, s), lt in zip(axes, samples.items(), "abcd"):
        Ps = pca.transform(np.nan_to_num(_pca_tr(s, mu, sd)))
        ax.scatter(Pr[:, 0], Pr[:, 1], s=1.2, c=REF_COLOR, alpha=0.35, lw=0, rasterized=True)
        ax.scatter(Ps[:, 0], Ps[:, 1], s=1.2, c=GEN_COLOR[k], alpha=0.35, lw=0, rasterized=True)
        ax.text(0.04, 0.97, GEN_LABEL[k], transform=ax.transAxes, va="top", color=GEN_COLOR[k])
        ax.set_xlim(xl[0] - padx, xl[1] + padx); ax.set_ylim(yl[0] - pady, yl[1] + pady)
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.0f}% of variance)")
        letter(ax, lt, -0.1)
    axes[0].set_ylabel(f"PC2 ({100 * ev[1]:.0f}% of variance)")
    hs = [Line2D([], [], marker="o", ls="", color=REF_COLOR, ms=3, label="Reference synthetic panel"),
          Line2D([], [], marker="o", ls="", color="#444444", ms=3, label="Generated (color by generator)")]
    fig.legend(handles=hs, loc="outside upper center", ncol=2)
    save(fig, "figS14_generative_pca")


def corr_metrics(ref_df, syn_df, cols):
    """As e4_generative.corr_metrics: MAD over the finite upper-triangle entries; Frobenius over the full matrix."""
    d = syn_df[cols].corr().values - ref_df[cols].corr().values
    du = d[np.triu_indices(len(cols), 1)]
    mad = float(np.mean(np.abs(du[np.isfinite(du)])))
    fro = float(np.sqrt(np.nansum(d ** 2)))
    return mad, fro, d


def s15():
    d = e4_data()
    ref, samples, runs = d["ref"], d["samples"], d["runs"]
    fig, axes = newfig(W2, 6.4, nrows=2, ncols=2)
    lab = [SHORT[c] for c in NUM]
    im = None
    for ax, k, lt in zip(axes.ravel(), samples, "abcd"):
        mad, fro, dd = corr_metrics(ref, samples[k], NUM)
        ep = 300 if k in ("CTGAN", "TVAE") else None
        r = runs[(runs.generator == k) & (runs.seed == 42)]
        if ep is not None:
            r = r[r.epochs == ep]
        assert len(r) == 1 and abs(r.corr_MAD.iloc[0] - mad) < 1e-9, (k, mad)
        im = ax.imshow(np.ma.masked_invalid(dd), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(NUM)), lab, rotation=90)
        ax.set_yticks(range(len(NUM)), lab)
        ax.tick_params(length=0, labelsize=FS)
        ax.set_xlabel(f"{GEN_LABEL[k]}: MAD = {mad:.3f}, Frobenius norm = {fro:.2f}")
        for sp in ax.spines.values():
            sp.set_visible(False)
        letter(ax, lt, -0.12)
    cb = fig.colorbar(im, ax=axes, shrink=0.45, aspect=25, pad=0.02)
    cb.set_label("Pearson r (generated) − r (reference)")
    cb.outline.set_linewidth(0.6)
    save(fig, "figS15_generative_corr_diff")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    jobs = [("S01", s01), ("S02", s02), ("S03", s03), ("S04", s04), ("S05", s05), ("S06", s06), ("S07", s07),
            ("S08", s08), ("S09", s09), ("S10", s10), ("S11", s11_s12), ("S13", s13), ("S14", s14),
            ("S15", s15)]
    for tag, fn in jobs:
        if not only or tag in only:
            print(tag, flush=True)
            try:
                fn()
            except FileNotFoundError as e:
                print(f"  skipped {tag}: input not found ({os.path.basename(e.filename or '')})")
            fsty.apply()
