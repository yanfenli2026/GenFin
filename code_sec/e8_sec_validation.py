"""
e8_sec_validation.py -- experiment E8: external validation on the SEC XBRL panel
(independently sourced, point-in-time 10-K data; Datavidence sample).

Outputs: output/e8_sec/  (CSV / JSON tables, figures PNG 300 dpi + PDF,
tables_autogen.md with every table rendered as markdown).

Sections
 (a) descriptives + correlations of debt_ratio with leakage-free determinants, sign check
 (b) panel regressions: pooled OLS (firm-clustered), firm FE (clustered), two-way FE (clustered),
     firm FE with Driscoll-Kraay SEs; full and core (Rajan-Zingales-type) specifications;
     debt_ratio and liabilities_ratio
 (c) predictive validation: naive baselines, LinearReg, Ridge, RF, LightGBM, XGBoost under
     (i) sector-stratified GroupKFold(5) and leave-one-firm-out, (ii) out-of-time
     (train FY2009-2019, test FY2020-2025, same firms), (iii) unseen firms in future years;
     firm-bootstrap CIs (1,000 reps); Nadeau-Bengio corrected t-test LightGBM vs Ridge (5x5)
 (d) transfer: trained on the reference synthetic panel, tested on the SEC panel
 (e) TreeSHAP on out-of-fold LightGBM (SEC) vs synthetic strict ranking (E3, S4)
 (f) pooled real evidence: SEC + 46-firm Yahoo panel, de-duplicated (SEC preferred)
 (g) sensitivity: vintage, missing-debt treatment, Financials/Utilities
Winsorisation thresholds for the ML protocols come from the training fold only.
Run:  python e8_sec_validation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "code")))
import sec_data as sd  # noqa: E402
import common as cm  # noqa: E402

from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
import shap  # noqa: E402
from scipy import stats  # noqa: E402
from linearmodels.panel import PanelOLS, PooledOLS  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
N_JOBS = 4
N_BOOT = 1000
OUT = os.path.join(cm.OUT_MAIN, "e8_sec")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
LOG = []
MD = []          # markdown tables (tables_autogen.md)
KEY = {}         # headline numbers (key_numbers.json)


def log(msg):
    s = f"[{time.time() - T0:7.1f}s] {msg}"
    print(s, flush=True)
    LOG.append(s)


def md_table(df: pd.DataFrame, title: str, floatfmt=4):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.astype(str).values]
    MD.append(f"### {title}\n\n" + "\n".join([head, sep] + rows) + "\n")


def save(df, name, title=None, floatfmt=4):
    df.to_csv(os.path.join(OUT, name), index=False)
    if title:
        md_table(df, f"{title} (`{name}`)", floatfmt)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
PRED = ["size_assets", "size_revenue", "tangibility", "profitability", "cash_ratio",
        "capex_assets", "da_assets", "sales_growth", "div_payer", "ocf_assets", "ca_assets"]
CORE = ["tangibility", "size_revenue", "profitability", "sales_growth"]
RATIO_PRED = ["tangibility", "profitability", "cash_ratio", "capex_assets", "da_assets",
              "sales_growth", "ocf_assets", "ca_assets"]
LABEL = {"size_assets": "Size: ln total assets", "size_revenue": "Size: ln revenue",
         "tangibility": "Tangibility (net PPE/TA)", "profitability": "Profitability (EBIT/TA)",
         "cash_ratio": "Cash/TA", "capex_assets": "Capex/TA", "da_assets": "D&A/TA",
         "sales_growth": "Sales growth", "div_payer": "Dividend payer", "ocf_assets": "OCF/TA",
         "ca_assets": "Current assets/TA", "log_assets": "ln total assets (USD m)",
         "ebit_to_assets": "EBIT/TA", "asset_turnover": "Revenue/TA"}
# Expected sign of the association with book leverage in the established literature
EXPECTED = {
    "tangibility": ("+", "Rajan & Zingales (1995); Frank & Goyal (2009) core factor"),
    "size_assets": ("+", "Rajan & Zingales (1995, log sales); Frank & Goyal (2009) core factor (log assets)"),
    "size_revenue": ("+", "Rajan & Zingales (1995) log sales"),
    "profitability": ("-", "Rajan & Zingales (1995); Frank & Goyal (2009) core factor (pecking order)"),
    "ocf_assets": ("-", "cash-flow profitability; same logic as profitability (pecking order)"),
    "sales_growth": ("-", "growth-opportunity proxy (market-to-book unavailable); Frank & Goyal (2009) report MB - "),
    "div_payer": ("-", "Frank & Goyal (2009) extended factor: dividend payers less levered"),
    "da_assets": ("-", "non-debt tax shield (DeAngelo & Masulis 1980); weak/mixed in Frank & Goyal (2009)"),
    "cash_ratio": ("-", "not a Frank-Goyal core factor; liquid assets substitute for debt (pecking order)"),
    "capex_assets": ("?", "not a core factor; no established sign"),
    "ca_assets": ("?", "not a core factor; asset-liquidity effect ambiguous"),
}
TARGET = "debt_ratio"


def load(point_in_time=True, debt_both_missing="nan", exclude=True, winsor=False):
    d = sd.load_sec_panel(point_in_time=point_in_time, debt_both_missing=debt_both_missing,
                          exclude_financials_utilities=exclude, winsorize_ratios=winsor)
    d["log_assets"] = np.log(d["total_assets"] / 1e6)          # USD millions (synthetic units)
    d["ebit_to_assets"] = d["profitability"]
    d["asset_turnover"] = d["total_revenue"] / d["total_assets"]
    # unwinsorised targets for evaluation
    d["debt_ratio_eval"] = d["debt_ratio_raw"]
    d["liabilities_ratio_eval"] = d["liabilities_ratio_raw"]
    return d


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SectorMedian:
    name = "SectorMedian (naive)"

    def fit(self, X, y, groups, meta):
        self.m = pd.Series(y, index=meta.index).groupby(meta["sector"].values).median()
        self.g = float(np.median(y))
        return self

    def predict(self, X, meta):
        return meta["sector"].map(self.m).fillna(self.g).values


class GlobalMean:
    name = "GlobalMean (naive, firm-mean-free)"

    def fit(self, X, y, groups, meta):
        self.mu = float(np.mean(y))
        return self

    def predict(self, X, meta):
        return np.full(len(X), self.mu)


class FirmMean:
    """Reference for out-of-time only: the firm's own historical (training-period) mean."""
    name = "FirmHistMean (reference)"

    def fit(self, X, y, groups, meta):
        self.m = pd.Series(y).groupby(np.asarray(groups)).mean()
        self.g = float(np.mean(y))
        return self

    def predict(self, X, meta):
        return meta["firm_id"].map(self.m).fillna(self.g).values


class Scaled:
    def __init__(self, est, name):
        self.est, self.name = est, name

    def fit(self, X, y, groups, meta):
        self.sc = StandardScaler().fit(X)
        self.est.fit(self.sc.transform(X), y)
        return self

    def predict(self, X, meta):
        return self.est.predict(self.sc.transform(X))


class RidgeGroupCV:
    """Ridge on standardised features, alpha chosen by inner GroupKFold (firms) on the training data."""
    name = "Ridge"
    ALPHAS = np.logspace(-3, 3, 13)

    def fit(self, X, y, groups, meta):
        groups = np.asarray(groups)
        ng = len(np.unique(groups))
        k = min(5, ng)
        best, best_a = np.inf, 1.0
        if k >= 2:
            for a in self.ALPHAS:
                sse = 0.0
                for tr, te in GroupKFold(k).split(X, y, groups):
                    sc = StandardScaler().fit(X[tr])
                    m = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr])
                    sse += np.sum((y[te] - m.predict(sc.transform(X[te]))) ** 2)
                if sse < best:
                    best, best_a = sse, a
        self.alpha = best_a
        self.sc = StandardScaler().fit(X)
        self.m = Ridge(alpha=best_a).fit(self.sc.transform(X), y)
        return self

    def predict(self, X, meta):
        return self.m.predict(self.sc.transform(X))


class Plain:
    def __init__(self, est, name):
        self.est, self.name = est, name

    def fit(self, X, y, groups, meta):
        self.est.fit(X, y)
        return self

    def predict(self, X, meta):
        return self.est.predict(X)


def lgbm():
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, max_depth=3,
                             min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                             n_jobs=N_JOBS, verbose=-1)


def make_models(which="all"):
    ms = {
        "SectorMedian (naive)": lambda: SectorMedian(),
        "GlobalMean (naive)": lambda: GlobalMean(),
        "LinearReg": lambda: Scaled(LinearRegression(), "LinearReg"),
        "Ridge": lambda: RidgeGroupCV(),
        "RandomForest": lambda: Plain(RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                            max_features=0.5, random_state=SEED,
                                                            n_jobs=N_JOBS), "RandomForest"),
        "LightGBM": lambda: Plain(lgbm(), "LightGBM"),
        "XGBoost": lambda: Plain(xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                                  min_child_weight=5, subsample=0.8,
                                                  colsample_bytree=0.8, reg_lambda=1.0,
                                                  random_state=SEED, n_jobs=N_JOBS), "XGBoost"),
    }
    if which == "all":
        return ms
    return {k: ms[k] for k in which}


MODEL_ORDER = ["SectorMedian (naive)", "GlobalMean (naive)", "FirmHistMean (reference)", "LinearReg",
               "Ridge", "RandomForest", "LightGBM", "XGBoost"]


# ---------------------------------------------------------------------------
# Fold-safe preprocessing and evaluation
# ---------------------------------------------------------------------------
def prep_fold(tr, te, feats, target):
    """Winsorise ratio features (1/99%) with thresholds from the training rows only; training target
    winsorised with training thresholds; test target left unwinsorised (evaluation on raw values)."""
    rat = [f for f in feats if f in RATIO_PRED + ["ebit_to_assets", "asset_turnover"]]
    trw = sd.winsorize(tr, rat + [target], ref=tr)
    tew = sd.winsorize(te, rat, ref=tr)
    return (trw[feats].values.astype(float), trw[target].values.astype(float),
            tew[feats].values.astype(float), te[target + "_eval"].values.astype(float))


def metrics(y, p):
    e = y - p
    sst = np.sum((y - y.mean()) ** 2)
    return {"R2": 1 - np.sum(e ** 2) / sst if sst > 0 else np.nan,
            "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))), "n": len(y)}


def run_splits(df, feats, target, splits, models, protocol, extra_models=None, keep_cols=()):
    """splits: list of (fold_label, train_idx, test_idx) positional. Returns predictions long df."""
    rows = []
    for fold, tr_i, te_i in splits:
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        Xtr, ytr, Xte, yte = prep_fold(tr, te, feats, target)
        ms = dict(models)
        if extra_models:
            ms.update(extra_models)
        for mname, mk in ms.items():
            m = mk().fit(Xtr, ytr, tr["firm_id"].values, tr)
            p = m.predict(Xte, te)
            rows.append(pd.DataFrame({"protocol": protocol, "fold": fold, "model": mname,
                                      "firm_id": te["firm_id"].values, "fyear": te["fyear"].values,
                                      "sector": te["sector"].values, "y": yte, "yhat": p,
                                      "n_train": len(tr),
                                      **{c: te[c].values for c in keep_cols}}))
    return pd.concat(rows, ignore_index=True)


def firm_bootstrap(pred, n_boot=N_BOOT, seed=SEED, ref_model=None):
    """Firm-cluster bootstrap of pooled R2/RMSE/MAE for each model (same firm draws for all models),
    plus the R2 difference relative to ref_model."""
    rng = np.random.RandomState(seed)
    firms = np.array(sorted(pred["firm_id"].unique()))
    draws = rng.randint(0, len(firms), size=(n_boot, len(firms)))
    W = np.zeros((n_boot, len(firms)))
    for b in range(n_boot):
        W[b] = np.bincount(draws[b], minlength=len(firms))
    out, r2s = [], {}
    for mname, g in pred.groupby("model", sort=False):
        a = g.assign(e=g.y - g.yhat).groupby("firm_id").agg(
            n=("y", "size"), sy=("y", "sum"), syy=("y", lambda v: np.sum(v ** 2)),
            se2=("e", lambda v: np.sum(v ** 2)), sae=("e", lambda v: np.sum(np.abs(v)))).reindex(firms).fillna(0)
        n = W @ a.n.values
        sst = W @ a.syy.values - (W @ a.sy.values) ** 2 / n
        sse = W @ a.se2.values
        r2 = 1 - sse / sst
        rmse = np.sqrt(sse / n)
        mae = (W @ a.sae.values) / n
        r2s[mname] = r2
        pt = metrics(g.y.values, g.yhat.values)
        out.append({"model": mname, "n_obs": pt["n"], "n_firms": g.firm_id.nunique(),
                    "R2": pt["R2"], "R2_lo": np.nanpercentile(r2, 2.5), "R2_hi": np.nanpercentile(r2, 97.5),
                    "RMSE": pt["RMSE"], "RMSE_lo": np.percentile(rmse, 2.5), "RMSE_hi": np.percentile(rmse, 97.5),
                    "MAE": pt["MAE"], "MAE_lo": np.percentile(mae, 2.5), "MAE_hi": np.percentile(mae, 97.5)})
    res = pd.DataFrame(out)
    if ref_model and ref_model in r2s:
        res["dR2_vs_" + ref_model] = np.nan
        res["dR2_lo"] = np.nan
        res["dR2_hi"] = np.nan
        for i, r in res.iterrows():
            d = r2s[r.model] - r2s[ref_model]
            pt = r.R2 - res.loc[res.model == ref_model, "R2"].iloc[0]
            res.loc[i, ["dR2_vs_" + ref_model, "dR2_lo", "dR2_hi"]] = [pt, np.nanpercentile(d, 2.5),
                                                                        np.nanpercentile(d, 97.5)]
    res["model"] = pd.Categorical(res["model"], [m for m in MODEL_ORDER if m in set(res.model)] +
                                  [m for m in res.model if m not in MODEL_ORDER])
    return res.sort_values("model").reset_index(drop=True)


def fold_metrics(pred):
    r = []
    for (p, f, m), g in pred.groupby(["protocol", "fold", "model"], sort=False):
        r.append({"protocol": p, "fold": f, "model": m, **metrics(g.y.values, g.yhat.values)})
    return pd.DataFrame(r)


def splits_grouped(df, n=5, seed=SEED):
    return [(f"g{k}", tr, te) for k, (tr, te) in enumerate(cm.grouped_kfold(df, n, seed))]


def splits_lofo(df):
    f = df["firm_id"].values
    return [(g, np.where(f != g)[0], np.where(f == g)[0]) for g in sorted(np.unique(f))]


def splits_oot(df, last_train=2019):
    y = df["fyear"].values
    tr = np.where(y <= last_train)[0]
    seen = set(df.iloc[tr]["firm_id"])
    te = np.where((y > last_train) & df["firm_id"].isin(seen).values)[0]
    return [("oot", tr, te)]


def splits_unseen_future(df, last_train=2019, n=5, seed=SEED):
    out = []
    y = df["fyear"].values
    for k, (tr_g, te_g) in enumerate(cm.grouped_kfold(df, n, seed)):
        test_firms = set(df.iloc[te_g]["firm_id"])
        is_test_firm = df["firm_id"].isin(test_firms).values
        tr = np.where(~is_test_firm & (y <= last_train))[0]
        te = np.where(is_test_firm & (y > last_train))[0]
        if len(te):
            out.append((f"u{k}", tr, te))
    return out


def protocol_suite(df, feats, target, label, models=None, lofo=True):
    """Runs the four protocols; returns (pred_all, summary_with_CI)."""
    models = models or make_models()
    preds = []
    preds.append(run_splits(df, feats, target, splits_grouped(df), models, "GroupKFold5"))
    if lofo:
        preds.append(run_splits(df, feats, target, splits_lofo(df), models, "LOFO"))
    preds.append(run_splits(df, feats, target, splits_oot(df), models, "OOT 2009-19 -> 2020-25",
                            extra_models={"FirmHistMean (reference)": lambda: FirmMean()}))
    preds.append(run_splits(df, feats, target, splits_unseen_future(df), models,
                            "Unseen firms, future years"))
    pred = pd.concat(preds, ignore_index=True)
    summ = []
    for p, g in pred.groupby("protocol", sort=False):
        b = firm_bootstrap(g, ref_model="Ridge")
        b.insert(0, "protocol", p)
        summ.append(b)
    summ = pd.concat(summ, ignore_index=True)
    summ.insert(0, "sample", label)
    return pred, summ


# ===========================================================================
log("loading SEC panel")
raw_main = load()                                    # PIT, NaN debt dropped, excl Fin/Util, unwinsorised
win_main = load(winsor=True)                         # full-sample winsorised (descriptives, regressions)
cc_mask = raw_main[PRED + [TARGET]].notna().all(axis=1)
main = raw_main[cc_mask].reset_index(drop=True)
assert (raw_main[["firm_id", "fyear"]].values == win_main[["firm_id", "fyear"]].values).all()
win = win_main[cc_mask.values].reset_index(drop=True)
excl_firms = sorted(set(raw_main.firm_id) - set(main.firm_id))
KEY["main_sample"] = {"firms": int(main.firm_id.nunique()), "firm_years": int(len(main)),
                      "fyear_min": int(main.fyear.min()), "fyear_max": int(main.fyear.max()),
                      "firms_lost_to_complete_cases": excl_firms,
                      "panel_rows_excl_fin_util": int(len(raw_main)),
                      "rows_with_debt_ratio": int(raw_main[TARGET].notna().sum())}
log(f"main sample {KEY['main_sample']}")
comp = (main.groupby(["sector"]).agg(firms=("firm_id", "nunique"), firm_years=("firm_id", "size"),
                                      mean_debt_ratio=("debt_ratio_raw", "mean"))
        .reset_index())
comp["tickers"] = comp.sector.map(lambda s: ", ".join(sorted(main.loc[main.sector == s, "firm_id"].unique())))
save(comp, "a_sample_composition.csv", "Main sample composition (complete cases, excl. Financials/Utilities)")

# ---------------------------------------------------------------------------
# (a) descriptives, correlations, sign check
# ---------------------------------------------------------------------------
log("(a) descriptives")
desc = win[[TARGET, "liabilities_ratio"] + PRED].describe().T.reset_index().rename(columns={"index": "variable"})
save(desc, "a_descriptives_main.csv", "Descriptives, main sample (pooled 1/99% winsorised)")
corr_rows = []
for f in PRED:
    pr, pp = stats.pearsonr(win[f], win[TARGET])
    sr, spv = stats.spearmanr(win[f], win[TARGET])
    # within-firm (demeaned) correlation
    wf = win[f] - win.groupby("firm_id")[f].transform("mean")
    wy = win[TARGET] - win.groupby("firm_id")[TARGET].transform("mean")
    wr = np.corrcoef(wf, wy)[0, 1] if wf.std() > 0 else np.nan
    # between-firm correlation of firm means
    fm = win.groupby("firm_id")[[f, TARGET]].mean()
    br, bp = stats.pearsonr(fm[f], fm[TARGET])
    exp, src = EXPECTED[f]
    obs = "+" if pr > 0 else "-"
    corr_rows.append({"variable": f, "pearson_r": pr, "pearson_p_naive": pp, "spearman_rho": sr,
                      "spearman_p_naive": spv, "within_firm_r": wr, "between_firm_r": br,
                      "between_p": bp, "expected_sign": exp, "observed_sign_pearson": obs,
                      "consistent": ("n/a" if exp == "?" else ("yes" if exp == obs else "no")),
                      "expectation_source": src})
corr = pd.DataFrame(corr_rows)
save(corr, "a_correlations_signs.csv", "Correlation of debt_ratio with each determinant and sign check "
     "(naive p-values ignore within-firm dependence; between-firm r uses the firm means, n = firms)")
cmat = win[[TARGET] + PRED].corr()
cmat.reset_index().rename(columns={"index": "variable"}).to_csv(os.path.join(OUT, "a_correlation_matrix.csv"), index=False)

# ---------------------------------------------------------------------------
# (b) panel regressions
# ---------------------------------------------------------------------------
log("(b) panel regressions")


def panel_regs(dfw, feats, target, label, standardise=True):
    d = dfw.dropna(subset=feats + [target]).copy()
    X = d[feats].astype(float)
    if standardise:
        X = (X - X.mean()) / X.std(ddof=0)
    X.index = pd.MultiIndex.from_arrays([d["firm_id"].values, d["fyear"].values.astype(int)])
    y = pd.Series(d[target].values, index=X.index)
    res = []
    Xc = X.copy()
    Xc.insert(0, "const", 1.0)
    fits = {
        "Pooled OLS (firm-clustered)": PooledOLS(y, Xc).fit(cov_type="clustered", cluster_entity=True),
        "Firm FE (firm-clustered)": PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit(
            cov_type="clustered", cluster_entity=True),
        "Two-way FE (firm-clustered)": PanelOLS(y, X, entity_effects=True, time_effects=True,
                                                drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True),
        "Firm FE (Driscoll-Kraay)": PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit(
            cov_type="kernel", kernel="bartlett"),
    }
    for est, f in fits.items():
        ci = f.conf_int()
        for v in feats:
            if v not in f.params.index:
                continue
            res.append({"sample": label, "target": target, "estimator": est, "variable": v,
                        "coef": f.params[v], "se": f.std_errors[v], "t": f.tstats[v], "p": f.pvalues[v],
                        "ci_lo": ci.loc[v, "lower"], "ci_hi": ci.loc[v, "upper"], "n_obs": int(f.nobs),
                        "n_firms": int(d.firm_id.nunique()), "r2_within_or_pooled": float(f.rsquared),
                        "standardised_X": standardise})
    return pd.DataFrame(res)


reg_all = []
for tgt in [TARGET, "liabilities_ratio"]:
    for spec, feats in [("full", PRED), ("core", CORE)]:
        for stdz in [True, False]:
            r = panel_regs(win, feats, tgt, "main", stdz)
            r.insert(2, "spec", spec)
            reg_all.append(r)
# Profitability measure variants (FE, debt_ratio, core spec with alternative profit proxy)
win_alt = win.copy()
reg_prof = []
for prof_var, note in [("profitability", "EBIT/TA (pre-interest; main)"),
                       ("ocf_assets", "OCF/TA (cash-flow profitability)"),
                       ("roa_net", "Net income/TA (post-interest: mechanically debt-linked, reference only)")]:
    feats = ["tangibility", "size_revenue", prof_var, "sales_growth"]
    d_ = win_main.dropna(subset=feats + [TARGET])
    r = panel_regs(d_, feats, TARGET, "main (rows with these 4 vars)", True)
    r = r[r.variable == prof_var].copy()
    r["profit_measure"] = note
    reg_prof.append(r)
reg = pd.concat(reg_all, ignore_index=True)
save(reg, "b_panel_regressions_all.csv")
tab = reg[(reg.target == TARGET) & reg.standardised_X].copy()
tab["coef_se"] = tab.apply(lambda r: f"{r.coef:.4f} ({r.se:.4f}){'***' if r.p < .01 else '**' if r.p < .05 else '*' if r.p < .1 else ''}", axis=1)
for spec in ["full", "core"]:
    pv = tab[tab.spec == spec].pivot(index="variable", columns="estimator", values="coef_se")
    feats = PRED if spec == "full" else CORE
    pv = pv.reindex([f for f in feats if f in pv.index]).reset_index()
    nrow = tab[tab.spec == spec].groupby("estimator")[["n_obs", "n_firms", "r2_within_or_pooled"]].first()
    pv = pd.concat([pv, pd.DataFrame([{"variable": "N obs / firms", **{e: f"{int(nrow.loc[e, 'n_obs'])} / {int(nrow.loc[e, 'n_firms'])}" for e in nrow.index}},
                                      {"variable": "R2 (within for FE)", **{e: f"{nrow.loc[e, 'r2_within_or_pooled']:.3f}" for e in nrow.index}}])])
    save(pv, f"b_table_debt_ratio_{spec}.csv",
         f"Panel regressions, debt_ratio, {spec} spec; coefficient per 1 s.d. of X (SE); *** p<.01 ** p<.05 * p<.1")
tabl = reg[(reg.target == "liabilities_ratio") & reg.standardised_X & (reg.spec == "full")].copy()
tabl["coef_se"] = tabl.apply(lambda r: f"{r.coef:.4f} ({r.se:.4f}){'***' if r.p < .01 else '**' if r.p < .05 else '*' if r.p < .1 else ''}", axis=1)
pvl = tabl.pivot(index="variable", columns="estimator", values="coef_se").reindex(PRED).reset_index()
save(pvl, "b_table_liabilities_ratio_full.csv", "Panel regressions, liabilities_ratio (robustness), full spec, per 1 s.d. of X")
rp = pd.concat(reg_prof, ignore_index=True)[["profit_measure", "estimator", "coef", "se", "p", "n_obs", "n_firms"]]
save(rp, "b_profitability_measures.csv", "Profitability coefficient (per 1 s.d.) in the core spec "
     "(tangibility, ln revenue, profit measure, sales growth), debt_ratio, by profit measure")


# small-cluster inference. Wild cluster restricted (WCR) bootstrap-t (Cameron, Gelbach & Miller
# 2008; Webb 6-point weights, null imposed, CR1 cluster SEs) for the key coefficients, 21 firm clusters.
def wild_cluster_boot(dfw, feats, target, test_vars, fe="none", B=9999, seed=SEED, standardise=True):
    d = dfw.dropna(subset=feats + [target]).reset_index(drop=True)
    X = d[feats].astype(float)
    if standardise:
        X = (X - X.mean()) / X.std(ddof=0)
    if fe == "twoway":
        yd = pd.get_dummies(d["fyear"].astype(int), prefix="yr", drop_first=True).astype(float)
        X = pd.concat([X, yd], axis=1)
    y = d[target].astype(float).values
    Xm = X.values
    g = d["firm_id"].values
    if fe == "none":
        Xm = np.column_stack([np.ones(len(d)), Xm])
        names = ["const"] + list(X.columns)
    else:  # entity demeaning (FWL; with year dummies included this gives two-way FE)
        dm = lambda a: a - pd.DataFrame(a).groupby(g).transform("mean").values
        Xm, y = dm(Xm), dm(y.reshape(-1, 1)).ravel()
        names = list(X.columns)
    ug, gi = np.unique(g, return_inverse=True)
    G, N = len(ug), len(y)
    K = Xm.shape[1] + (G if fe != "none" else 0)
    cfac = G / (G - 1) * (N - 1) / max(N - K, 1)

    XtXi_full = np.linalg.pinv(Xm.T @ Xm)

    def ols_t(Xa, ya, j):
        XtXi = XtXi_full if Xa is Xm else np.linalg.pinv(Xa.T @ Xa)
        b = XtXi @ Xa.T @ ya
        u = ya - Xa @ b
        S = np.zeros((G, Xa.shape[1]))
        np.add.at(S, gi, Xa * u[:, None])
        V = cfac * XtXi @ (S.T @ S) @ XtXi
        return b[j], np.sqrt(V[j, j])

    rng = np.random.RandomState(seed)
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    W = webb[rng.randint(0, 6, size=(B, G))]
    out = []
    for v in test_vars:
        j = names.index(v)
        b_hat, se_hat = ols_t(Xm, y, j)
        t_hat = b_hat / se_hat
        Xr = np.delete(Xm, j, axis=1)
        br = np.linalg.lstsq(Xr, y, rcond=None)[0]
        fit_r, u_r = Xr @ br, y - Xr @ br
        tb = np.empty(B)
        for bb in range(B):
            ys = fit_r + u_r * W[bb, gi]
            bs_, ses_ = ols_t(Xm, ys, j)
            tb[bb] = bs_ / ses_
        p_sym = float(np.mean(np.abs(tb) >= abs(t_hat)))
        out.append({"variable": v, "coef": b_hat, "se_CR1": se_hat, "t_CR1": t_hat,
                    "p_t_dist_G-1": float(2 * stats.t.sf(abs(t_hat), G - 1)),
                    "p_wild_cluster_boot": p_sym, "B": B, "n_obs": N, "n_clusters": G})
    return out


log("(b) wild cluster bootstrap (small-cluster inference)")
wcb_rows = []
for spec, feats in [("full", PRED), ("core", CORE)]:
    for fe, est in [("none", "Pooled OLS"), ("entity", "Firm FE"), ("twoway", "Two-way FE")]:
        tv = [v for v in ["profitability", "tangibility", "size_assets", "size_revenue", "sales_growth", "ocf_assets"]
              if v in feats]
        for r in wild_cluster_boot(win, feats, TARGET, tv, fe=fe):
            wcb_rows.append({"spec": spec, "estimator": est, **r})
wcb = pd.DataFrame(wcb_rows)
save(wcb, "b_wild_cluster_bootstrap.csv", "Small-cluster inference: wild cluster restricted bootstrap-t p-values "
     "(Webb weights, 9,999 reps, null imposed, CR1 SEs, 21 firm clusters), debt_ratio, per 1 s.d. of X")

# ---------------------------------------------------------------------------
# (c) predictive validation, main sample
# ---------------------------------------------------------------------------
log("(c) predictive protocols, main sample")
pred_main, summ_main = protocol_suite(main, PRED, TARGET, "main (excl Fin/Util)")
pred_main.to_csv(os.path.join(OUT, "c_predictions_main.csv.gz"), index=False)
fm_main = fold_metrics(pred_main)
fm_main.to_csv(os.path.join(OUT, "c_fold_metrics_main.csv"), index=False)
save(summ_main, "c_predictive_main.csv", "Predictive validation, main sample, debt_ratio; pooled out-of-sample "
     "metrics with 95% firm-bootstrap CIs (1,000 reps); dR2 relative to Ridge")
oot_tr = splits_oot(main)[0]
KEY["oot_split"] = {"n_train": int(len(oot_tr[1])), "n_test": int(len(oot_tr[2])),
                    "firms_test": int(main.iloc[oot_tr[2]].firm_id.nunique())}
uf = splits_unseen_future(main)
KEY["unseen_future_split"] = {"n_train_per_fold": [int(len(s[1])) for s in uf],
                              "n_test_per_fold": [int(len(s[2])) for s in uf]}
gf = fm_main[fm_main.protocol == "GroupKFold5"].groupby("model")["R2"].agg(["mean", "std"]).reset_index()
save(gf, "c_groupkfold_fold_mean_r2_main.csv", "GroupKFold5: mean and s.d. of fold-level R2, main sample")

# seed stability of the firm-bootstrap CIs (lower bounds depend on the RNG stream with 21 firms)
log("(c) bootstrap seed stability")
bs_rows = []
BS_SEEDS = [SEED, 1, 7, 123, 2024, 31337]
for p_, g in pred_main.groupby("protocol", sort=False):
    for sd_ in BS_SEEDS:
        b = firm_bootstrap(g, seed=sd_)
        for _, r in b.iterrows():
            bs_rows.append({"protocol": p_, "model": str(r.model), "boot_seed": sd_, "n_boot": N_BOOT,
                            "R2": r.R2, "R2_lo": r.R2_lo, "R2_hi": r.R2_hi})
    b = firm_bootstrap(g, n_boot=10000, seed=SEED)
    for _, r in b.iterrows():
        bs_rows.append({"protocol": p_, "model": str(r.model), "boot_seed": SEED, "n_boot": 10000,
                        "R2": r.R2, "R2_lo": r.R2_lo, "R2_hi": r.R2_hi})
bs = pd.DataFrame(bs_rows)
bs.to_csv(os.path.join(OUT, "c_bootstrap_seed_stability_long.csv"), index=False)
b1 = bs[bs.n_boot == N_BOOT]
bss = (b1.groupby(["protocol", "model"], sort=False)
       .agg(R2=("R2", "first"), R2_lo_min=("R2_lo", "min"), R2_lo_max=("R2_lo", "max"),
            R2_hi_min=("R2_hi", "min"), R2_hi_max=("R2_hi", "max")).reset_index())
b10 = bs[bs.n_boot == 10000].set_index(["protocol", "model"])
bss["R2_lo_10k"] = [b10.loc[(p_, m_), "R2_lo"] for p_, m_ in zip(bss.protocol, bss.model)]
bss["R2_hi_10k"] = [b10.loc[(p_, m_), "R2_hi"] for p_, m_ in zip(bss.protocol, bss.model)]
bss["n_seeds"] = len(BS_SEEDS)
save(bss, "c_bootstrap_seed_stability.csv", f"Seed stability of 95% firm-bootstrap R2 CIs: range of bounds over "
     f"{len(BS_SEEDS)} RNG seeds x {N_BOOT} reps, and a 10,000-rep run (seed {SEED})")
KEY["bootstrap_seed_stability"] = {"seeds": BS_SEEDS,
                                   "max_range_R2_lo": float((bss.R2_lo_max - bss.R2_lo_min).max()),
                                   "max_range_R2_hi": float((bss.R2_hi_max - bss.R2_hi_min).max())}

log("(c) liabilities_ratio robustness")
pred_liab, summ_liab = protocol_suite(main.dropna(subset=["liabilities_ratio"]).reset_index(drop=True),
                                      PRED, "liabilities_ratio", "main, target liabilities_ratio")
save(summ_liab, "c_predictive_main_liabilities_ratio.csv",
     "Predictive validation, main sample, target liabilities_ratio (robustness)")

log("(c) Nadeau-Bengio 5x5 LightGBM vs Ridge")
nb_rows = []
models_nb = make_models(["Ridge", "LightGBM", "SectorMedian (naive)"])
for r, k, tr, te in cm.repeated_grouped_kfold(main, 5, 5, SEED):
    p = run_splits(main, PRED, TARGET, [(f"r{r}k{k}", tr, te)], models_nb, "5x5")
    for m, g in p.groupby("model"):
        nb_rows.append({"repeat": r, "fold": k, "model": m, "n_train": len(tr), "n_test": len(te),
                        **metrics(g.y.values, g.yhat.values)})
nb = pd.DataFrame(nb_rows)
nb.to_csv(os.path.join(OUT, "c_repeated_5x5_fold_metrics.csv"), index=False)
nbt = []
for a, b in [("LightGBM", "Ridge"), ("LightGBM", "SectorMedian (naive)"), ("Ridge", "SectorMedian (naive)")]:
    for met in ["R2", "RMSE"]:
        A = nb[nb.model == a].sort_values(["repeat", "fold"])
        B = nb[nb.model == b].sort_values(["repeat", "fold"])
        d = A[met].values - B[met].values
        ntr, nte = A.n_train.mean(), A.n_test.mean()
        # primary: exact (unrounded) mean n_test / mean n_train (not rounded to int)
        t, p = cm.corrected_resampled_ttest(d, float(ntr), float(nte))
        # variant: mean of the per-fold n_test/n_train ratios (folds are unequal in size)
        rat = float((A.n_test / A.n_train).mean())
        t2, p2 = cm.corrected_resampled_ttest(d, 1.0, rat)
        t0, p0 = cm.corrected_resampled_ttest(d, int(round(ntr)), int(round(nte)))
        nbt.append({"model_A": a, "model_B": b, "metric": met, "mean_A": A[met].mean(), "mean_B": B[met].mean(),
                    "mean_diff_A_minus_B": d.mean(), "sd_diff": d.std(ddof=1), "J": len(d),
                    "mean_n_train": ntr, "mean_n_test": nte, "t_corrected": t, "p_corrected": p,
                    "mean_fold_ratio_test_train": rat, "t_corrected_mean_ratio": t2, "p_corrected_mean_ratio": p2,
                    "t_corrected_int_rounded_n": t0, "p_corrected_int_rounded_n": p0,
                    "naive_paired_t_p": stats.ttest_rel(A[met].values, B[met].values).pvalue})
nbt = pd.DataFrame(nbt)
save(nbt, "c_nadeau_bengio_5x5.csv", "Nadeau-Bengio corrected resampled t-test on repeated grouped 5x5 CV (main sample)")

# ---------------------------------------------------------------------------
# (d) transfer from the reference synthetic panel
# ---------------------------------------------------------------------------
log("(d) transfer synthetic -> SEC")
syn = cm.generate_panel(seed=SEED)
SECTOR_D = [f"sector_{s}" for s in sorted(cm.SECTORS)]
for s in sorted(cm.SECTORS):
    raw_main[f"sector_{s}"] = (raw_main["sector"] == s).astype(int)
T_FIRM = ["log_assets", "ebit_to_assets", "cash_ratio", "asset_turnover", "sales_growth"]
T_SETS = {"T1: firm characteristics (5)": T_FIRM, "T2: T1 + 8 sector dummies": T_FIRM + SECTOR_D}
mapping = pd.DataFrame([
    {"feature": "log_assets", "synthetic": "ln(total_assets); total_assets = exp(N(7.5,1.9)) read as USD m",
     "SEC": "ln(total_assets / 1e6)", "identical_definition": "yes, up to the unit assumption"},
    {"feature": "ebit_to_assets", "synthetic": "ebit/total_assets, ebit = max(1.7*NI, 0)",
     "SEC": "operating_income/TA (fallback pretax + interest)", "identical_definition": "same ratio; synthetic EBIT is a scaled NI floored at 0"},
    {"feature": "cash_ratio", "synthetic": "cash_holdings/total_assets", "SEC": "cash_and_equivalents/TA",
     "identical_definition": "yes"},
    {"feature": "asset_turnover", "synthetic": "revenue/total_assets", "SEC": "total_revenue/TA",
     "identical_definition": "yes"},
    {"feature": "sales_growth", "synthetic": "revenue_t/revenue_{t-1} - 1 (consecutive years)",
     "SEC": "same (consecutive fyears only)", "identical_definition": "yes"},
    {"feature": "sector_* (8)", "synthetic": "sector dummy", "SEC": "paper-sector dummy (SECTOR_MAP)",
     "identical_definition": "same coding; Financials/Utilities/Materials are all-zero in the SEC main sample"},
    {"feature": "roa (not used)", "synthetic": "NI/TA, NI has no interest deduction (clean)",
     "SEC": "NI/TA is after interest (debt-linked)", "identical_definition": "no -> excluded"},
    {"feature": "size_code, macro, tangibility, capex, D&A, OCF, dividends, current assets (not used)",
     "synthetic": "size_code is a sample tertile; macro is year-level; other items do not exist",
     "SEC": "-", "identical_definition": "no -> excluded"},
])
save(mapping, "d_transfer_feature_mapping.csv", "Transfer feature mapping (synthetic -> SEC)")
transfer_rows, within_rows, overlap_rows, transfer_preds = [], [], [], []
for tname, feats in T_SETS.items():
    s_tr = syn.dropna(subset=feats + [TARGET]).reset_index(drop=True)
    sec_t = raw_main.dropna(subset=feats + [TARGET]).reset_index(drop=True)
    for f in T_FIRM:
        if tname.startswith("T1"):
            overlap_rows.append({"feature": f, "synthetic_mean": s_tr[f].mean(), "synthetic_sd": s_tr[f].std(),
                                 "synthetic_p01": s_tr[f].quantile(.01), "synthetic_p99": s_tr[f].quantile(.99),
                                 "SEC_mean": sec_t[f].mean(), "SEC_sd": sec_t[f].std(),
                                 "SEC_share_outside_synth_p01_p99": float(((sec_t[f] < s_tr[f].quantile(.01)) |
                                                                           (sec_t[f] > s_tr[f].quantile(.99))).mean())})
    s_tr = s_tr.assign(debt_ratio_eval=s_tr[TARGET], fyear=s_tr["year"])
    s_tr["firm_id"] = "syn" + s_tr["firm_id"].astype(str)
    both = pd.concat([s_tr[["firm_id", "fyear", "sector", TARGET, "debt_ratio_eval"] + feats],
                      sec_t[["firm_id", "fyear", "sector", TARGET, "debt_ratio_eval"] + feats]], ignore_index=True)
    ntr = len(s_tr)
    for test_name, te_idx in [("SEC all years", np.arange(ntr, len(both))),
                              ("SEC FY2020-2025", ntr + np.where(sec_t.fyear.values >= 2020)[0])]:
        p = run_splits(both, feats, TARGET, [("transfer", np.arange(ntr), te_idx)],
                       make_models(["SectorMedian (naive)", "GlobalMean (naive)", "LinearReg", "Ridge",
                                    "RandomForest", "LightGBM", "XGBoost"]), f"Transfer -> {test_name}")
        p["feature_set"] = tname
        transfer_preds.append(p)
        b = firm_bootstrap(p, ref_model="Ridge")
        b.insert(0, "protocol", f"Transfer synthetic -> {test_name}")
        b.insert(0, "feature_set", tname)
        transfer_rows.append(b)
    # within-SEC comparison with identical features
    p = run_splits(sec_t, feats, TARGET, splits_grouped(sec_t), make_models(), "Within-SEC GroupKFold5")
    b = firm_bootstrap(p, ref_model="Ridge")
    b.insert(0, "protocol", "Within-SEC GroupKFold5 (same features)")
    b.insert(0, "feature_set", tname)
    within_rows.append(b)
    p = run_splits(sec_t, feats, TARGET, splits_oot(sec_t), make_models(), "Within-SEC OOT")
    b = firm_bootstrap(p, ref_model="Ridge")
    b.insert(0, "protocol", "Within-SEC OOT 2009-19 -> 2020-25 (same features)")
    b.insert(0, "feature_set", tname)
    within_rows.append(b)
tr_res = pd.concat(transfer_rows + within_rows, ignore_index=True)
save(tr_res, "d_transfer_vs_within.csv", "Transfer (trained on the reference synthetic panel) vs within-SEC models, "
     "same features; naive baselines in the transfer rows are fitted on the synthetic panel")
pd.concat(transfer_preds).to_csv(os.path.join(OUT, "d_transfer_predictions.csv.gz"), index=False)
save(pd.DataFrame(overlap_rows), "d_covariate_overlap.csv", "Covariate overlap, transfer features (T1 rows)")
# also: mean target in each sample
KEY["target_means"] = {"synthetic_debt_ratio_mean": float(syn[TARGET].mean()),
                       "SEC_main_debt_ratio_raw_mean": float(main["debt_ratio_raw"].mean())}

# ---------------------------------------------------------------------------
# (e) TreeSHAP on out-of-fold LightGBM (SEC)
# ---------------------------------------------------------------------------
log("(e) TreeSHAP OOF LightGBM")
shap_rows, oof_pred = [], np.full(len(main), np.nan)
SV = np.full((len(main), len(PRED)), np.nan)
for fold, tr_i, te_i in splits_grouped(main):
    tr, te = main.iloc[tr_i], main.iloc[te_i]
    Xtr, ytr, Xte, yte = prep_fold(tr, te, PRED, TARGET)
    m = lgbm().fit(Xtr, ytr)
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(Xte)
    add_err = np.max(np.abs(ex.expected_value + sv.sum(1) - m.predict(Xte)))
    SV[te_i] = sv
    oof_pred[te_i] = m.predict(Xte)
    shap_rows.append({"fold": fold, "additivity_max_abs_err": float(add_err)})
imp = pd.DataFrame({"feature": PRED, "label": [LABEL[f] for f in PRED],
                    "mean_abs_shap": np.abs(SV).mean(0),
                    "mean_shap": SV.mean(0),
                    "corr_feature_shap": [np.corrcoef(main[f].values, SV[:, j])[0, 1] if main[f].std() > 0 else np.nan
                                          for j, f in enumerate(PRED)]})
for k, (fold, tr_i, te_i) in enumerate(splits_grouped(main)):
    imp[f"mean_abs_shap_{fold}"] = np.abs(SV[te_i]).mean(0)
imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp.insert(0, "rank", np.arange(1, len(imp) + 1))
fold_cols = [c for c in imp.columns if c.startswith("mean_abs_shap_g")]
rk = imp[fold_cols].rank(ascending=False)
sp = [stats.spearmanr(rk.iloc[:, i], rk.iloc[:, j])[0] for i in range(5) for j in range(i + 1, 5)]
KEY["shap_fold_rank_spearman_mean"] = float(np.mean(sp))
KEY["shap_additivity_max_err"] = float(max(r["additivity_max_abs_err"] for r in shap_rows))
save(imp.drop(columns=fold_cols), "e_shap_ranking_sec.csv",
     "TreeSHAP, out-of-fold LightGBM (GroupKFold5), main SEC sample; corr_feature_shap = direction of effect")
pd.DataFrame(SV, columns=PRED).assign(firm_id=main.firm_id, fyear=main.fyear).to_csv(
    os.path.join(OUT, "e_shap_values_oof.csv.gz"), index=False)
# synthetic strict ranking (E3, S4)
s4p = os.path.join(cm.OUT_MAIN, "e3_shap", "shap_ranking_S4.csv")
if os.path.exists(s4p):
    s4 = pd.read_csv(s4p)
    s4src = "output/e3_shap/shap_ranking_S4.csv (E3, FEATURES_STRICT, OOF grouped LightGBM)"
else:  # fallback: quick recomputation
    s4src = "recomputed in E8"
    X = syn[cm.FEATURES_STRICT].values
    sv_s = np.zeros_like(X, dtype=float)
    for tr, te in cm.grouped_kfold(syn, 5, SEED):
        m = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=SEED,
                              verbose=-1, n_jobs=N_JOBS).fit(X[tr], syn[TARGET].values[tr])
        sv_s[te] = shap.TreeExplainer(m).shap_values(X[te])
    s4 = pd.DataFrame({"feature": cm.FEATURES_STRICT, "mean_abs_shap_oof": np.abs(sv_s).mean(0)})
    s4 = s4.sort_values("mean_abs_shap_oof", ascending=False)
    s4["rank_oof"] = np.arange(1, len(s4) + 1)
concept = pd.DataFrame([
    {"concept": "Profitability", "synthetic_feature": "roa", "SEC_feature": "profitability (EBIT/TA); ocf_assets"},
    {"concept": "Size", "synthetic_feature": "size_code", "SEC_feature": "size_assets; size_revenue"},
    {"concept": "Cash holdings", "synthetic_feature": "cash_ratio", "SEC_feature": "cash_ratio"},
    {"concept": "Sector", "synthetic_feature": "sector_code / sector dummies (best rank)", "SEC_feature": "not in the SEC predictor set"},
    {"concept": "Macro (year-level)", "synthetic_feature": "gdp_growth (best rank of 4)", "SEC_feature": "not in the SEC predictor set"},
    {"concept": "Asset structure", "synthetic_feature": "not generated", "SEC_feature": "tangibility; ca_assets; capex; D&A"},
])
srank = dict(zip(s4.feature, s4.rank_oof))
erank = dict(zip(imp.feature, imp["rank"]))
def best(d, keys):
    v = [d[k] for k in keys if k in d]
    return min(v) if v else np.nan
concept["synthetic_rank_of_16"] = [best(srank, ["roa"]), best(srank, ["size_code"]), best(srank, ["cash_ratio"]),
                                   best(srank, ["sector_code"] + SECTOR_D), best(srank, ["gdp_growth", "inflation", "interest_rate", "credit_spread"]), np.nan]
concept["SEC_rank_of_11"] = [best(erank, ["profitability", "ocf_assets"]), best(erank, ["size_assets", "size_revenue"]),
                             best(erank, ["cash_ratio"]), np.nan, np.nan,
                             best(erank, ["tangibility", "ca_assets", "capex_assets", "da_assets"])]
concept["synthetic_source"] = s4src
save(concept, "e_shap_concept_comparison.csv", "SHAP ranking by concept: SEC (11 features) vs synthetic strict (16 features)")

# direction of effect, computed (not asserted) for BOTH datasets.
# Synthetic: E3 out-of-fold S4 SHAP values merged with common.generate_panel(42) on (firm_id, year).
s4v = os.path.join(cm.OUT_MAIN, "e3_shap", "shap_values_oof_S4.csv.gz")
if os.path.exists(s4v):
    sv4 = pd.read_csv(s4v)
    sfeat = [c for c in cm.FEATURES_STRICT if c in sv4.columns]
    mg = sv4[["firm_id", "year"] + sfeat].merge(syn[["firm_id", "year"] + sfeat], on=["firm_id", "year"],
                                                suffixes=("_shap", ""), validate="one_to_one")
    syn_dir_src = "output/e3_shap/shap_values_oof_S4.csv.gz merged with common.generate_panel(42)"
    syn_X = {f: mg[f].values for f in sfeat}
    syn_S = {f: mg[f + "_shap"].values for f in sfeat}
else:
    syn_dir_src = "recomputed in E8 (fallback)"
    syn_X = {f: syn[f].values for f in cm.FEATURES_STRICT}
    syn_S = {f: sv_s[:, j] for j, f in enumerate(cm.FEATURES_STRICT)}
KEY["synthetic_shap_direction_source"] = syn_dir_src


def direction_row(dataset, feat, x, s):
    ok = np.isfinite(x) & np.isfinite(s)
    x, s = x[ok], s[ok]
    q = pd.qcut(pd.Series(x).rank(method="first"), 5, labels=False)
    qm = pd.Series(s).groupby(q.values).mean().values
    dq = np.diff(qm)
    mono = "increasing" if (dq > 0).all() else "decreasing" if (dq < 0).all() else "non-monotonic"
    pr = np.corrcoef(x, s)[0, 1] if np.std(x) > 0 else np.nan
    sr = stats.spearmanr(x, s)[0] if np.std(x) > 0 else np.nan
    return {"dataset": dataset, "feature": feat, "n": int(ok.sum()), "pearson_x_shap": pr, "spearman_x_shap": sr,
            **{f"mean_shap_q{i + 1}": qm[i] for i in range(5)},
            "q5_minus_q1": qm[4] - qm[0], "quintile_pattern": mono,
            "mean_abs_shap": float(np.mean(np.abs(s)))}


dir_rows = []
for f in ["profitability", "ocf_assets", "size_assets", "size_revenue", "cash_ratio"]:
    dir_rows.append(direction_row("SEC (E8, OOF LightGBM)", f, main[f].values.astype(float),
                                  SV[:, PRED.index(f)]))
for f in ["roa", "size_code", "cash_ratio"]:
    if f in syn_X:
        dir_rows.append(direction_row("Reference synthetic panel (E3 S4, OOF LightGBM)", f,
                                      syn_X[f].astype(float), syn_S[f].astype(float)))
shap_dir = pd.DataFrame(dir_rows)
save(shap_dir, "e_shap_direction_comparison.csv", "Direction of the SHAP effect (feature vs own SHAP value; mean SHAP "
     "by feature quintile), SEC vs reference synthetic panel strict model")
KEY["shap_direction"] = shap_dir[["dataset", "feature", "pearson_x_shap", "spearman_x_shap",
                                  "quintile_pattern"]].to_dict(orient="records")

# ---------------------------------------------------------------------------
# (f) pooled real evidence: SEC + Yahoo 46-firm panel
# ---------------------------------------------------------------------------
log("(f) pooled SEC + Yahoo")
POOL_F = ["size_assets", "size_revenue", "profitability", "cash_ratio"]
POOL_F5 = POOL_F + ["sales_growth"]
sec_p = raw_main[["firm_id", "fyear", "sector", "debt_ratio", "debt_ratio_raw"] + POOL_F5].copy()
sec_p["source"] = "SEC"
sec_p = sec_p.dropna(subset=["debt_ratio"])
yh = cm.load_real_panel()
yh = yh[~yh.sector.isin(["Financials", "Utilities"])].copy()
yp = pd.DataFrame({"firm_id": yh.ticker.values, "fyear": yh.year.values, "sector": yh.sector.values,
                   "debt_ratio": yh.debt_ratio.values, "debt_ratio_raw": yh.debt_ratio.values,
                   "size_assets": np.log(yh.total_assets.values),
                   "size_revenue": np.log(yh.revenue.where(yh.revenue > 0).values),
                   "profitability": yh.ebit_to_assets.values, "cash_ratio": yh.cash_ratio.values,
                   "sales_growth": yh.sales_growth.values, "source": "Yahoo"})
key_sec = set(zip(sec_p.firm_id, sec_p.fyear))
dup = np.array([(a, b) in key_sec for a, b in zip(yp.firm_id, yp.fyear)])
# overlapping firm-years: SEC vs Yahoo debt_ratio comparison
ov = yp[dup].merge(sec_p, on=["firm_id", "fyear"], suffixes=("_yahoo", "_sec"))
pool_info = {"sec_rows_with_debt_ratio": int(len(sec_p)), "yahoo_rows_excl_fin_util": int(len(yp)),
             "yahoo_rows_dropped_as_duplicates": int(dup.sum()),
             "overlap_corr_debt_ratio": float(np.corrcoef(ov.debt_ratio_raw_yahoo, ov.debt_ratio_raw_sec)[0, 1]),
             "overlap_median_yahoo_minus_sec_debt_ratio": float(np.median(ov.debt_ratio_raw_yahoo - ov.debt_ratio_raw_sec))}
pooled = pd.concat([sec_p, yp[~dup]], ignore_index=True)
pooled["debt_ratio_eval"] = pooled["debt_ratio_raw"]
pool_sum, lfl_rows = [], []
for fname, feats in [("4 vars: size_assets, size_revenue, profitability, cash_ratio", POOL_F),
                     ("5 vars: + sales_growth", POOL_F5)]:
    pdf = pooled.dropna(subset=feats + [TARGET]).reset_index(drop=True)
    info = {"feature_set": fname, "n_obs": len(pdf), "n_firms": pdf.firm_id.nunique(),
            "n_obs_SEC": int((pdf.source == "SEC").sum()), "n_obs_Yahoo": int((pdf.source == "Yahoo").sum()),
            "n_firms_SEC_only": len(set(pdf[pdf.source == "SEC"].firm_id) - set(pdf[pdf.source == "Yahoo"].firm_id)),
            "n_firms_Yahoo_only": len(set(pdf[pdf.source == "Yahoo"].firm_id) - set(pdf[pdf.source == "SEC"].firm_id))}
    pool_info[fname] = info
    pooled_preds = {}
    for prot, spl in [("GroupKFold5", splits_grouped(pdf)), ("LOFO", splits_lofo(pdf))]:
        p = run_splits(pdf, feats, TARGET, spl, make_models(), prot, keep_cols=("source",))
        pooled_preds[prot] = p
        b = firm_bootstrap(p, ref_model="Ridge")
        b.insert(0, "protocol", prot)
        b.insert(0, "feature_set", fname)
        b.insert(0, "rows", "all pooled")
        pool_sum.append(b)
        for src in ["SEC", "Yahoo"]:
            q = p[p.source == src]
            b = firm_bootstrap(q, ref_model="Ridge")
            b.insert(0, "protocol", prot)
            b.insert(0, "feature_set", fname)
            b.insert(0, "rows", f"{src}-origin rows only")
            pool_sum.append(b)
    # SEC-only, same features, same protocol (for comparison)
    sdf = pdf[pdf.source == "SEC"].reset_index(drop=True)
    for prot, spl in [("GroupKFold5", splits_grouped(sdf)), ("LOFO", splits_lofo(sdf))]:
        p = run_splits(sdf, feats, TARGET, spl, make_models(), prot)
        b = firm_bootstrap(p, ref_model="Ridge")
        b.insert(0, "protocol", prot)
        b.insert(0, "feature_set", fname)
        b.insert(0, "rows", "SEC-only training and testing (comparison)")
        pool_sum.append(b)
        # like-for-like comparison on the SAME SEC-origin test rows:
        # pooled (SEC+Yahoo) training vs SEC-only training; paired firm bootstrap of the R2 difference.
        q = pooled_preds[prot]
        q = q[q.source == "SEC"]
        for mname in p.model.unique():
            a_ = q[q.model == mname].assign(model="pooled-trained")
            b_ = p[p.model == mname].assign(model="SEC-only-trained")
            assert len(a_) == len(b_) and set(zip(a_.firm_id, a_.fyear)) == set(zip(b_.firm_id, b_.fyear))
            fb = firm_bootstrap(pd.concat([a_, b_], ignore_index=True), ref_model="SEC-only-trained")
            ra = fb[fb.model == "pooled-trained"].iloc[0]
            rb = fb[fb.model == "SEC-only-trained"].iloc[0]
            lfl_rows.append({"feature_set": fname, "protocol": prot, "model": mname, "n_test_SEC_rows": len(a_),
                             "n_firms": a_.firm_id.nunique(),
                             "R2_pooled_trained": ra.R2, "R2_SEC_only_trained": rb.R2,
                             "dR2_pooled_minus_SEC_only": ra["dR2_vs_SEC-only-trained"],
                             "dR2_lo": ra.dR2_lo, "dR2_hi": ra.dR2_hi,
                             "RMSE_pooled_trained": ra.RMSE, "RMSE_SEC_only_trained": rb.RMSE})
pool_sum = pd.concat(pool_sum, ignore_index=True)
lfl = pd.DataFrame(lfl_rows)
save(lfl, "f_pooled_vs_sec_only_like_for_like.csv", "Like-for-like: R2 on the same SEC-origin test rows when trained "
     "on pooled SEC+Yahoo data vs on SEC data only (fold assignment differs between the two runs); dR2 with "
     "95% paired firm-bootstrap CI")
# explicit check of the Yahoo year convention (common.load_real_panel 'year') against SEC fyear:
# correlations on the overlapping firms with the Yahoo year shifted by -1/0/+1.
yal = []
for shift in [-1, 0, 1]:
    ys = yp.assign(fyear=yp.fyear + shift)
    mm = ys.merge(sec_p, on=["firm_id", "fyear"], suffixes=("_yahoo", "_sec"))
    mm = mm.dropna(subset=["debt_ratio_raw_yahoo", "debt_ratio_raw_sec", "size_assets_yahoo", "size_assets_sec"])
    yal.append({"yahoo_year_shift": shift, "n_matched": len(mm),
                "corr_debt_ratio": float(np.corrcoef(mm.debt_ratio_raw_yahoo, mm.debt_ratio_raw_sec)[0, 1]),
                "corr_ln_assets": float(np.corrcoef(mm.size_assets_yahoo, mm.size_assets_sec)[0, 1]),
                "median_abs_diff_ln_assets": float(np.median(np.abs(mm.size_assets_yahoo - mm.size_assets_sec))),
                "share_ln_assets_within_0.01": float((np.abs(mm.size_assets_yahoo - mm.size_assets_sec) < 0.01).mean())})
yal = pd.DataFrame(yal)
save(yal, "f_yahoo_year_alignment.csv", "Yahoo year convention check: Yahoo 'year' shifted by -1/0/+1 vs SEC fyear "
     "on overlapping firms (shift 0 is the merge used)")
pool_info["year_alignment_best_shift_by_ln_assets"] = int(yal.loc[yal.median_abs_diff_ln_assets.idxmin(), "yahoo_year_shift"])
save(pool_sum, "f_pooled_real.csv", "Pooled real data (SEC + Yahoo, excl. Fin/Util, de-duplicated preferring SEC)")
with open(os.path.join(OUT, "f_pooled_info.json"), "w") as fh:
    json.dump(pool_info, fh, indent=2, default=str)
KEY["pooled"] = pool_info

# ---------------------------------------------------------------------------
# (g) sensitivity to data choices
# ---------------------------------------------------------------------------
log("(g) sensitivity")
variants = [("V0 baseline: point-in-time, missing debt dropped, excl Fin/Util", dict()),
            ("V1 latest/restated values", dict(point_in_time=False)),
            ("V2 missing debt (both components) = 0", dict(debt_both_missing="zero")),
            ("V3 incl. Financials + Utilities (full sample)", dict(exclude=False)),
            ("V4 = V2 + V3", dict(debt_both_missing="zero", exclude=False)),
            ("V5 = V0 without the flagged original-LTD-zero firm-years", dict(drop_flagged=True))]
# the 4 firm-years flagged in sec_panel_summary.md (original LTD = 0, current part missing)
FLAGGED = [("TSLA", 2013), ("MA", 2013), ("META", 2013), ("META", 2021)]
fl_rows = []
for pit_ in [True, False]:
    a_ = load(point_in_time=pit_, exclude=False)
    for t_, y_ in FLAGGED:
        rr_ = a_[(a_.firm_id == t_) & (a_.fyear == y_)]
        for _, x_ in rr_.iterrows():
            fl_rows.append({"firm_id": t_, "fyear": y_, "vintage": "point-in-time (original)" if pit_ else "latest/restated",
                            "sector": x_.sector, "long_term_debt": x_.long_term_debt,
                            "long_term_debt_current": x_.long_term_debt_current,
                            "debt_component_imputed": x_.debt_component_imputed,
                            "total_assets": x_.total_assets, "debt_ratio_raw": x_.debt_ratio_raw,
                            "in_main_sample_V0": bool(pit_ and ((main.firm_id == t_) & (main.fyear == y_)).any())})
fl = pd.DataFrame(fl_rows)
save(fl, "g_flagged_zero_ltd_rows.csv", "Firm-years with original LTD = 0 flagged for manual checking: values in both "
     "vintages and main-sample membership")
sens, sens_reg = [], []
full_pred = full_summ = None
for vname, kw in variants:
    kw = dict(kw)
    drop_fl = kw.pop("drop_flagged", False)
    r = load(**kw)
    m = r[PRED + [TARGET]].notna().all(axis=1)
    wfull = load(winsor=True, **kw)
    assert (r[["firm_id", "fyear"]].values == wfull[["firm_id", "fyear"]].values).all()
    if drop_fl:
        m = m & ~pd.Series([(a, b) in set(FLAGGED) for a, b in zip(r.firm_id, r.fyear)], index=r.index)
    dv = r[m].reset_index(drop=True)
    w = wfull[m.values].reset_index(drop=True)
    if vname.startswith("V3"):
        full_pred, full_summ = protocol_suite(dv, PRED, TARGET, "full sample incl Fin/Util")
        g = full_summ[full_summ.protocol == "GroupKFold5"].copy()
    else:
        p = run_splits(dv, PRED, TARGET, splits_grouped(dv),
                       make_models(["SectorMedian (naive)", "Ridge", "LightGBM"]), "GroupKFold5")
        g = firm_bootstrap(p, ref_model="Ridge")
    for _, row in g.iterrows():
        if row.model in ["SectorMedian (naive)", "Ridge", "LightGBM"]:
            sens.append({"variant": vname, "n_obs": len(dv), "n_firms": dv.firm_id.nunique(),
                         "mean_debt_ratio": dv.debt_ratio_raw.mean(), "model": row.model,
                         "R2_GroupKFold5": row.R2, "R2_lo": row.R2_lo, "R2_hi": row.R2_hi, "RMSE": row.RMSE})
    rr = panel_regs(w, PRED, TARGET, vname, True)
    rr = rr[rr.variable.isin(["tangibility", "size_assets", "profitability", "size_revenue"]) &
            rr.estimator.isin(["Pooled OLS (firm-clustered)", "Firm FE (firm-clustered)"])]
    sens_reg.append(rr)
    if vname.startswith("V0"):
        assert len(dv) == len(main)
sens = pd.DataFrame(sens)
save(sens, "g_sensitivity_predictive.csv", "Sensitivity of GroupKFold5 R2 to data choices")
sr = pd.concat(sens_reg, ignore_index=True)[["sample", "estimator", "variable", "coef", "se", "p", "n_obs", "n_firms"]]
sr = sr.rename(columns={"sample": "variant"})
save(sr, "g_sensitivity_regressions.csv", "Sensitivity of key coefficients (full spec, per 1 s.d.) to data choices")
save(full_summ, "c_predictive_full_sample_supplementary.csv",
     "Supplementary: predictive validation on the full sample incl. Financials + Utilities")
# vintage differences in the main panel
pit, lat = load(), load(point_in_time=False)
vm = pit.merge(lat, on=["firm_id", "fyear"], suffixes=("_pit", "_lat"))
vd = {"n_matched": int(len(vm)),
      "n_debt_ratio_differs_gt_0.001": int((np.abs(vm.debt_ratio_raw_pit - vm.debt_ratio_raw_lat) > 1e-3).sum()),
      "corr_debt_ratio_pit_latest": float(vm[["debt_ratio_raw_pit", "debt_ratio_raw_lat"]].dropna().corr().iloc[0, 1])}
KEY["vintage_main_panel"] = vd

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
log("figures")
plt = cm.set_style()
OI = cm.OKABE_ITO
# 1) coefficient plot
fig, ax = plt.subplots(figsize=(cm.FIG_W_DOUBLE, 3.6))
cp = reg[(reg.target == TARGET) & reg.standardised_X & (reg.spec == "full")]
ests = ["Pooled OLS (firm-clustered)", "Firm FE (firm-clustered)", "Two-way FE (firm-clustered)", "Firm FE (Driscoll-Kraay)"]
cols = [OI[7], OI[4], OI[5], OI[2]]
marks = ["o", "s", "D", "^"]
order = PRED[::-1]
for k, (e, c, mk) in enumerate(zip(ests, cols, marks)):
    s = cp[cp.estimator == e].set_index("variable").reindex(order)
    yy = np.arange(len(order)) + (k - 1.5) * 0.18
    ax.errorbar(s.coef, yy, xerr=[s.coef - s.ci_lo, s.ci_hi - s.coef], fmt=mk, ms=3.5, color=c, lw=0.9,
                capsize=1.5, label=e)
ax.axvline(0, color="grey", lw=0.6, ls="--")
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels([f"{LABEL[f]} [{EXPECTED[f][0]}]" for f in order])
ax.set_xlabel("Change in book debt ratio per 1 s.d. of the determinant (95% CI)")
ax.legend(frameon=False, loc="lower right", fontsize=8)
cm.savefig(fig, os.path.join(FIG, "fig_e8_coefficients"))
plt.close(fig)

# 2) predicted vs actual, out-of-time
oot = pred_main[pred_main.protocol.str.startswith("OOT")]
fig, axes = plt.subplots(1, 3, figsize=(cm.FIG_W_DOUBLE, 2.7), sharex=True, sharey=True)
for ax, mname, c in zip(axes, ["Ridge", "LightGBM", "FirmHistMean (reference)"], [OI[4], OI[5], OI[7]]):
    g = oot[oot.model == mname]
    mt = summ_main[(summ_main.protocol.str.startswith("OOT")) & (summ_main.model == mname)].iloc[0]
    ax.scatter(g.y, g.yhat, s=7, color=c, alpha=0.7, lw=0)
    ax.plot([0, 0.9], [0, 0.9], color="grey", lw=0.6, ls="--")
    ax.text(0.03, 0.97, f"{mname}\nR$^2$ = {mt.R2:.2f} [{mt.R2_lo:.2f}, {mt.R2_hi:.2f}]\nRMSE = {mt.RMSE:.3f}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel("Actual debt ratio (FY2020-25)")
axes[0].set_ylabel("Predicted debt ratio")
cm.savefig(fig, os.path.join(FIG, "fig_e8_pred_vs_actual_oot"))
plt.close(fig)

# 3) R2 by protocol
rows = []
for p_ in ["GroupKFold5", "LOFO", "OOT 2009-19 -> 2020-25", "Unseen firms, future years"]:
    for mname in ["SectorMedian (naive)", "Ridge", "LightGBM"]:
        r = summ_main[(summ_main.protocol == p_) & (summ_main.model == mname)].iloc[0]
        rows.append({"group": {"GroupKFold5": "Within-SEC\nGroupKFold(5)", "LOFO": "Within-SEC\nleave-one-firm-out",
                               "OOT 2009-19 -> 2020-25": "Within-SEC\nout-of-time",
                               "Unseen firms, future years": "Within-SEC\nunseen firms,\nfuture years"}[p_],
                     "model": mname, "R2": r.R2, "lo": r.R2_lo, "hi": r.R2_hi})
for mname in ["SectorMedian (naive)", "Ridge", "LightGBM"]:
    r = tr_res[(tr_res.protocol == "Transfer synthetic -> SEC all years") & (tr_res.feature_set.str.startswith("T1"))
               & (tr_res.model == mname)].iloc[0]
    rows.append({"group": "Transfer:\nsynthetic → SEC\n(T1 features)", "model": mname, "R2": r.R2, "lo": r.R2_lo, "hi": r.R2_hi})
fr = pd.DataFrame(rows)
fr.to_csv(os.path.join(OUT, "fig_e8_r2_by_protocol_data.csv"), index=False)
fig, ax = plt.subplots(figsize=(cm.FIG_W_DOUBLE, 3.2))
groups = list(dict.fromkeys(fr.group))
LO, HI = -1.0, 1.0
for k, (mname, c, mk) in enumerate(zip(["SectorMedian (naive)", "Ridge", "LightGBM"], [OI[7], OI[4], OI[5]], ["o", "s", "D"])):
    s = fr[fr.model == mname].set_index("group").reindex(groups)
    x = np.arange(len(groups)) + (k - 1) * 0.22
    yv = s.R2.clip(LO, HI)
    ax.errorbar(x, yv, yerr=[(s.R2 - s.lo).where(s.R2 > LO, 0).clip(lower=0),
                             (s.hi - s.R2).where(s.R2 > LO, 0).clip(lower=0)],
                fmt=mk, ms=4, color=c, lw=0.9, capsize=1.5, label=mname.replace(" (naive)", " (naive baseline)"))
    for xi, rv in zip(x, s.R2):
        if rv < LO:
            ax.annotate(f"{rv:.1f}", (xi, LO), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=8, color=c)
            ax.plot(xi, LO, marker="v", color=c, ms=4)
ax.axhline(0, color="grey", lw=0.6, ls="--")
ax.set_ylim(LO - 0.05, HI)
ax.set_xticks(np.arange(len(groups)))
ax.set_xticklabels(groups, fontsize=8)
ax.set_ylabel("Out-of-sample R$^2$ (95% firm-bootstrap CI)")
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
cm.savefig(fig, os.path.join(FIG, "fig_e8_r2_by_protocol"))
plt.close(fig)

# 4) SHAP bar
fig, ax = plt.subplots(figsize=(cm.FIG_W_SINGLE, 2.9))
s = imp.iloc[::-1]
colr = [OI[5] if v > 0 else OI[4] for v in s.corr_feature_shap]
ax.barh(np.arange(len(s)), s.mean_abs_shap, color=colr)
ax.set_yticks(np.arange(len(s)))
ax.set_yticklabels(s.label)
ax.set_xlabel("Mean |SHAP| (debt-ratio units), out-of-fold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=OI[5], label="higher value → higher prediction"),
                   Patch(color=OI[4], label="higher value → lower prediction")], frameon=False, fontsize=8,
          loc="lower right")
cm.savefig(fig, os.path.join(FIG, "fig_e8_shap_bar"))
plt.close(fig)

# ---------------------------------------------------------------------------
# headline numbers
# ---------------------------------------------------------------------------
def pick(df, prot, model, col="R2"):
    r = df[(df.protocol == prot) & (df.model == model)]
    return float(r[col].iloc[0]) if len(r) else None


for p_ in summ_main.protocol.unique():
    KEY[f"R2 main {p_}"] = {m: [pick(summ_main, p_, m), pick(summ_main, p_, m, "R2_lo"), pick(summ_main, p_, m, "R2_hi")]
                            for m in summ_main[summ_main.protocol == p_].model.astype(str)}
KEY["nadeau_bengio"] = nbt.to_dict(orient="records")
KEY["fe_profitability"] = reg[(reg.target == TARGET) & reg.standardised_X & (reg.spec == "full") &
                              (reg.variable == "profitability")][["estimator", "coef", "se", "p"]].to_dict(orient="records")
KEY["pooled_like_for_like"] = lfl.to_dict(orient="records")
KEY["yahoo_year_alignment"] = yal.to_dict(orient="records")
KEY["wild_cluster_boot_profitability"] = wcb[wcb.variable == "profitability"].to_dict(orient="records")
KEY["flagged_zero_ltd_rows_in_main"] = fl[fl.in_main_sample_V0][["firm_id", "fyear"]].values.tolist()
KEY["runtime_s"] = time.time() - T0
KEY["versions"] = {"lightgbm": lgb.__version__, "xgboost": xgb.__version__, "shap": shap.__version__,
                   "linearmodels": __import__("linearmodels").__version__,
                   "sklearn": __import__("sklearn").__version__, "pandas": pd.__version__}
with open(os.path.join(OUT, "key_numbers.json"), "w") as fh:
    json.dump(KEY, fh, indent=2, default=str)
with open(os.path.join(OUT, "tables_autogen.md"), "w", encoding="utf-8") as fh:
    fh.write("# E8 tables\n\nGenerated by code_sec/e8_sec_validation.py\n\n" + "\n".join(MD))
log("done")
with open(os.path.join(OUT, "run_log.txt"), "w") as fh:
    fh.write("\n".join(LOG))
