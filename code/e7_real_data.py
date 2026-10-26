"""
E7 -- External validation on real data: 46 S&P-100 firms (Yahoo Finance statements).

(a) descriptive comparison: real sample vs reference synthetic panel (shared variables, KS)
(b) within-real prediction of debt_ratio with leakage-free features
    protocols: sector-stratified GroupKFold by firm (5 folds, 5 repeats), leave-one-firm-out,
    out-of-time (train fiscal years <= 2024, test 2025); firm-bootstrap CIs (1,000 reps)
(c) transfer: models trained on the reference synthetic panel, tested on the real sample
(d) TreeSHAP of the real-data LightGBM (out-of-fold) vs a synthetic strict LightGBM
(e) descriptive check: each real firm's debt ratio vs its sector median
(f) limitations (written to RESULTS.md by hand; supporting counts saved here)

Run:  python code/e7_real_data.py
Outputs: output/e7_real_data/
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
N_BOOT = 1000
N_JOBS = 4
OUT = os.path.join(common.OUT_MAIN, "e7_real_data")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
real = common.load_real_panel()
syn = common.generate_panel(seed=SEED)
real = real.sort_values(["ticker", "year"]).reset_index(drop=True)
log(f"real panel {real.shape}, synthetic panel {syn.shape}")

# Leakage-free feature sets on real data.
# REAL_STRICT mirrors FEATURES_STRICT as far as it can be computed (12 of 16; no macro data).
STRICT_REAL, align_strict = common.align_features(real, common.FEATURES_STRICT)
EXT_CANDIDATES = [f for f in common.FEATURES_STRICT_EXT]
EXT_REAL, align_ext = common.align_features(real, EXT_CANDIDATES)
# Sensitivity set without roa: real net income is after interest expense, so real roa
# is not strictly free of debt (see foundation RESULTS.md, point 5).
STRICT_NO_ROA = [f for f in STRICT_REAL if f != "roa"]
FEATURE_SETS = {"strict": STRICT_REAL, "strict_ext": EXT_REAL, "strict_no_roa": STRICT_NO_ROA}
BANNED_TOKENS = ("debt", "interest", "market", "equity", "roe", "tobin", "altman", "wacc", "leverage")
for name, fs in FEATURE_SETS.items():
    bad = [f for f in fs if any(tok in f for tok in BANNED_TOKENS)]
    assert not bad, (name, bad)
pd.DataFrame([{"feature_set": k, "n_features": len(v), "features": ";".join(v)} for k, v in FEATURE_SETS.items()]
             ).to_csv(os.path.join(OUT, "feature_sets_real.csv"), index=False)
align_ext.to_csv(os.path.join(OUT, "feature_alignment_strict_ext.csv"), index=False)
log(f"feature sets: { {k: len(v) for k, v in FEATURE_SETS.items()} }")

# ---------------------------------------------------------------------------
# (a) Descriptive comparison
# ---------------------------------------------------------------------------
DESC_VARS = {
    "debt_ratio": "Debt ratio (total debt / total assets)",
    "roa": "ROA (net income / total assets)",
    "cash_ratio": "Cash ratio (cash / total assets)",
    "interest_coverage": "Interest coverage (EBIT / interest expense)",
    "log_assets": "Size: ln(total assets, USD m)",
    "ebit_to_assets": "EBIT / total assets",
    "asset_turnover": "Asset turnover (revenue / total assets)",
}
rows = []
for v, lab in DESC_VARS.items():
    for src, d in (("real", real), ("synthetic", syn)):
        x = d[v].replace([np.inf, -np.inf], np.nan).dropna()
        q1, q2, q3 = np.percentile(x, [25, 50, 75])
        rows.append({"variable": v, "label": lab, "sample": src, "n": len(x), "mean": x.mean(), "sd": x.std(ddof=1),
                     "min": x.min(), "p25": q1, "median": q2, "p75": q3, "iqr": q3 - q1, "max": x.max()})
desc = pd.DataFrame(rows)
ks_rows = []
for v in DESC_VARS:
    a = real[v].replace([np.inf, -np.inf], np.nan).dropna()
    b = syn[v].replace([np.inf, -np.inf], np.nan).dropna()
    ks = stats.ks_2samp(a, b)
    # firm-level KS (one value per firm: firm mean) -- avoids treating firm-years as independent
    af = real.groupby("ticker")[v].mean().dropna()
    bf = syn.groupby("firm_id")[v].mean().dropna()
    ksf = stats.ks_2samp(af, bf)
    ks_rows.append({"variable": v, "n_real": len(a), "n_synthetic": len(b), "ks_stat": ks.statistic,
                    "ks_pvalue_naive": ks.pvalue, "n_real_firms": len(af), "n_synthetic_firms": len(bf),
                    "ks_stat_firm_means": ksf.statistic, "ks_pvalue_firm_means": ksf.pvalue})
ks_df = pd.DataFrame(ks_rows)
desc_wide = desc.pivot_table(index="variable", columns="sample",
                             values=["mean", "sd", "median", "iqr", "p25", "p75", "n"], aggfunc="first")
desc_wide.columns = [f"{s}_{m}" for m, s in desc_wide.columns]
desc_wide = desc_wide.reindex(list(DESC_VARS)).reset_index().merge(ks_df, on="variable")
desc.to_csv(os.path.join(OUT, "a_descriptives_long.csv"), index=False)
desc_wide.to_csv(os.path.join(OUT, "a_descriptives_real_vs_synthetic.csv"), index=False)

# sector mix and sector mean debt ratios
sec_rows = []
for s in sorted(common.SECTORS):
    r_s, y_s = real[real.sector == s], syn[syn.sector == s]
    sec_rows.append({
        "sector": s,
        "real_firms": r_s.ticker.nunique(), "real_firm_years": len(r_s),
        "real_share_firm_years": len(r_s) / len(real),
        "synthetic_firms": y_s.firm_id.nunique(), "synthetic_firm_years": len(y_s),
        "synthetic_share_firm_years": len(y_s) / len(syn),
        "real_mean_dr": r_s.debt_ratio.mean() if len(r_s) else np.nan,
        "real_sd_dr": r_s.debt_ratio.std(ddof=1) if len(r_s) > 1 else np.nan,
        "real_median_dr": r_s.debt_ratio.median() if len(r_s) else np.nan,
        "synthetic_mean_dr": y_s.debt_ratio.mean(), "synthetic_sd_dr": y_s.debt_ratio.std(ddof=1),
        "synthetic_median_dr": y_s.debt_ratio.median(),
        "synthetic_baseline_dr_cap_x_0.45": common.SECTOR_CAPS[s] * 0.45,
    })
sec_df = pd.DataFrame(sec_rows)
sec_df["diff_mean_real_minus_synthetic"] = sec_df.real_mean_dr - sec_df.synthetic_mean_dr
sec_df.to_csv(os.path.join(OUT, "a_sector_mix_and_mean_dr.csv"), index=False)
r_s, p_s = stats.spearmanr(sec_df.real_mean_dr, sec_df.synthetic_mean_dr)
chi_obs = sec_df.real_firms.values
chi_exp = sec_df.synthetic_share_firm_years.values * chi_obs.sum()
chi = stats.chisquare(chi_obs, chi_exp)
gics = (real.drop_duplicates("ticker").groupby("gics_sector").agg(firms=("ticker", "count"),
        mapped_to=("sector", lambda s: ";".join(sorted(set(s))))).reset_index())
gics_dr = real.groupby("gics_sector").debt_ratio.agg(["mean", "median", "std"]).reset_index()
gics = gics.merge(gics_dr, on="gics_sector")
gics.to_csv(os.path.join(OUT, "a_gics_sector_coverage.csv"), index=False)


# variance decomposition of real debt_ratio (same quantities as foundation section 2)
def r2_dummies(y, groups):
    g = pd.Series(y).groupby(np.asarray(groups)).transform("mean")
    return 1 - np.sum((y - g) ** 2) / np.sum((y - y.mean()) ** 2)


vd = {
    "real_between_sector_share_mapped8": r2_dummies(real.debt_ratio.values, real.sector.values),
    "real_between_sector_share_gics11": r2_dummies(real.debt_ratio.values, real.gics_sector.values),
    "real_between_firm_share": r2_dummies(real.debt_ratio.values, real.ticker.values),
    "synthetic_between_sector_share": r2_dummies(syn.debt_ratio.values, syn.sector.values),
    "synthetic_between_firm_share": r2_dummies(syn.debt_ratio.values, syn.firm_id.values),
    "sector_mean_dr_spearman_real_vs_synthetic": r_s, "sector_mean_dr_spearman_p": p_s,
    "sector_mix_chisq_firms_vs_synthetic_shares": chi.statistic, "sector_mix_chisq_p": chi.pvalue,
}
# within-firm year-to-year persistence of real debt ratio
rr = real.sort_values(["ticker", "year"]).copy()
rr["dr_lag"] = rr.groupby("ticker").debt_ratio.shift(1)
rr["year_lag"] = rr.groupby("ticker").year.shift(1)
pp = rr[(rr.year - rr.year_lag) == 1]
vd["real_dr_ar1_corr"] = float(np.corrcoef(pp.debt_ratio, pp.dr_lag)[0, 1])
ss = syn.sort_values(["firm_id", "year"]).copy()
ss["dr_lag"] = ss.groupby("firm_id").debt_ratio.shift(1)
ps = ss.dropna(subset=["dr_lag"])
vd["synthetic_dr_ar1_corr"] = float(np.corrcoef(ps.debt_ratio, ps.dr_lag)[0, 1])
pd.DataFrame([{"quantity": k, "value": v} for k, v in vd.items()]).to_csv(
    os.path.join(OUT, "a_variance_decomposition.csv"), index=False)
log("(a) descriptives done")

# ---------------------------------------------------------------------------
# Models and baselines
# ---------------------------------------------------------------------------
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
RF_PARAMS = dict(n_estimators=500, max_depth=4, min_samples_leaf=5, max_features=0.5,
                 random_state=SEED, n_jobs=N_JOBS)
LGB_PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=10,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, n_jobs=N_JOBS, verbose=-1)
ML_MODELS = ["LinearReg", "Ridge", "RandomForest", "LightGBM"]
BASELINES = ["SectorMean", "IndustryMedian", "TrainMean"]
ALL_MODELS = BASELINES + ML_MODELS


def build(name):
    imp = ("imp", SimpleImputer(strategy="median"))
    if name == "LinearReg":
        return Pipeline([imp, ("sc", StandardScaler()), ("m", LinearRegression())])
    if name == "Ridge":
        return Pipeline([imp, ("sc", StandardScaler()), ("m", Ridge())])
    if name == "RandomForest":
        return Pipeline([imp, ("m", RandomForestRegressor(**RF_PARAMS))])
    if name == "LightGBM":
        return LGBMRegressor(**LGB_PARAMS)
    raise ValueError(name)


def fit_model(name, Xtr, ytr, groups_tr):
    """Fit one ML model.  Ridge alpha is chosen by inner GroupKFold(4) on the training firms."""
    est = build(name)
    if name == "Ridge":
        n_inner = min(4, len(np.unique(groups_tr)))
        gs = GridSearchCV(est, {"m__alpha": RIDGE_ALPHAS}, cv=GroupKFold(n_splits=n_inner),
                          scoring="neg_mean_squared_error")
        gs.fit(Xtr, ytr, groups=groups_tr)
        return gs.best_estimator_, {"alpha": gs.best_params_["m__alpha"]}
    est.fit(Xtr, ytr)
    return est, {}


def baseline_predict(name, tr: pd.DataFrame, te: pd.DataFrame, y="debt_ratio"):
    glob_mean, glob_med = tr[y].mean(), tr[y].median()
    if name == "TrainMean":
        return np.full(len(te), glob_mean)
    if name == "SectorMean":       # mean of training debt ratios in the mapped (8-sector) sector
        m = tr.groupby("sector")[y].mean()
        return te["sector"].map(m).fillna(glob_mean).values
    if name == "IndustryMedian":   # median of training debt ratios in the firm's GICS sector (11 sectors)
        col = "gics_sector" if "gics_sector" in te.columns and "gics_sector" in tr.columns else "sector"
        m = tr.groupby(col)[y].median()
        return te[col].map(m).fillna(glob_med).values
    raise ValueError(name)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    sst = np.sum((y - y.mean()) ** 2)
    return {"r2": 1 - np.sum(e ** 2) / sst, "rmse": np.sqrt(np.mean(e ** 2)), "mae": np.mean(np.abs(e)),
            "spearman": stats.spearmanr(y, p).correlation, "bias_mean_pred_minus_actual": float(np.mean(p - y))}


def firm_bootstrap(pred_df: pd.DataFrame, model_cols, ref_col="SectorMean", n_boot=N_BOOT, seed=SEED):
    """Percentile CIs for R2/RMSE/MAE, resampling firms (all of a firm's rows together).
    Also the paired difference in R2 vs ref_col on the same resamples."""
    rng = np.random.RandomState(seed)
    firms = pred_df["ticker"].unique()
    idx_by_firm = [np.where(pred_df["ticker"].values == f)[0] for f in firms]
    y = pred_df["y"].values
    P = {m: pred_df[m].values for m in model_cols}
    out = {m: {"r2": [], "rmse": [], "mae": [], "dr2": []} for m in model_cols}
    for _ in range(n_boot):
        pick = rng.randint(0, len(firms), len(firms))
        idx = np.concatenate([idx_by_firm[i] for i in pick])
        yy = y[idx]
        sst = np.sum((yy - yy.mean()) ** 2)
        r2ref = 1 - np.sum((yy - P[ref_col][idx]) ** 2) / sst if ref_col in P else np.nan
        for m in model_cols:
            e = yy - P[m][idx]
            r2 = 1 - np.sum(e ** 2) / sst
            out[m]["r2"].append(r2)
            out[m]["rmse"].append(np.sqrt(np.mean(e ** 2)))
            out[m]["mae"].append(np.mean(np.abs(e)))
            out[m]["dr2"].append(r2 - r2ref)
    res = {}
    for m in model_cols:
        d = {}
        for k, v in out[m].items():
            v = np.asarray(v)
            d[f"{k}_lo"], d[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
        d["p_r2_not_above_ref"] = float(np.mean(np.asarray(out[m]["dr2"]) <= 0))
        res[m] = d
    return res


def summarise(pred_df, protocol, feature_set, extra=None):
    rows = []
    cols = [m for m in ALL_MODELS if m in pred_df.columns]
    ci = firm_bootstrap(pred_df, cols)
    for m in cols:
        r = {"protocol": protocol, "feature_set": feature_set, "model": m, "n_obs": len(pred_df),
             "n_firms": pred_df.ticker.nunique()}
        r.update(metrics(pred_df.y, pred_df[m]))
        r.update(ci[m])
        if extra:
            r.update(extra)
        rows.append(r)
    return rows


def run_split(df, tr_idx, te_idx, feats, models=ALL_MODELS):
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    preds, info = {}, {}
    for m in models:
        if m in BASELINES:
            preds[m] = baseline_predict(m, tr, te)
        else:
            est, inf = fit_model(m, tr[feats], tr["debt_ratio"].values, tr["firm_id"].values)
            preds[m] = est.predict(te[feats])
            info[m] = inf
    return preds, info


def oof_predictions(df, splits, feats):
    out = pd.DataFrame({"ticker": df.ticker.values, "year": df.year.values, "sector": df.sector.values,
                        "y": df.debt_ratio.values})
    for m in ALL_MODELS:
        out[m] = np.nan
    out["fold"] = -1
    alphas = []
    for k, (tr, te) in enumerate(splits):
        p, info = run_split(df, tr, te, feats)
        for m in ALL_MODELS:
            out.loc[te, m] = p[m]
        out.loc[te, "fold"] = k
        if "Ridge" in info:
            alphas.append(info["Ridge"]["alpha"])
    assert out[ALL_MODELS].notna().all().all()
    return out, alphas


# ---------------------------------------------------------------------------
# (b) Within-real validation
# ---------------------------------------------------------------------------
summary_rows, alpha_rows, repeat_rows = [], [], []
pred_store = {}
N_REPEATS = 5
for fs_name, feats in FEATURE_SETS.items():
    # 5-fold sector-stratified GroupKFold by firm, repeated 5 times (repeat r: seed 42 + r)
    rep_preds = []
    for r in range(N_REPEATS):
        splits = common.grouped_kfold(real, n_splits=5, seed=SEED + r, group_col="firm_id", strat_col="sector")
        for tr, te in splits:
            assert not set(real.ticker.iloc[tr]) & set(real.ticker.iloc[te])
        oof, alphas = oof_predictions(real, splits, feats)
        rep_preds.append(oof)
        alpha_rows += [{"protocol": "grouped5", "feature_set": fs_name, "repeat": r, "alpha": a} for a in alphas]
        for m in ALL_MODELS:
            mm = metrics(oof.y, oof[m])
            repeat_rows.append({"protocol": "grouped5", "feature_set": fs_name, "repeat": r, "model": m, **mm})
    pred_store[("grouped5", fs_name)] = rep_preds[0]
    rep_preds[0].to_csv(os.path.join(OUT, f"b_oof_grouped5_rep0_{fs_name}.csv"), index=False)
    summary_rows += summarise(rep_preds[0], "grouped5 (repeat 0)", fs_name)
    # leave-one-firm-out
    firms = np.sort(real.firm_id.unique())
    lofo = [(np.where(real.firm_id != f)[0], np.where(real.firm_id == f)[0]) for f in firms]
    oof, alphas = oof_predictions(real, lofo, feats)
    alpha_rows += [{"protocol": "lofo", "feature_set": fs_name, "repeat": 0, "alpha": a} for a in alphas]
    pred_store[("lofo", fs_name)] = oof
    oof.to_csv(os.path.join(OUT, f"b_oof_lofo_{fs_name}.csv"), index=False)
    summary_rows += summarise(oof, "LOFO", fs_name)
    # out-of-time: train fiscal years <= 2024, test 2025
    test_year = 2025 if (real.year == 2025).any() else int(real.year.max())
    tr = np.where(real.year < test_year)[0]
    te = np.where(real.year == test_year)[0]
    p, info = run_split(real, tr, te, feats)
    oot = pd.DataFrame({"ticker": real.ticker.values[te], "year": real.year.values[te],
                        "sector": real.sector.values[te], "y": real.debt_ratio.values[te], **p})
    if "Ridge" in info:
        alpha_rows.append({"protocol": "oot", "feature_set": fs_name, "repeat": 0, "alpha": info["Ridge"]["alpha"]})
    pred_store[("oot", fs_name)] = oot
    oot.to_csv(os.path.join(OUT, f"b_pred_oot{test_year}_{fs_name}.csv"), index=False)
    summary_rows += summarise(oot, f"out-of-time ({test_year})", fs_name,
                              {"n_train": len(tr), "train_years": f"{real.year.iloc[tr].min()}-{real.year.iloc[tr].max()}"})
    log(f"(b) within-real done for {fs_name}")

# Robustness: drop the 4 banks (no EBIT, debt has a different role) and the firms with
# non-positive book equity in any year (ABBV, BA, MCD); strict features only.
ROB_DROP = sorted(set(["AXP", "BAC", "GS", "JPM"]) | set(real.loc[real.total_equity <= 0, "ticker"]))
real_rob = real[~real.ticker.isin(ROB_DROP)].reset_index(drop=True)
rob_rows = []
for r in range(N_REPEATS):
    splits = common.grouped_kfold(real_rob, n_splits=5, seed=SEED + r, group_col="firm_id", strat_col="sector")
    oof, _ = oof_predictions(real_rob, splits, STRICT_REAL)
    for m in ALL_MODELS:
        rob_rows.append({"protocol": "grouped5", "repeat": r, "model": m, **metrics(oof.y, oof[m])})
    if r == 0:
        summary_rows += summarise(oof, "robustness grouped5 (repeat 0): excl. banks and negative-equity firms", "strict",
                                  {"dropped": ";".join(ROB_DROP)})
firms_r = np.sort(real_rob.firm_id.unique())
lofo_r = [(np.where(real_rob.firm_id != f)[0], np.where(real_rob.firm_id == f)[0]) for f in firms_r]
oof, _ = oof_predictions(real_rob, lofo_r, STRICT_REAL)
summary_rows += summarise(oof, "robustness LOFO: excl. banks and negative-equity firms", "strict",
                          {"dropped": ";".join(ROB_DROP)})
rob_df = pd.DataFrame(rob_rows)
rob_df.groupby("model").agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"), rmse_mean=("rmse", "mean")).reset_index(
    ).to_csv(os.path.join(OUT, "b_robustness_grouped5_repeat_summary.csv"), index=False)
log(f"(b) robustness subsample done ({real_rob.ticker.nunique()} firms, {len(real_rob)} firm-years)")

rep_df = pd.DataFrame(repeat_rows)
rep_df.to_csv(os.path.join(OUT, "b_grouped5_by_repeat.csv"), index=False)
rep_sum = rep_df.groupby(["feature_set", "model"]).agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                                                       r2_min=("r2", "min"), r2_max=("r2", "max"),
                                                       rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean")
                                                       ).reset_index()
rep_sum.to_csv(os.path.join(OUT, "b_grouped5_repeat_summary.csv"), index=False)
pd.DataFrame(alpha_rows).to_csv(os.path.join(OUT, "b_ridge_alpha_selected.csv"), index=False)

# ---------------------------------------------------------------------------
# (c) Transfer: train on the reference synthetic panel, test on real firms
# ---------------------------------------------------------------------------
transfer_rows = []
transfer_info = []
test_year = 2025 if (real.year == 2025).any() else int(real.year.max())
for fs_name, feats in FEATURE_SETS.items():
    feats_syn = [f for f in feats if f in syn.columns]
    assert feats_syn == feats, (fs_name, set(feats) - set(feats_syn))
    preds = {}
    for m in ML_MODELS:
        est, inf = fit_model(m, syn[feats], syn["debt_ratio"].values, syn["firm_id"].values)
        preds[m] = est.predict(real[feats])
        transfer_info.append({"feature_set": fs_name, "model": m, **inf})
    for m in BASELINES:
        if m == "IndustryMedian":   # the synthetic panel has no GICS label: use its 8-sector median
            med = syn.groupby("sector").debt_ratio.median()
            preds[m] = real.sector.map(med).values
        else:
            preds[m] = baseline_predict(m, syn, real)
    tp = pd.DataFrame({"ticker": real.ticker.values, "year": real.year.values, "sector": real.sector.values,
                       "y": real.debt_ratio.values, **preds})
    tp.to_csv(os.path.join(OUT, f"c_transfer_pred_{fs_name}.csv"), index=False)
    pred_store[("transfer_all", fs_name)] = tp
    transfer_rows += summarise(tp, "transfer: synthetic -> real (all firm-years)", fs_name)
    tpo = tp[tp.year == test_year].reset_index(drop=True)
    pred_store[("transfer_oot", fs_name)] = tpo
    transfer_rows += summarise(tpo, f"transfer: synthetic -> real ({test_year})", fs_name)
    # transfer + linear recalibration fitted on real training firms (grouped 5-fold, repeat 0)
    rec = tp[["ticker", "year", "sector", "y"]].copy()
    splits = common.grouped_kfold(real, n_splits=5, seed=SEED, group_col="firm_id", strat_col="sector")
    for m in ML_MODELS:
        rec[m] = np.nan
        for tr, te in splits:
            lr = LinearRegression().fit(tp[[m]].iloc[tr], tp.y.iloc[tr])
            rec.loc[te, m] = lr.predict(tp[[m]].iloc[te])
    # reference baseline on the same folds
    rec["SectorMean"] = pred_store[("grouped5", fs_name)]["SectorMean"].values
    pred_store[("transfer_recal", fs_name)] = rec
    transfer_rows += summarise(rec, "transfer + linear recalibration on real (grouped5 repeat 0)", fs_name)
    log(f"(c) transfer done for {fs_name}")
pd.DataFrame(transfer_info).to_csv(os.path.join(OUT, "c_transfer_model_info.csv"), index=False)

# covariate overlap of real features with the synthetic training range
ov_rows = []
for f in EXT_REAL:
    lo, hi = syn[f].min(), syn[f].max()
    x = real[f].dropna()
    ov_rows.append({"feature": f, "synthetic_min": lo, "synthetic_max": hi, "real_min": x.min(), "real_max": x.max(),
                    "share_real_outside_synthetic_range": float(np.mean((x < lo) | (x > hi)))})
pd.DataFrame(ov_rows).to_csv(os.path.join(OUT, "c_covariate_overlap.csv"), index=False)

res = pd.DataFrame(summary_rows + transfer_rows)
res.to_csv(os.path.join(OUT, "bc_results_all.csv"), index=False)
log("(b)+(c) results saved")

# ---------------------------------------------------------------------------
# (d) TreeSHAP: real LightGBM (out-of-fold) vs synthetic strict LightGBM
# ---------------------------------------------------------------------------
import shap  # noqa: E402


def oof_shap(df, feats, params, seed_split):
    splits = common.grouped_kfold(df, n_splits=5, seed=seed_split, group_col="firm_id", strat_col="sector")
    S = np.zeros((len(df), len(feats)))
    for tr, te in splits:
        mdl = LGBMRegressor(**params).fit(df[feats].iloc[tr], df.debt_ratio.iloc[tr])
        S[te] = shap.TreeExplainer(mdl).shap_values(df[feats].iloc[te])
    return S


def shap_rank(S, feats, label):
    imp = np.abs(S).mean(0)
    d = pd.DataFrame({"feature": feats, f"{label}_mean_abs_shap": imp})
    d[f"{label}_share"] = imp / imp.sum()
    d[f"{label}_rank"] = d[f"{label}_mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    return d


def grouped_importance(S, feats, label):
    """Sector variables (code + dummies) summed per row before averaging |SHAP|."""
    sec_idx = [i for i, f in enumerate(feats) if f.startswith("sector")]
    mac_idx = [i for i, f in enumerate(feats) if f in ("gdp_growth", "inflation", "interest_rate", "credit_spread")]
    other = [i for i in range(len(feats)) if i not in sec_idx + mac_idx]
    cols = {feats[i]: S[:, i] for i in other}
    cols["sector (code + dummies)"] = S[:, sec_idx].sum(1)
    if mac_idx:
        cols["macro (4 variables)"] = S[:, mac_idx].sum(1)
    imp = {k: np.abs(v).mean() for k, v in cols.items()}
    tot = sum(imp.values())
    d = pd.DataFrame({"feature_group": list(imp), f"{label}_mean_abs_shap": list(imp.values())})
    d[f"{label}_share"] = d[f"{label}_mean_abs_shap"] / tot
    d[f"{label}_rank"] = d[f"{label}_mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    return d


S_real = oof_shap(real, STRICT_REAL, LGB_PARAMS, SEED)
S_real_ext = oof_shap(real, EXT_REAL, LGB_PARAMS, SEED)
# synthetic strict LightGBM: original paper hyperparameters (n_estimators=100, max_depth=5, lr=0.1)
LGB_PAPER = dict(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=SEED, verbose=-1, n_jobs=N_JOBS)
S_syn16 = oof_shap(syn, common.FEATURES_STRICT, LGB_PAPER, SEED)
S_syn12 = oof_shap(syn, STRICT_REAL, LGB_PAPER, SEED)
S_syn_ext = oof_shap(syn, EXT_REAL, LGB_PAPER, SEED)

sh = (shap_rank(S_real, STRICT_REAL, "real12")
      .merge(shap_rank(S_syn12, STRICT_REAL, "synthetic12"), on="feature")
      .merge(shap_rank(S_syn16, common.FEATURES_STRICT, "synthetic16"), on="feature", how="left"))
sh = sh.sort_values("real12_rank")
sh.to_csv(os.path.join(OUT, "d_shap_ranking_strict.csv"), index=False)
sh16_full = shap_rank(S_syn16, common.FEATURES_STRICT, "synthetic16").sort_values("synthetic16_rank")
sh16_full.to_csv(os.path.join(OUT, "d_shap_synthetic_strict16_full.csv"), index=False)
she = (shap_rank(S_real_ext, EXT_REAL, "real15").merge(shap_rank(S_syn_ext, EXT_REAL, "synthetic15"), on="feature")
       ).sort_values("real15_rank")
she.to_csv(os.path.join(OUT, "d_shap_ranking_strict_ext.csv"), index=False)
shg = (grouped_importance(S_real, STRICT_REAL, "real12")
       .merge(grouped_importance(S_syn12, STRICT_REAL, "synthetic12"), on="feature_group")
       .merge(grouped_importance(S_syn16, common.FEATURES_STRICT, "synthetic16"), on="feature_group", how="outer"))
shg = shg.sort_values("real12_rank")
shg.to_csv(os.path.join(OUT, "d_shap_grouped_strict.csv"), index=False)
shge = (grouped_importance(S_real_ext, EXT_REAL, "real15")
        .merge(grouped_importance(S_syn_ext, EXT_REAL, "synthetic15"), on="feature_group")).sort_values("real15_rank")
shge.to_csv(os.path.join(OUT, "d_shap_grouped_strict_ext.csv"), index=False)
shap_corr = {
    "spearman_rank_real12_vs_synthetic12": stats.spearmanr(sh.real12_rank, sh.synthetic12_rank).correlation,
    "spearman_rank_real15_vs_synthetic15": stats.spearmanr(she.real15_rank, she.synthetic15_rank).correlation,
    "spearman_rank_grouped_real12_vs_synthetic12": stats.spearmanr(shg.dropna(subset=["real12_rank"]).real12_rank,
                                                                   shg.dropna(subset=["real12_rank"]).synthetic12_rank).correlation,
    "spearman_rank_grouped_real15_vs_synthetic15": stats.spearmanr(shge.real15_rank, shge.synthetic15_rank).correlation,
}
log("(d) SHAP done")

# ---------------------------------------------------------------------------
# (e) Real firm debt ratio vs sector median (descriptive only)
# ---------------------------------------------------------------------------
fm = real.groupby(["ticker", "sector", "gics_sector"]).agg(mean_dr=("debt_ratio", "mean"),
                                                           n_years=("year", "count")).reset_index()
fm["sector_median_dr"] = fm.groupby("sector").mean_dr.transform("median")
fm["sector_n_firms"] = fm.groupby("sector").ticker.transform("count")
fm["dev_from_sector_median"] = fm.mean_dr - fm.sector_median_dr
fm["gics_median_dr"] = fm.groupby("gics_sector").mean_dr.transform("median")
fm["gics_n_firms"] = fm.groupby("gics_sector").ticker.transform("count")
fm["dev_from_gics_median"] = fm.mean_dr - fm.gics_median_dr
# leave-one-out sector median (firm excluded from its own benchmark)
fm["sector_median_loo"] = [np.median(fm.mean_dr[(fm.sector == s) & (fm.ticker != t)]) if
                           ((fm.sector == s) & (fm.ticker != t)).any() else np.nan
                           for s, t in zip(fm.sector, fm.ticker)]
fm["dev_from_sector_median_loo"] = fm.mean_dr - fm.sector_median_loo
fm = fm.sort_values("dev_from_sector_median")
fm.to_csv(os.path.join(OUT, "e_firm_vs_sector_median.csv"), index=False)
ok = fm[fm.sector_n_firms >= 3]
e_stats = {
    "n_firms_in_sectors_with_3plus_firms": len(ok),
    "median_abs_dev_pp": float(np.median(np.abs(ok.dev_from_sector_median)) * 100),
    "share_within_5pp": float(np.mean(np.abs(ok.dev_from_sector_median) <= 0.05)),
    "share_within_10pp": float(np.mean(np.abs(ok.dev_from_sector_median) <= 0.10)),
    "median_abs_dev_loo_pp": float(np.nanmedian(np.abs(ok.dev_from_sector_median_loo)) * 100),
}
# do firms move toward the sector median from one year to the next?  (pooled OLS, firm-clustered SE)
import statsmodels.api as sm  # noqa: E402

rr = real.sort_values(["ticker", "year"]).copy()
rr["sec_med_year"] = rr.groupby(["sector", "year"]).debt_ratio.transform("median")
rr["dr_lag"] = rr.groupby("ticker").debt_ratio.shift(1)
rr["gap_lag"] = rr.groupby("ticker").sec_med_year.shift(1) - rr.dr_lag
rr["d_dr"] = rr.debt_ratio - rr.dr_lag
rr["year_lag"] = rr.groupby("ticker").year.shift(1)
pa = rr[((rr.year - rr.year_lag) == 1) & (rr.sector.map(rr.drop_duplicates("ticker").sector.value_counts()) >= 3)
        ].dropna(subset=["d_dr", "gap_lag"])
ols = sm.OLS(pa.d_dr, sm.add_constant(pa.gap_lag)).fit(cov_type="cluster", cov_kwds={"groups": pa.ticker})
e_stats.update({
    "pa_toward_sector_median_coef": float(ols.params["gap_lag"]), "pa_se_clustered": float(ols.bse["gap_lag"]),
    "pa_ci_lo": float(ols.conf_int().loc["gap_lag", 0]), "pa_ci_hi": float(ols.conf_int().loc["gap_lag", 1]),
    "pa_p": float(ols.pvalues["gap_lag"]), "pa_n_obs": int(ols.nobs), "pa_n_firms": int(pa.ticker.nunique()),
    "pa_r2": float(ols.rsquared),
})
pd.DataFrame([{"quantity": k, "value": v} for k, v in e_stats.items()]).to_csv(
    os.path.join(OUT, "e_sector_median_summary.csv"), index=False)
log("(e) sector-median check done")

# ---------------------------------------------------------------------------
# (f) Limitations: supporting counts
# ---------------------------------------------------------------------------
lim = {
    "n_firms": int(real.ticker.nunique()), "n_firm_years": int(len(real)),
    "fiscal_years": ";".join(f"{y}:{n}" for y, n in real.year.value_counts().sort_index().items()),
    "mapped_sectors_with_lt3_firms": ";".join(sec_df.loc[sec_df.real_firms < 3, "sector"]),
    "gics_sectors_with_lt3_firms": ";".join(gics.loc[gics.firms < 3, "gics_sector"]),
    "gics_sectors_not_in_synthetic_8": "Communication Services;Consumer Staples",
    "n_firms_missing_ebit": int(real.loc[real.ebit.isna(), "ticker"].nunique()),
    "firms_missing_ebit": ";".join(sorted(real.loc[real.ebit.isna(), "ticker"].unique())),
    "firms_missing_interest_expense": ";".join(sorted(real.loc[real.interest_expense.isna(), "ticker"].unique())),
    "firms_negative_book_equity": ";".join(sorted(real.loc[real.total_equity <= 0, "ticker"].unique())),
    "min_total_assets_usd_bn": float(real.total_assets.min() / 1e9),
    "median_total_assets_usd_bn": float(real.total_assets.median() / 1e9),
    "macro_variables_available": 0,
}
pd.DataFrame([{"item": k, "value": v} for k, v in lim.items()]).to_csv(
    os.path.join(OUT, "f_limitations_counts.csv"), index=False)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
plt = common.set_style()
plt.rcParams["mathtext.fontset"] = "stix"
OI = common.OKABE_ITO
C_REAL, C_SYN = OI[5], OI[4]   # vermillion = real, blue = synthetic

# Fig 1: distribution overlays
fig, axes = plt.subplots(2, 3, figsize=(common.FIG_W_DOUBLE, 4.3))
panels = [("debt_ratio", "Debt ratio (total debt / total assets)", None),
          ("roa", "ROA (net income / total assets)", None),
          ("cash_ratio", "Cash ratio (cash / total assets)", None),
          ("interest_coverage", "log$_{10}$ interest coverage (EBIT / interest)", "log"),
          ("log_assets", "ln(total assets, USD million)", None),
          ("asset_turnover", "log$_{10}$ asset turnover (revenue / assets)", "log")]
ks_left = {"log_assets"}
for ax, (v, lab, scale), letter in zip(axes.ravel(), panels, "abcdef"):
    a = real[v].replace([np.inf, -np.inf], np.nan).dropna()
    b = syn[v].replace([np.inf, -np.inf], np.nan).dropna()
    if scale == "log":   # zeros (synthetic EBIT floored at 0) cannot be shown on a log axis
        a, b = np.log10(a[a > 0]), np.log10(b[b > 0])
    # range covers every real value and the central 99% of the synthetic panel;
    # synthetic values beyond it are drawn in the edge bins
    lo = min(a.min(), np.percentile(b, 0.5))
    hi = max(a.max(), np.percentile(b, 99.5))
    bins = np.linspace(lo, hi, 40)
    ax.hist(b.clip(bins[0], bins[-1]), bins=bins, density=True, color=C_SYN, alpha=0.45,
            label=f"Reference synthetic panel (n = {len(b):,})")
    ax.hist(a.clip(bins[0], bins[-1]), bins=bins, density=True, histtype="step", lw=1.4, color=C_REAL,
            label=f"Real sample, 46 firms (n = {len(a)})")
    ks = ks_df.set_index("variable").loc[v, "ks_stat"]
    if v in ks_left:
        ax.text(0.03, 0.95, f"KS = {ks:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    else:
        ax.text(0.97, 0.95, f"KS = {ks:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ax.set_xlabel(lab)
    ax.set_ylabel("Density")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.05, 1, 1))
common.savefig(fig, os.path.join(OUT, "fig_e7_distributions_real_vs_synthetic"))
plt.close(fig)

# Fig 2: sector mix and sector mean debt ratio
fig, axes = plt.subplots(1, 2, figsize=(common.FIG_W_DOUBLE, 3.1))
sd = sec_df.sort_values("synthetic_mean_dr")
y = np.arange(len(sd))
axes[0].barh(y - 0.2, sd.synthetic_share_firm_years * 100, height=0.4, color=C_SYN, label="Reference synthetic panel")
axes[0].barh(y + 0.2, sd.real_share_firm_years * 100, height=0.4, color=C_REAL, label="Real sample")
for yi, nf in zip(y, sd.real_firms):
    axes[0].text(sd.real_share_firm_years.iloc[yi] * 100 + 0.5, yi + 0.2, f"{nf}", va="center", fontsize=7)
axes[0].set_yticks(y)
axes[0].set_yticklabels(sd.sector)
axes[0].set_xlabel("Share of firm-years (%)")
axes[0].text(-0.3, 1.03, "(a)", transform=axes[0].transAxes, fontweight="bold")
# real firm-mean DR points
fmean = real.groupby(["ticker", "sector"]).debt_ratio.mean().reset_index()
for yi, s in enumerate(sd.sector):
    pts = fmean[fmean.sector == s].debt_ratio.values
    axes[1].scatter(pts, np.full(len(pts), yi) + np.linspace(-0.12, 0.12, len(pts)) if len(pts) > 1 else [yi],
                    s=9, color=C_REAL, alpha=0.45, lw=0)
axes[1].errorbar(sd.synthetic_mean_dr, y - 0.28, xerr=sd.synthetic_sd_dr, fmt="s", ms=4, color=C_SYN, capsize=2,
                 lw=0.9, label="Reference synthetic panel: mean ± s.d.")
axes[1].errorbar(sd.real_mean_dr, y + 0.28, xerr=sd.real_sd_dr.fillna(0), fmt="o", ms=4, color=C_REAL, capsize=2,
                 lw=0.9, label="Real sample: mean ± s.d.")
axes[1].scatter([], [], s=9, color=C_REAL, alpha=0.45, label="Real firm (4-year mean)")
axes[1].set_yticks(y)
axes[1].set_yticklabels([])
axes[1].set_xlabel("Debt ratio (total debt / total assets)")
axes[1].text(-0.02, 1.03, "(b)", transform=axes[1].transAxes, fontweight="bold")
axes[0].legend(frameon=False, loc="lower center", fontsize=7, ncol=2, bbox_to_anchor=(0.5, 1.04))
axes[1].legend(frameon=False, loc="lower center", fontsize=7, ncol=2, bbox_to_anchor=(0.5, 1.04))
axes[0].set_xlim(0, sd.real_share_firm_years.max() * 100 + 4)
fig.tight_layout()
common.savefig(fig, os.path.join(OUT, "fig_e7_sector_mix_and_debt_ratio"))
plt.close(fig)

# Fig 3: predicted vs actual (grouped 5-fold, strict features) + transfer
g5 = pred_store[("grouped5", "strict")]
tr_all = pred_store[("transfer_all", "strict")]
res_i = res.set_index(["protocol", "feature_set", "model"])
pan = [("SectorMean", g5, "grouped5 (repeat 0)", "Sector-mean baseline\n(leave-firms-out)"),
       ("Ridge", g5, "grouped5 (repeat 0)", "Ridge (leave-firms-out)"),
       ("LightGBM", g5, "grouped5 (repeat 0)", "LightGBM (leave-firms-out)"),
       ("LightGBM", tr_all, "transfer: synthetic -> real (all firm-years)", "LightGBM trained on the\nsynthetic panel")]
fig, axes = plt.subplots(1, 4, figsize=(common.FIG_W_DOUBLE, 2.25), sharey=True)
lim_ = (0, 1.0)
secs = sorted(real.sector.unique())
markers = dict(zip(secs, ["o", "s", "^", "D", "v", "P", "X", "*"]))
colors = dict(zip(secs, OI))
for ax, (m, d, proto, title), letter in zip(axes, pan, "abcd"):
    for s in secs:
        dd = d[d.sector == s]
        ax.scatter(dd[m], dd.y, s=10, marker=markers[s], color=colors[s], alpha=0.8, lw=0.3,
                   edgecolor="white", label=s)
    ax.plot(lim_, lim_, color="0.4", lw=0.8, ls="--")
    ax.set_xlim(lim_)
    ax.set_ylim(lim_)
    ax.set_aspect("equal")
    r2 = res_i.loc[(proto, "strict", m), "r2"]
    ax.text(0.96, 0.04, f"$R^2$ = {r2:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("Predicted debt ratio")
    ax.set_title(f"({letter}) {title}", fontsize=8, loc="left")
axes[0].set_ylabel("Actual debt ratio")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=7,
           handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
common.savefig(fig, os.path.join(OUT, "fig_e7_pred_vs_actual"))
plt.close(fig)

# Fig 4: R2 by protocol, within-real vs transfer (strict features)
protos = [("grouped5 (repeat 0)", "Within-real:\nleave-firms-out (5-fold)"),
          ("LOFO", "Within-real:\nleave-one-firm-out"),
          (f"out-of-time ({test_year})", f"Within-real:\nout-of-time ({test_year})"),
          ("transfer: synthetic -> real (all firm-years)", "Transfer: synthetic\n→ real (all years)"),
          (f"transfer: synthetic -> real ({test_year})", f"Transfer: synthetic\n→ real ({test_year})")]
models_fig = ["SectorMean", "IndustryMedian", "LinearReg", "Ridge", "RandomForest", "LightGBM"]
mlabels = {"SectorMean": "Sector mean", "IndustryMedian": "Industry median", "LinearReg": "Linear regression",
           "Ridge": "Ridge", "RandomForest": "Random forest", "LightGBM": "LightGBM"}
mcol = dict(zip(models_fig, [OI[7], "0.6", OI[0], OI[1], OI[2], OI[5]]))
fig, ax = plt.subplots(figsize=(common.FIG_W_DOUBLE, 2.8))
XMIN = -1.0
w = 0.13
for i, (p, plab) in enumerate(protos):
    for j, m in enumerate(models_fig):
        r = res_i.loc[(p, "strict", m)]
        x = i + (j - (len(models_fig) - 1) / 2) * w
        val = r.r2
        shown = max(val, XMIN)
        ax.bar(x, shown, width=w, color=mcol[m], label=mlabels[m] if i == 0 else None, edgecolor="white", lw=0.3)
        lo, hi = max(r.r2_lo, XMIN), max(r.r2_hi, XMIN)
        ax.plot([x, x], [lo, hi], color="0.15", lw=0.7)
        if val < XMIN:
            ax.text(x, XMIN + 0.03, f"{val:.1f}", rotation=90, ha="center", va="bottom", fontsize=6, color="white")
ax.axhline(0, color="0.2", lw=0.7)
ax.set_xticks(range(len(protos)))
ax.set_xticklabels([pl for _, pl in protos], fontsize=7.5)
ax.set_ylabel("Out-of-sample $R^2$ (debt ratio)")
ax.set_ylim(XMIN, 1.0)
ax.legend(frameon=False, ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.tight_layout()
common.savefig(fig, os.path.join(OUT, "fig_e7_r2_by_protocol"))
plt.close(fig)

# Fig 5: SHAP grouped importance shares, real vs synthetic (strict, 12 aligned features)
g = shg.dropna(subset=["real12_share"]).sort_values("real12_share")
fig, ax = plt.subplots(figsize=(common.FIG_W_SINGLE, 2.4))
y = np.arange(len(g))
ax.barh(y - 0.2, g.synthetic12_share * 100, height=0.4, color=C_SYN, label="Reference synthetic panel")
ax.barh(y + 0.2, g.real12_share * 100, height=0.4, color=C_REAL, label="Real sample")
ax.set_yticks(y)
ax.set_yticklabels(g.feature_group.str.replace("_", " "))
ax.set_xlabel("Share of mean |SHAP| (%)")
ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout()
common.savefig(fig, os.path.join(OUT, "fig_e7_shap_real_vs_synthetic"))
plt.close(fig)

# Fig 6: firm deviation from sector median
f6 = fm.sort_values("dev_from_sector_median")
fig, ax = plt.subplots(figsize=(common.FIG_W_SINGLE, 5.4))
y = np.arange(len(f6))
cols6 = [colors[s] for s in f6.sector]
ax.hlines(y, 0, f6.dev_from_sector_median * 100, color="0.75", lw=0.8)
for s in secs:
    msk = (f6.sector == s).values
    ax.scatter(f6.dev_from_sector_median.values[msk] * 100, y[msk], s=14, marker=markers[s], color=colors[s],
               label=f"{s} ({int(f6[f6.sector == s].sector_n_firms.iloc[0])})", zorder=3)
ax.axvline(0, color="0.3", lw=0.7)
ax.axvspan(-5, 5, color="0.92", zorder=0)
ax.set_yticks(y)
ax.set_yticklabels(f6.ticker, fontsize=6)
ax.set_xlabel("Firm mean debt ratio minus\nsector median (percentage points)")
ax.set_ylim(-0.8, len(f6) - 0.2)
ax.legend(frameon=False, fontsize=6.5, loc="lower right", title="Sector (firms)", title_fontsize=6.5)
fig.tight_layout()
common.savefig(fig, os.path.join(OUT, "fig_e7_firm_vs_sector_median"))
plt.close(fig)
log("figures saved")

# ---------------------------------------------------------------------------
# Compact key-number JSON
# ---------------------------------------------------------------------------
key = {"variance_decomposition": vd, "shap_rank_correlations": shap_corr, "sector_median_check": e_stats,
       "limitations": lim, "feature_sets": FEATURE_SETS, "test_year_oot": test_year,
       "hyperparameters": {"ridge_alphas": RIDGE_ALPHAS, "rf": {k: v for k, v in RF_PARAMS.items()},
                           "lightgbm": LGB_PARAMS, "lightgbm_synthetic_shap": LGB_PAPER},
       "n_boot": N_BOOT, "seed": SEED, "runtime_seconds": time.time() - T0}
with open(os.path.join(OUT, "key_numbers.json"), "w") as fh:
    json.dump(key, fh, indent=2, default=float)
log("done")
