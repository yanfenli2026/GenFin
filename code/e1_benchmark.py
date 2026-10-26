"""
e1_benchmark.py -- experiment E1: predictive benchmark.

Predictive benchmark for debt_ratio under leakage-free features and panel-aware validation.

Stages (all run by default; ``--stages`` selects a subset; ``--quick`` shrinks everything for a smoke test):
  bench        12 models x {original, strict} x {legacy 80/20, random 5-fold, repeated grouped 5x5
               (repeat 0 = grouped 5-fold), temporal, unseen-firm-future}
  uncertainty  per-model fold distribution on strict / repeated grouped 5x5, Nadeau-Bengio tests among top 3
  panels       LightGBM on 10 independently generated panels (seeds 0-9)
  sensitivity  LightGBM over a lambda x sigma_zeta^2 grid (3 panel seeds per cell)
  secondary    default_risk_score and market_leverage, strict features, grouped 5-fold, top-5 models
  wacc         WACC leakage demonstration (LightGBM, grouped 5-fold)
  learning     learning curve (LightGBM, grouped 5-fold, fraction of training firms)
  figures      all figures from the saved CSVs

Run:  python code/e1_benchmark.py
Outputs: output/e1_benchmark/
"""
from __future__ import annotations

import argparse
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

from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

warnings.filterwarnings("ignore")

OUT = os.path.join(common.OUT_MAIN, "e1_benchmark")
FIG = os.path.join(OUT, "figures")
N_JOBS = 4
SEED = 42

MODEL_ORDER = ["LinearReg", "Ridge", "Lasso", "ElasticNet", "RandomForest", "GradBoost",
               "XGBoost", "LightGBM", "CatBoost", "KNN", "SVR", "MLP"]
SCALED = {"KNN", "SVR", "MLP"}   # as in run_all_models(): only these three were fitted on standardised inputs


def make_model(name: str):
    """Same estimators and hyperparameters as run_all_models() in run_pipeline_enhanced.py.
    Only change: n_jobs / thread count capped at 4.  KNN, SVR and MLP are wrapped in a Pipeline so the
    StandardScaler is fitted on the training fold only."""
    m = {
        "LinearReg": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=N_JOBS),
        "GradBoost": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
                                verbosity=0, n_jobs=N_JOBS),
        "LightGBM": LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
                                  verbose=-1, n_jobs=N_JOBS),
        "CatBoost": CatBoostRegressor(n_estimators=100, depth=5, learning_rate=0.1, random_seed=42,
                                      verbose=0, thread_count=N_JOBS, allow_writing_files=False),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel="rbf", C=1.0),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    }[name]
    if name in SCALED:
        return Pipeline([("scaler", StandardScaler()), ("model", m)])
    return m


def metrics(y, p):
    return dict(R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=mean_absolute_error(y, p))


def fit_predict(name, X, y, tr, te):
    md = make_model(name)
    md.fit(X[tr], y[tr])
    return md.predict(X[te]), md


def spec_features(spec):
    return {"original": common.FEATURES_ORIGINAL, "strict": common.FEATURES_STRICT}[spec]


def protocol_splits(df, quick=False):
    """Returns dict protocol -> list of (repeat, fold, tr, te)."""
    out = {}
    tr, te = common.split_random(df, seed=SEED)
    out["legacy_random_80_20"] = [(0, 0, tr, te)]
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    out["random_5fold"] = [(0, k, a, b) for k, (a, b) in enumerate(kf.split(np.arange(len(df))))]
    n_rep = 2 if quick else 5
    out["grouped_5x5"] = list(common.repeated_grouped_kfold(df, 5, n_rep, seed=SEED))
    tr, te = common.temporal_split(df)
    out["temporal"] = [(0, 0, tr, te)]
    tr, te = common.unseen_firm_future_split(df, seed=SEED)
    out["unseen_firm_future"] = [(0, 0, tr, te)]
    return out


PROTOCOL_LABEL = {
    "legacy_random_80_20": "Random 80/20 (legacy)",
    "random_5fold": "Random 5-fold",
    "grouped_5fold": "Grouped 5-fold",
    "grouped_5x5": "Grouped 5x5",
    "temporal": "Temporal",
    "unseen_firm_future": "Unseen firms, future years",
}


# ---------------------------------------------------------------------------
# Stage: bench
# ---------------------------------------------------------------------------
def stage_bench(df, quick=False, max_seconds=None, fresh=False):
    """Fits every (spec, model) chunk and stores it in OUT/bench_parts/.  Finished chunks are skipped on a
    re-run, so the stage can be completed over several invocations (max_seconds stops after the chunk that
    crosses the budget; returns False if chunks remain).  --fresh deletes the parts first."""
    splits = protocol_splits(df, quick)
    y = df[common.TARGET].values.astype(float)
    models = MODEL_ORDER if not quick else ["LinearReg", "LightGBM", "SVR"]
    parts = os.path.join(OUT, "bench_parts")
    if fresh and os.path.isdir(parts):
        for f in os.listdir(parts):
            os.remove(os.path.join(parts, f))
    os.makedirs(parts, exist_ok=True)
    t_stage = time.time()
    for spec in ["original", "strict"]:
        X = df[spec_features(spec)].values.astype(float)
        for nm in models:
            f_m = os.path.join(parts, f"folds_{spec}_{nm}.csv")
            f_p = os.path.join(parts, f"preds_{spec}_{nm}.csv")
            if os.path.exists(f_m):
                continue
            if max_seconds is not None and time.time() - t_stage > max_seconds:
                print("  bench: time budget reached, re-run to continue", flush=True)
                return False
            rows = []
            preds_keep = []
            t0 = time.time()
            for prot, sp in splits.items():
                for rep, fold, tr, te in sp:
                    p, _ = fit_predict(nm, X, y, tr, te)
                    r = dict(spec=spec, model=nm, protocol=prot, repeat=rep, fold=fold,
                             n_train=len(tr), n_test=len(te), **metrics(y[te], p))
                    rows.append(r)
                    if nm == "LightGBM" and prot in ("temporal", "unseen_firm_future", "legacy_random_80_20"):
                        preds_keep.append(pd.DataFrame({"spec": spec, "protocol": prot,
                                                        "row": te, "firm_id": df["firm_id"].values[te],
                                                        "year": df["year"].values[te], "sector": df["sector"].values[te],
                                                        "y_true": y[te], "y_pred": p}))
            if preds_keep:
                pd.concat(preds_keep).to_csv(f_p, index=False)
            pd.DataFrame(rows).to_csv(f_m, index=False)   # written last = chunk complete
            print(f"  bench {spec:8s} {nm:12s} {time.time() - t0:6.1f}s", flush=True)
    folds = pd.concat([pd.read_csv(os.path.join(parts, f"folds_{s}_{m}.csv"))
                       for s in ["original", "strict"] for m in models], ignore_index=True)
    folds.to_csv(os.path.join(OUT, "bench_fold_metrics.csv"), index=False)
    pd.concat([pd.read_csv(os.path.join(parts, f"preds_{s}_LightGBM.csv")) for s in ["original", "strict"]]) \
        .to_csv(os.path.join(OUT, "lightgbm_test_predictions.csv"), index=False)
    return True


def legacy_reproduction(summ):
    """Original-spec legacy 80/20 results against the Baseline_Models sheet of the original workbook."""
    xl = os.path.join(common.ROOT, "outputs_original", "GenFin_AI_Capital_Structure_Analysis_ENHANCED.xlsx")
    orig = pd.read_excel(xl, "Baseline_Models")
    orig = orig[orig.Target == "debt_ratio"].set_index("Model")
    s = summ[(summ.spec == "original") & (summ.protocol.astype(str) == "legacy_random_80_20")].copy()
    s["model"] = s["model"].astype(str)
    rows = []
    for _, r in s.iterrows():
        o = orig.loc[r.model]
        rows.append(dict(model=r.model, R2_rerun=r.R2, R2_original=o.R2, RMSE_rerun=r.RMSE, RMSE_original=o.RMSE,
                         MAE_rerun=r.MAE, MAE_original=o.MAE, abs_diff_R2=abs(r.R2 - o.R2),
                         reproduces_to_6dp=bool(round(r.R2, 6) == round(o.R2, 6))))
    lr = pd.DataFrame(rows)
    lr.to_csv(os.path.join(OUT, "legacy_reproduction_check.csv"), index=False)
    return lr


def summarise_bench(folds):
    """Per model x spec x protocol: fold mean / sd.  grouped_5fold = repeat 0 of grouped_5x5 (seed 42)."""
    g5 = folds[(folds.protocol == "grouped_5x5") & (folds.repeat == 0)].copy()
    g5["protocol"] = "grouped_5fold"
    allf = pd.concat([folds, g5])
    summ = (allf.groupby(["spec", "model", "protocol"])
            .agg(R2=("R2", "mean"), R2_sd=("R2", "std"), RMSE=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                 MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), n_folds=("R2", "size"),
                 n_train=("n_train", "mean"), n_test=("n_test", "mean"))
            .reset_index())
    order_p = ["legacy_random_80_20", "random_5fold", "grouped_5fold", "grouped_5x5", "temporal", "unseen_firm_future"]
    summ["protocol"] = pd.Categorical(summ["protocol"], order_p, ordered=True)
    summ["model"] = pd.Categorical(summ["model"], [m for m in MODEL_ORDER if m in summ.model.unique()], ordered=True)
    summ = summ.sort_values(["spec", "protocol", "model"]).reset_index(drop=True)
    summ["spec"] = summ["spec"].astype(str)
    summ.to_csv(os.path.join(OUT, "bench_summary_long.csv"), index=False)
    wide = summ.pivot_table(index=["spec", "model"], columns="protocol", values="R2", observed=True)
    wide.to_csv(os.path.join(OUT, "bench_r2_wide.csv"))
    for met in ["RMSE", "MAE"]:
        summ.pivot_table(index=["spec", "model"], columns="protocol", values=met, observed=True) \
            .to_csv(os.path.join(OUT, f"bench_{met.lower()}_wide.csv"))
    return summ


def inflation_table(summ):
    s = summ.copy()
    s["protocol"] = s["protocol"].astype(str)
    s["model"] = s["model"].astype(str)
    strict_g = s[(s.spec == "strict") & (s.protocol == "grouped_5x5")].sort_values("R2", ascending=False)
    best = strict_g.iloc[0]["model"]
    cells = [("original", "legacy_random_80_20"), ("original", "random_5fold"), ("original", "grouped_5x5"),
             ("original", "temporal"), ("original", "unseen_firm_future"),
             ("strict", "legacy_random_80_20"), ("strict", "random_5fold"), ("strict", "grouped_5x5"),
             ("strict", "temporal"), ("strict", "unseen_firm_future")]
    rows = []
    for mdl in dict.fromkeys(["LightGBM", best]):
        ref = s[(s.spec == "original") & (s.protocol == "legacy_random_80_20") & (s.model == mdl)]["R2"].iloc[0]
        for spec, prot in cells:
            r = s[(s.spec == spec) & (s.protocol == prot) & (s.model == mdl)].iloc[0]
            rows.append(dict(model=mdl, spec=spec, protocol=prot, R2=r.R2, R2_sd=r.R2_sd, RMSE=r.RMSE, MAE=r.MAE,
                             n_folds=int(r.n_folds), drop_vs_original_random=ref - r.R2,
                             pct_of_original_random=100 * r.R2 / ref))
    inf = pd.DataFrame(rows)
    inf.to_csv(os.path.join(OUT, "inflation_table.csv"), index=False)
    return inf, best


# ---------------------------------------------------------------------------
# Stage: uncertainty
# ---------------------------------------------------------------------------
def stage_uncertainty(folds):
    f = folds[(folds.spec == "strict") & (folds.protocol == "grouped_5x5")].copy()
    J = f.groupby("model").size().iloc[0]
    rows = []
    for nm, g in f.groupby("model"):
        r2 = g["R2"].values
        ntr, nte = g["n_train"].mean(), g["n_test"].mean()
        se_c = np.sqrt((1 / len(r2) + nte / ntr) * r2.var(ddof=1))
        tq = stats.t.ppf(0.975, len(r2) - 1)
        rows.append(dict(model=nm, n_folds=len(r2), R2_mean=r2.mean(), R2_sd=r2.std(ddof=1),
                         R2_p2_5=np.percentile(r2, 2.5), R2_p97_5=np.percentile(r2, 97.5),
                         R2_min=r2.min(), R2_max=r2.max(),
                         R2_ci95_corrected_lo=r2.mean() - tq * se_c, R2_ci95_corrected_hi=r2.mean() + tq * se_c,
                         RMSE_mean=g["RMSE"].mean(), RMSE_sd=g["RMSE"].std(ddof=1),
                         MAE_mean=g["MAE"].mean(), MAE_sd=g["MAE"].std(ddof=1)))
    u = pd.DataFrame(rows).sort_values("R2_mean", ascending=False).reset_index(drop=True)
    u["rank"] = np.arange(1, len(u) + 1)
    u.to_csv(os.path.join(OUT, "uncertainty_strict_grouped5x5.csv"), index=False)

    top3 = u["model"].iloc[:3].tolist()
    piv = f.pivot_table(index=["repeat", "fold"], columns="model", values="R2")
    piv_rmse = f.pivot_table(index=["repeat", "fold"], columns="model", values="RMSE")
    ntr = f.groupby(["repeat", "fold"])["n_train"].first().mean()
    nte = f.groupby(["repeat", "fold"])["n_test"].first().mean()
    prs = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = top3[i], top3[j]
            for met, pv in [("R2", piv), ("RMSE", piv_rmse)]:
                d = (pv[a] - pv[b]).values
                t, p = common.corrected_resampled_ttest(d, ntr, nte)
                tn, pn = stats.ttest_1samp(d, 0.0)
                prs.append(dict(metric=met, model_a=a, model_b=b, mean_diff_a_minus_b=d.mean(), sd_diff=d.std(ddof=1),
                                n_folds=len(d), n_train_mean=ntr, n_test_mean=nte, t_corrected=t, p_corrected=p,
                                t_naive=float(tn), p_naive=float(pn), a_better_in_folds=int((d > 0).sum() if met == "R2" else (d < 0).sum())))
    pr = pd.DataFrame(prs)
    pr.to_csv(os.path.join(OUT, "pairwise_nadeau_bengio_top3.csv"), index=False)
    return u, pr


# ---------------------------------------------------------------------------
# Stage: panels (across independently generated panels)
# ---------------------------------------------------------------------------
def grouped_cv_score(df, feats, model="LightGBM", target=None, n_splits=5, seed=SEED, return_folds=False):
    target = target or common.TARGET
    X = df[feats].values.astype(float)
    y = df[target].values.astype(float)
    res = []
    for k, (tr, te) in enumerate(common.grouped_kfold(df, n_splits, seed)):
        p, _ = fit_predict(model, X, y, tr, te)
        res.append(dict(fold=k, n_train=len(tr), n_test=len(te), **metrics(y[te], p)))
    r = pd.DataFrame(res)
    return r if return_folds else r["R2"].mean()


def legacy_score(df, feats, model="LightGBM"):
    X = df[feats].values.astype(float)
    y = df[common.TARGET].values.astype(float)
    tr, te = common.split_random(df, seed=SEED)
    p, _ = fit_predict(model, X, y, tr, te)
    return metrics(y[te], p)["R2"]


def stage_panels(quick=False):
    seeds = range(3) if quick else range(10)
    rows = []
    for s in seeds:
        df = common.generate_panel(seed=s)
        for spec in ["strict", "original"]:
            fr = grouped_cv_score(df, spec_features(spec), return_folds=True)
            rows.append(dict(panel_seed=s, spec=spec, protocol="grouped_5fold", R2=fr.R2.mean(), R2_fold_sd=fr.R2.std(ddof=1),
                             RMSE=fr.RMSE.mean(), MAE=fr.MAE.mean(), dr_mean=df.debt_ratio.mean(), dr_sd=df.debt_ratio.std()))
        rows.append(dict(panel_seed=s, spec="original", protocol="legacy_random_80_20",
                         R2=legacy_score(df, common.FEATURES_ORIGINAL), dr_mean=df.debt_ratio.mean(), dr_sd=df.debt_ratio.std()))
        print(f"  panel seed {s} done", flush=True)
    p = pd.DataFrame(rows)
    p.to_csv(os.path.join(OUT, "across_panels_lightgbm.csv"), index=False)
    agg = (p.groupby(["spec", "protocol"])["R2"]
           .agg(n_panels="size", R2_mean="mean", R2_sd="std", R2_min="min", R2_max="max").reset_index())
    agg.to_csv(os.path.join(OUT, "across_panels_lightgbm_summary.csv"), index=False)
    return p, agg


# ---------------------------------------------------------------------------
# Stage: sensitivity
# ---------------------------------------------------------------------------
LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.6]
SIGMA2 = [0.0004, 0.005, 0.01, 0.02, 0.03, 0.05]   # 0.0004 = code default (sigma_zeta = 0.02)
SENS_SEEDS = [42, 43, 44]


def stage_sensitivity(quick=False):
    lams = [0.1, 0.3] if quick else LAMBDAS
    sig2 = [0.0004, 0.02] if quick else SIGMA2
    seeds = [42] if quick else SENS_SEEDS
    rows = []
    for lam in lams:
        for s2 in sig2:
            for sd in seeds:
                df = common.generate_panel(seed=sd, lam=lam, sigma_zeta=float(np.sqrt(s2)))
                fr = grouped_cv_score(df, common.FEATURES_STRICT, return_folds=True)
                fo = grouped_cv_score(df, common.FEATURES_ORIGINAL, return_folds=True)
                firm_mean = df.groupby("firm_id")["debt_ratio"].transform("mean")
                between = 1 - ((df.debt_ratio - firm_mean) ** 2).sum() / ((df.debt_ratio - df.debt_ratio.mean()) ** 2).sum()
                # share of firm-years at the DGP clip bounds (dr_floor or sector cap), measured on the
                # un-winsorised series; winsorize=False does not change the random stream, so this is
                # the same panel before the 1%/99% winsorisation step.
                raw = common.generate_panel(seed=sd, lam=lam, sigma_zeta=float(np.sqrt(s2)), winsorize=False,
                                            add_extra_features=False)
                cap = raw["sector"].map(common.SECTOR_CAPS).values
                rdr = raw["debt_ratio"].values
                at_floor = rdr <= common.DGP_DEFAULTS["dr_floor"] + 1e-9
                at_cap = rdr >= cap - 1e-9
                rows.append(dict(lam=lam, sigma_zeta2=s2, sigma_zeta=np.sqrt(s2), panel_seed=sd,
                                 R2_strict_grouped=fr.R2.mean(), RMSE_strict_grouped=fr.RMSE.mean(),
                                 R2_original_grouped=fo.R2.mean(),
                                 R2_original_legacy_random=legacy_score(df, common.FEATURES_ORIGINAL),
                                 dr_mean=df.debt_ratio.mean(), dr_sd=df.debt_ratio.std(),
                                 between_firm_share=between, share_at_dgp_floor=float(at_floor.mean()),
                                 share_at_dgp_cap=float(at_cap.mean()),
                                 share_at_dgp_bounds=float((at_floor | at_cap).mean())))
        print(f"  sensitivity lam={lam} done", flush=True)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "sensitivity_lambda_sigma_by_seed.csv"), index=False)
    agg = s.groupby(["lam", "sigma_zeta2"]).agg(
        sigma_zeta=("sigma_zeta", "first"), n_seeds=("panel_seed", "size"),
        R2_strict_grouped=("R2_strict_grouped", "mean"), R2_strict_grouped_sd=("R2_strict_grouped", "std"),
        R2_original_grouped=("R2_original_grouped", "mean"),
        R2_original_legacy_random=("R2_original_legacy_random", "mean"),
        dr_sd=("dr_sd", "mean"), between_firm_share=("between_firm_share", "mean"),
        share_at_dgp_floor=("share_at_dgp_floor", "mean"), share_at_dgp_cap=("share_at_dgp_cap", "mean"),
        share_at_dgp_bounds=("share_at_dgp_bounds", "mean")).reset_index()
    agg.to_csv(os.path.join(OUT, "sensitivity_lambda_sigma.csv"), index=False)
    return s, agg


# ---------------------------------------------------------------------------
# Stage: secondary targets
# ---------------------------------------------------------------------------
def stage_secondary(df, top5):
    rows = []
    for tgt in ["default_risk_score", "market_leverage"]:
        for nm in top5:
            fr = grouped_cv_score(df, common.FEATURES_STRICT, model=nm, target=tgt, return_folds=True)
            rows.append(dict(target=tgt, model=nm, spec="strict", protocol="grouped_5fold",
                             R2=fr.R2.mean(), R2_sd=fr.R2.std(ddof=1), RMSE=fr.RMSE.mean(), MAE=fr.MAE.mean()))
        # mechanical-link check for default_risk_score: only its three generator inputs
        if tgt == "default_risk_score":
            fr = grouped_cv_score(df, ["roa", "gdp_growth", "credit_spread"], model="LightGBM", target=tgt, return_folds=True)
            rows.append(dict(target=tgt, model="LightGBM", spec="generator_inputs_only(roa,gdp_growth,credit_spread)",
                             protocol="grouped_5fold", R2=fr.R2.mean(), R2_sd=fr.R2.std(ddof=1),
                             RMSE=fr.RMSE.mean(), MAE=fr.MAE.mean()))
        if tgt == "market_leverage":
            fr = grouped_cv_score(df, common.FEATURES_STRICT + ["debt_ratio"], model="LightGBM", target=tgt, return_folds=True)
            rows.append(dict(target=tgt, model="LightGBM", spec="strict+debt_ratio", protocol="grouped_5fold",
                             R2=fr.R2.mean(), R2_sd=fr.R2.std(ddof=1), RMSE=fr.RMSE.mean(), MAE=fr.MAE.mean()))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "secondary_targets_strict_grouped.csv"), index=False)
    return s


# ---------------------------------------------------------------------------
# Stage: WACC leakage
# ---------------------------------------------------------------------------
def stage_wacc(df):
    no_formula = [f for f in common.FEATURES_STRICT if f not in ("interest_rate", "credit_spread")]
    no_macro = [f for f in common.FEATURES_STRICT if f not in ("interest_rate", "credit_spread", "gdp_growth", "inflation")]
    specs = [
        ("original (paper feature list)", common.FEATURES_ORIGINAL, "yes (via roe, tobin_q, altman_zscore, interest_rate, credit_spread)"),
        ("formula inputs only: debt_ratio, interest_rate, credit_spread", ["debt_ratio", "interest_rate", "credit_spread"], "yes"),
        ("strict + debt_ratio", common.FEATURES_STRICT + ["debt_ratio"], "yes"),
        ("strict without interest_rate, credit_spread (no formula inputs)", no_formula, "no"),
        ("strict without any macro variable (roa, cash_ratio, sector, size)", no_macro, "no"),
    ]
    rows = []
    for lab, feats, has in specs:
        fr = grouped_cv_score(df, feats, target="wacc", return_folds=True)
        rows.append(dict(feature_set=lab, n_features=len(feats), contains_formula_inputs=has,
                         R2=fr.R2.mean(), R2_sd=fr.R2.std(ddof=1), RMSE=fr.RMSE.mean(), MAE=fr.MAE.mean()))
    w = pd.DataFrame(rows)
    w.to_csv(os.path.join(OUT, "wacc_leakage_lightgbm_grouped.csv"), index=False)
    return w


# ---------------------------------------------------------------------------
# Stage: learning curve
# ---------------------------------------------------------------------------
FRACS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def stage_learning(df, quick=False):
    fracs = [0.2, 1.0] if quick else FRACS
    y = df[common.TARGET].values.astype(float)
    rows = []
    firm_sector = df[["firm_id", "sector"]].drop_duplicates("firm_id").set_index("firm_id")["sector"]
    for spec in ["strict", "original"]:
        X = df[spec_features(spec)].values.astype(float)
        for k, (tr, te) in enumerate(common.grouped_kfold(df, 5, SEED)):
            tr_firms = np.sort(df["firm_id"].values[tr].astype(int))
            tr_firms = np.unique(tr_firms)
            for fr in fracs:
                rng = np.random.RandomState(1000 + k)
                keep = []
                for sec in sorted(firm_sector.unique()):
                    f = np.array([g for g in tr_firms if firm_sector[g] == sec])
                    n = max(1, int(round(fr * len(f))))
                    keep.extend(rng.choice(f, n, replace=False).tolist())
                sub = tr[np.isin(df["firm_id"].values[tr], keep)]
                md = make_model("LightGBM")
                md.fit(X[sub], y[sub])
                rows.append(dict(spec=spec, fold=k, frac_train_firms=fr, n_train_firms=len(keep), n_train_obs=len(sub),
                                 R2_train=r2_score(y[sub], md.predict(X[sub])),
                                 R2_test=r2_score(y[te], md.predict(X[te])),
                                 RMSE_test=float(np.sqrt(mean_squared_error(y[te], md.predict(X[te]))))))
    lc = pd.DataFrame(rows)
    lc.to_csv(os.path.join(OUT, "learning_curve_lightgbm_grouped_folds.csv"), index=False)
    agg = lc.groupby(["spec", "frac_train_firms"]).agg(
        n_train_firms=("n_train_firms", "mean"), n_train_obs=("n_train_obs", "mean"),
        R2_train=("R2_train", "mean"), R2_train_sd=("R2_train", "std"),
        R2_test=("R2_test", "mean"), R2_test_sd=("R2_test", "std")).reset_index()
    agg.to_csv(os.path.join(OUT, "learning_curve_lightgbm_grouped.csv"), index=False)
    return lc, agg


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def stage_figures():
    plt = common.set_style()
    OI = common.OKABE_ITO
    os.makedirs(FIG, exist_ok=True)

    # 1. R2 by model, strict spec, repeated grouped 5x5 (+ original spec for reference)
    folds = pd.read_csv(os.path.join(OUT, "bench_fold_metrics.csv"))
    g = folds[folds.protocol == "grouped_5x5"].groupby(["spec", "model"])["R2"].agg(["mean", "std"]).reset_index()
    order = g[g.spec == "strict"].sort_values("mean")["model"].tolist()
    fig, ax = plt.subplots(figsize=(common.FIG_W_SINGLE, 3.3))
    ypos = np.arange(len(order))
    for off, spec, col, mk, lab in [(-0.17, "original", OI[5], "s", "Original features"),
                                    (0.17, "strict", OI[4], "o", "Strict features")]:
        d = g[g.spec == spec].set_index("model").loc[order]
        ax.errorbar(d["mean"], ypos + off, xerr=d["std"], fmt=mk, color=col, ms=4, capsize=2, lw=0.8, label=lab)
    ax.axvline(0, color="0.6", lw=0.6, ls=":")
    ax.set_yticks(ypos)
    ax.set_yticklabels(order)
    ax.set_xlabel(r"Out-of-sample $R^2$ (mean $\pm$ s.d., 25 folds)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2, handletextpad=0.3)
    common.savefig(fig, os.path.join(FIG, "fig_r2_by_model_grouped5x5"))
    plt.close(fig)

    # 2. inflation chart: LightGBM, protocol x spec
    summ = pd.read_csv(os.path.join(OUT, "bench_summary_long.csv"))
    prots = ["legacy_random_80_20", "random_5fold", "grouped_5x5", "temporal", "unseen_firm_future"]
    labs = ["Random\n80/20", "Random\n5-fold", "Grouped\n5x5", "Temporal", "Unseen\nfirms,\nfuture"]
    fig, ax = plt.subplots(figsize=(common.FIG_W_SINGLE, 2.6))
    x = np.arange(len(prots))
    w = 0.38
    for i, (spec, col, hatch, lab) in enumerate([("original", OI[5], "", "Original features"),
                                                 ("strict", OI[4], "//", "Strict features")]):
        d = summ[(summ.spec == spec) & (summ.model == "LightGBM")].set_index("protocol").loc[prots]
        err = d["R2_sd"].values            # NaN for single-split protocols -> no error bar drawn
        bars = ax.bar(x + (i - 0.5) * w, d["R2"], w, color=col, edgecolor="black", lw=0.4,
                      hatch=hatch, label=lab)
        has = ~np.isnan(err)
        ax.errorbar((x + (i - 0.5) * w)[has], d["R2"].values[has], yerr=err[has], fmt="none",
                    ecolor="black", elinewidth=0.7, capsize=2)
        for b, v, e in zip(bars, d["R2"], np.nan_to_num(err)):
            ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + e + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel(r"LightGBM out-of-sample $R^2$")
    lg = summ[(summ.model == "LightGBM") & summ.protocol.isin(prots)]
    lo = min(0, (lg["R2"] - lg["R2_sd"].fillna(0)).min() - 0.03)
    ax.set_ylim(lo, 1.08)
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    common.savefig(fig, os.path.join(FIG, "fig_inflation_lightgbm"))
    plt.close(fig)

    # 3. sensitivity heatmaps
    sens = pd.read_csv(os.path.join(OUT, "sensitivity_lambda_sigma.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(common.FIG_W_DOUBLE, 2.7))
    for ax, col, lab in [(axes[0], "R2_strict_grouped", "Strict features, grouped 5-fold"),
                         (axes[1], "R2_original_legacy_random", "Original features, random 80/20")]:
        piv = sens.pivot(index="lam", columns="sigma_zeta2", values=col)
        im = ax.imshow(piv.values, cmap="cividis", aspect="auto", origin="lower",
                       vmin=min(0, sens[col].min()), vmax=1)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.55 else "black")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns], fontsize=7)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([f"{r:g}" for r in piv.index])
        ax.set_xlabel(r"Shock variance $\sigma_\zeta^2$")
        ax.set_ylabel(r"Adjustment speed $\lambda$")
        ax.text(0.0, 1.03, lab, transform=ax.transAxes, fontsize=8, ha="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"LightGBM $R^2$", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    common.savefig(fig, os.path.join(FIG, "fig_sensitivity_heatmap"))
    plt.close(fig)

    # 4. actual vs predicted + residuals, LightGBM strict / temporal
    pr = pd.read_csv(os.path.join(OUT, "lightgbm_test_predictions.csv"))
    d = pr[(pr.spec == "strict") & (pr.protocol == "temporal")]
    fig, axes = plt.subplots(1, 2, figsize=(common.FIG_W_DOUBLE, 2.8))
    secs = sorted(d.sector.unique())
    for k, s in enumerate(secs):
        dd = d[d.sector == s]
        axes[0].scatter(dd.y_true, dd.y_pred, s=5, alpha=0.6, color=OI[k % 8], lw=0, label=s)
        axes[1].scatter(dd.y_pred, dd.y_true - dd.y_pred, s=5, alpha=0.6, color=OI[k % 8], lw=0)
    lim = [0, max(d.y_true.max(), d.y_pred.max()) * 1.05]
    axes[0].plot(lim, lim, color="black", lw=0.7, ls="--")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("Actual debt ratio (debt / assets)")
    axes[0].set_ylabel("Predicted debt ratio (debt / assets)")
    r2 = r2_score(d.y_true, d.y_pred)
    axes[0].text(0.03, 0.95, f"$R^2$ = {r2:.3f}, n = {len(d)}", transform=axes[0].transAxes, va="top", fontsize=8)
    axes[1].axhline(0, color="black", lw=0.7, ls="--")
    axes[1].set_xlabel("Predicted debt ratio (debt / assets)")
    axes[1].set_ylabel("Residual (actual - predicted)")
    axes[0].legend(frameon=False, fontsize=6.5, markerscale=2, loc="lower right", ncol=2, handletextpad=0.1, columnspacing=0.5)
    for ax, t in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.18, 1.02, t, transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    common.savefig(fig, os.path.join(FIG, "fig_actual_vs_pred_lightgbm_strict_temporal"))
    plt.close(fig)

    # 5. learning curve
    lc = pd.read_csv(os.path.join(OUT, "learning_curve_lightgbm_grouped.csv"))
    fig, ax = plt.subplots(figsize=(common.FIG_W_SINGLE, 2.6))
    for spec, col, mk, lab in [("original", OI[5], "s", "Original"), ("strict", OI[4], "o", "Strict")]:
        d = lc[lc.spec == spec]
        ax.errorbar(d.n_train_firms, d.R2_test, yerr=d.R2_test_sd, color=col, marker=mk, ms=3.5, lw=1, capsize=2,
                    label=f"{lab}, held-out firms")
        ax.plot(d.n_train_firms, d.R2_train, color=col, marker=mk, ms=3.5, lw=1, ls="--", mfc="white",
                label=f"{lab}, training")
    ax.set_xlabel("Training firms (count; 15 years each)")
    ax.set_ylabel(r"LightGBM $R^2$")
    ax.set_ylim(min(0, (lc.R2_test - lc.R2_test_sd).min() - 0.05), 1.02)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              columnspacing=0.8, handletextpad=0.4)
    common.savefig(fig, os.path.join(FIG, "fig_learning_curve_grouped"))
    plt.close(fig)

    # 6. across-panel variability (supplement)
    ap = pd.read_csv(os.path.join(OUT, "across_panels_lightgbm.csv"))
    fig, ax = plt.subplots(figsize=(common.FIG_W_SINGLE, 2.2))
    cats = [("original", "legacy_random_80_20", "Original,\nrandom 80/20", OI[5]),
            ("original", "grouped_5fold", "Original,\ngrouped 5-fold", OI[0]),
            ("strict", "grouped_5fold", "Strict,\ngrouped 5-fold", OI[4])]
    for i, (sp, pt, lab, col) in enumerate(cats):
        v = ap[(ap.spec == sp) & (ap.protocol == pt)]["R2"].values
        jit = np.linspace(-0.12, 0.12, len(v))
        ax.scatter(np.full(len(v), i) + jit, v, s=10, color=col, edgecolor="black", lw=0.3, zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color="black", lw=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c[2] for c in cats], fontsize=7.5)
    ax.set_ylabel(r"LightGBM $R^2$ (10 panels)")
    ax.set_ylim(min(0, ap.R2.min() - 0.05), 1.0)
    common.savefig(fig, os.path.join(FIG, "fig_across_panels_lightgbm"))
    plt.close(fig)
    print("  figures written", flush=True)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="bench,uncertainty,panels,sensitivity,secondary,wacc,learning,figures")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-bench-seconds", type=float, default=None,
                    help="stop the bench stage after this many seconds; re-run to resume from saved chunks")
    ap.add_argument("--fresh", action="store_true", help="discard saved bench chunks and refit everything")
    a = ap.parse_args()
    global OUT, FIG
    if a.quick:
        OUT = os.path.join(common.OUT_MAIN, "e1_benchmark_quick")
        FIG = os.path.join(OUT, "figures")
    os.makedirs(OUT, exist_ok=True)
    stages = a.stages.split(",")
    t_all = time.time()
    timing = {}
    df = common.generate_panel(seed=SEED)
    assert len(df) == 4440 and abs(df.debt_ratio.mean() - 0.229511) < 1e-5

    if "bench" in stages:
        t0 = time.time()
        done = stage_bench(df, a.quick, a.max_bench_seconds, a.fresh)
        timing["bench"] = time.time() - t0
        if not done:
            print(json.dumps(timing, indent=1))
            return
    folds = pd.read_csv(os.path.join(OUT, "bench_fold_metrics.csv"))
    summ = summarise_bench(folds)
    inf, best = inflation_table(summ)
    if not a.quick:
        legacy_reproduction(summ)
    if "uncertainty" in stages:
        t0 = time.time()
        u, pr = stage_uncertainty(folds)
        timing["uncertainty"] = time.time() - t0
    u_path = os.path.join(OUT, "uncertainty_strict_grouped5x5.csv")
    u = pd.read_csv(u_path) if os.path.exists(u_path) else stage_uncertainty(folds)[0]
    if "panels" in stages:
        t0 = time.time()
        stage_panels(a.quick)
        timing["panels"] = time.time() - t0
    if "sensitivity" in stages:
        t0 = time.time()
        stage_sensitivity(a.quick)
        timing["sensitivity"] = time.time() - t0
    if "secondary" in stages:
        t0 = time.time()
        stage_secondary(df, u["model"].iloc[:5].tolist())
        timing["secondary"] = time.time() - t0
    if "wacc" in stages:
        t0 = time.time()
        stage_wacc(df)
        timing["wacc"] = time.time() - t0
    if "learning" in stages:
        t0 = time.time()
        stage_learning(df, a.quick)
        timing["learning"] = time.time() - t0
    if "figures" in stages:
        t0 = time.time()
        stage_figures()
        timing["figures"] = time.time() - t0
    timing["total_this_invocation"] = time.time() - t_all
    # merge with earlier invocations (the stages can be run in separate calls)
    ri_path = os.path.join(OUT, "run_info.json")
    prev = {}
    if os.path.exists(ri_path) and not a.fresh:
        try:
            with open(ri_path) as f:
                prev = json.load(f)
        except Exception:
            prev = {}
    tim_all = {k: v for k, v in prev.get("timing_seconds", {}).items()}
    tim_all.update({k: v for k, v in timing.items() if k != "total_this_invocation"})
    stage_keys = ["bench", "uncertainty", "panels", "sensitivity", "secondary", "wacc", "learning", "figures"]
    completed = sorted(set(prev.get("completed_stages", [])) | set(stages), key=stage_keys.index)
    if all(k in completed for k in stage_keys):
        tim_all["total"] = sum(tim_all.get(k, 0.0) for k in stage_keys)
    invocations = prev.get("invocations", []) + [dict(stages=stages, seconds=timing["total_this_invocation"],
                                                      finished=time.strftime("%Y-%m-%d %H:%M:%S"))]
    with open(ri_path, "w") as f:
        json.dump(dict(timing_seconds=tim_all, completed_stages=completed, invocations=invocations,
                       quick=a.quick, seed=SEED, n_jobs=N_JOBS,
                       best_strict_grouped_model=best,
                       secondary_model_rule="top 5 models by mean R2 on debt_ratio, strict spec, repeated grouped 5x5",
                       secondary_models=u["model"].iloc[:5].tolist(),
                       versions=dict(numpy=np.__version__, pandas=pd.__version__,
                                     sklearn=__import__("sklearn").__version__,
                                     lightgbm=__import__("lightgbm").__version__,
                                     xgboost=__import__("xgboost").__version__,
                                     catboost=__import__("catboost").__version__)), f, indent=2)
    print(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
