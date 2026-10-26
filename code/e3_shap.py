"""
e3_shap.py -- experiment E3: TreeSHAP on LightGBM and separated leakage ablations (target debt_ratio).

Run:  python code/e3_shap.py
Out:  output/e3_shap/

Steps
  1. Specifications S1-S4 (+ supplementary sequential ablation and reference sets):
     LightGBM (n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42) evaluated on
     the legacy random 80/20 split, grouped 5-fold CV by firm (seed 42) and repeated grouped 5x5 CV.
  2. Out-of-fold TreeSHAP on the LightGBM model itself (fit on 4 folds, explain the held-out fold).
  3. Stability of the S4 ranking across the 5 folds and across 5 panel seeds (42-46).
  4. Reproduction of the original Random-Forest-surrogate SHAP ranking and rank agreement.
  5. Provenance checks: capital_structure_efficiency in the original feature list; source of Table XV.
  6. Figures.
All seeds fixed; n_jobs = 4.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

warnings.filterwarnings("ignore")
T0 = time.time()
OUT = os.path.join(common.OUT_MAIN, "e3_shap")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
N_JOBS = 4
SEED = 42
PANEL_SEEDS = [42, 43, 44, 45, 46]
LGBM_PARAMS = dict(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=SEED, verbose=-1, n_jobs=N_JOBS)
TARGET = common.TARGET

import shap  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------
SPECS = {
    "S1": dict(label="S1 Original", features=common.FEATURES_ORIGINAL),
    "S2": dict(label="S2 Original minus Altman Z", features=common.FEATURES_NO_Z),
    "S3": dict(label="S3 Original minus Altman Z and default_risk_score", features=common.FEATURES_NO_Z_DR),
    "S4": dict(label="S4 Strict, leakage-free (primary)", features=common.FEATURES_STRICT),
}
_orig = common.FEATURES_ORIGINAL
_seq = ["altman_zscore", "roe", "tobin_q", "interest_coverage", "capital_structure_efficiency"]
SUPP = {
    "A1": dict(label="S1 - altman_zscore", features=[f for f in _orig if f not in _seq[:1]]),
    "A2": dict(label="S1 - altman_zscore, roe", features=[f for f in _orig if f not in _seq[:2]]),
    "A3": dict(label="S1 - altman_zscore, roe, tobin_q", features=[f for f in _orig if f not in _seq[:3]]),
    "A4": dict(label="S1 - altman_zscore, roe, tobin_q, interest_coverage", features=[f for f in _orig if f not in _seq[:4]]),
    "A5": dict(label="S1 - all 5 debt-dependent features (= S4)", features=[f for f in _orig if f not in _seq[:5]]),
    "R_roa_roe": dict(label="Reference: roa + roe only", features=["roa", "roe"]),
    "R_sector": dict(label="Reference: sector_code + 8 sector dummies only", features=["sector_code"] + common.SECTOR_DUMMIES),
    "S4_ext": dict(label="Strict extended (FEATURES_STRICT_EXT)", features=common.FEATURES_STRICT_EXT),
}

DISPLAY = {"roa": "ROA", "roe": "ROE", "interest_coverage": "Interest coverage", "cash_ratio": "Cash ratio",
           "tobin_q": "Tobin's Q", "altman_zscore": "Altman Z-score",
           "capital_structure_efficiency": "Cap. structure efficiency", "sector_code": "Sector code",
           "size_code": "Size tertile", "gdp_growth": "GDP growth", "inflation": "Inflation",
           "interest_rate": "Interest rate", "credit_spread": "Credit spread",
           "default_risk_score": "Default risk score", "log_assets": "log(assets)",
           "ebit_to_assets": "EBIT/assets", "asset_turnover": "Asset turnover"}
for _s in common.SECTORS:
    DISPLAY[f"sector_{_s}"] = f"Sector: {_s}"
MACRO = ["gdp_growth", "inflation", "interest_rate", "credit_spread"]
SECTOR_BLOCK = ["sector_code"] + common.SECTOR_DUMMIES


def disp(f):
    return DISPLAY.get(f, f)


def metrics(y, p):
    return dict(R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=mean_absolute_error(y, p))


def fit_lgbm(X, y):
    m = LGBMRegressor(**LGBM_PARAMS)
    m.fit(X, y)
    return m


def tree_shap(model, X):
    expl = shap.TreeExplainer(model)
    sv = np.asarray(expl.shap_values(X))
    ev = float(np.ravel(expl.expected_value)[0])
    add_err = float(np.max(np.abs(ev + sv.sum(axis=1) - model.predict(X))))
    return sv, ev, add_err


def grouped_oof(df, feats, seed=SEED, do_shap=True):
    """Grouped 5-fold (firm-level, sector-balanced). Returns OOF predictions, OOF SHAP matrix,
    per-fold metrics and per-fold mean |SHAP| vectors."""
    X = df[feats].astype(float)
    y = df[TARGET].astype(float).values
    oof = np.full(len(df), np.nan)
    sv_all = np.full((len(df), len(feats)), np.nan) if do_shap else None
    fold_rows, fold_imp, add_errs = [], [], []
    for k, (tr, te) in enumerate(common.grouped_kfold(df, 5, seed)):
        m = fit_lgbm(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        oof[te] = p
        r = metrics(y[te], p)
        r.update(fold=k, n_train=len(tr), n_test=len(te))
        fold_rows.append(r)
        if do_shap:
            sv, ev, ae = tree_shap(m, X.iloc[te])
            sv_all[te] = sv
            fold_imp.append(np.abs(sv).mean(axis=0))
            add_errs.append(ae)
    return dict(oof=oof, shap=sv_all, folds=pd.DataFrame(fold_rows),
                fold_imp=np.array(fold_imp) if do_shap else None,
                add_err=max(add_errs) if add_errs else np.nan)


def legacy_eval(df, feats, scaled=False, do_shap=False):
    X = df[feats].astype(float)
    y = df[TARGET].astype(float).values
    tr, te = common.split_random(df, SEED)
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    if scaled:  # sensitivity only: run_all_models fits tree models on UNSCALED Xt (scaling used only for SVR/MLP/KNN)
        sc = StandardScaler().fit(Xtr)
        Xtr = pd.DataFrame(sc.transform(Xtr), columns=feats)
        Xte = pd.DataFrame(sc.transform(Xte), columns=feats)
    m = fit_lgbm(Xtr, y[tr])
    p = m.predict(Xte)
    out = metrics(y[te], p)
    if do_shap:
        sv, ev, ae = tree_shap(m, Xte)
        out["_imp"] = np.abs(sv).mean(axis=0)
        out["_add_err"] = ae
    return out


def repeated_eval(df, feats):
    rows = []
    for r, k, tr, te in common.repeated_grouped_kfold(df, 5, 5, SEED):
        X = df[feats].astype(float)
        y = df[TARGET].astype(float).values
        m = fit_lgbm(X.iloc[tr], y[tr])
        d = metrics(y[te], m.predict(X.iloc[te]))
        d.update(repeat=r, fold=k, n_train=len(tr), n_test=len(te))
        rows.append(d)
    return pd.DataFrame(rows)


def block_importance(sv, feats, blocks):
    """mean_i |sum_{j in block} phi_ij| for each block (SHAP values are additive within a row)."""
    out = {}
    for name, cols in blocks.items():
        idx = [feats.index(c) for c in cols if c in feats]
        if idx:
            out[name] = float(np.abs(sv[:, idx].sum(axis=1)).mean())
    return out


def pairwise_spearman(mat):
    vals = [stats.spearmanr(mat[i], mat[j]).correlation for i, j in combinations(range(len(mat)), 2)]
    return np.array(vals)


def topk_sets(mat, feats, k=5):
    return [set(np.array(feats)[np.argsort(-v)[:k]]) for v in mat]


# ===========================================================================
print("Generating reference synthetic panel (seed 42) ...")
df = common.generate_panel(seed=SEED)
assert len(df) == 4440
assert common.FEATURES_NO_Z == common.FEATURES_NO_Z_DR, "S2 and S3 expected identical"
assert "default_risk_score" not in common.FEATURES_ORIGINAL

# ---------------------------------------------------------------------------
# (0) legacy reproduction check
# ---------------------------------------------------------------------------
ORIG_LGBM = dict(R2=0.894922, RMSE=0.037794, MAE=0.026852)  # Baseline_Models sheet, original workbook
leg_scaled = legacy_eval(df, common.FEATURES_ORIGINAL, scaled=True)
leg_raw = legacy_eval(df, common.FEATURES_ORIGINAL, scaled=False)
repro = pd.DataFrame([
    dict(run="original workbook (Baseline_Models, LightGBM, debt_ratio)", **ORIG_LGBM),
    dict(run="this script, unscaled features as in run_all_models (used throughout)", **{k: leg_raw[k] for k in ("R2", "RMSE", "MAE")}),
    dict(run="sensitivity: StandardScaler fitted on train (not used by the original for trees)", **{k: leg_scaled[k] for k in ("R2", "RMSE", "MAE")}),
])
repro.to_csv(os.path.join(OUT, "legacy_reproduction_check.csv"), index=False)
print(repro.to_string(index=False))

# ---------------------------------------------------------------------------
# (a) + (b) main specifications
# ---------------------------------------------------------------------------
perf_rows, fold_rows_all, rep_rows_all = [], [], []
oof_store, rankings = {}, {}
for key, spec in {**SPECS, **SUPP}.items():
    feats = list(spec["features"])
    main = key in SPECS
    print(f"[{time.time()-T0:6.1f}s] {key}: {spec['label']} ({len(feats)} features)")
    g = grouped_oof(df, feats, do_shap=main)
    y = df[TARGET].values
    pooled = metrics(y, g["oof"])
    f = g["folds"]
    leg = legacy_eval(df, feats, do_shap=main)
    rep = repeated_eval(df, feats)
    rep["spec"] = key
    rep_rows_all.append(rep)
    f2 = f.copy()
    f2["spec"] = key
    fold_rows_all.append(f2)
    perf_rows.append(dict(
        spec=key, label=spec["label"], n_features=len(feats), main_spec=main,
        legacy_R2=leg["R2"], legacy_RMSE=leg["RMSE"], legacy_MAE=leg["MAE"],
        grouped_R2_mean=f["R2"].mean(), grouped_R2_sd=f["R2"].std(ddof=1),
        grouped_RMSE_mean=f["RMSE"].mean(), grouped_RMSE_sd=f["RMSE"].std(ddof=1),
        grouped_MAE_mean=f["MAE"].mean(), grouped_MAE_sd=f["MAE"].std(ddof=1),
        grouped_pooled_oof_R2=pooled["R2"], grouped_pooled_oof_RMSE=pooled["RMSE"], grouped_pooled_oof_MAE=pooled["MAE"],
        rep5x5_R2_mean=rep["R2"].mean(), rep5x5_R2_sd=rep["R2"].std(ddof=1),
        rep5x5_RMSE_mean=rep["RMSE"].mean(), rep5x5_MAE_mean=rep["MAE"].mean(),
        legacy_minus_grouped_R2=leg["R2"] - f["R2"].mean(),
        shap_additivity_max_abs_err=g["add_err"] if main else np.nan,
    ))
    if main:
        oof_store[key] = dict(feats=feats, g=g)
        imp_oof = np.abs(g["shap"]).mean(axis=0)
        rk = pd.DataFrame({"feature": feats, "display_name": [disp(x) for x in feats],
                           "audit_class": [common.AUDIT_CLASS[x] for x in feats],
                           "mean_abs_shap_oof": imp_oof,
                           "mean_abs_shap_oof_fold_sd": g["fold_imp"].std(axis=0, ddof=1),
                           "mean_shap_oof": g["shap"].mean(axis=0),
                           "mean_abs_shap_legacy_test": leg["_imp"]})
        for k in range(5):
            rk[f"mean_abs_shap_fold{k}"] = g["fold_imp"][k]
        rk["share_of_total_oof"] = rk["mean_abs_shap_oof"] / rk["mean_abs_shap_oof"].sum()
        rk = rk.sort_values("mean_abs_shap_oof", ascending=False).reset_index(drop=True)
        rk.insert(0, "rank_oof", np.arange(1, len(rk) + 1))
        rk["rank_legacy_test"] = rk["mean_abs_shap_legacy_test"].rank(ascending=False, method="min").astype(int)
        rk.to_csv(os.path.join(OUT, f"shap_ranking_{key}.csv"), index=False)
        rankings[key] = rk
        pd.DataFrame(g["shap"], columns=feats).assign(firm_id=df["firm_id"].values, year=df["year"].values,
                                                       debt_ratio=df[TARGET].values, oof_pred=g["oof"]) \
            .to_csv(os.path.join(OUT, f"shap_values_oof_{key}.csv.gz"), index=False, compression="gzip")

perf = pd.DataFrame(perf_rows)
perf.to_csv(os.path.join(OUT, "spec_performance.csv"), index=False)
pd.concat(fold_rows_all).to_csv(os.path.join(OUT, "spec_performance_by_fold.csv"), index=False)
rep_all = pd.concat(rep_rows_all)
rep_all.to_csv(os.path.join(OUT, "spec_performance_repeated_5x5.csv"), index=False)
print(perf[["spec", "n_features", "legacy_R2", "grouped_R2_mean", "grouped_pooled_oof_R2", "rep5x5_R2_mean"]].to_string(index=False))

# S2 vs S3 identical by construction -> verify identical outputs
s2s3_identical = bool(np.array_equal(oof_store["S2"]["g"]["oof"], oof_store["S3"]["g"]["oof"])
                      and np.array_equal(oof_store["S2"]["g"]["shap"], oof_store["S3"]["g"]["shap"]))

# Paired comparisons on repeated 5x5 (Nadeau-Bengio corrected t-test), R2 differences vs S4
ttest_rows = []
r4 = rep_all[rep_all.spec == "S4"].sort_values(["repeat", "fold"])
for key in ["S1", "S2", "S3"]:
    rk_ = rep_all[rep_all.spec == key].sort_values(["repeat", "fold"])
    d = rk_["R2"].values - r4["R2"].values
    t, p = common.corrected_resampled_ttest(d, int(r4["n_train"].mean()), int(r4["n_test"].mean()))
    ttest_rows.append(dict(comparison=f"{key} - S4", metric="R2", mean_diff=d.mean(), sd_diff=d.std(ddof=1),
                           J=len(d), t_corrected=t, p_corrected=p))
pd.DataFrame(ttest_rows).to_csv(os.path.join(OUT, "spec_r2_difference_tests.csv"), index=False)

# Grouped (block) importance per spec
blocks_rows = []
for key, st in oof_store.items():
    feats = st["feats"]
    blocks = {"Sector block (sector_code + 8 dummies)": SECTOR_BLOCK, "Macro block (4 variables)": MACRO}
    for f_ in feats:
        if f_ not in SECTOR_BLOCK and f_ not in MACRO:
            blocks[f_] = [f_]
    debt_dep = [f_ for f_ in feats if common.AUDIT_CLASS[f_] != "clean"]
    if debt_dep:
        blocks["All debt-dependent features combined"] = debt_dep
    bi = block_importance(st["g"]["shap"], feats, blocks)
    for b, v in bi.items():
        blocks_rows.append(dict(spec=key, block=b, mean_abs_block_shap_oof=v))
blocks_df = pd.DataFrame(blocks_rows)
blocks_df.to_csv(os.path.join(OUT, "shap_block_importance.csv"), index=False)

# Top-8 comparison table across specs
top8 = []
for key, rk in rankings.items():
    for _, r in rk.head(8).iterrows():
        top8.append(dict(spec=key, rank=r["rank_oof"], feature=r["feature"], audit_class=r["audit_class"],
                         mean_abs_shap_oof=r["mean_abs_shap_oof"]))
pd.DataFrame(top8).to_csv(os.path.join(OUT, "top8_by_spec.csv"), index=False)

# ---------------------------------------------------------------------------
# (c) stability of S4
# ---------------------------------------------------------------------------
print(f"[{time.time()-T0:6.1f}s] stability across folds and panel seeds (S4)")
feats4 = list(common.FEATURES_STRICT)
fold_imp_42 = oof_store["S4"]["g"]["fold_imp"]
seed_imp, seed_fold_imp, seed_perf = [], [], []
for s in PANEL_SEEDS:
    dfs = df if s == SEED else common.generate_panel(seed=s)
    g = oof_store["S4"]["g"] if s == SEED else grouped_oof(dfs, feats4, seed=SEED)
    seed_imp.append(np.abs(g["shap"]).mean(axis=0))
    seed_fold_imp.extend(list(g["fold_imp"]))
    pm = metrics(dfs[TARGET].values, g["oof"])
    seed_perf.append(dict(panel_seed=s, grouped_R2_mean=g["folds"]["R2"].mean(), grouped_R2_sd=g["folds"]["R2"].std(ddof=1),
                          pooled_oof_R2=pm["R2"], pooled_oof_RMSE=pm["RMSE"], pooled_oof_MAE=pm["MAE"]))
seed_imp = np.array(seed_imp)
seed_fold_imp = np.array(seed_fold_imp)

imp_by_seed = pd.DataFrame(seed_imp.T, columns=[f"seed{s}" for s in PANEL_SEEDS])
imp_by_seed.insert(0, "feature", feats4)
rank_by_seed = imp_by_seed.copy()
for c in rank_by_seed.columns[1:]:
    rank_by_seed[c] = imp_by_seed[c].rank(ascending=False, method="min").astype(int)
imp_by_seed.to_csv(os.path.join(OUT, "stability_S4_importance_by_seed.csv"), index=False)
pd.DataFrame(seed_perf).to_csv(os.path.join(OUT, "stability_S4_performance_by_seed.csv"), index=False)

stab = pd.DataFrame({"feature": feats4, "display_name": [disp(x) for x in feats4]})
fold_ranks = np.array([pd.Series(v).rank(ascending=False, method="min").values for v in fold_imp_42])
seed_ranks = np.array([pd.Series(v).rank(ascending=False, method="min").values for v in seed_imp])
sf_ranks = np.array([pd.Series(v).rank(ascending=False, method="min").values for v in seed_fold_imp])
stab["rank_seed42_oof"] = pd.Series(seed_imp[0]).rank(ascending=False, method="min").astype(int).values
for k in range(5):
    stab[f"rank_fold{k}"] = fold_ranks[k].astype(int)
for i, s in enumerate(PANEL_SEEDS):
    stab[f"rank_seed{s}"] = seed_ranks[i].astype(int)
stab["top5_count_folds_of5"] = (fold_ranks <= 5).sum(axis=0)
stab["top5_count_seeds_of5"] = (seed_ranks <= 5).sum(axis=0)
stab["top5_count_seedfold_of25"] = (sf_ranks <= 5).sum(axis=0)
stab["rank_min_seedfold"] = sf_ranks.min(axis=0).astype(int)
stab["rank_max_seedfold"] = sf_ranks.max(axis=0).astype(int)
stab["mean_abs_shap_mean_over_seeds"] = seed_imp.mean(axis=0)
stab["mean_abs_shap_sd_over_seeds"] = seed_imp.std(axis=0, ddof=1)
stab = stab.sort_values("rank_seed42_oof").reset_index(drop=True)
stab.to_csv(os.path.join(OUT, "stability_S4_ranks.csv"), index=False)


def jacc(sets):
    return np.array([len(a & b) / len(a | b) for a, b in combinations(sets, 2)])


sp_f, sp_s, sp_sf = pairwise_spearman(fold_imp_42), pairwise_spearman(seed_imp), pairwise_spearman(seed_fold_imp)
top5_f, top5_s = topk_sets(fold_imp_42, feats4), topk_sets(seed_imp, feats4)
stab_summary = pd.DataFrame([
    dict(comparison="5 folds, panel seed 42", n_vectors=5, n_pairs=len(sp_f), spearman_mean=sp_f.mean(),
         spearman_min=sp_f.min(), spearman_max=sp_f.max(), top5_jaccard_mean=jacc(top5_f).mean(),
         top5_identical_all=len(set.intersection(*top5_f)) == 5),
    dict(comparison="5 panel seeds (OOF importance)", n_vectors=5, n_pairs=len(sp_s), spearman_mean=sp_s.mean(),
         spearman_min=sp_s.min(), spearman_max=sp_s.max(), top5_jaccard_mean=jacc(top5_s).mean(),
         top5_identical_all=len(set.intersection(*top5_s)) == 5),
    dict(comparison="25 seed x fold vectors", n_vectors=25, n_pairs=len(sp_sf), spearman_mean=sp_sf.mean(),
         spearman_min=sp_sf.min(), spearman_max=sp_sf.max(),
         top5_jaccard_mean=jacc(topk_sets(seed_fold_imp, feats4)).mean(),
         top5_identical_all=len(set.intersection(*topk_sets(seed_fold_imp, feats4))) == 5),
])
stab_summary.to_csv(os.path.join(OUT, "stability_S4_summary.csv"), index=False)
print(stab_summary.to_string(index=False))

# ---------------------------------------------------------------------------
# (d) reproduce the original RF-surrogate SHAP (S1) and compare with LightGBM SHAP
# ---------------------------------------------------------------------------
print(f"[{time.time()-T0:6.1f}s] RF surrogate reproduction")
feats1 = list(common.FEATURES_ORIGINAL)
v = df[feats1 + [TARGET]].dropna()
X1, y1 = v[feats1], v[TARGET]
rf = RandomForestRegressor(50, max_depth=8, random_state=42, n_jobs=N_JOBS).fit(X1, y1)
Xs = X1.sample(min(500, len(X1)), random_state=42)
sv_rf = np.asarray(shap.TreeExplainer(rf).shap_values(Xs))
imp_rf = np.abs(sv_rf).mean(axis=0)
orig_shap = pd.read_excel(os.path.join(common.ROOT, "outputs_original", "GenFin_AI_Capital_Structure_Analysis_ENHANCED.xlsx"),
                          sheet_name="SHAP").set_index("Feature")["Importance"]
# same protocol, LightGBM (fit on all rows, explain the same 500 rows; in-sample, like the original)
lgb_full = fit_lgbm(X1.astype(float), y1.values)
sv_lgb_ins = np.asarray(shap.TreeExplainer(lgb_full).shap_values(Xs.astype(float)))
imp_lgb_ins = np.abs(sv_lgb_ins).mean(axis=0)
imp_lgb_oof = rankings["S1"].set_index("feature").loc[feats1, "mean_abs_shap_oof"].values

cmp = pd.DataFrame({"feature": feats1, "audit_class": [common.AUDIT_CLASS[x] for x in feats1],
                    "rf_surrogate_reproduced": imp_rf,
                    "rf_surrogate_original_workbook": orig_shap.reindex(feats1).values,
                    "lgbm_same_protocol_insample": imp_lgb_ins,
                    "lgbm_oof_grouped": imp_lgb_oof})
for c in ["rf_surrogate_reproduced", "lgbm_same_protocol_insample", "lgbm_oof_grouped"]:
    cmp[f"rank_{c}"] = cmp[c].rank(ascending=False, method="min").astype(int)
cmp = cmp.sort_values("rank_rf_surrogate_reproduced").reset_index(drop=True)
cmp.to_csv(os.path.join(OUT, "rf_surrogate_vs_lgbm_S1.csv"), index=False)
rf_repro_maxdiff = float(np.max(np.abs(cmp["rf_surrogate_reproduced"] - cmp["rf_surrogate_original_workbook"])))


def agree(a, b, name):
    top5a = set(np.array(feats1)[np.argsort(-a)[:5]])
    top5b = set(np.array(feats1)[np.argsort(-b)[:5]])
    return dict(comparison=name, spearman=stats.spearmanr(a, b).correlation, kendall_tau=stats.kendalltau(a, b).correlation,
                top5_overlap=len(top5a & top5b), same_top1=bool(np.argmax(a) == np.argmax(b)),
                top1_a=feats1[int(np.argmax(a))], top1_b=feats1[int(np.argmax(b))])


agree_df = pd.DataFrame([
    agree(imp_rf, imp_lgb_ins, "RF surrogate vs LightGBM (same in-sample protocol, 500 rows)"),
    agree(imp_rf, imp_lgb_oof, "RF surrogate vs LightGBM (out-of-fold, grouped 5-fold)"),
    agree(imp_lgb_ins, imp_lgb_oof, "LightGBM in-sample vs LightGBM out-of-fold"),
])
agree_df.to_csv(os.path.join(OUT, "rf_surrogate_rank_agreement.csv"), index=False)
rf_r2_ins = r2_score(y1, rf.predict(X1))
print(agree_df.to_string(index=False))

# ---------------------------------------------------------------------------
# (e) provenance checks
# ---------------------------------------------------------------------------
print(f"[{time.time()-T0:6.1f}s] provenance checks")
prov = {}
prov["capital_structure_efficiency_in_FEATURES_ORIGINAL"] = "capital_structure_efficiency" in common.FEATURES_ORIGINAL
prov["capital_structure_efficiency_position_in_feature_list"] = common.FEATURES_ORIGINAL.index("capital_structure_efficiency") + 1
prov["original_workbook_rank_of_cse"] = int(orig_shap.rank(ascending=False)["capital_structure_efficiency"])
prov["original_workbook_mean_abs_shap_cse"] = float(orig_shap["capital_structure_efficiency"])
prov["reproduced_rf_mean_abs_shap_cse"] = float(cmp.set_index("feature").loc["capital_structure_efficiency", "rf_surrogate_reproduced"])
prov["corr_cse_debt_ratio_synthetic"] = float(np.corrcoef(df["capital_structure_efficiency"], df[TARGET])[0, 1])
prov["rf_surrogate_max_abs_diff_vs_workbook"] = rf_repro_maxdiff
prov["rf_surrogate_insample_R2_full_panel"] = float(rf_r2_ins)
prov["rf_surrogate_shap_rows_explained"] = int(len(Xs))
prov["original_shap_background"] = "none: shap.TreeExplainer(rf) default (tree_path_dependent); no background sample was passed"

# Source of paper Table XV: re-run the original code path (GenFin_Hybrid/reporting/implement_future_work.py, item 4)
real = pd.read_csv(os.path.join(common.REAL_DATA_DIR, "fundamentals_2022_2025.csv"))
prov["real_csv_default_risk_score_equals_clip_0.01+0.15*debt_ratio_maxabsdiff"] = float(
    np.nanmax(np.abs((0.01 + 0.15 * real["debt_ratio"]).clip(0, 0.5) - real["default_risk_score"])))
fa = ["debt_ratio", "total_assets", "total_debt", "total_equity", "roa", "roe", "interest_coverage", "wacc",
      "default_risk_score", "altman_z_score", "tobin_q", "cash_ratio"]
d = real[fa].dropna().copy()
for c in ["total_assets", "total_debt", "total_equity"]:
    d[c] = np.log1p(d[c])
d = d[np.isfinite(d).all(axis=1)]
fn = [c for c in fa if c != "debt_ratio"]
Xr = StandardScaler().fit_transform(d[fn].values)
fn_noz = [c for c in fn if c != "altman_z_score"]
Xr_noz = Xr[:, [i for i, c in enumerate(fn) if c != "altman_z_score"]]
from sklearn.inspection import permutation_importance  # noqa: E402
rf_r = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=N_JOBS).fit(Xr_noz, d["debt_ratio"].values)
pi = permutation_importance(rf_r, Xr_noz, d["debt_ratio"].values, n_repeats=10, random_state=42, scoring="r2")
t15 = pd.DataFrame({"feature": fn_noz, "permutation_importance_reproduced": pi.importances_mean}) \
    .sort_values("permutation_importance_reproduced", ascending=False)
orig_t15 = pd.read_csv(os.path.join(common.ROOT, "outputs_original", "future_work", "permutation_importance_no_z.csv")).set_index("Feature")["Importance"]
t15["permutation_importance_original_file"] = orig_t15.reindex(t15["feature"]).values
t15.to_csv(os.path.join(OUT, "table_XV_provenance_reproduction.csv"), index=False)
prov["tableXV_n_rows_real"] = int(len(d))
prov["tableXV_reproduction_max_abs_diff"] = float(np.nanmax(np.abs(t15["permutation_importance_reproduced"] - t15["permutation_importance_original_file"])))
prov["tableXV_insample_R2_rf_no_z"] = float(r2_score(d["debt_ratio"], rf_r.predict(Xr_noz)))
with open(os.path.join(OUT, "provenance_checks.json"), "w") as fh:
    json.dump(prov, fh, indent=2)
print(json.dumps(prov, indent=2))

# ---------------------------------------------------------------------------
# (f) figures
# ---------------------------------------------------------------------------
print(f"[{time.time()-T0:6.1f}s] figures")
plt = common.set_style()
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CLASS_COL = {"clean": common.OKABE_ITO[4], "direct": common.OKABE_ITO[5], "indirect": common.OKABE_ITO[0]}
sv4 = oof_store["S4"]["g"]["shap"]
X4 = df[feats4].astype(float)

# F1 beeswarm S4
plt.figure()
shap.summary_plot(sv4, X4.values, feature_names=[disp(f) for f in feats4], show=False, max_display=len(feats4),
                  plot_size=(common.FIG_W_SINGLE, 4.2), cmap=plt.get_cmap("viridis"), color_bar_label="Feature value (low to high)",
                  alpha=0.6)
fig = plt.gcf()
for ax in fig.axes:
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    for t in ax.texts:
        t.set_fontsize(8)
main_ax = fig.axes[0]
main_ax.set_xlabel("SHAP value (effect on predicted debt ratio)")
if len(fig.axes) > 1:
    fig.axes[-1].set_ylabel("Feature value (low to high)", fontsize=8)
    fig.axes[-1].tick_params(labelsize=8)
fig.patch.set_facecolor("white")
common.savefig(fig, os.path.join(FIG, "fig_e3_S4_beeswarm"))
plt.close(fig)

# F2 bar mean |SHAP| S4 with fold s.d.
rk4 = rankings["S4"].iloc[::-1]
fig, ax = plt.subplots(figsize=(common.FIG_W_SINGLE, 3.6))
ax.barh([disp(f) for f in rk4["feature"]], rk4["mean_abs_shap_oof"], xerr=rk4["mean_abs_shap_oof_fold_sd"],
        color=common.OKABE_ITO[4], error_kw=dict(lw=0.8, capsize=2, ecolor="#333333"))
ax.set_xlabel("Mean |SHAP value| (debt ratio units)")
ax.tick_params(axis="y", length=0)
common.savefig(fig, os.path.join(FIG, "fig_e3_S4_bar"))
plt.close(fig)

# F3 four-panel comparison of top-8 features across S1-S4
fig, axes = plt.subplots(2, 2, figsize=(common.FIG_W_DOUBLE, 5.4))
for ax, (key, letter) in zip(axes.ravel(), zip(["S1", "S2", "S3", "S4"], "abcd")):
    rk = rankings[key].head(8).iloc[::-1]
    pr = perf.set_index("spec").loc[key]
    ax.barh([disp(f) for f in rk["feature"]], rk["mean_abs_shap_oof"], color=[CLASS_COL[c] for c in rk["audit_class"]])
    ax.set_xlabel("Mean |SHAP value| (debt ratio units)")
    ax.tick_params(axis="y", length=0)
    ax.text(0.98, 0.04, f"({letter}) {SPECS[key]['label'].split(' ', 1)[0]}\n"
            f"grouped-CV $R^2$ = {pr['grouped_R2_mean']:.3f}\nrandom-split $R^2$ = {pr['legacy_R2']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
handles = [Patch(color=CLASS_COL[c], label=f"{c} (leakage audit)") for c in ["direct", "indirect", "clean"]]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.04, 1, 1))
common.savefig(fig, os.path.join(FIG, "fig_e3_top8_S1_S4"))
plt.close(fig)

# F4 dependence plots for top-2 S4 features (colour = sector); F4b same for the top-2 non-sector features
rng = np.random.RandomState(SEED)
sectors_sorted = sorted(common.SECTORS)


def dependence_fig(feat_list, name):
    fig, axes = plt.subplots(1, 2, figsize=(common.FIG_W_DOUBLE, 2.9))
    for ax, feat in zip(axes, feat_list):
        j = feats4.index(feat)
        xv = X4[feat].values.copy()
        nunique = len(np.unique(xv))
        if nunique <= 10:  # discrete feature: horizontal jitter for visibility
            xv = xv + rng.uniform(-0.15, 0.15, len(xv)) * (np.ptp(X4[feat].values) / max(nunique - 1, 1))
        for si, s_ in enumerate(sectors_sorted):
            msk = df["sector"].values == s_
            ax.scatter(xv[msk], sv4[msk, j], s=3, alpha=0.5, color=common.OKABE_ITO[si], label=s_, rasterized=True)
        ax.axhline(0, color="#777777", lw=0.6)
        ax.set_ylabel(f"SHAP value for {disp(feat)}" + chr(10) + "(debt ratio units)")
        if feat == "sector_code":
            ax.set_xticks(range(8))
            ax.set_xticklabels(sectors_sorted, rotation=45, ha="right")
            ax.set_xlabel("Sector code (alphabetical)")
        elif feat.startswith("sector_"):
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["0 (other sectors)", "1 (" + feat.split("_", 1)[1] + ")"])
            ax.set_xlabel(disp(feat) + " dummy")
        elif feat == "size_code":
            ax.set_xticks(sorted(np.unique(X4[feat].values)))
            ax.set_xlabel("Size tertile (sample tertile of total assets)")
        elif feat in ("roa", "cash_ratio"):
            ax.set_xlabel(disp(feat) + " (ratio to total assets)")
        elif feat in MACRO:
            ax.set_xlabel(disp(feat) + " (%)")
        else:
            ax.set_xlabel(disp(feat))
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, markerscale=3, fontsize=8)
    fig.tight_layout()
    common.savefig(fig, os.path.join(FIG, name))
    plt.close(fig)


top2 = rankings["S4"]["feature"].head(2).tolist()
dependence_fig(top2, "fig_e3_S4_dependence_top2")
top2_nonsector = [f for f in rankings["S4"]["feature"] if f not in SECTOR_BLOCK][:2]
dependence_fig(top2_nonsector, "fig_e3_S4_dependence_top2_nonsector")

# F5 rank stability S4 across seeds and folds
fig, ax = plt.subplots(figsize=(common.FIG_W_SINGLE, 3.6))
order = stab["feature"].tolist()[::-1]
ypos = {f: i for i, f in enumerate(order)}
fi_idx = {f: feats4.index(f) for f in order}
for q in range(sf_ranks.shape[0]):
    ax.scatter([sf_ranks[q, fi_idx[f]] for f in order], [ypos[f] + rng.uniform(-0.15, 0.15) for f in order],
               s=6, color=common.OKABE_ITO[1], alpha=0.6, lw=0)
for i in range(len(PANEL_SEEDS)):
    ax.scatter([seed_ranks[i, fi_idx[f]] for f in order], [ypos[f] for f in order], s=14, marker="D",
               facecolor="none", edgecolor=common.OKABE_ITO[5], lw=0.8)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([disp(f) for f in order])
ax.set_xlabel("Rank by mean |SHAP value| (1 = most important)")
ax.set_xticks([1, 4, 8, 12, 16])
ax.invert_xaxis()
ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=3, color=common.OKABE_ITO[1], label="fold (25)"),
                   plt.Line2D([], [], ls="", marker="D", ms=4, mfc="none", mec=common.OKABE_ITO[5], label="panel seed, OOF (5)")],
          loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=8)
common.savefig(fig, os.path.join(FIG, "fig_e3_S4_rank_stability"))
plt.close(fig)

# ---------------------------------------------------------------------------
summary = dict(
    runtime_minutes=(time.time() - T0) / 60,
    s2_s3_outputs_identical=s2s3_identical,
    legacy_reproduction=repro.to_dict(orient="records"),
    performance=perf.drop(columns=["label"]).to_dict(orient="records"),
    top5={k: rankings[k]["feature"].head(5).tolist() for k in rankings},
    stability=stab_summary.to_dict(orient="records"),
    rf_vs_lgbm=agree_df.to_dict(orient="records"),
    provenance=prov,
    versions=dict(shap=shap.__version__, lightgbm=__import__("lightgbm").__version__,
                  sklearn=__import__("sklearn").__version__, numpy=np.__version__, pandas=pd.__version__),
)
with open(os.path.join(OUT, "e3_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"done in {(time.time()-T0)/60:.2f} min")
