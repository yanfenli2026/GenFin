"""
Tests for common.py and generation of foundation outputs.
Run:  python test_common.py
Writes to output/foundation/ (or $GENFIN_OUT/foundation).
Section 5 (46-firm Yahoo Finance panel) is skipped when the raw statement CSVs are absent,
unless GENFIN_REQUIRE_REAL=1.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = C.OUT_FOUNDATION
os.makedirs(OUT, exist_ok=True)
results = {}


# ---------------------------------------------------------------------------
# Verbatim copy of run_pipeline_enhanced.generate_dataset() + get_features() (reference oracle)
# ---------------------------------------------------------------------------
def original_generate_dataset():
    from sklearn.preprocessing import LabelEncoder
    N_FIRMS, N_YEARS = 300, 15
    SECTORS = C.SECTORS; SECTOR_CAPS = C.SECTOR_CAPS
    np.random.seed(42)
    years = list(range(2010, 2010+N_YEARS))
    n = len(years)
    t = np.arange(n)
    gdp = 2.2 + 0.5*np.sin(2*np.pi*t/7) + 0.3*np.sin(2*np.pi*t/3.5) + np.random.normal(0,0.6,n)
    infl = 2.0 + 0.3*gdp + 0.1*np.cumsum(np.random.normal(0,0.2,n)) + np.random.normal(0,0.4,n)
    infl = np.clip(infl,0.2,7)
    rate = 2.5 + 0.5*(infl-2) - 0.3*(gdp-2) + np.random.normal(0,0.3,n)
    rate = np.clip(rate,0.1,6)
    spread = 1.5 - 0.3*(gdp-2) + 0.2*np.abs(rate-2.5) + np.random.normal(0,0.25,n)
    spread = np.clip(spread,0.5,5)
    macro = pd.DataFrame({"year":years,"gdp_growth":gdp,"inflation":infl,
                          "interest_rate":rate,"credit_spread":spread})
    records = []; fid = 0
    for sector in SECTORS:
        cap = SECTOR_CAPS[sector]; base_dr = cap*0.45
        sec_prof = {"Tech":0.08,"Healthcare":0.07,"Financials":0.04,"ConsDisc":0.06,
                    "Industrials":0.05,"Energy":0.05,"Utilities":0.03,"Materials":0.04}
        sec_growth = {"Tech":0.10,"Healthcare":0.07,"Financials":0.04,"ConsDisc":0.06,
                      "Industrials":0.05,"Energy":0.04,"Utilities":0.02,"Materials":0.03}
        for _ in range(N_FIRMS//len(SECTORS)):
            fid += 1
            sz = np.exp(np.random.normal(7.5,1.9))
            prof = np.random.normal(sec_prof[sector],0.08)
            risk = np.random.beta(2,5)
            growth = np.random.normal(sec_growth[sector],0.10)
            dr = np.clip(base_dr+np.random.normal(0,0.1),0.02,cap)
            for yi, yr in enumerate(years):
                m = macro.iloc[yi]
                rev_g = growth+0.5*m["gdp_growth"]/100+np.random.normal(0,0.04)
                rev = sz*np.exp(rev_g*yi)*np.random.uniform(0.35,1.6)
                roa = prof+0.02*m["gdp_growth"]/100+np.random.normal(0,0.02)
                ni = sz*roa*np.exp(rev_g*yi); ebit = ni*1.7
                target_dr = dr-0.03*(m["interest_rate"]-2.5)/100-0.02*m["credit_spread"]/1.0
                target_dr = np.clip(target_dr,0.02,cap)
                cur_dr = target_dr+np.random.normal(0,0.03) if yi==0 else \
                         dr_prev+0.3*(target_dr-dr_prev)+np.random.normal(0,0.02)
                cur_dr = np.clip(cur_dr,0.02,cap); dr_prev = cur_dr
                debt = sz*cur_dr; eq = sz-debt
                int_exp = debt*max(m["interest_rate"]/100+m["credit_spread"]/100*risk*2,0.005)
                cash = sz*np.clip(0.08+0.02*risk,0.01,0.4)*np.random.uniform(0.8,1.2)
                mcap = eq*np.clip(np.exp(np.random.normal(0.8,0.7)),0.5,9)
                sret = np.random.normal(0.08,0.25)+0.5*m["gdp_growth"]/100
                records.append({"firm_id":fid,"year":yr,"sector":sector,
                    "total_assets":sz,"total_debt":debt,"total_equity":eq,
                    "revenue":rev,"ebit":max(ebit,0),"net_income":ni,
                    "interest_expense":max(int_exp,0.001*debt),"cash_holdings":cash,
                    "market_cap":mcap,"stock_return":sret,
                    "gdp_growth":m["gdp_growth"],"inflation":m["inflation"],
                    "interest_rate":m["interest_rate"],"credit_spread":m["credit_spread"]})
    df = pd.DataFrame(records)
    df["debt_ratio"] = df["total_debt"]/df["total_assets"]
    df["debt_to_equity"] = df["total_debt"]/df["total_equity"].clip(1e-6)
    df["equity_ratio"] = df["total_equity"]/df["total_assets"]
    df["roa"] = df["net_income"]/df["total_assets"]
    df["roe"] = df["net_income"]/df["total_equity"].clip(1e-6)
    df["interest_coverage"] = df["ebit"]/df["interest_expense"].clip(1e-6)
    df["cash_ratio"] = df["cash_holdings"]/df["total_assets"]
    df["market_leverage"] = df["total_debt"]/(df["total_debt"]+df["market_cap"]).clip(1e-6)
    df["tobin_q"] = (df["market_cap"]+df["total_debt"])/df["total_assets"]
    rf = df["interest_rate"]/100; beta = np.clip(1.0+0.3*df["debt_ratio"]*3,0.5,3)
    Ke = rf+beta*0.055; Kd = (rf+df["credit_spread"]/100)*0.79
    ew = df["total_equity"]/(df["total_debt"]+df["total_equity"]); dw = 1-ew
    df["wacc"] = (ew*Ke+dw*Kd).clip(0.02,0.25)
    z = (df["roa"]-df["roa"].mean())/df["roa"].std()
    df["default_risk_score"] = np.clip(
        1/(1+np.exp(-(-2.5-0.8*z-0.5*df["gdp_growth"]/100+0.3*df["credit_spread"]))),0.001,0.999)
    df["altman_zscore"] = (1.2*(df["total_assets"]-df["total_debt"]-df["cash_holdings"])/df["total_assets"]
        +1.4*df["net_income"]/df["total_assets"]+3.3*df["ebit"]/df["total_assets"]
        +0.6*df["market_cap"]/df["total_debt"].clip(1e-6)+1.0*df["revenue"]/df["total_assets"]).clip(-10,20)
    df["capital_structure_efficiency"] = np.clip(
        0.3/(1+df["wacc"]*10)+0.2*df["roe"].clip(-1,2)/2
        +0.25*(1-df["default_risk_score"])+0.1*df["cash_ratio"]+0.15*df["tobin_q"]/5,0,1)
    le = LabelEncoder(); df["sector_code"] = le.fit_transform(df["sector"])
    df["size_code"] = pd.qcut(df["total_assets"],3,labels=[0,1,2],duplicates="drop").astype(int)
    for c in ["debt_ratio","debt_to_equity","roa","roe","interest_coverage","tobin_q","wacc"]:
        lo,hi = df[c].quantile([0.01,0.99]); df[c] = df[c].clip(lo,hi)
    fcols = ["roa","roe","interest_coverage","cash_ratio","tobin_q","altman_zscore",
             "capital_structure_efficiency","sector_code","size_code",
             "gdp_growth","inflation","interest_rate","credit_spread"]
    for sec in sorted(df["sector"].unique()):
        df[f"sector_{sec}"] = (df["sector"]==sec).astype(int)
        fcols.append(f"sector_{sec}")
    return df, fcols


# ---------------------------------------------------------------------------
# 1. Reproduction
# ---------------------------------------------------------------------------
df = C.generate_panel(seed=42)
assert len(df) == 4440, len(df)
assert df["firm_id"].nunique() == 296
assert df["year"].min() == 2010 and df["year"].max() == 2024
assert round(df["debt_ratio"].mean(), 4) == 0.2295, df["debt_ratio"].mean()
assert round(df["debt_ratio"].std(), 4) == 0.1120, df["debt_ratio"].std()
assert round(df["wacc"].mean(), 4) == 0.0788, df["wacc"].mean()

ref, ref_fcols = original_generate_dataset()
assert ref_fcols == C.FEATURES_ORIGINAL, (ref_fcols, C.FEATURES_ORIGINAL)
df_cmp = df.sort_values(["firm_id", "year"]).reset_index(drop=True)
ref = ref.sort_values(["firm_id", "year"]).reset_index(drop=True)
max_abs = {}
for c in ref.columns:
    if c == "sector":
        assert (ref[c] == df_cmp[c]).all()
        continue
    d = np.nanmax(np.abs(ref[c].astype(float).values - df_cmp[c].astype(float).values))
    max_abs[c] = float(d)
    assert d <= 1e-12 * max(1.0, np.nanmax(np.abs(ref[c].astype(float).values))), (c, d)
results["reproduction"] = {"n_rows": len(df), "n_firms": int(df.firm_id.nunique()),
                           "years": [int(df.year.min()), int(df.year.max())],
                           "debt_ratio_mean": float(df.debt_ratio.mean()), "debt_ratio_std": float(df.debt_ratio.std()),
                           "wacc_mean": float(df.wacc.mean()), "wacc_std": float(df.wacc.std()),
                           "max_abs_diff_vs_original_code": max(max_abs.values())}
print("[1] reproduction OK; max abs diff vs verbatim original =", max(max_abs.values()))

# determinism and parameter effect
assert C.generate_panel(seed=42).equals(df)
assert not C.generate_panel(seed=123)["debt_ratio"].equals(df["debt_ratio"])
d_lam = C.generate_panel(seed=42, lam=0.6)
assert len(d_lam) == 4440 and not np.allclose(d_lam.debt_ratio, df.debt_ratio)
assert np.allclose(d_lam.roa, df.roa)  # random stream unaffected by lam
d_w = C.generate_panel(seed=42, tax_rate=0.30, erp=0.06, beta_lev=1.2)
assert not np.allclose(d_w.wacc, df.wacc) and np.allclose(d_w.debt_ratio, df.debt_ratio)
try:
    C.generate_panel(seed=42, not_a_param=1); raise AssertionError("unknown param accepted")
except TypeError:
    pass
print("[1b] determinism / parameter checks OK")

# ---------------------------------------------------------------------------
# 2. Leakage audit: code-based classes verified empirically by perturbing DR*
#    (same random stream, winsorisation off so columns are compared raw)
# ---------------------------------------------------------------------------
base = C.generate_panel(seed=42, winsorize=False)
pert = C.generate_panel(seed=42, winsorize=False, _dr_perturb=0.05)
assert (base.debt_ratio != pert.debt_ratio).mean() > 0.9
rows = []
for d in C.FEATURE_AUDIT:
    f = d["feature"]
    a, b = base[f].astype(float), pert[f].astype(float)
    mask = a.notna() & b.notna()
    changed = float((~np.isclose(a[mask], b[mask], rtol=1e-12, atol=0)).mean())
    corr = float(np.corrcoef(base.loc[mask, f], base.loc[mask, "debt_ratio"])[0, 1]) if a[mask].std() > 0 else np.nan
    rows.append({"feature": f, "class": d["class"], "share_rows_changed_when_DR_shifted": changed,
                 "pearson_r_with_debt_ratio": corr})
    if d["class"] == "clean":
        assert changed == 0.0, (f, changed)
    else:
        assert changed > 0.5, (f, changed)
pc = pd.DataFrame(rows)
pc.to_csv(os.path.join(OUT, "leakage_perturbation_check.csv"), index=False)
print("[2] perturbation check agrees with audit for all", len(pc), "features")

# check identities claimed in the audit (unwinsorised panel)
b = base
assert np.allclose(b.equity_ratio, 1 - b.debt_ratio)
assert np.allclose(1 - b.roa / b.roe, b.debt_ratio)
assert np.allclose(b.debt_to_equity, b.debt_ratio / (1 - b.debt_ratio))
print("[2b] algebraic identities hold (equity_ratio = 1-DR; DR = 1 - roa/roe; D/E = DR/(1-DR))")

audit_path = C.write_feature_audit()
for fs_name, fs in C.FEATURE_SETS.items():
    assert all(f in df.columns for f in fs), fs_name
assert all(C.AUDIT_CLASS[f] == "clean" for f in C.FEATURES_STRICT_EXT)
assert "default_risk_score" not in C.FEATURES_ORIGINAL
assert C.FEATURES_NO_Z_DR == C.FEATURES_NO_Z
pd.DataFrame([{"set": k, "n": len(v), "features": ";".join(v)} for k, v in C.FEATURE_SETS.items()]).to_csv(
    os.path.join(OUT, "feature_sets.csv"), index=False)
results["feature_sets"] = {k: v for k, v in C.FEATURE_SETS.items()}

# ---------------------------------------------------------------------------
# 3. Splits
# ---------------------------------------------------------------------------
split_rows = []
tr, te = C.split_random(df, seed=42)
assert len(tr) == 3552 and len(te) == 888 and len(set(tr) & set(te)) == 0
# identical to original protocol (train_test_split on the feature frame)
from sklearn.model_selection import train_test_split
Xo = ref[ref_fcols + ["debt_ratio"]].dropna()
_, Xe, _, _ = train_test_split(Xo[ref_fcols], Xo["debt_ratio"], test_size=0.2, random_state=42)
assert np.array_equal(np.sort(Xe.index.values), np.sort(te))
tr2, te2 = C.split_random(df, seed=42)
assert np.array_equal(tr, tr2)
firm_overlap = len(set(df.firm_id.iloc[tr]) & set(df.firm_id.iloc[te]))
split_rows.append({"split": "random_80_20 (legacy)", "fold": "-", "n_train": len(tr), "n_test": len(te),
                   "firms_train": df.firm_id.iloc[tr].nunique(), "firms_test": df.firm_id.iloc[te].nunique(),
                   "firms_in_both": firm_overlap, "years_train": "2010-2024", "years_test": "2010-2024"})

folds = C.grouped_kfold(df, 5, seed=42)
all_te = np.concatenate([f[1] for f in folds])
assert len(all_te) == len(df) and len(np.unique(all_te)) == len(df)
for k, (a, bb) in enumerate(folds):
    assert len(set(df.firm_id.iloc[a]) & set(df.firm_id.iloc[bb])) == 0
    split_rows.append({"split": "grouped_kfold (seed 42)", "fold": k, "n_train": len(a), "n_test": len(bb),
                       "firms_train": df.firm_id.iloc[a].nunique(), "firms_test": df.firm_id.iloc[bb].nunique(),
                       "firms_in_both": 0, "years_train": "2010-2024", "years_test": "2010-2024"})
f2 = C.grouped_kfold(df, 5, seed=42)
assert all(np.array_equal(x[1], y[1]) for x, y in zip(folds, f2))
rep = list(C.repeated_grouped_kfold(df, 5, 5, seed=42))
assert len(rep) == 25
assert not np.array_equal(rep[0][3], rep[5][3])
assert np.array_equal(rep[0][3], folds[0][1])

tr, te = C.temporal_split(df)
assert set(df.year.iloc[tr]) == set(range(2010, 2021)) and set(df.year.iloc[te]) == {2021, 2022, 2023, 2024}
split_rows.append({"split": "temporal (train 2010-2020, test 2021-2024)", "fold": "-", "n_train": len(tr), "n_test": len(te),
                   "firms_train": df.firm_id.iloc[tr].nunique(), "firms_test": df.firm_id.iloc[te].nunique(),
                   "firms_in_both": len(set(df.firm_id.iloc[tr]) & set(df.firm_id.iloc[te])),
                   "years_train": "2010-2020", "years_test": "2021-2024"})
tr, te = C.unseen_firm_future_split(df)
assert len(set(df.firm_id.iloc[tr]) & set(df.firm_id.iloc[te])) == 0
assert df.year.iloc[tr].max() < df.year.iloc[te].min()
held_by_sector = df.iloc[te].groupby("sector").firm_id.nunique().to_dict()
split_rows.append({"split": "unseen-firm future (20% firms, 2021-2024)", "fold": "-", "n_train": len(tr), "n_test": len(te),
                   "firms_train": df.firm_id.iloc[tr].nunique(), "firms_test": df.firm_id.iloc[te].nunique(),
                   "firms_in_both": 0, "years_train": "2010-2020", "years_test": "2021-2024"})
sp = pd.DataFrame(split_rows)
sp.to_csv(os.path.join(OUT, "split_sizes.csv"), index=False)
results["splits"] = sp.to_dict(orient="records")
results["unseen_firm_held_out_by_sector"] = {k: int(v) for k, v in held_by_sector.items()}
print("[3] splits OK\n", sp.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Corrected resampled t-test
# ---------------------------------------------------------------------------
rng = np.random.RandomState(0)
d = rng.normal(0.01, 0.02, 25)
t, p = C.corrected_resampled_ttest(d, 3552, 888)
t_manual = d.mean() / np.sqrt((1 / 25 + 888 / 3552) * d.var(ddof=1))
from scipy import stats as st
t_naive, p_naive = st.ttest_1samp(d, 0)
assert np.isclose(t, t_manual) and abs(t) < abs(t_naive) and p > p_naive
assert C.corrected_resampled_ttest(np.zeros(5), 80, 20) == (0.0, 1.0)
results["ttest_check"] = {"t_corrected": t, "p_corrected": p, "t_naive": float(t_naive), "p_naive": float(p_naive)}
print(f"[4] corrected t-test OK: t={t:.3f} p={p:.4f} (naive t={t_naive:.3f} p={p_naive:.4f})")

# ---------------------------------------------------------------------------
# 5. Real panel
# ---------------------------------------------------------------------------
import config as _cfg
REQUIRE_REAL = os.environ.get("GENFIN_REQUIRE_REAL", "0") == "1"
if not _cfg.have_yahoo_data():
    if REQUIRE_REAL:
        raise SystemExit(f"Yahoo Finance statement CSVs not found in {_cfg.REAL_DATA_DIR}")
    print(f"[5] SKIPPED: real panel (no Yahoo Finance CSVs in {_cfg.REAL_DATA_DIR}; run data/download_recent_data.py or set GENFIN_YAHOO_DIR)")
    results["real_panel"] = "skipped: data not present"
else:
    real, log = C.load_real_panel(return_log=True)
    assert real.ticker.nunique() == 46, real.ticker.nunique()
    assert real.debt_ratio.between(0, 1.5).all()
    assert set(real.sector) <= set(C.SECTORS)
    assert not real.duplicated(["ticker", "year"]).any()
    real.to_csv(os.path.join(OUT, "real_panel.csv"), index=False)
    with open(os.path.join(OUT, "real_panel_load_log.txt"), "w") as fh:
        fh.write("\n".join(log))
    num = ["debt_ratio", "cash_ratio", "roa", "ebit_to_assets", "asset_turnover", "log_assets", "roe",
           "interest_coverage", "sales_growth", "book_equity_ratio"]
    desc = real[num].describe().T
    desc["n_missing"] = len(real) - desc["count"]
    desc.to_csv(os.path.join(OUT, "real_panel_descriptives.csv"))
    by_year = real.groupby("year").agg(firms=("ticker", "nunique"), debt_ratio_mean=("debt_ratio", "mean")).reset_index()
    by_year.to_csv(os.path.join(OUT, "real_panel_by_year.csv"), index=False)
    by_sector = real.groupby("sector").agg(firms=("ticker", "nunique"), firm_years=("ticker", "size"),
                                           debt_ratio_mean=("debt_ratio", "mean"),
                                           debt_ratio_sd=("debt_ratio", "std")).reset_index()
    by_sector.to_csv(os.path.join(OUT, "real_panel_by_sector.csv"), index=False)
    smap = pd.DataFrame([{"ticker": k, "gics_sector": v[0], "mapped_sector": v[1], "note": v[2]}
                         for k, v in sorted(C.REAL_SECTOR_MAP.items())])
    smap.to_csv(os.path.join(OUT, "real_sector_map.csv"), index=False)
    src_counts = {c: real[c].value_counts(dropna=False).to_dict() for c in real.columns if c.startswith("src_")}
    results["real_panel"] = {"n_firms": int(real.ticker.nunique()), "n_firm_years": len(real),
                             "years": [int(real.year.min()), int(real.year.max())],
                             "firm_years_by_year": dict(zip(by_year.year.astype(int), by_year.firms.astype(int))),
                             "debt_ratio_mean": float(real.debt_ratio.mean()), "debt_ratio_std": float(real.debt_ratio.std()),
                             "line_item_sources": {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in src_counts.items()},
                             "log": log}

    # align
    all_feats = list(dict.fromkeys(C.FEATURES_ORIGINAL + C.FEATURES_STRICT_EXT + ["sales_growth"]))
    usable, mapping = C.align_features(real, all_feats)
    mapping["class_in_synthetic_audit"] = mapping.feature.map(C.AUDIT_CLASS)
    mapping.to_csv(os.path.join(OUT, "real_feature_alignment.csv"), index=False)
    u_strict, _ = C.align_features(real, C.FEATURES_STRICT)
    u_ext, _ = C.align_features(real, C.FEATURES_STRICT_EXT)
    u_orig, _ = C.align_features(real, C.FEATURES_ORIGINAL)
    results["real_alignment"] = {"strict": u_strict, "strict_ext": u_ext, "original": u_orig}
    print(f"[5] real panel: {real.ticker.nunique()} firms, {len(real)} firm-years, {real.year.min()}-{real.year.max()}")
    print("    usable strict:", u_strict)
    print("    usable strict_ext:", u_ext)

# DGP parameter table
pd.DataFrame([{"parameter": k, "default": json.dumps(v) if isinstance(v, (dict, tuple, list)) else v}
              for k, v in C.DGP_DEFAULTS.items()]).to_csv(os.path.join(OUT, "dgp_parameters.csv"), index=False)
# synthetic descriptive check for strict features
df[["debt_ratio"] + C.FEATURES_STRICT_EXT].describe().T.to_csv(os.path.join(OUT, "synthetic_strict_descriptives.csv"))

# ---------------------------------------------------------------------------
# 6. Style
# ---------------------------------------------------------------------------
plt = C.set_style()
assert plt.rcParams["font.size"] == 9 and plt.rcParams["savefig.dpi"] == 300

with open(os.path.join(OUT, "test_results.json"), "w") as fh:
    json.dump(results, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("\nALL TESTS PASSED")

# ---------------------------------------------------------------------------
# 7. Variance decomposition of debt_ratio in the reference synthetic panel
#    (how much of the target is explainable from sector / year information alone)
# ---------------------------------------------------------------------------
def _r2_dummies(y, *cols):
    X = pd.get_dummies(df[list(cols)].astype(str), drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    resid = y.values - X.values @ beta
    return float(1 - resid.var() / y.values.var())
vd = {}
tot = df.debt_ratio.var(ddof=0)
vd["share_between_sector"] = df.groupby("sector").debt_ratio.transform("mean").var(ddof=0) / tot
vd["share_between_firm"] = df.groupby("firm_id").debt_ratio.transform("mean").var(ddof=0) / tot
vd["share_within_firm"] = 1 - vd["share_between_firm"]
vd["r2_ols_sector_dummies_insample"] = _r2_dummies(df.debt_ratio, "sector")
vd["r2_ols_sector_plus_year_insample"] = _r2_dummies(df.debt_ratio, "sector", "year")
vd["r2_ols_firm_fe_insample"] = _r2_dummies(df.debt_ratio, "firm_id")
pd.DataFrame([vd]).T.rename(columns={0: "value"}).to_csv(os.path.join(OUT, "synthetic_debt_ratio_variance_decomposition.csv"))
results["variance_decomposition"] = vd
print("[7] variance decomposition:", {k: round(v, 4) for k, v in vd.items()})
with open(os.path.join(OUT, "test_results.json"), "w") as fh:
    json.dump(results, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("ALL TESTS PASSED (incl. section 7)")
