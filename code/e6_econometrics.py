"""E6 - Panel econometrics for the debt_ratio regression.

Re-run:  python code/e6_econometrics.py
Outputs: output/e6_econometrics/

Models on the reference synthetic panel (seed 42, 4,440 firm-years, 296 firms, 2010-2024):
  M1 pooled OLS, HC1 SEs (original specification, cleaned of duplicated / collinear columns)
  M2 pooled OLS, SEs clustered by firm
  M3 firm FE (within), clustered by firm
  M4 firm + year FE, clustered by firm
  M5 firm FE, Driscoll-Kraay SEs
  M6 firm FE, clustered by firm, leakage-free regressors (common.FEATURES_STRICT, time-varying part)
  M7 pooled OLS, clustered by firm, leakage-free regressors (to separate the FE effect from the
     effect of dropping the debt-dependent regressors)
  M8 firm + year FE, clustered by firm, leakage-free regressors
Real 46-firm panel: pooled OLS (year + sector dummies, firm-clustered) and firm + year FE (firm-clustered).
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = os.path.join(common.OUT_MAIN, "e6_econometrics")
os.makedirs(OUT, exist_ok=True)
SEED = 42
np.random.seed(SEED)

MACRO = ["gdp_growth", "inflation", "interest_rate", "credit_spread"]
ORIG_TV = ["roa", "tobin_q", "interest_coverage"]           # time-varying firm regressors of the original model
SECTOR_REF = "ConsDisc"                                      # pd.get_dummies(drop_first=True) drops the first (alphabetical) level
SECTOR_DUMS = [f"sector_{s}" for s in sorted(common.SECTORS) if s != SECTOR_REF]
STRICT_TV = [f for f in common.FEATURES_STRICT if f in ("roa", "cash_ratio")]  # time-varying clean firm variables
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


# ---------------------------------------------------------------------------
# 0. Reproduce the original econometrics() call exactly and diagnose its design matrix
# ---------------------------------------------------------------------------
def reproduce_original(df_in):
    """Verbatim logic of econometrics() in run_pipeline_enhanced.py (pooled OLS part)."""
    df = df_in.copy()
    de = df.dropna(subset=["debt_ratio", "roa", "tobin_q", "gdp_growth"])
    de = pd.get_dummies(de, columns=["sector"], drop_first=True)
    y = de["debt_ratio"]
    xv = ["roa", "tobin_q", "interest_coverage", "gdp_growth", "inflation", "interest_rate", "credit_spread"]
    xv += [c for c in de.columns if c.startswith("sector_")]
    xv = [c for c in xv if c in de.columns]
    X = de[xv].fillna(0).astype(float)
    y = y.astype(float)
    X = sm.add_constant(X)
    ols = sm.OLS(y, X).fit(cov_type="HC1")
    return ols, X, xv


def design_diagnostics(X):
    Xv = X.values
    rank = np.linalg.matrix_rank(Xv)
    sv = np.linalg.svd(Xv, compute_uv=False)
    names = list(X.columns)
    dup_names = sorted({n for n in names if names.count(n) > 1})
    return {"n_columns": Xv.shape[1], "rank": int(rank), "rank_deficiency": int(Xv.shape[1] - rank),
            "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
            "smallest_singular_value": float(sv[-1]),
            "n_distinct_column_names": len(set(names)), "duplicated_column_names": ";".join(dup_names)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def tidy_sm(res, model, se_label):
    ci = res.conf_int()
    return pd.DataFrame({"model": model, "variable": res.params.index, "coef": res.params.values,
                         "se": res.bse.values, "t": res.tvalues.values, "p": res.pvalues.values,
                         "ci_low": ci.iloc[:, 0].values, "ci_high": ci.iloc[:, 1].values, "se_type": se_label})


def tidy_lm(res, model, se_label):
    ci = res.conf_int()
    return pd.DataFrame({"model": model, "variable": res.params.index, "coef": res.params.values,
                         "se": res.std_errors.values, "t": res.tstats.values, "p": res.pvalues.values,
                         "ci_low": ci.iloc[:, 0].values, "ci_high": ci.iloc[:, 1].values, "se_type": se_label})


def r2_from_lm(res):
    out = {"r2_within": float(res.rsquared_within), "r2_between": float(res.rsquared_between),
           "r2_overall": float(res.rsquared_overall), "r2_model": float(res.rsquared)}
    if hasattr(res, "rsquared_inclusive"):
        out["r2_inclusive"] = float(res.rsquared_inclusive)
    return out


def panel_index(df, entity="firm_id", time="year"):
    return df.set_index([entity, time]).sort_index()


def dk_bandwidth(T):
    """Newey-West rule of thumb floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def wooldridge_serial_test(df, y, xvars, entity="firm_id", time="year"):
    """Wooldridge (2002, sec. 10.6.3) test for AR(1) in idiosyncratic errors of a FE/linear panel model.
    Regress dy on dX (no constant), take residuals e, regress e_it on e_i,t-1 with SEs clustered by firm,
    and test H0: coefficient = -0.5."""
    d = df.sort_values([entity, time]).copy()
    g = d.groupby(entity)
    consecutive = (g[time].diff() == 1)
    dy = g[y].diff()
    dX = g[xvars].diff()
    m = consecutive & dy.notna() & dX.notna().all(axis=1)
    fd = sm.OLS(dy[m], dX[m]).fit()
    e = pd.Series(np.nan, index=d.index)
    e[m] = fd.resid
    d["_e"] = e
    d["_e_lag"] = d.groupby(entity)["_e"].shift(1)
    lag_ok = d.groupby(entity)[time].diff() == 1
    mm = d["_e"].notna() & d["_e_lag"].notna() & lag_ok
    r = sm.OLS(d.loc[mm, "_e"], d.loc[mm, ["_e_lag"]]).fit(cov_type="cluster",
                                                           cov_kwds={"groups": d.loc[mm, entity]})
    b, se = float(r.params.iloc[0]), float(r.bse.iloc[0])
    F = ((b + 0.5) / se) ** 2
    G = d.loc[mm, entity].nunique()
    p = float(stats.f.sf(F, 1, G - 1))
    return {"test": "Wooldridge AR(1) (FD residual regression, H0: rho_fd = -0.5)", "regressors": ";".join(xvars),
            "rho_fd": b, "se_cluster": se, "F": F, "df1": 1, "df2": G - 1, "p": p, "n_obs": int(mm.sum()),
            "implied_rho_levels_note": "rho_fd = -0.5 under no serial correlation in levels"}


def hausman(fe_res, re_res, vars_, sigmamore=False):
    """Classic Hausman test, non-robust (homoskedastic) covariance matrices of FE and RE.
    Uses the Moore-Penrose inverse of V_FE - V_RE; df = number of positive eigenvalues of that matrix.
    sigmamore=True rescales both covariance matrices to the RE estimate of the idiosyncratic error
    variance (as Stata's hausman, sigmamore), which makes V_FE - V_RE positive semi-definite in practice."""
    b = fe_res.params[vars_] - re_res.params[vars_]
    Vfe, Vre = fe_res.cov.loc[vars_, vars_], re_res.cov.loc[vars_, vars_]
    if sigmamore:
        sig_e2 = float(re_res.variance_decomposition["Residual"])
        Vfe = Vfe * sig_e2 / float(fe_res.s2)
        Vre = Vre * sig_e2 / float(re_res.s2)
    V = Vfe - Vre
    Vi = np.linalg.pinv(V.values)
    H = float(b.values @ Vi @ b.values)
    eig = np.linalg.eigvalsh(V.values)
    df_ = int(np.sum(eig > 1e-12 * max(abs(eig).max(), 1e-300)))
    return H, df_, float(stats.chi2.sf(H, df_)), bool((eig > 0).all())


def mundlak_test(df, y, tv_vars, other_vars, entity="firm_id", time="year", return_fit=False):
    """Cluster-robust Mundlak (correlated random effects) version of the Hausman test:
    pooled OLS of y on X, firm means of the time-varying X, other regressors; Wald test that the
    coefficients on the firm means are jointly zero, SEs clustered by firm."""
    d = df.copy()
    means = []
    for v in tv_vars:
        d[f"mean_{v}"] = d.groupby(entity)[v].transform("mean")
        means.append(f"mean_{v}")
    X = sm.add_constant(d[tv_vars + means + other_vars].astype(float))
    r = sm.OLS(d[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": d[entity]})
    R = np.zeros((len(means), X.shape[1]))
    for i, mname in enumerate(means):
        R[i, list(X.columns).index(mname)] = 1.0
    w = r.wald_test(R, scalar=True, use_f=True)
    res = (float(w.statistic), int(w.df_num), int(w.df_denom), float(w.pvalue))
    if return_fit:
        return res, r
    d_out = pd.DataFrame({"variable": r.params.index, "coef": r.params.values, "se": r.bse.values,
                          "p": r.pvalues.values})
    d_out.to_csv(os.path.join(OUT, f"mundlak_regression_{'_'.join(tv_vars)}.csv"), index=False)
    return res


def stars(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


def fmt_num(x, nd=4):
    if pd.isna(x):
        return ""
    if x != 0 and abs(x) < 10 ** (-nd):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def wide_table(coefs, stats_df, models, var_order, var_labels=None):
    """LaTeX-ready wide table: coefficient row (with stars) and SE row (in parentheses) per variable,
    then summary statistics rows."""
    var_labels = var_labels or {}
    rows = []
    for v in var_order:
        sub = coefs[coefs.variable == v]
        if sub.empty:
            continue
        r1 = {"row": var_labels.get(v, v)}
        r2 = {"row": ""}
        for m in models:
            s = sub[sub.model == m]
            if s.empty:
                r1[m], r2[m] = "", ""
            else:
                s = s.iloc[0]
                r1[m] = fmt_num(s.coef) + stars(s.p)
                r2[m] = "(" + fmt_num(s.se) + ")"
        rows += [r1, r2]
    st = stats_df.set_index("model")
    stat_rows = [("Firm effects", "firm_effects", None), ("Year effects", "year_effects", None),
                 ("Standard errors", "se_type", None), ("N (firm-years)", "n_obs", 0), ("Firms", "n_firms", 0),
                 ("R2 within", "r2_within", 3), ("R2 between", "r2_between", 3), ("R2 overall", "r2_overall", 3)]
    for lab, col, nd in stat_rows:
        r = {"row": lab}
        for m in models:
            val = st.at[m, col] if (m in st.index and col in st.columns) else ""
            if nd is None or val == "" or pd.isna(val):
                r[m] = "" if (isinstance(val, float) and pd.isna(val)) else str(val)
            elif nd == 0:
                r[m] = f"{int(val):,}"
            else:
                r[m] = f"{float(val):.{nd}f}"
        rows.append(r)
    return pd.DataFrame(rows)


def to_latex(wide, path, caption_note):
    cols = [c for c in wide.columns if c != "row"]
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("R2", r"$R^2$")
    lines = [r"% generated by e6_econometrics.py; " + caption_note,
             r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & ".join(["Variable"] + [esc(c) for c in cols]) + r" \\", r"\midrule"]
    first_stat = True
    for _, r in wide.iterrows():
        if r["row"] == "Firm effects" and first_stat:
            lines.append(r"\midrule")
            first_stat = False
        lines.append(" & ".join([esc(r["row"])] + [esc(r[c]) for c in cols]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def run_synthetic():
    df = common.generate_panel(seed=SEED, add_extra_features=False)
    df = df.sort_values(["firm_id", "year"]).reset_index(drop=True)
    log(f"Synthetic panel: {len(df)} rows, {df.firm_id.nunique()} firms, {df.year.min()}-{df.year.max()}")

    # --- 0. original reproduction and design diagnostics
    ols0, X0, xv0 = reproduce_original(df)
    diag0 = design_diagnostics(X0)
    diag0.update({"model": "original econometrics() design", "r2": ols0.rsquared, "adj_r2": ols0.rsquared_adj,
                  "dw": float(sm.stats.durbin_watson(ols0.resid)), "roa_coef": float(ols0.params["roa"].iloc[0]
                  if isinstance(ols0.params["roa"], pd.Series) else ols0.params["roa"]),
                  "column_list": ";".join(X0.columns)})
    orig_tab = tidy_sm(ols0, "original_reproduced", "HC1")
    orig_tab["duplicate_of_previous_row"] = orig_tab.duplicated(subset=["variable"], keep="first")
    orig_tab.to_csv(os.path.join(OUT, "original_ols_reproduction.csv"), index=False)
    log(f"Original reproduction: R2={ols0.rsquared:.4f} adjR2={ols0.rsquared_adj:.4f} "
        f"DW={diag0['dw']:.4f}; columns={diag0['n_columns']} rank={diag0['rank']} "
        f"cond={diag0['condition_number']:.3g}")

    # Collinearity facts: sector_code is an exact linear combination of the constant and the dummies
    S = df[[f"sector_{s}" for s in sorted(common.SECTORS)]].values
    codes = np.arange(len(common.SECTORS))
    max_dev_code = float(np.abs(S @ codes - df["sector_code"].values).max())
    max_dev_sum = float(np.abs(S.sum(axis=1) - 1).max())
    ic_na = int(df["interest_coverage"].isna().sum())

    # --- cleaned design: const + roa, tobin_q, interest_coverage + 4 macro + 7 sector dummies (ref ConsDisc)
    xs_clean = ORIG_TV + MACRO + SECTOR_DUMS
    Xc = sm.add_constant(df[xs_clean].astype(float))
    diag_c = design_diagnostics(Xc)
    m1 = sm.OLS(df["debt_ratio"], Xc).fit(cov_type="HC1")
    diag_c.update({"model": "cleaned design (M1)", "r2": m1.rsquared, "adj_r2": m1.rsquared_adj,
                   "dw": float(sm.stats.durbin_watson(m1.resid)), "roa_coef": float(m1.params["roa"]),
                   "column_list": ";".join(Xc.columns)})
    fitted_diff = float(np.abs(m1.fittedvalues.values - ols0.fittedvalues.values).max())
    diag = pd.DataFrame([diag0, diag_c])
    diag["max_abs_diff_fitted_vs_original"] = [0.0, fitted_diff]
    diag["sector_code_minus_dummy_combination_maxabs"] = max_dev_code
    diag["dummy_sum_minus_one_maxabs"] = max_dev_sum
    diag["interest_coverage_missing_rows"] = ic_na
    diag.to_csv(os.path.join(OUT, "design_matrix_diagnostics.csv"), index=False)
    log(f"Cleaned design: columns={diag_c['n_columns']} rank={diag_c['rank']} R2={m1.rsquared:.4f}; "
        f"max |fitted - original fitted| = {fitted_diff:.2e}")

    coefs, mstats = [], []
    pdf = panel_index(df)
    G = df.firm_id.nunique()
    T = df.year.nunique()

    def add_stats(model, res_lm, n, firm_fx, year_fx, se_type, extra=None):
        d = {"model": model, "n_obs": n, "n_firms": G, "firm_effects": firm_fx, "year_effects": year_fx,
             "se_type": se_type}
        d.update(r2_from_lm(res_lm))
        if extra:
            d.update(extra)
        mstats.append(d)

    # M1 pooled OLS HC1 ; M2 pooled OLS clustered by firm (statsmodels, CR1 small-sample correction)
    coefs.append(tidy_sm(m1, "M1", "HC1"))
    m2 = sm.OLS(df["debt_ratio"], Xc).fit(cov_type="cluster", cov_kwds={"groups": df["firm_id"]})
    coefs.append(tidy_sm(m2, "M2", "cluster(firm)"))
    pooled_lm = PooledOLS(pdf["debt_ratio"], sm.add_constant(pdf[xs_clean].astype(float))).fit()
    assert np.allclose(pooled_lm.params.values, m1.params.values, atol=1e-10)
    bp = sm.stats.diagnostic.het_breuschpagan(m1.resid, m1.model.exog)
    add_stats("M1", pooled_lm, len(df), "No", "No", "HC1",
              {"adj_r2": m1.rsquared_adj, "durbin_watson": float(sm.stats.durbin_watson(m1.resid)),
               "breusch_pagan_LM": float(bp[0]), "breusch_pagan_p": float(bp[1])})
    add_stats("M2", pooled_lm, len(df), "No", "No", "cluster(firm)",
              {"adj_r2": m2.rsquared_adj, "durbin_watson": float(sm.stats.durbin_watson(m2.resid))})

    # M3 firm FE, clustered; time-invariant sector dummies are absorbed and not included
    x_fe = ORIG_TV + MACRO
    m3 = PanelOLS(pdf["debt_ratio"], sm.add_constant(pdf[x_fe]), entity_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    coefs.append(tidy_lm(m3, "M3", "cluster(firm)"))
    add_stats("M3", m3, int(m3.nobs), "Yes", "No", "cluster(firm)",
              {"F_poolability": float(m3.f_pooled.stat), "F_poolability_p": float(m3.f_pooled.pval)})

    # M4 two-way FE (firm + year), clustered; year-constant macro variables are absorbed
    m4 = PanelOLS(pdf["debt_ratio"], pdf[ORIG_TV], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    coefs.append(tidy_lm(m4, "M4", "cluster(firm)"))
    add_stats("M4", m4, int(m4.nobs), "Yes", "Yes", "cluster(firm)",
              {"F_poolability": float(m4.f_pooled.stat), "F_poolability_p": float(m4.f_pooled.pval)})

    # M5 firm FE with Driscoll-Kraay SEs (Bartlett kernel)
    bw = dk_bandwidth(T)
    m5 = PanelOLS(pdf["debt_ratio"], sm.add_constant(pdf[x_fe]), entity_effects=True).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=bw)
    coefs.append(tidy_lm(m5, "M5", f"Driscoll-Kraay (Bartlett, bw={bw})"))
    add_stats("M5", m5, int(m5.nobs), "Yes", "No", f"Driscoll-Kraay (bw={bw})")

    # M6 firm FE, leakage-free regressors: time-varying part of FEATURES_STRICT
    #     (roa, cash_ratio, 4 macro).  sector_code, size_code and sector dummies are constant within firm.
    x_strict = STRICT_TV + MACRO
    m6 = PanelOLS(pdf["debt_ratio"], sm.add_constant(pdf[x_strict]), entity_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    coefs.append(tidy_lm(m6, "M6", "cluster(firm)"))
    add_stats("M6", m6, int(m6.nobs), "Yes", "No", "cluster(firm)",
              {"F_poolability": float(m6.f_pooled.stat), "F_poolability_p": float(m6.f_pooled.pval)})

    # M7 pooled OLS, leakage-free regressors (full FEATURES_STRICT less the collinear sector_code
    #     and with 7 sector dummies), clustered by firm
    x_strict_pooled = STRICT_TV + ["size_code"] + MACRO + SECTOR_DUMS
    Xs = sm.add_constant(df[x_strict_pooled].astype(float))
    m7 = sm.OLS(df["debt_ratio"], Xs).fit(cov_type="cluster", cov_kwds={"groups": df["firm_id"]})
    coefs.append(tidy_sm(m7, "M7", "cluster(firm)"))
    m7_lm = PooledOLS(pdf["debt_ratio"], sm.add_constant(pdf[x_strict_pooled].astype(float))).fit()
    add_stats("M7", m7_lm, len(df), "No", "No", "cluster(firm)",
              {"adj_r2": m7.rsquared_adj, "durbin_watson": float(sm.stats.durbin_watson(m7.resid))})

    # M8 two-way FE, leakage-free
    m8 = PanelOLS(pdf["debt_ratio"], pdf[STRICT_TV], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    coefs.append(tidy_lm(m8, "M8", "cluster(firm)"))
    add_stats("M8", m8, int(m8.nobs), "Yes", "Yes", "cluster(firm)")

    coefs = pd.concat(coefs, ignore_index=True)
    mstats = pd.DataFrame(mstats)
    coefs.to_csv(os.path.join(OUT, "synthetic_panel_coefficients_long.csv"), index=False)
    mstats.to_csv(os.path.join(OUT, "synthetic_panel_model_stats.csv"), index=False)

    order = ["const", "roa", "tobin_q", "interest_coverage", "cash_ratio", "size_code"] + MACRO + SECTOR_DUMS
    w_main = wide_table(coefs, mstats, ["M1", "M2", "M3", "M4", "M5", "M6"], order)
    w_main.to_csv(os.path.join(OUT, "table_synthetic_panel_M1_M6.csv"), index=False)
    to_latex(w_main, os.path.join(OUT, "table_synthetic_panel_M1_M6.tex"),
             "dependent variable debt_ratio; reference synthetic panel; *** p<0.01 ** p<0.05 * p<0.10")
    w_strict = wide_table(coefs, mstats, ["M7", "M6", "M8"], order)
    w_strict.to_csv(os.path.join(OUT, "table_synthetic_panel_strict_M7_M6_M8.csv"), index=False)
    to_latex(w_strict, os.path.join(OUT, "table_synthetic_panel_strict_M7_M6_M8.tex"),
             "dependent variable debt_ratio; leakage-free regressors; reference synthetic panel")

    # --- diagnostics: serial correlation, Hausman, Mundlak
    tests = []
    tests.append({"test": "Durbin-Watson (M1 pooled OLS residuals, firm-year order)",
                  "statistic": float(sm.stats.durbin_watson(m1.resid)), "df": "", "p": "",
                  "note": "2 = no first-order autocorrelation; computed over the stacked panel as in the original"})
    resid = pd.Series(m1.resid.values, index=df.index)
    df["_r"] = resid
    df["_r_lag"] = df.groupby("firm_id")["_r"].shift(1)
    rho_within = float(df[["_r", "_r_lag"]].dropna().corr().iloc[0, 1])
    tests.append({"test": "AR(1) correlation of M1 residuals within firm", "statistic": rho_within, "df": "", "p": "",
                  "note": "corr(e_it, e_i,t-1), 296 x 14 pairs"})
    for name, xs in [("original regressors (roa, tobin_q, interest_coverage, macro)", ORIG_TV + MACRO),
                     ("strict regressors (roa, cash_ratio, macro)", STRICT_TV + MACRO)]:
        w = wooldridge_serial_test(df, "debt_ratio", xs)
        tests.append({"test": w["test"] + " - " + name, "statistic": w["F"], "df": f"F(1,{w['df2']})",
                      "p": w["p"], "note": f"rho_fd = {w['rho_fd']:.4f} (cluster SE {w['se_cluster']:.4f}); "
                                          f"n = {w['n_obs']}"})
    # Hausman FE vs RE (non-robust covariances) and Mundlak (cluster-robust).
    # Macro variables are constant within year and the panel is balanced, so their firm means are the same
    # constant for every firm (collinear with the intercept); the Mundlak test therefore uses firm means of the
    # firm-level time-varying regressors only.  The Hausman contrast is reported both over all time-varying
    # regressors and over the firm-level ones only.
    for name, firm_tv, other_re in [("original regressors", ORIG_TV, SECTOR_DUMS),
                                    ("strict regressors", STRICT_TV, ["size_code"] + SECTOR_DUMS)]:
        tv = firm_tv + MACRO
        fe = PanelOLS(pdf["debt_ratio"], sm.add_constant(pdf[tv]), entity_effects=True).fit(cov_type="unadjusted")
        re = RandomEffects(pdf["debt_ratio"], sm.add_constant(pdf[tv + other_re].astype(float))).fit(
            cov_type="unadjusted")
        theta_val = float(np.asarray(re.theta).ravel()[0])
        coefs_cmp = pd.DataFrame({"variable": tv, "fe_coef": fe.params[tv].values, "re_coef": re.params[tv].values,
                                  "fe_se": fe.std_errors[tv].values, "re_se": re.std_errors[tv].values})
        coefs_cmp["spec"] = name
        coefs_cmp.to_csv(os.path.join(OUT, f"hausman_fe_vs_re_{name.split()[0]}.csv"), index=False)
        for cl, vv in [("all time-varying regressors", tv), ("firm-level regressors only", firm_tv)]:
            for sm_flag in (False, True):
                H, dfh, ph, posdef = hausman(fe, re, vv, sigmamore=sm_flag)
                tests.append({"test": f"Hausman FE vs RE{' (sigmamore)' if sm_flag else ''} - {name}, "
                                      f"contrast over {cl}", "statistic": H,
                              "df": f"chi2({dfh})", "p": ph if dfh > 0 else np.nan,
                              "note": f"non-robust covariances; V_FE - V_RE positive definite: {posdef}; "
                                      f"RE theta = {theta_val:.4f}"})
        F, d1, d2, pm = mundlak_test(df, "debt_ratio", firm_tv, MACRO + other_re)
        tests.append({"test": f"Mundlak / correlated-RE Wald test (cluster-robust) - {name}", "statistic": F,
                      "df": f"F({d1},{d2})", "p": pm,
                      "note": "H0: firm means of " + ", ".join(firm_tv) + " jointly zero (RE consistent)"})
    tests.append({"test": "F test poolability (M3: firm effects jointly zero)", "statistic": float(m3.f_pooled.stat),
                  "df": f"F({m3.f_pooled.df},{m3.f_pooled.df_denom})", "p": float(m3.f_pooled.pval), "note": ""})
    tests.append({"test": "F test poolability (M6: firm effects jointly zero)", "statistic": float(m6.f_pooled.stat),
                  "df": f"F({m6.f_pooled.df},{m6.f_pooled.df_denom})", "p": float(m6.f_pooled.pval), "note": ""})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(OUT, "synthetic_panel_specification_tests.csv"), index=False)

    # --- ROA across specifications
    roa = coefs[coefs.variable == "roa"][["model", "coef", "se", "t", "p", "ci_low", "ci_high", "se_type"]].copy()
    desc = {"M1": "Pooled OLS, original regressors, HC1", "M2": "Pooled OLS, original regressors, firm cluster",
            "M3": "Firm FE, original regressors, firm cluster", "M4": "Firm + year FE, original regressors, firm cluster",
            "M5": "Firm FE, original regressors, Driscoll-Kraay",
            "M6": "Firm FE, strict regressors, firm cluster", "M7": "Pooled OLS, strict regressors, firm cluster",
            "M8": "Firm + year FE, strict regressors, firm cluster"}
    roa["description"] = roa["model"].map(desc)

    # Mechanism check: drop interest_coverage only / tobin_q only from the pooled and FE models
    extra = []
    for lab, xs in [("pooled, original minus interest_coverage", ["roa", "tobin_q"] + MACRO + SECTOR_DUMS),
                    ("pooled, original minus tobin_q", ["roa", "interest_coverage"] + MACRO + SECTOR_DUMS),
                    ("pooled, roa + macro + sector only", ["roa"] + MACRO + SECTOR_DUMS)]:
        r = sm.OLS(df["debt_ratio"], sm.add_constant(df[xs].astype(float))).fit(
            cov_type="cluster", cov_kwds={"groups": df["firm_id"]})
        ci = r.conf_int().loc["roa"]
        extra.append({"model": "aux", "description": lab + ", firm cluster", "coef": r.params["roa"],
                      "se": r.bse["roa"], "t": r.tvalues["roa"], "p": r.pvalues["roa"], "ci_low": ci.iloc[0],
                      "ci_high": ci.iloc[1], "se_type": "cluster(firm)"})
    for lab, xs in [("firm FE, original minus interest_coverage", ["roa", "tobin_q"] + MACRO),
                    ("firm FE, roa + macro only", ["roa"] + MACRO)]:
        r = PanelOLS(pdf["debt_ratio"], sm.add_constant(pdf[xs]), entity_effects=True).fit(
            cov_type="clustered", cluster_entity=True)
        ci = r.conf_int().loc["roa"]
        extra.append({"model": "aux", "description": lab + ", firm cluster", "coef": r.params["roa"],
                      "se": r.std_errors["roa"], "t": r.tstats["roa"], "p": r.pvalues["roa"], "ci_low": ci.iloc[0],
                      "ci_high": ci.iloc[1], "se_type": "cluster(firm)"})
    roa = pd.concat([roa, pd.DataFrame(extra)], ignore_index=True)
    # simple correlations
    df["_dr_dm"] = df["debt_ratio"] - df.groupby("firm_id")["debt_ratio"].transform("mean")
    df["_roa_dm"] = df["roa"] - df.groupby("firm_id")["roa"].transform("mean")
    fm = df.groupby("firm_id")[["debt_ratio", "roa"]].mean()
    corr = pd.DataFrame([
        {"quantity": "pooled corr(debt_ratio, roa)", "value": float(df[["debt_ratio", "roa"]].corr().iloc[0, 1])},
        {"quantity": "within-firm corr (demeaned)", "value": float(df[["_dr_dm", "_roa_dm"]].corr().iloc[0, 1])},
        {"quantity": "between-firm corr (firm means)", "value": float(fm.corr().iloc[0, 1])},
        {"quantity": "pooled corr(roa, interest_coverage)",
         "value": float(df[["roa", "interest_coverage"]].corr().iloc[0, 1])},
        {"quantity": "pooled corr(debt_ratio, interest_coverage)",
         "value": float(df[["debt_ratio", "interest_coverage"]].corr().iloc[0, 1])},
    ])
    corr.to_csv(os.path.join(OUT, "synthetic_roa_debt_correlations.csv"), index=False)
    roa.to_csv(os.path.join(OUT, "roa_coefficient_across_specifications.csv"), index=False)
    return dict(coefs=coefs, mstats=mstats, tests=tests, roa=roa, diag=diag, corr=corr, bw=bw,
                orig_r2=ols0.rsquared, orig_adj=ols0.rsquared_adj, orig_dw=diag0["dw"], m1=m1)


# ---------------------------------------------------------------------------
# Real 46-firm panel
# ---------------------------------------------------------------------------
def run_real():
    dr = common.load_real_panel()
    dr = dr.sort_values(["firm_id", "year"]).reset_index(drop=True)
    usable, _ = common.align_features(dr, common.FEATURES_STRICT)
    log(f"Real panel: {len(dr)} rows, {dr.ticker.nunique()} firms; strict features usable: {usable}")
    years = sorted(dr.year.unique())
    ref_year = 2022  # year dummies relative to 2022 (2021 contains only MDT)
    ydums = []
    for y in years:
        if y != ref_year:
            dr[f"year_{y}"] = (dr.year == y).astype(int)
            ydums.append(f"year_{y}")
    # sector dummies present in the real sample (ref ConsDisc)
    sdums = [c for c in SECTOR_DUMS if dr[c].sum() > 0]

    specs = {
        "A": ["roa", "cash_ratio"],                                          # time-varying part of the strict set
        "B": ["roa", "cash_ratio", "log_assets", "asset_turnover"],          # extended, no missing values
    }
    coefs, mstats, roa_rows = [], [], []
    for samp_lab, d in [("all", dr), ("excl_MCD", dr[dr.ticker != "MCD"].copy())]:
        pdr = panel_index(d)
        G = d.firm_id.nunique()
        for sp, xs in specs.items():
            dd = d.dropna(subset=xs + ["debt_ratio"])
            pdd = panel_index(dd)
            pooled_x = xs + (["size_code"] if sp == "A" else []) + sdums + ydums
            X = sm.add_constant(dd[pooled_x].astype(float))
            r2 = sm.OLS(dd["debt_ratio"], X).fit(cov_type="cluster", cov_kwds={"groups": dd["firm_id"]})
            r2lm = PooledOLS(pdd["debt_ratio"], sm.add_constant(pdd[pooled_x].astype(float))).fit()
            name2 = f"Real_M2_{sp}_{samp_lab}"
            coefs.append(tidy_sm(r2, name2, "cluster(firm)"))
            st = {"model": name2, "sample": samp_lab, "spec": sp, "n_obs": len(dd), "n_firms": dd.firm_id.nunique(),
                  "firm_effects": "No", "year_effects": "Yes (dummies)", "se_type": "cluster(firm)",
                  "adj_r2": r2.rsquared_adj}
            st.update(r2_from_lm(r2lm))
            mstats.append(st)
            r3 = PanelOLS(pdd["debt_ratio"], pdd[xs], entity_effects=True, time_effects=True).fit(
                cov_type="clustered", cluster_entity=True)
            name3 = f"Real_M3_{sp}_{samp_lab}"
            coefs.append(tidy_lm(r3, name3, "cluster(firm)"))
            st = {"model": name3, "sample": samp_lab, "spec": sp, "n_obs": int(r3.nobs),
                  "n_firms": dd.firm_id.nunique(), "firm_effects": "Yes", "year_effects": "Yes",
                  "se_type": "cluster(firm)", "F_poolability": float(r3.f_pooled.stat),
                  "F_poolability_p": float(r3.f_pooled.pval)}
            st.update(r2_from_lm(r3))
            mstats.append(st)
            for nm, res, kind in [(name2, r2, "sm"), (name3, r3, "lm")]:
                ci = res.conf_int().loc["roa"]
                roa_rows.append({"model": nm, "sample": samp_lab, "spec": sp,
                                 "coef": res.params["roa"], "se": (res.bse if kind == "sm" else res.std_errors)["roa"],
                                 "p": res.pvalues["roa"], "ci_low": ci.iloc[0], "ci_high": ci.iloc[1]})
    coefs = pd.concat(coefs, ignore_index=True)
    mstats = pd.DataFrame(mstats)
    coefs.to_csv(os.path.join(OUT, "real_panel_coefficients_long.csv"), index=False)
    mstats.to_csv(os.path.join(OUT, "real_panel_model_stats.csv"), index=False)
    roa_real = pd.DataFrame(roa_rows)
    roa_real.to_csv(os.path.join(OUT, "real_panel_roa_coefficient.csv"), index=False)
    order = ["const", "roa", "cash_ratio", "log_assets", "asset_turnover", "size_code"]
    models = [f"Real_M2_A_all", "Real_M3_A_all", "Real_M2_B_all", "Real_M3_B_all"]
    w = wide_table(coefs, mstats, models, order)
    w.to_csv(os.path.join(OUT, "table_real_panel.csv"), index=False)
    to_latex(w, os.path.join(OUT, "table_real_panel.tex"),
             "dependent variable debt_ratio; 46-firm Yahoo Finance panel; year dummies (ref 2022) and sector "
             "dummies (ref ConsDisc) in pooled models not shown")
    # within-firm variation available
    wv = pd.DataFrame([{"variable": v,
                        "total_sd": float(dr[v].std()),
                        "within_sd": float((dr[v] - dr.groupby("firm_id")[v].transform("mean")).std()),
                        "between_sd": float(dr.groupby("firm_id")[v].mean().std())}
                       for v in ["debt_ratio", "roa", "cash_ratio", "log_assets", "asset_turnover"]])
    wv.to_csv(os.path.join(OUT, "real_panel_within_between_sd.csv"), index=False)
    obs_per_firm = dr.groupby("firm_id").size().value_counts().sort_index()
    return dict(coefs=coefs, mstats=mstats, roa=roa_real, wv=wv, obs_per_firm=obs_per_firm, sdums=sdums,
                ydums=ydums)


def coef_plot(roa_syn, roa_real):
    import matplotlib.pyplot as plt
    common.set_style()
    keep = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
    s = roa_syn[roa_syn.model.isin(keep)].set_index("model").loc[keep].reset_index()
    labels = {"M1": "M1 Pooled, HC1", "M2": "M2 Pooled, cluster", "M3": "M3 Firm FE", "M4": "M4 Firm+year FE",
              "M5": "M5 Firm FE, DK", "M6": "M6 Firm FE, strict", "M7": "M7 Pooled, strict",
              "M8": "M8 Firm+year FE, strict"}
    fig, ax = plt.subplots(figsize=(common.FIG_W_SINGLE, 2.8))
    yy = np.arange(len(s))[::-1]
    col = [common.OKABE_ITO[4] if m in ("M1", "M2", "M3", "M4", "M5") else common.OKABE_ITO[5] for m in s.model]
    for yi, (_, r), c in zip(yy, s.iterrows(), col):
        ax.errorbar(r.coef, yi, xerr=[[r.coef - r.ci_low], [r.ci_high - r.coef]], fmt="o", color=c, ms=3.5,
                    capsize=2, lw=1)
    ax.axvline(0, color="0.4", lw=0.7, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([labels[m] for m in s.model])
    ax.set_xlabel("ROA coefficient on debt ratio (95% CI)")
    fig.tight_layout()
    common.savefig(fig, os.path.join(OUT, "fig_roa_coefficient_synthetic"))
    plt.close(fig)


def main():
    syn = run_synthetic()
    real = run_real()
    coef_plot(syn["roa"], real["roa"])
    summary = {
        "original_reproduction": {"r2": syn["orig_r2"], "adj_r2": syn["orig_adj"], "dw": syn["orig_dw"]},
        "driscoll_kraay_bandwidth": syn["bw"],
        "roa_synthetic": syn["roa"].to_dict(orient="records"),
        "roa_real": real["roa"].to_dict(orient="records"),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    real["obs_per_firm"].rename("n_firms").rename_axis("obs_per_firm").to_csv(
        os.path.join(OUT, "real_panel_obs_per_firm.csv"))
    with open(os.path.join(OUT, "run_log.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    pd.set_option("display.width", 250)
    print(syn["diag"].drop(columns=["column_list"]).T)
    print(syn["mstats"].T)
    print(syn["tests"].to_string())
    print(syn["roa"].to_string())
    print(syn["corr"].to_string())
    print(real["mstats"].T)
    print(real["roa"].to_string())
    print(real["wv"].to_string())


if __name__ == "__main__":
    main()
