"""
common.py -- shared foundation module (data-generating process, feature sets, splits, tests).

Contents
  generate_panel()              reference synthetic panel (exact copy of the original DGP, parameterised)
  FEATURE_AUDIT / FEATURES_*    leakage audit and feature sets for the debt_ratio target
  split_random, grouped_kfold, repeated_grouped_kfold, temporal_split, unseen_firm_future_split
  corrected_resampled_ttest()   Nadeau & Bengio (2003) corrected resampled t-test
  load_real_panel(), align_features()   46-firm Yahoo Finance panel
  set_style(), OKABE_ITO        figure style

The DGP code is copied (not imported) from run_pipeline_enhanced.py::generate_dataset() and
get_features(), because importing that file creates folders and sets global state.  With default
arguments and seed=42 generate_panel() reproduces the original 4,440-row panel.

No function here writes files except write_feature_audit().
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# All locations come from config.py (overridable with GENFIN_* environment variables).
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as _config  # noqa: E402
ROOT = _config.ROOT
REAL_DATA_DIR = _config.REAL_DATA_DIR
OUT_MAIN = _config.OUT_MAIN
OUT_FOUNDATION = os.path.join(OUT_MAIN, "foundation")

# ---------------------------------------------------------------------------
# DGP defaults (values taken verbatim from run_pipeline_enhanced.py)
# ---------------------------------------------------------------------------
SECTORS = ["Tech", "Healthcare", "Financials", "ConsDisc", "Industrials", "Energy", "Utilities", "Materials"]
SECTOR_CAPS = {"Tech": 0.40, "Healthcare": 0.45, "Financials": 0.80, "ConsDisc": 0.50,
               "Industrials": 0.55, "Energy": 0.55, "Utilities": 0.70, "Materials": 0.55}
SECTOR_PROF = {"Tech": 0.08, "Healthcare": 0.07, "Financials": 0.04, "ConsDisc": 0.06,
               "Industrials": 0.05, "Energy": 0.05, "Utilities": 0.03, "Materials": 0.04}
SECTOR_GROWTH = {"Tech": 0.10, "Healthcare": 0.07, "Financials": 0.04, "ConsDisc": 0.06,
                 "Industrials": 0.05, "Energy": 0.04, "Utilities": 0.02, "Materials": 0.03}

DGP_DEFAULTS = dict(
    n_firms=300,            # firms requested; n_firms // 8 per sector -> 37 x 8 = 296 firms
    n_years=15,
    start_year=2010,
    sector_caps=SECTOR_CAPS,          # upper clip of debt ratio by sector
    base_dr_frac=0.45,                # sector baseline leverage = cap * base_dr_frac
    sector_prof=SECTOR_PROF,          # mean firm profitability by sector
    sector_growth=SECTOR_GROWTH,      # mean firm growth by sector
    size_mu=7.5, size_sd=1.9,         # log total assets ~ N(size_mu, size_sd)
    prof_sd=0.08,                     # firm profitability dispersion
    risk_a=2.0, risk_b=5.0,           # latent risk ~ Beta(a, b)
    growth_sd=0.10,                   # firm growth dispersion
    firm_dr_sd=0.10,                  # firm-specific long-run leverage dispersion
    dr_floor=0.02,                    # lower clip of debt ratio
    target_rate_coef=0.03,            # DR* sensitivity to (interest_rate - 2.5)/100
    target_spread_coef=0.02,          # DR* sensitivity to credit_spread (in %)
    lam=0.30,                         # partial-adjustment speed (paper: lambda)
    sigma_zeta=0.02,                  # s.d. of adjustment shock (paper: zeta ~ N(0, 0.02); 0.02 is an s.d.)
    sigma_init=0.03,                  # s.d. of first-year deviation from DR*
    roa_noise_sd=0.02,
    rev_noise_sd=0.04,
    ebit_ni_mult=1.7,                 # EBIT = 1.7 * net income (no interest deducted from NI)
    # WACC parameterisation (CAPM cost of equity, after-tax cost of debt, book weights)
    erp=0.055,                        # equity risk premium
    beta0=1.0,                        # unlevered-like intercept of equity beta
    beta_lev=0.9,                     # beta = beta0 + beta_lev * DR  (original code: 1 + 0.3*DR*3)
    beta_clip=(0.5, 3.0),
    tax_rate=0.21,                    # Kd after tax = (rf + spread) * (1 - tax_rate)
    wacc_clip=(0.02, 0.25),
    winsorize=True,
    winsor_q=(0.01, 0.99),
    _dr_perturb=0.0,                  # audit-only: additive shift to DR* (consumes no random numbers)
)

WINSOR_COLS = ["debt_ratio", "debt_to_equity", "roa", "roe", "interest_coverage", "tobin_q", "wacc"]


def generate_panel(seed: int = 42, add_extra_features: bool = True, **dgp_params) -> pd.DataFrame:
    """Reference synthetic firm-year panel.

    Exact transcription of generate_dataset() + get_features() (sector dummies) from
    run_pipeline_enhanced.py, with the hard-coded constants exposed as keyword arguments
    (see DGP_DEFAULTS).  Uses a private np.random.RandomState(seed), whose draw sequence is
    identical to np.random.seed(seed) + np.random.* in the original code.

    add_extra_features adds clean derived columns that are not in the original paper
    (log_assets, ebit_to_assets, asset_turnover, sales_growth); they do not alter any
    original column or the random stream.
    """
    unknown = set(dgp_params) - set(DGP_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown DGP parameters: {sorted(unknown)}")
    p = {**DGP_DEFAULTS, **dgp_params}
    rs = np.random.RandomState(seed)

    years = list(range(p["start_year"], p["start_year"] + p["n_years"]))
    n = len(years)
    t = np.arange(n)
    gdp = 2.2 + 0.5 * np.sin(2 * np.pi * t / 7) + 0.3 * np.sin(2 * np.pi * t / 3.5) + rs.normal(0, 0.6, n)
    infl = 2.0 + 0.3 * gdp + 0.1 * np.cumsum(rs.normal(0, 0.2, n)) + rs.normal(0, 0.4, n)
    infl = np.clip(infl, 0.2, 7)
    rate = 2.5 + 0.5 * (infl - 2) - 0.3 * (gdp - 2) + rs.normal(0, 0.3, n)
    rate = np.clip(rate, 0.1, 6)
    spread = 1.5 - 0.3 * (gdp - 2) + 0.2 * np.abs(rate - 2.5) + rs.normal(0, 0.25, n)
    spread = np.clip(spread, 0.5, 5)
    macro = pd.DataFrame({"year": years, "gdp_growth": gdp, "inflation": infl,
                          "interest_rate": rate, "credit_spread": spread})

    lam, s_zeta, floor = p["lam"], p["sigma_zeta"], p["dr_floor"]
    records = []
    fid = 0
    for sector in SECTORS:
        cap = p["sector_caps"][sector]
        base_dr = cap * p["base_dr_frac"]
        for _ in range(p["n_firms"] // len(SECTORS)):
            fid += 1
            sz = np.exp(rs.normal(p["size_mu"], p["size_sd"]))
            prof = rs.normal(p["sector_prof"][sector], p["prof_sd"])
            risk = rs.beta(p["risk_a"], p["risk_b"])
            growth = rs.normal(p["sector_growth"][sector], p["growth_sd"])
            dr = np.clip(base_dr + rs.normal(0, p["firm_dr_sd"]), floor, cap)
            dr_prev = None
            for yi, yr in enumerate(years):
                m = macro.iloc[yi]
                rev_g = growth + 0.5 * m["gdp_growth"] / 100 + rs.normal(0, p["rev_noise_sd"])
                rev = sz * np.exp(rev_g * yi) * rs.uniform(0.35, 1.6)
                roa = prof + 0.02 * m["gdp_growth"] / 100 + rs.normal(0, p["roa_noise_sd"])
                ni = sz * roa * np.exp(rev_g * yi)
                ebit = ni * p["ebit_ni_mult"]
                target_dr = (dr - p["target_rate_coef"] * (m["interest_rate"] - 2.5) / 100
                             - p["target_spread_coef"] * m["credit_spread"] / 1.0 + p["_dr_perturb"])
                target_dr = np.clip(target_dr, floor, cap)
                cur_dr = target_dr + rs.normal(0, p["sigma_init"]) if yi == 0 else \
                    dr_prev + lam * (target_dr - dr_prev) + rs.normal(0, s_zeta)
                cur_dr = np.clip(cur_dr, floor, cap)
                dr_prev = cur_dr
                debt = sz * cur_dr
                eq = sz - debt
                int_exp = debt * max(m["interest_rate"] / 100 + m["credit_spread"] / 100 * risk * 2, 0.005)
                cash = sz * np.clip(0.08 + 0.02 * risk, 0.01, 0.4) * rs.uniform(0.8, 1.2)
                mcap = eq * np.clip(np.exp(rs.normal(0.8, 0.7)), 0.5, 9)
                sret = rs.normal(0.08, 0.25) + 0.5 * m["gdp_growth"] / 100
                records.append({"firm_id": fid, "year": yr, "sector": sector,
                                "total_assets": sz, "total_debt": debt, "total_equity": eq,
                                "revenue": rev, "ebit": max(ebit, 0), "net_income": ni,
                                "interest_expense": max(int_exp, 0.001 * debt), "cash_holdings": cash,
                                "market_cap": mcap, "stock_return": sret,
                                "gdp_growth": m["gdp_growth"], "inflation": m["inflation"],
                                "interest_rate": m["interest_rate"], "credit_spread": m["credit_spread"]})

    df = pd.DataFrame(records)
    df["debt_ratio"] = df["total_debt"] / df["total_assets"]
    df["debt_to_equity"] = df["total_debt"] / df["total_equity"].clip(1e-6)
    df["equity_ratio"] = df["total_equity"] / df["total_assets"]
    df["roa"] = df["net_income"] / df["total_assets"]
    df["roe"] = df["net_income"] / df["total_equity"].clip(1e-6)
    df["interest_coverage"] = df["ebit"] / df["interest_expense"].clip(1e-6)
    df["cash_ratio"] = df["cash_holdings"] / df["total_assets"]
    df["market_leverage"] = df["total_debt"] / (df["total_debt"] + df["market_cap"]).clip(1e-6)
    df["tobin_q"] = (df["market_cap"] + df["total_debt"]) / df["total_assets"]
    rf = df["interest_rate"] / 100
    if p["beta_lev"] == 0.9 and p["beta0"] == 1.0:
        beta_raw = 1.0 + 0.3 * df["debt_ratio"] * 3          # bit-identical to the original expression
    else:
        beta_raw = p["beta0"] + p["beta_lev"] * df["debt_ratio"]
    beta = np.clip(beta_raw, *p["beta_clip"])
    Ke = rf + beta * p["erp"]
    Kd = (rf + df["credit_spread"] / 100) * (0.79 if p["tax_rate"] == 0.21 else 1 - p["tax_rate"])
    ew = df["total_equity"] / (df["total_debt"] + df["total_equity"])
    dw = 1 - ew
    df["wacc"] = (ew * Ke + dw * Kd).clip(*p["wacc_clip"])
    z = (df["roa"] - df["roa"].mean()) / df["roa"].std()
    df["default_risk_score"] = np.clip(
        1 / (1 + np.exp(-(-2.5 - 0.8 * z - 0.5 * df["gdp_growth"] / 100 + 0.3 * df["credit_spread"]))), 0.001, 0.999)
    df["altman_zscore"] = (1.2 * (df["total_assets"] - df["total_debt"] - df["cash_holdings"]) / df["total_assets"]
                           + 1.4 * df["net_income"] / df["total_assets"] + 3.3 * df["ebit"] / df["total_assets"]
                           + 0.6 * df["market_cap"] / df["total_debt"].clip(1e-6)
                           + 1.0 * df["revenue"] / df["total_assets"]).clip(-10, 20)
    df["capital_structure_efficiency"] = np.clip(
        0.3 / (1 + df["wacc"] * 10) + 0.2 * df["roe"].clip(-1, 2) / 2
        + 0.25 * (1 - df["default_risk_score"]) + 0.1 * df["cash_ratio"] + 0.15 * df["tobin_q"] / 5, 0, 1)
    # LabelEncoder on sector names == alphabetical codes
    sec_sorted = sorted(df["sector"].unique())
    df["sector_code"] = df["sector"].map({s: i for i, s in enumerate(sec_sorted)}).astype(int)
    df["size_code"] = pd.qcut(df["total_assets"], 3, labels=[0, 1, 2], duplicates="drop").astype(int)
    if p["winsorize"]:
        qlo, qhi = p["winsor_q"]
        for c in WINSOR_COLS:
            lo, hi = df[c].quantile([qlo, qhi])
            df[c] = df[c].clip(lo, hi)
    # get_features(): sector dummies (sorted)
    for sec in sec_sorted:
        df[f"sector_{sec}"] = (df["sector"] == sec).astype(int)

    if add_extra_features:
        df["log_assets"] = np.log(df["total_assets"])
        df["ebit_to_assets"] = df["ebit"] / df["total_assets"]
        df["asset_turnover"] = df["revenue"] / df["total_assets"]
        df = df.sort_values(["firm_id", "year"]).reset_index(drop=True)
        prev = df.groupby("firm_id")["revenue"].shift(1)
        prev_year = df.groupby("firm_id")["year"].shift(1)
        df["sales_growth"] = np.where(prev_year == df["year"] - 1, df["revenue"] / prev - 1, np.nan)
    return df


# ---------------------------------------------------------------------------
# Leakage audit (target = debt_ratio).  Each entry was checked against the DGP code above and
# verified empirically in test_common.py (perturbing DR* with the random stream held fixed).
#   direct   : the formula contains total_debt, total_equity (= assets - debt) or debt_ratio
#   indirect : no explicit debt term, but built from generator quantities that depend on debt
#              (interest_expense, market_cap = equity * multiple, wacc, roe, tobin_q)
#   clean    : not a function of debt anywhere in the generator
# ---------------------------------------------------------------------------
FEATURE_AUDIT = [
    # direct
    dict(feature="total_debt", cls="direct", reason="total_debt = total_assets * debt_ratio (numerator of the target)."),
    dict(feature="total_equity", cls="direct", reason="total_equity = total_assets - total_debt = total_assets * (1 - debt_ratio)."),
    dict(feature="debt_to_equity", cls="direct", reason="total_debt / total_equity = DR / (1 - DR); one-to-one with the target before winsorisation."),
    dict(feature="equity_ratio", cls="direct", reason="total_equity / total_assets = 1 - debt_ratio exactly."),
    dict(feature="market_leverage", cls="direct", reason="total_debt / (total_debt + market_cap), market_cap = equity * multiple; = DR / (DR + (1-DR)*M)."),
    dict(feature="wacc", cls="direct", reason="Book weights E/(D+E) = 1 - DR, and equity beta = 1 + 0.9*DR; given interest_rate and credit_spread WACC is a deterministic function of DR."),
    dict(feature="roe", cls="direct", reason="net_income / total_equity = roa / (1 - DR); together with roa it recovers DR = 1 - roa/roe."),
    dict(feature="tobin_q", cls="direct", reason="(market_cap + total_debt) / total_assets = DR + (1 - DR) * M, with M a lognormal multiple; contains total_debt explicitly."),
    dict(feature="altman_zscore", cls="direct", reason="Term 1.2*(TA - total_debt - cash)/TA contains 1 - DR; term 0.6*market_cap/total_debt = 0.6*(1-DR)*M/DR."),
    # indirect
    dict(feature="interest_expense", cls="indirect", reason="interest_expense = total_debt * (rf + 2*risk*spread), floored at 0.001*debt."),
    dict(feature="interest_coverage", cls="indirect", reason="ebit / interest_expense = 1.7*roa*g / (DR * r_eff); inversely proportional to DR given profitability and rates."),
    dict(feature="market_cap", cls="indirect", reason="market_cap = total_equity * clip(exp(N(0.8,0.7)), 0.5, 9), so it scales with 1 - DR."),
    dict(feature="capital_structure_efficiency", cls="indirect", reason="Composite of wacc, roe and tobin_q (all debt-dependent) plus default_risk_score and cash_ratio."),
    # clean
    dict(feature="roa", cls="clean", reason="net_income = size * profitability draw * growth factor; interest is never deducted, so NI and ROA do not depend on debt."),
    dict(feature="cash_ratio", cls="clean", reason="cash = size * clip(0.08 + 0.02*risk) * U(0.8,1.2); depends on latent risk, not debt."),
    dict(feature="default_risk_score", cls="clean", reason="Logistic in z-scored roa, gdp_growth and credit_spread; no debt term (z uses full-sample roa mean/s.d.)."),
    dict(feature="sector_code", cls="clean", reason="Alphabetical code of the sector label; sector sets the baseline leverage, so it is informative but not leaky."),
    dict(feature="size_code", cls="clean", reason="Sample tertile of total_assets; size is drawn independently of leverage."),
    dict(feature="gdp_growth", cls="clean", reason="Year-level macro series (common to all firms in a year)."),
    dict(feature="inflation", cls="clean", reason="Year-level macro series."),
    dict(feature="interest_rate", cls="clean", reason="Year-level macro series; enters DR* as an exogenous driver."),
    dict(feature="credit_spread", cls="clean", reason="Year-level macro series; enters DR* as an exogenous driver."),
    dict(feature="total_assets", cls="clean", reason="Firm size draw exp(N(7.5,1.9)); constant over time within firm."),
    dict(feature="revenue", cls="clean", reason="size * exp(growth*t) * U(0.35,1.6); no debt term."),
    dict(feature="net_income", cls="clean", reason="See roa: no interest deduction."),
    dict(feature="ebit", cls="clean", reason="ebit = 1.7 * net_income (floored at 0); no debt term."),
    dict(feature="cash_holdings", cls="clean", reason="See cash_ratio."),
    dict(feature="stock_return", cls="clean", reason="N(0.08, 0.25) + 0.5*gdp_growth/100; no debt term."),
    dict(feature="log_assets", cls="clean", reason="log(total_assets); additional derived feature."),
    dict(feature="ebit_to_assets", cls="clean", reason="ebit / total_assets (= 1.7*roa when NI > 0, before roa winsorisation); additional derived feature."),
    dict(feature="asset_turnover", cls="clean", reason="revenue / total_assets; additional derived feature."),
    dict(feature="sales_growth", cls="clean", reason="revenue_t / revenue_{t-1} - 1 within firm (NaN in first year); additional derived feature."),
]
for _s in sorted(SECTORS):
    FEATURE_AUDIT.append(dict(feature=f"sector_{_s}", cls="clean", reason="Sector dummy."))
# rename key to 'class' for export while keeping attribute-safe 'cls' internally
for _d in FEATURE_AUDIT:
    _d["class"] = _d["cls"]

AUDIT_CLASS = {d["feature"]: d["class"] for d in FEATURE_AUDIT}

SECTOR_DUMMIES = [f"sector_{s}" for s in sorted(SECTORS)]
# exactly get_features() output (order preserved); run_all_models uses it unchanged for debt_ratio
FEATURES_ORIGINAL = ["roa", "roe", "interest_coverage", "cash_ratio", "tobin_q", "altman_zscore",
                     "capital_structure_efficiency", "sector_code", "size_code",
                     "gdp_growth", "inflation", "interest_rate", "credit_spread"] + SECTOR_DUMMIES
FEATURES_NO_Z = [f for f in FEATURES_ORIGINAL if f != "altman_zscore"]
# default_risk_score is NOT in the original debt_ratio feature list, so this equals FEATURES_NO_Z
FEATURES_NO_Z_DR = [f for f in FEATURES_ORIGINAL if f not in ("altman_zscore", "default_risk_score")]
# primary specification: clean subset of the original list
FEATURES_STRICT = [f for f in FEATURES_ORIGINAL if AUDIT_CLASS[f] == "clean"]
# extended clean set: strict + clean variables that were not in the original list
FEATURES_STRICT_EXT = FEATURES_STRICT + ["default_risk_score", "log_assets", "ebit_to_assets", "asset_turnover"]

FEATURE_SETS = {"strict": FEATURES_STRICT, "strict_ext": FEATURES_STRICT_EXT, "original": FEATURES_ORIGINAL,
                "no_z": FEATURES_NO_Z, "no_z_dr": FEATURES_NO_Z_DR}
TARGET = "debt_ratio"


def write_feature_audit(path: str | None = None) -> str:
    path = path or os.path.join(OUT_FOUNDATION, "feature_audit.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for d in FEATURE_AUDIT:
        rows.append({"feature": d["feature"], "class": d["class"], "reason": d["reason"],
                     "in_original": d["feature"] in FEATURES_ORIGINAL,
                     "in_strict": d["feature"] in FEATURES_STRICT,
                     "in_strict_ext": d["feature"] in FEATURES_STRICT_EXT})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Splits.  All return positional integer index arrays (use df.iloc).
# ---------------------------------------------------------------------------
def split_random(df: pd.DataFrame, seed: int = 42, test_size: float = 0.2):
    """Legacy protocol: sklearn train_test_split(test_size=0.2, random_state=seed) over rows."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    return np.asarray(tr), np.asarray(te)


def _firm_fold_assignment(df, n_splits, seed, group_col="firm_id", strat_col="sector"):
    """Shuffle firms within each sector (seeded) and deal them round-robin to folds,
    continuing the deal across sectors so fold sizes differ by at most one firm."""
    rng = np.random.RandomState(seed)
    firms = df[[group_col, strat_col]].drop_duplicates(group_col)
    assign = {}
    k = 0
    for s in sorted(firms[strat_col].unique()):
        f = np.sort(firms.loc[firms[strat_col] == s, group_col].values)
        rng.shuffle(f)
        for g in f:
            assign[g] = k % n_splits
            k += 1
    return df[group_col].map(assign).values


def grouped_kfold(df: pd.DataFrame, n_splits: int = 5, seed: int = 42, group_col="firm_id", strat_col="sector"):
    """Sector-stratified grouped K-fold: every firm appears in exactly one test fold.
    Returns list of (train_idx, test_idx)."""
    fold = _firm_fold_assignment(df, n_splits, seed, group_col, strat_col)
    out = []
    for k in range(n_splits):
        te = np.where(fold == k)[0]
        tr = np.where(fold != k)[0]
        out.append((tr, te))
    return out


def repeated_grouped_kfold(df: pd.DataFrame, n_splits: int = 5, n_repeats: int = 5, seed: int = 42,
                           group_col="firm_id", strat_col="sector"):
    """Yields (repeat, fold, train_idx, test_idx); repeat r uses seed + r."""
    for r in range(n_repeats):
        for k, (tr, te) in enumerate(grouped_kfold(df, n_splits, seed + r, group_col, strat_col)):
            yield r, k, tr, te


DEFAULT_TEST_YEARS = (2021, 2022, 2023, 2024)


def temporal_split(df: pd.DataFrame, test_years=DEFAULT_TEST_YEARS):
    """Out-of-time split: train = years not in test_years and earlier than min(test_years);
    test = test_years (future observations of firms seen in training)."""
    test_years = set(test_years)
    missing = test_years - set(df["year"].unique())
    if missing:
        raise ValueError(f"test years not present: {sorted(missing)}")
    y = df["year"].values
    te = np.where(np.isin(y, list(test_years)))[0]
    tr = np.where(y < min(test_years))[0]
    return tr, te


def unseen_firm_future_split(df: pd.DataFrame, test_years=DEFAULT_TEST_YEARS, firm_frac: float = 0.2,
                             seed: int = 42, group_col="firm_id", strat_col="sector"):
    """Strictest split: hold out firm_frac of firms (sector-stratified, round(frac*n_sector) per sector);
    test = held-out firms in test years; train = remaining firms in years before min(test_years)."""
    rng = np.random.RandomState(seed)
    firms = df[[group_col, strat_col]].drop_duplicates(group_col)
    held = []
    for s in sorted(firms[strat_col].unique()):
        f = np.sort(firms.loc[firms[strat_col] == s, group_col].values)
        n_hold = int(round(firm_frac * len(f)))
        held.extend(rng.choice(f, n_hold, replace=False).tolist())
    held = set(held)
    is_held = df[group_col].isin(held).values
    y = df["year"].values
    is_test_year = np.isin(y, list(test_years))
    te = np.where(is_held & is_test_year)[0]
    tr = np.where((~is_held) & (y < min(test_years)))[0]
    return tr, te


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def corrected_resampled_ttest(diffs, n_train: int, n_test: int):
    """Nadeau & Bengio (2003) corrected resampled t-test for J = repeats x folds paired
    differences in a performance metric.
        t = mean(d) / sqrt((1/J + n_test/n_train) * var(d, ddof=1)),   df = J - 1
    Returns (t, two-sided p).  If var(d) == 0, returns (0, 1) when mean is 0 else (+-inf, 0)."""
    d = np.asarray(diffs, dtype=float)
    J = d.size
    if J < 2:
        raise ValueError("need at least two differences")
    m = d.mean()
    v = d.var(ddof=1)
    if v == 0:
        return (0.0, 1.0) if m == 0 else (float(np.sign(m) * np.inf), 0.0)
    t = m / np.sqrt((1.0 / J + n_test / n_train) * v)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return float(t), float(p)


# ---------------------------------------------------------------------------
# Real data: 46 S&P-100 firms (Yahoo Finance statements)
# ---------------------------------------------------------------------------
# GICS sector (current) and mapping to the 8 synthetic sectors
REAL_SECTOR_MAP = {
    # ticker: (gics_sector, synthetic_sector, note)
    "AAPL": ("Information Technology", "Tech", ""),
    "ADBE": ("Information Technology", "Tech", ""),
    "CRM": ("Information Technology", "Tech", ""),
    "CSCO": ("Information Technology", "Tech", ""),
    "IBM": ("Information Technology", "Tech", ""),
    "INTC": ("Information Technology", "Tech", ""),
    "MSFT": ("Information Technology", "Tech", ""),
    "NVDA": ("Information Technology", "Tech", ""),
    "QCOM": ("Information Technology", "Tech", ""),
    "TXN": ("Information Technology", "Tech", ""),
    "GOOGL": ("Communication Services", "Tech", "Communication Services not in the 8; mapped to Tech (classified IT before the 2018 GICS revision)."),
    "META": ("Communication Services", "Tech", "Communication Services not in the 8; mapped to Tech (classified IT before the 2018 GICS revision)."),
    "NFLX": ("Communication Services", "ConsDisc", "Communication Services not in the 8; mapped to ConsDisc (classified Consumer Discretionary before 2018)."),
    "DIS": ("Communication Services", "ConsDisc", "Communication Services not in the 8; mapped to ConsDisc (classified Consumer Discretionary before 2018)."),
    "ABBV": ("Health Care", "Healthcare", ""),
    "JNJ": ("Health Care", "Healthcare", ""),
    "LLY": ("Health Care", "Healthcare", ""),
    "MDT": ("Health Care", "Healthcare", ""),
    "MRK": ("Health Care", "Healthcare", ""),
    "PFE": ("Health Care", "Healthcare", ""),
    "TMO": ("Health Care", "Healthcare", ""),
    "UNH": ("Health Care", "Healthcare", ""),
    "AXP": ("Financials", "Financials", ""),
    "BAC": ("Financials", "Financials", ""),
    "GS": ("Financials", "Financials", ""),
    "JPM": ("Financials", "Financials", ""),
    "MA": ("Financials", "Financials", "Classified IT until the 2023 GICS revision."),
    "V": ("Financials", "Financials", "Classified IT until the 2023 GICS revision."),
    "AMZN": ("Consumer Discretionary", "ConsDisc", ""),
    "HD": ("Consumer Discretionary", "ConsDisc", ""),
    "MCD": ("Consumer Discretionary", "ConsDisc", ""),
    "TSLA": ("Consumer Discretionary", "ConsDisc", ""),
    "COST": ("Consumer Staples", "ConsDisc", "Consumer Staples not in the 8; mapped to the nearest consumer sector (ConsDisc)."),
    "KO": ("Consumer Staples", "ConsDisc", "Consumer Staples not in the 8; mapped to the nearest consumer sector (ConsDisc)."),
    "PEP": ("Consumer Staples", "ConsDisc", "Consumer Staples not in the 8; mapped to the nearest consumer sector (ConsDisc)."),
    "PG": ("Consumer Staples", "ConsDisc", "Consumer Staples not in the 8; mapped to the nearest consumer sector (ConsDisc)."),
    "WMT": ("Consumer Staples", "ConsDisc", "Consumer Staples not in the 8; mapped to the nearest consumer sector (ConsDisc)."),
    "BA": ("Industrials", "Industrials", ""),
    "CAT": ("Industrials", "Industrials", ""),
    "GE": ("Industrials", "Industrials", ""),
    "HON": ("Industrials", "Industrials", ""),
    "MMM": ("Industrials", "Industrials", ""),
    "CVX": ("Energy", "Energy", ""),
    "XOM": ("Energy", "Energy", ""),
    "NEE": ("Utilities", "Utilities", ""),
    "LIN": ("Materials", "Materials", ""),
}

# line items: first available in each list is used
_ITEMS = {
    "total_assets": ("balance_sheet", ["Total Assets"]),
    "total_debt": ("balance_sheet", ["Total Debt"]),
    "_ltd": ("balance_sheet", ["Long Term Debt And Capital Lease Obligation", "Long Term Debt"]),
    "_cd": ("balance_sheet", ["Current Debt And Capital Lease Obligation", "Current Debt"]),
    "total_equity": ("balance_sheet", ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
    "cash_holdings": ("balance_sheet", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
                                        "Cash Cash Equivalents And Federal Funds Sold", "Cash Financial"]),
    "net_income": ("income_stmt", ["Net Income", "Net Income Common Stockholders"]),
    "revenue": ("income_stmt", ["Total Revenue", "Operating Revenue"]),
    "ebit": ("income_stmt", ["EBIT", "Operating Income"]),
    "interest_expense": ("income_stmt", ["Interest Expense", "Interest Expense Non Operating"]),
}


def _read_stmt(ticker, kind, data_dir):
    f = os.path.join(data_dir, f"{ticker}_{kind}.csv")
    if not os.path.exists(f):
        return None
    s = pd.read_csv(f, index_col=0)
    s.columns = pd.to_datetime(s.columns)
    return s


def _fiscal_year(d: pd.Timestamp) -> int:
    """Compustat convention: fiscal years ending January-May are assigned to the previous calendar year
    (e.g. WMT FYE 2025-01-31 -> 2024).  Labelling by the calendar year of the year-end instead would
    create a 2026 year for the four January year-end firms."""
    return d.year if d.month >= 6 else d.year - 1


def load_real_panel(data_dir: str = REAL_DATA_DIR, return_log: bool = False):
    """Firm-year panel for the 46 real firms recomputed from the raw statement CSVs.

    total_debt = 'Total Debt' (Yahoo; includes capital/operating lease obligations where reported),
    falling back to long-term + current debt (and lease) lines.  Rows with missing or non-positive
    total_assets or missing total_debt are dropped.  market_cap is not used (only a current
    snapshot is available)."""
    rows, log = [], []
    tickers = sorted({f.split("_")[0] for f in os.listdir(data_dir) if f.endswith("_balance_sheet.csv")})
    for tk in tickers:
        stm = {k: _read_stmt(tk, k, data_dir) for k in ("balance_sheet", "income_stmt")}
        dates = sorted(set(stm["balance_sheet"].columns) | set(stm["income_stmt"].columns))
        for d in dates:
            rec = {"ticker": tk, "fiscal_year_end": d.date().isoformat(), "year": _fiscal_year(d)}
            for var, (kind, labels) in _ITEMS.items():
                val, src = np.nan, None
                s = stm[kind]
                if s is not None and d in s.columns:
                    for lab in labels:
                        if lab in s.index and pd.notna(s.at[lab, d]):
                            val, src = float(s.at[lab, d]), lab
                            break
                rec[var] = val
                rec[f"src_{var}"] = src
            if pd.isna(rec["total_debt"]) and (pd.notna(rec["_ltd"]) or pd.notna(rec["_cd"])):
                rec["total_debt"] = np.nansum([rec["_ltd"], rec["_cd"]])
                rec["src_total_debt"] = "LTD+CurrentDebt"
            rows.append(rec)
    raw = pd.DataFrame(rows)
    n_raw = len(raw)
    keep = raw["total_assets"].gt(0) & raw["total_debt"].notna()
    for _, r in raw.loc[~keep].iterrows():
        log.append(f"dropped {r.ticker} FYE {r.fiscal_year_end}: missing total_assets or total_debt")
    df = raw.loc[keep].drop(columns=["_ltd", "_cd", "src__ltd", "src__cd"]).copy()
    # duplicate fiscal-year labels (should not occur) -> keep latest FYE
    df = df.sort_values(["ticker", "fiscal_year_end"]).drop_duplicates(["ticker", "year"], keep="last")

    df["gics_sector"] = df["ticker"].map(lambda t: REAL_SECTOR_MAP[t][0])
    df["sector"] = df["ticker"].map(lambda t: REAL_SECTOR_MAP[t][1])
    df["sector_note"] = df["ticker"].map(lambda t: REAL_SECTOR_MAP[t][2])
    sec_sorted = sorted(SECTORS)
    df["sector_code"] = df["sector"].map({s: i for i, s in enumerate(sec_sorted)}).astype(int)
    for s in sec_sorted:
        df[f"sector_{s}"] = (df["sector"] == s).astype(int)

    ta = df["total_assets"]
    df["debt_ratio"] = df["total_debt"] / ta
    df["cash_ratio"] = df["cash_holdings"] / ta
    df["roa"] = df["net_income"] / ta
    df["ebit_to_assets"] = df["ebit"] / ta
    df["asset_turnover"] = df["revenue"] / ta
    df["log_assets"] = np.log(ta / 1e6)          # USD millions (synthetic total_assets ~ exp(N(7.5,1.9)))
    df["roe"] = np.where(df["total_equity"] > 0, df["net_income"] / df["total_equity"], np.nan)
    df["book_equity_ratio"] = df["total_equity"] / ta
    df["interest_coverage"] = np.where(df["interest_expense"] > 0, df["ebit"] / df["interest_expense"], np.nan)
    df["size_code"] = pd.qcut(ta, 3, labels=[0, 1, 2]).astype(int)   # tertile within the real sample
    df = df.sort_values(["ticker", "year"]).reset_index(drop=True)
    prev_rev = df.groupby("ticker")["revenue"].shift(1)
    prev_year = df.groupby("ticker")["year"].shift(1)
    df["sales_growth"] = np.where(prev_year == df["year"] - 1, df["revenue"] / prev_rev - 1, np.nan)
    df["firm_id"] = df["ticker"].astype("category").cat.codes + 1
    log.append(f"raw firm-year columns: {n_raw}; kept: {len(df)}; firms: {df.ticker.nunique()}; "
               f"years: {df.year.min()}-{df.year.max()}")
    return (df, log) if return_log else df


# synthetic feature -> real-data equivalent
REAL_FEATURE_MAP = {
    "roa": ("net_income / total_assets", True, "Same definition. Real NI is after interest and tax; synthetic NI has no interest deduction."),
    "cash_ratio": ("cash_and_equivalents / total_assets", True, "Same definition."),
    "sector_code": ("alphabetical code of mapped sector", True, "Same coding; Comm. Services and Cons. Staples mapped (see REAL_SECTOR_MAP)."),
    "size_code": ("tertile of total_assets within the real sample", True, "Same rule (sample tertiles) but cut points differ between samples."),
    "log_assets": ("ln(total_assets in USD millions)", True, "Unit assumption for synthetic total_assets."),
    "ebit_to_assets": ("EBIT / total_assets", True, "Real EBIT from statements; synthetic EBIT = 1.7*NI."),
    "asset_turnover": ("revenue / total_assets", True, "Same definition."),
    "sales_growth": ("revenue_t / revenue_{t-1} - 1", True, "Needs consecutive fiscal years."),
    "roe": ("net_income / stockholders_equity", True, "Real book equity != assets - debt; NaN when equity <= 0."),
    "interest_coverage": ("EBIT / interest_expense", True, "NaN where interest expense not reported (e.g. some banks)."),
    "tobin_q": (None, False, "Needs historical market cap; only a current snapshot is available."),
    "altman_zscore": (None, False, "Needs historical market cap (term 0.6*MVE/TD)."),
    "capital_structure_efficiency": (None, False, "Needs wacc, tobin_q and default_risk_score."),
    "default_risk_score": (None, False, "Needs gdp_growth and credit_spread (no macro data)."),
    "wacc": (None, False, "Needs rf, spread and market data."),
    "market_leverage": (None, False, "Needs historical market cap."),
    "gdp_growth": (None, False, "data/raw/fred_macro is empty."),
    "inflation": (None, False, "data/raw/fred_macro is empty."),
    "interest_rate": (None, False, "data/raw/fred_macro is empty."),
    "credit_spread": (None, False, "data/raw/fred_macro is empty."),
}
for _s in sorted(SECTORS):
    REAL_FEATURE_MAP[f"sector_{_s}"] = ("sector dummy (mapped sector)", True, "Same definition.")


def align_features(df_real: pd.DataFrame, feature_list):
    """Return (usable_features, mapping_table) for a synthetic feature list.
    A feature is usable if it has a same-definition real counterpart and the column exists and
    is not entirely missing in df_real."""
    rows, usable = [], []
    for f in feature_list:
        real_def, ok, note = REAL_FEATURE_MAP.get(f, (None, False, "no mapping defined"))
        present = ok and f in df_real.columns and df_real[f].notna().any()
        n_avail = int(df_real[f].notna().sum()) if present else 0
        rows.append({"feature": f, "real_definition": real_def, "computable": bool(present),
                     "n_non_missing": n_avail, "note": note})
        if present:
            usable.append(f)
    return usable, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
FIG_W_SINGLE = 3.5
FIG_W_DOUBLE = 7.0


def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.prop_cycle": cycler(color=OKABE_ITO),
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def savefig(fig, path_no_ext: str):
    """Save PNG (300 dpi) and PDF with white background."""
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig.savefig(path_no_ext + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight", facecolor="white")
