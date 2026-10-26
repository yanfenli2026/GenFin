"""
E5 -- Re-specified multi-objective capital-structure optimisation.

Run:  python code/e5_optimization.py
Outputs: output/e5_optimization/

Parts
  (a) legacy optimisation() reproduced; Pareto point count versus grid size
  (b) trade-off specification: Hamada relevering, leverage-dependent credit spread linked to a Merton
      default probability, tax shield, expected distress cost. Two debt-beta conventions:
        zero        : debt beta = 0 (r_e relevered with r_f)                        [main, as before]
        spread_prem : debt beta implied by the priced part of the spread
                      (expected debt return = r_d - expected loss)                   [variant]
  (c) two-objective Pareto analysis (WACC vs default probability) per sector and scenario
  (d) Latin-hypercube Monte Carlo over objective-function coefficients, nested with calibration
      (panel-seed) uncertainty; one-at-a-time tornado (Base and Recession); channel decomposition of
      the adverse scenarios; sensitivity to the DGP (lambda, sigma_zeta^2) and to the panel seed
  (e) sector table: actual mean debt ratio vs model-implied intervals
  (f) figures (PNG 300 dpi + PDF)

Design notes:
  * price of default risk = max(lam_cs * k_cs, 1): the priced spread can no longer fall below the
    actuarial expected loss (Strong Growth previously implied a negative credit-risk premium)
  * main calibration = sector parameters averaged over 50 panel seeds (42-91); the Monte Carlo draws
    a panel seed per draw (nested calibration uncertainty); seed-42-only results are kept as a check
  * debt-beta variant added; value-maximising optimum reported alongside the WACC minimum
  * GDP-to-drift mapping is now an explicit coefficient g_to_mu (base 0.5, range 0-1), included in
    the Monte Carlo and in the tornado; scenario channels decomposed
  * lower/upper corner shares reported separately for x_W and x_V
All random numbers come from fixed seeds.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

OUT = os.path.join(common.OUT_MAIN, "e5_optimization")
os.makedirs(OUT, exist_ok=True)
SEED = 42
SECTORS = sorted(common.SECTORS)
SECTOR_LABEL = {"ConsDisc": "Consumer Disc.", "Energy": "Energy", "Financials": "Financials",
                "Healthcare": "Healthcare", "Industrials": "Industrials", "Materials": "Materials",
                "Tech": "Technology", "Utilities": "Utilities"}

# decision-variable grid (debt ratio D/V); same feasible range as the original specification
X_LO, X_HI, X_STEP = 0.05, 0.85, 0.0005
XGRID = np.round(np.arange(X_LO, X_HI + X_STEP / 2, X_STEP), 6)
BAND_BP = 0.0010   # near-optimal WACC band: WACC <= min WACC + 10 basis points

# macro scenarios: shifts in percentage points (identical to the original optimization())
SCENARIOS = [("Base", 0.0, 0.0, 0.0), ("Recession", -4.0, -2.0, 3.0), ("High_Rate", 0.0, 4.0, 2.0),
             ("Credit_Crunch", -2.0, 2.0, 4.0), ("Strong_Growth", 3.0, 0.0, -1.0)]
SCEN_NAMES = [s[0] for s in SCENARIOS]
SCEN_LABEL = {"Base": "Base", "Recession": "Recession", "High_Rate": "High rate",
              "Credit_Crunch": "Credit crunch", "Strong_Growth": "Strong growth"}

PRICE_FLOOR = 1.0    # price of default risk (spread / expected loss) is never below 1
CONVENTIONS = ["zero", "spread_prem"]
CONV_LABEL = {"zero": "zero debt beta", "spread_prem": "debt beta from spread premium"}

# objective-function coefficients: base value, Monte Carlo range (uniform), source / rationale
THETA = [
    # name,        base,   low,    high,  description
    ("tau",        0.21,   0.15,   0.30,  "Corporate tax rate. Base = DGP tax_rate (US federal statutory rate since 2018); range covers effective to combined federal+state rates."),
    ("erp",        0.055,  0.040,  0.070, "Equity risk premium. Base = DGP erp."),
    ("beta_u",     1.00,   0.80,   1.20,  "Unlevered (asset) beta, common to all sectors. Base = DGP beta0 (equity beta at zero leverage)."),
    ("rf_shift",   0.000, -0.010,  0.010, "Additive shift to the risk-free rate (decimal) on top of the scenario level."),
    ("sigma_bar",  0.25,   0.18,   0.35,  "Cross-sector average asset volatility (annual). Sector sigma_A = sigma_bar x relative earnings volatility. Assumed; range spans typical estimates for US non-financial firms."),
    ("s0",         0.006,  0.003,  0.010, "Credit spread at negligible default risk (decimal). Assumed (roughly AA/AAA spread level)."),
    ("lam_cs",     2.5,    1.5,    4.0,   "Ratio of credit spread to annual expected loss at base macro conditions (credit-risk premium multiplier). Assumed; spreads exceed expected losses (credit-spread puzzle). Scenario price = max(lam_cs x k_cs, 1)."),
    ("alpha_scale", 1.00,  0.50,   1.50,  "Multiplier on sector distress cost alpha_s (fraction of unlevered value lost in default)."),
    ("recov_shift", 0.00, -0.10,   0.10,  "Additive shift to sector recovery rate R_s (LGD_s = 1 - R_s)."),
    ("T",          5.0,    3.0,    10.0,  "Debt horizon in years (Merton horizon, tax-shield horizon, spread annualisation)."),
    ("mu_shift",   0.000, -0.020,  0.020, "Additive shift to the sector asset drift mu_s (decimal)."),
    ("g_to_mu",    0.50,   0.00,   1.00,  "Asset-drift response to the scenario GDP shift (mu_s += g_to_mu x dGDP). Assumed. Base 0.5 = GDP loading of stock returns in the DGP code (stock_return = N(0.08, 0.25) + 0.5 gdp/100); the DGP's direct ROA loading is 0.02 (roa = prof + 0.02 gdp/100 + noise); 1 = one-for-one mapping; 0 = drift channel switched off."),
]
THETA_NAMES = [t[0] for t in THETA]
THETA_BASE = {t[0]: t[1] for t in THETA}

# distress cost and recovery as linear functions of asset tangibility (Net PPE / total assets)
ALPHA_MIN, ALPHA_MAX = 0.10, 0.40     # alpha_s = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (1 - tang_s)
RECOV_MIN, RECOV_MAX = 0.20, 0.60     # R_s = RECOV_MIN + (RECOV_MAX - RECOV_MIN) * tang_s

N_MC = 2000                          # Monte Carlo draws (Latin hypercube)
CAL_SEEDS = list(range(42, 92))      # 50 panel seeds for calibration uncertainty
DGP_LAMS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
DGP_SZ2 = [0.0004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]   # sigma_zeta^2; 0.0004 = code default (sd 0.02)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# Calibration
# ===========================================================================
def real_tangibility():
    """Median Net PPE / Total Assets by synthetic sector in the 46-firm Yahoo Finance sample.
    Firm value = median over the firm's available fiscal years; sector value = median over firms."""
    rows = []
    for tic, (gics, sec, _) in common.REAL_SECTOR_MAP.items():
        b = pd.read_csv(os.path.join(common.REAL_DATA_DIR, f"{tic}_balance_sheet.csv"), index_col=0)
        if "Net PPE" in b.index and "Total Assets" in b.index:
            r = (pd.to_numeric(b.loc["Net PPE"], errors="coerce") /
                 pd.to_numeric(b.loc["Total Assets"], errors="coerce")).dropna()
            rows.append({"ticker": tic, "gics_sector": gics, "sector": sec,
                         "net_ppe_to_assets_median": float(r.median()) if len(r) else np.nan, "n_years": len(r)})
        else:
            rows.append({"ticker": tic, "gics_sector": gics, "sector": sec,
                         "net_ppe_to_assets_median": np.nan, "n_years": 0})
    firm = pd.DataFrame(rows)
    sec = firm.groupby("sector")["net_ppe_to_assets_median"].agg(["median", "count"]).rename(
        columns={"median": "tangibility", "count": "n_firms"})
    return firm, sec


def finish_cal(cal, tang):
    cal = cal.copy()
    cal["rel_vol"] = cal["earn_vol"] / cal["earn_vol"].mean()
    cal["tangibility"] = tang.loc[SECTORS, "tangibility"].values
    cal["alpha"] = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (1 - cal["tangibility"])
    cal["recovery"] = RECOV_MIN + (RECOV_MAX - RECOV_MIN) * cal["tangibility"]
    return cal


def calibrate(df, tang):
    """Sector parameters from one synthetic panel (earnings volatility, median ROA, macro levels) and
    from real-sample tangibility. Leverage moments are NOT used, so the optimum is not calibrated to
    the debt ratios it is later compared with."""
    f = df.sort_values(["firm_id", "year"]).groupby(["sector", "firm_id"])["roa"]
    dsd = f.apply(lambda s: s.diff().std()).groupby("sector").median()     # earnings volatility
    med_roa = df.groupby("sector")["roa"].median()
    cal = pd.DataFrame({"earn_vol": dsd, "median_roa": med_roa}).loc[SECTORS]
    cal = finish_cal(cal, tang)
    cal["actual_mean_dr"] = df.groupby("sector")["debt_ratio"].mean().loc[SECTORS]   # comparison only
    macro = {"rf": df["interest_rate"].mean() / 100, "cs": df["credit_spread"].mean() / 100,
             "g": df["gdp_growth"].mean() / 100}
    return cal, macro


def seed_average(cals, macros, tang):
    """Main calibration: earnings volatility, median ROA, actual mean D/A and macro levels averaged
    over the panel seeds; relative volatility recomputed from the averaged earnings volatility."""
    ev = pd.concat([c["earn_vol"] for c in cals.values()], axis=1).mean(axis=1)
    mr = pd.concat([c["median_roa"] for c in cals.values()], axis=1).mean(axis=1)
    ad = pd.concat([c["actual_mean_dr"] for c in cals.values()], axis=1).mean(axis=1)
    cal = finish_cal(pd.DataFrame({"earn_vol": ev, "median_roa": mr}).loc[SECTORS], tang)
    cal["actual_mean_dr"] = ad.loc[SECTORS]
    macro = {k: float(np.mean([m[k] for m in macros.values()])) for k in ["rf", "cs", "g"]}
    return cal, macro


# ===========================================================================
# Trade-off model
# ===========================================================================
def scenario_levels(macro, scen):
    _, dg, dr, ds = scen
    rf = max(macro["rf"] + dr / 100, 0.0)
    cs = max(macro["cs"] + ds / 100, 0.001)
    return {"rf": rf, "cs": cs, "k_cs": cs / macro["cs"], "dg": dg / 100}


def evaluate(x, sec_row, lev, th, conv="zero"):
    """Vectorised model. x: (n_x,) grid; th: dict of scalars or (n,1) arrays. Returns dict of arrays."""
    tau, T = th["tau"], th["T"]
    rf = np.maximum(lev["rf"] + th["rf_shift"], 0.0)
    sigma = th["sigma_bar"] * sec_row["rel_vol"]
    mu = sec_row["median_roa"] + th["g_to_mu"] * lev["dg"] + th["mu_shift"]
    alpha = sec_row["alpha"] * th["alpha_scale"]
    lgd = np.clip(1 - (sec_row["recovery"] + th["recov_shift"]), 0.05, 0.95)
    # Merton probability that asset value falls below face value of debt within T years
    dd = (np.log(1.0 / x) + (mu - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    pd_T = stats.norm.cdf(-dd)
    el = lgd * pd_T / T                                        # annualised expected loss
    # price of default risk: base multiplier lam_cs scaled by the macro spread, floored at 1 so the
    # priced spread is never below the actuarial expected loss
    price = np.maximum(lev["k_cs"] * th["lam_cs"], PRICE_FLOOR)
    spread = th["s0"] + price * el
    r_d = rf + spread
    r_u = rf + th["beta_u"] * th["erp"]
    if conv == "zero":
        r_d_exp = rf + 0.0 * spread                             # debt beta = 0
    elif conv == "spread_prem":
        r_d_exp = r_d - el                                      # expected debt return = yield - expected loss
    else:
        raise ValueError(conv)
    r_e = r_u + (r_u - r_d_exp) * (1 - tau) * x / (1 - x)      # Hamada / MM with corporate tax
    wacc = (1 - x) * r_e + x * r_d * (1 - tau)
    # zero:        WACC = r_u (1 - tau x) + (1 - tau) x spread
    # spread_prem: WACC = r_u (1 - tau x) + (1 - tau) x expected loss
    # PV of interest tax shields over T years / V_U: computed on the default-free coupon (rf + s0) and
    # received only if the firm survives the horizon (shields are lost in default)
    shield = tau * x * (1 - (1 + rf + th["s0"]) ** (-T)) * (1 - pd_T)
    value = 1.0 + shield - alpha * pd_T                        # V_L / V_U (independent of convention)
    return {"wacc": wacc, "pd": pd_T, "spread": spread, "el": el, "r_d": r_d, "r_e": r_e,
            "r_u": r_u * np.ones_like(x), "value": value, "shield": shield, "distress": alpha * pd_T,
            "price": price * np.ones_like(x)}


def nondominated_2d(f1, f2):
    """Boolean mask of non-dominated points (minimise f1 and f2). O(n log n)."""
    order = np.lexsort((f1, f2))      # sort by f2, then f1
    keep = np.zeros(len(f1), dtype=bool)
    best = np.inf
    for i in order:
        if f1[i] < best:
            keep[i] = True
            best = f1[i]
    return keep


def summarise_curve(x, r):
    w, p, v = r["wacc"], r["pd"], r["value"]
    iw, iv = int(np.argmin(w)), int(np.argmax(v))
    nd = nondominated_2d(w, p)
    xs = x[nd]
    contiguous = bool(np.all(np.diff(np.where(nd)[0]) == 1))
    band = x[w <= w[iw] + BAND_BP]
    return {"x_wacc_min": x[iw], "wacc_min": w[iw], "pd_at_x_wacc": p[iw], "spread_at_x_wacc": r["spread"][iw],
            "el_at_x_wacc": r["el"][iw], "price_of_risk": float(np.ravel(r["price"])[0]),
            "x_value_max": x[iv], "value_max": v[iv], "pd_at_x_value": p[iv], "wacc_at_x_value": w[iv],
            "pareto_x_low": xs.min(), "pareto_x_high": xs.max(), "pareto_contiguous": contiguous,
            "pareto_n_gridpoints": int(nd.sum()),
            "band10bp_x_low": band.min(), "band10bp_x_high": band.max(),
            "wacc_at_xlo": w[0], "wacc_at_xhi": w[-1],
            "x_wacc_corner": bool(iw in (0, len(x) - 1)), "x_value_corner": bool(iv in (0, len(x) - 1))}


def batch_optima(x, sec_row, lev, thb, conv):
    """thb: dict of (n,1) arrays -> optima for each draw (each (n,))."""
    r = evaluate(x[None, :], sec_row, lev, thb, conv)
    iw = np.argmin(r["wacc"], axis=1)
    iv = np.argmax(r["value"], axis=1)
    rows = np.arange(len(iw))
    last = len(x) - 1
    return {"x_wacc_min": x[iw], "x_value_max": x[iv], "wacc_min": r["wacc"][rows, iw],
            "pd_at_x_wacc": r["pd"][rows, iw], "price_of_risk": r["price"][:, 0],
            "corner_w_lo": iw == 0, "corner_w_hi": iw == last,
            "corner_v_lo": iv == 0, "corner_v_hi": iv == last}


# ===========================================================================
# (a) Legacy optimisation, reproduced from run_pipeline_enhanced.py::optimization()
# ===========================================================================
def legacy_ev(dr, base):
    """Verbatim objective functions of the original ev() (base = frame of sampled rows)."""
    bw_m = np.mean(base["wacc"].values); bd_m = np.mean(base["default_risk_score"].values)
    br_m = np.mean(base["roe"].fillna(0).values); rat_m = np.mean(base["interest_rate"].fillna(2.5).values) / 100
    gdp_m = np.mean(base["gdp_growth"].fillna(2).values) / 100
    wacc = bw_m + 0.02 * np.abs(dr - 0.3) ** 2 + 0.01 * dr * rat_m * 2
    drisk = np.clip(bd_m + 0.3 * dr ** 2 - 0.05 * gdp_m, 0.01, 0.5)
    roe_v = np.clip(br_m * (1 + dr / (1 - dr + 0.01)), -0.5, 1.0)
    fv = np.clip(1.0 - 0.5 * np.abs(dr - 0.35) - drisk, 0, 1)
    return {"wacc": wacc, "default_risk": drisk, "roe": roe_v, "firm_value": fv}


def legacy_pareto_mask(objs):
    """Original O(n^2) non-dominated sorting loop."""
    n = len(objs)
    is_pf = np.ones(n, dtype=bool)
    for i in range(n):
        dom = np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)
        dom[i] = False
        if dom.any():
            is_pf[i] = False
    return is_pf


def run_legacy(df):
    log("(a) legacy specification")
    rows_count = []
    samples = {"full_panel": df}
    rs = np.random.RandomState(SEED)
    for k in range(20):
        samples[f"sample400_rs{k}"] = df.sample(400, random_state=rs.randint(0, 2 ** 31 - 1))
    for sname, s in samples.items():
        grids = [100, 500, 2000] if sname == "full_panel" else [500]
        for n in grids:
            drg = np.linspace(0.05, 0.85, n)
            res = legacy_ev(drg, s)
            o3 = np.column_stack([res["wacc"], res["default_risk"], -res["firm_value"]])
            o2 = np.column_stack([res["wacc"], res["default_risk"]])
            m3, m2 = legacy_pareto_mask(o3), legacy_pareto_mask(o2)
            rows_count.append({"base_sample": sname, "grid_points": n, "grid_step": drg[1] - drg[0],
                               "n_pareto_3obj": int(m3.sum()), "x_low_3obj": drg[m3].min(), "x_high_3obj": drg[m3].max(),
                               "n_pareto_2obj_wacc_risk": int(m2.sum()), "x_low_2obj": drg[m2].min(),
                               "x_high_2obj": drg[m2].max(),
                               "wacc_range_on_front_3obj": res["wacc"][m3].max() - res["wacc"][m3].min()})
    cnt = pd.DataFrame(rows_count)
    cnt.to_csv(os.path.join(OUT, "legacy_pareto_count_vs_grid.csv"), index=False)

    rbar = df["interest_rate"].mean() / 100
    rows = []
    for sec in SECTORS:
        sd = df[df["sector"] == sec]
        sdr = np.linspace(0.05, 0.70, 100); r = legacy_ev(sdr, sd); bi = int(np.argmin(r["wacc"]))
        rb = sd["interest_rate"].mean() / 100
        rows.append({"block": "sector", "name": sec, "grid": "linspace(0.05,0.70,100)",
                     "optimal_debt_ratio": sdr[bi], "analytic_optimum_0.3-0.5r": 0.3 - 0.5 * rb,
                     "min_wacc": r["wacc"][bi], "default_risk": r["default_risk"][bi],
                     "sector_mean_dr": sd["debt_ratio"].mean()})
    for name, gm, rm, sm in SCENARIOS:
        sd = df.copy()
        sd["gdp_growth"] = df["gdp_growth"] + gm; sd["interest_rate"] = df["interest_rate"] + rm
        sd["credit_spread"] = df["credit_spread"] + sm
        sdr = np.linspace(0.05, 0.85, 200); r = legacy_ev(sdr, sd); bi = int(np.argmin(r["wacc"]))
        rows.append({"block": "scenario", "name": name, "grid": "linspace(0.05,0.85,200)",
                     "optimal_debt_ratio": sdr[bi], "analytic_optimum_0.3-0.5r": 0.3 - 0.5 * (rbar + rm / 100),
                     "min_wacc": r["wacc"][bi], "default_risk": r["default_risk"][bi], "sector_mean_dr": np.nan})
    leg = pd.DataFrame(rows)
    leg.to_csv(os.path.join(OUT, "legacy_optima.csv"), index=False)
    return cnt, leg, rbar


# ===========================================================================
# Main analyses
# ===========================================================================
def theta_dict(base_overrides=None, arrays=None, idx=None):
    th = dict(THETA_BASE)
    if base_overrides:
        th.update(base_overrides)
    if arrays is not None:
        th = {k: (arrays[k][idx] if idx is not None else arrays[k])[:, None] if k in arrays else v
              for k, v in th.items()}
    return th


def deterministic(cal, macro, label, x=XGRID, keep_curves=False):
    rows, curves = [], []
    th = theta_dict()
    for conv in CONVENTIONS:
        for sname in SCEN_NAMES:
            lev = scenario_levels(macro, SCENARIOS[SCEN_NAMES.index(sname)])
            for sec in SECTORS:
                r = evaluate(x, cal.loc[sec], lev, th, conv)
                s = summarise_curve(x, r)
                rows.append({"calibration": label, "convention": conv, "scenario": sname, "sector": sec,
                             "rf": lev["rf"], "macro_spread": lev["cs"], "k_cs": lev["k_cs"],
                             "growth_shift": lev["dg"], **s})
                if keep_curves:
                    sub = slice(None, None, 10)   # store curves at 0.005 steps
                    curves.append(pd.DataFrame({
                        "convention": conv, "scenario": sname, "sector": sec, "x": x[sub], "wacc": r["wacc"][sub],
                        "pd_T": r["pd"][sub], "spread": r["spread"][sub], "expected_loss": r["el"][sub],
                        "r_e": r["r_e"][sub], "r_d": r["r_d"][sub], "value_rel": r["value"][sub],
                        "tax_shield_pv": r["shield"][sub], "exp_distress_cost": r["distress"][sub]}))
    return pd.DataFrame(rows), (pd.concat(curves, ignore_index=True) if keep_curves else None)


def new_model_grid_counts(cal, macro):
    rows = []
    lev = scenario_levels(macro, SCENARIOS[0])
    for n in [100, 500, 2000]:
        x = np.linspace(X_LO, X_HI, n)
        for sec in SECTORS:
            r = evaluate(x, cal.loc[sec], lev, theta_dict())
            nd = nondominated_2d(r["wacc"], r["pd"])
            rows.append({"grid_points": n, "sector": sec, "n_nondominated": int(nd.sum()),
                         "pareto_x_low": x[nd].min(), "pareto_x_high": x[nd].max()})
    return pd.DataFrame(rows)


def q(s, p):
    return float(np.quantile(s, p))


def monte_carlo(cal_avg, macro_avg, cals, macros):
    log(f"(d) Monte Carlo, {N_MC} Latin-hypercube draws, nested with {len(CAL_SEEDS)} calibration seeds")
    sampler = qmc.LatinHypercube(d=len(THETA), seed=SEED)
    u = sampler.random(N_MC)
    lo = np.array([t[2] for t in THETA]); hi = np.array([t[3] for t in THETA])
    draws = qmc.scale(u, lo, hi)
    dpar = pd.DataFrame(draws, columns=THETA_NAMES); dpar.insert(0, "draw", np.arange(N_MC))
    # balanced seed assignment: each calibration seed gets N_MC / 50 = 40 draws, in random order
    rng = np.random.default_rng(SEED)
    dpar["cal_seed"] = np.array(CAL_SEEDS)[rng.permutation(N_MC) % len(CAL_SEEDS)]
    dpar.to_csv(os.path.join(OUT, "mc_parameter_draws.csv"), index=False)
    arrays = {k: dpar[k].values for k in THETA_NAMES}

    out = []
    # (i) nested: coefficients + calibration seed
    for sd_ in CAL_SEEDS:
        idx = np.where(dpar.cal_seed.values == sd_)[0]
        thb = theta_dict(arrays=arrays, idx=idx)
        for conv in CONVENTIONS:
            for sname in SCEN_NAMES:
                lev = scenario_levels(macros[sd_], SCENARIOS[SCEN_NAMES.index(sname)])
                for sec in SECTORS:
                    o = batch_optima(XGRID, cals[sd_].loc[sec], lev, thb, conv)
                    out.append(pd.DataFrame({"mc_type": "nested", "convention": conv, "draw": idx,
                                             "cal_seed": sd_, "scenario": sname, "sector": sec,
                                             "rel_vol": cals[sd_].loc[sec, "rel_vol"],
                                             "median_roa": cals[sd_].loc[sec, "median_roa"], **o}))
    # (ii) coefficients only, seed-averaged calibration
    thb = theta_dict(arrays=arrays)
    for conv in CONVENTIONS:
        for sname in SCEN_NAMES:
            lev = scenario_levels(macro_avg, SCENARIOS[SCEN_NAMES.index(sname)])
            for sec in SECTORS:
                o = batch_optima(XGRID, cal_avg.loc[sec], lev, thb, conv)
                out.append(pd.DataFrame({"mc_type": "coef_only", "convention": conv, "draw": np.arange(N_MC),
                                         "cal_seed": -1, "scenario": sname, "sector": sec,
                                         "rel_vol": cal_avg.loc[sec, "rel_vol"],
                                         "median_roa": cal_avg.loc[sec, "median_roa"], **o}))
    mc = pd.concat(out, ignore_index=True).sort_values(["mc_type", "convention", "scenario", "sector", "draw"])
    mc.to_csv(os.path.join(OUT, "mc_optima_long.csv.gz"), index=False, compression="gzip")

    summ = []
    for (mt, conv, sname, sec), g in mc.groupby(["mc_type", "convention", "scenario", "sector"], sort=False):
        summ.append({"mc_type": mt, "convention": conv, "scenario": sname, "sector": sec, "n": len(g),
                     "xW_median": g.x_wacc_min.median(), "xW_p05": q(g.x_wacc_min, .05), "xW_p95": q(g.x_wacc_min, .95),
                     "xW_p25": q(g.x_wacc_min, .25), "xW_p75": q(g.x_wacc_min, .75),
                     "xW_share_corner_lo": g.corner_w_lo.mean(), "xW_share_corner_hi": g.corner_w_hi.mean(),
                     "xV_median": g.x_value_max.median(), "xV_p05": q(g.x_value_max, .05), "xV_p95": q(g.x_value_max, .95),
                     "xV_p25": q(g.x_value_max, .25), "xV_p75": q(g.x_value_max, .75),
                     "xV_share_corner_lo": g.corner_v_lo.mean(), "xV_share_corner_hi": g.corner_v_hi.mean(),
                     "waccmin_median": g.wacc_min.median(), "pd_at_xW_median": g.pd_at_x_wacc.median(),
                     "pd_at_xW_p95": q(g.pd_at_x_wacc, .95),
                     "share_price_floor_binding": float((g.price_of_risk <= PRICE_FLOOR + 1e-12).mean())})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, "mc_summary_sector_scenario.csv"), index=False)

    # paired scenario differences (same draw, same sector, same calibration seed) relative to Base
    diffs = []
    for (mt, conv), gg in mc.groupby(["mc_type", "convention"]):
        piv = gg.pivot_table(index=["draw", "sector"], columns="scenario", values="x_wacc_min")
        piv_v = gg.pivot_table(index=["draw", "sector"], columns="scenario", values="x_value_max")
        for sname in SCEN_NAMES[1:]:
            for sec in SECTORS + ["All sectors"]:
                sel = slice(None) if sec == "All sectors" else (piv.index.get_level_values("sector") == sec)
                d = (piv[sname] - piv["Base"])[sel]
                dv = (piv_v[sname] - piv_v["Base"])[sel]
                diffs.append({"mc_type": mt, "convention": conv, "scenario": sname, "sector": sec, "n": len(d),
                              "dxW_median": d.median(), "dxW_p05": q(d, .05), "dxW_p95": q(d, .95),
                              "dxW_share_negative": (d < 0).mean(), "dxW_share_positive": (d > 0).mean(),
                              "dxV_median": dv.median(), "dxV_p05": q(dv, .05), "dxV_p95": q(dv, .95),
                              "dxV_share_negative": (dv < 0).mean(), "dxV_share_positive": (dv > 0).mean()})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(os.path.join(OUT, "mc_scenario_differences.csv"), index=False)

    # rank correlation of each coefficient (and the calibrated sector inputs) with the optima,
    # nested MC, Base scenario, zero-debt-beta convention
    base = mc[(mc.mc_type == "nested") & (mc.convention == "zero") & (mc.scenario == "Base")].merge(
        dpar.drop(columns="cal_seed"), on="draw")
    src = []
    for sec in SECTORS + ["All sectors"]:
        g = base if sec == "All sectors" else base[base.sector == sec]
        for k in THETA_NAMES + ["rel_vol", "median_roa"]:
            rw = stats.spearmanr(g[k], g.x_wacc_min).statistic
            rv = stats.spearmanr(g[k], g.x_value_max).statistic
            src.append({"sector": sec, "parameter": k, "spearman_xW": rw, "spearman_xV": rv})
    src = pd.DataFrame(src)
    src.to_csv(os.path.join(OUT, "mc_spearman_base.csv"), index=False)

    # robustness of the sector ranking: rank sectors within each nested draw (Base, both conventions)
    rk = []
    for conv in CONVENTIONS:
        for col in ["x_wacc_min", "x_value_max"]:
            g = mc[(mc.mc_type == "nested") & (mc.convention == conv) & (mc.scenario == "Base")]
            piv = g.pivot_table(index="draw", columns="sector", values=col)[SECTORS]
            ranks = piv.rank(axis=1, method="average")   # 1 = lowest optimum
            for sec in SECTORS:
                rk.append({"convention": conv, "optimum": col, "sector": sec, "mean_rank": ranks[sec].mean(),
                           "rank_p05": q(ranks[sec], .05), "rank_p95": q(ranks[sec], .95),
                           "share_lowest": float((ranks[sec] <= 1.0).mean()),
                           "share_highest": float((ranks[sec] >= len(SECTORS)).mean())})
    rk = pd.DataFrame(rk)
    rk.to_csv(os.path.join(OUT, "mc_sector_rank_robustness.csv"), index=False)
    return dpar, mc, summ, diffs, src, rk


def tornado(cal, macro):
    log("(d) one-at-a-time tornado (Base and Recession)")
    avg = cal[["median_roa", "rel_vol", "alpha", "recovery"]].mean()
    rows = []
    targets = [("Average firm", avg)] + [(s, cal.loc[s]) for s in SECTORS]
    for sname in ["Base", "Recession"]:
        lev = scenario_levels(macro, SCENARIOS[SCEN_NAMES.index(sname)])
        for name, row in targets:
            b = summarise_curve(XGRID, evaluate(XGRID, row, lev, theta_dict()))
            bp = summarise_curve(XGRID, evaluate(XGRID, row, lev, theta_dict(), "spread_prem"))
            for (k, base, lo, hi, _) in THETA:
                res, resp = {}, {}
                for tag, val in (("low", lo), ("high", hi)):
                    res[tag] = summarise_curve(XGRID, evaluate(XGRID, row, lev, theta_dict({k: val})))
                    resp[tag] = summarise_curve(XGRID, evaluate(XGRID, row, lev, theta_dict({k: val}), "spread_prem"))
                rows.append({"scenario": sname, "target": name, "parameter": k, "base_value": base,
                             "low_value": lo, "high_value": hi,
                             "xW_base": b["x_wacc_min"], "xW_at_low": res["low"]["x_wacc_min"],
                             "xW_at_high": res["high"]["x_wacc_min"],
                             "xW_swing": abs(res["high"]["x_wacc_min"] - res["low"]["x_wacc_min"]),
                             "xV_base": b["x_value_max"], "xV_at_low": res["low"]["x_value_max"],
                             "xV_at_high": res["high"]["x_value_max"],
                             "xV_swing": abs(res["high"]["x_value_max"] - res["low"]["x_value_max"]),
                             "xW_spreadprem_base": bp["x_wacc_min"],
                             "xW_spreadprem_at_low": resp["low"]["x_wacc_min"],
                             "xW_spreadprem_at_high": resp["high"]["x_wacc_min"],
                             "xW_spreadprem_swing": abs(resp["high"]["x_wacc_min"] - resp["low"]["x_wacc_min"])})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "tornado_oat.csv"), index=False)
    return t, avg


def channel_decomposition(cal, macro, avg):
    """Which scenario channel moves the optimum: drift (g_to_mu x dGDP), price of default risk (k_cs),
    rate level. Deterministic, base coefficients, seed-averaged calibration."""
    log("(d) scenario channel decomposition")
    rows = []
    targets = [("Average firm", avg)] + [(s, cal.loc[s]) for s in SECTORS]
    variants = {"all_channels": (True, True), "no_drift_channel": (False, True),
                "no_price_channel": (True, False), "rate_only": (False, False)}
    for sname in SCEN_NAMES[1:]:
        lev0 = scenario_levels(macro, SCENARIOS[SCEN_NAMES.index(sname)])
        for vname, (drift_on, price_on) in variants.items():
            lev = dict(lev0)
            if not drift_on:
                lev["dg"] = 0.0
            if not price_on:
                lev["k_cs"] = 1.0
            for name, row in targets:
                s = summarise_curve(XGRID, evaluate(XGRID, row, lev, theta_dict()))
                sp = summarise_curve(XGRID, evaluate(XGRID, row, lev, theta_dict(), "spread_prem"))
                rows.append({"scenario": sname, "variant": vname, "target": name,
                             "x_wacc_min": s["x_wacc_min"], "x_value_max": s["x_value_max"],
                             "x_wacc_min_spreadprem": sp["x_wacc_min"], "price_of_risk": s["price_of_risk"]})
    cd = pd.DataFrame(rows)
    cd.to_csv(os.path.join(OUT, "scenario_channel_decomposition.csv"), index=False)
    return cd


def dgp_sensitivity(tang, cal42, macro42, cal_avg, macro_avg):
    log("(d) DGP grid sensitivity (lambda x sigma_zeta^2)")
    rows, chk = [], []
    lev_avg = scenario_levels(macro_avg, SCENARIOS[0])
    opt_avg = {sec: summarise_curve(XGRID, evaluate(XGRID, cal_avg.loc[sec], lev_avg, theta_dict()))
               for sec in SECTORS}
    for lam in DGP_LAMS:
        for sz2 in DGP_SZ2:
            df = common.generate_panel(seed=SEED, add_extra_features=False, lam=lam, sigma_zeta=float(np.sqrt(sz2)))
            cal, macro = calibrate(df, tang)
            chk.append({"lam": lam, "sigma_zeta2": sz2,
                        "max_abs_diff_earn_vol": float((cal["earn_vol"] - cal42["earn_vol"]).abs().max()),
                        "max_abs_diff_median_roa": float((cal["median_roa"] - cal42["median_roa"]).abs().max()),
                        "max_abs_diff_rf": abs(macro["rf"] - macro42["rf"]), "max_abs_diff_cs": abs(macro["cs"] - macro42["cs"])})
            rbar = df["interest_rate"].mean() / 100
            for sec in SECTORS:
                s = summarise_curve(XGRID, evaluate(XGRID, cal.loc[sec], scenario_levels(macro, SCENARIOS[0]), theta_dict()))
                rows.append({"lam": lam, "sigma_zeta2": sz2, "sigma_zeta": np.sqrt(sz2), "sector": sec,
                             "actual_mean_dr": cal.loc[sec, "actual_mean_dr"], "actual_sd_dr":
                             df.loc[df.sector == sec, "debt_ratio"].std(),
                             "x_wacc_min_own_calibration": s["x_wacc_min"],
                             "x_value_max_own_calibration": s["x_value_max"],
                             "x_wacc_min_seedavg": opt_avg[sec]["x_wacc_min"],
                             "x_value_max_seedavg": opt_avg[sec]["x_value_max"],
                             "gap_xW_seedavg_minus_actual": opt_avg[sec]["x_wacc_min"] - cal.loc[sec, "actual_mean_dr"],
                             "gap_xV_seedavg_minus_actual": opt_avg[sec]["x_value_max"] - cal.loc[sec, "actual_mean_dr"],
                             "legacy_optimum_analytic": 0.3 - 0.5 * rbar})
    dg = pd.DataFrame(rows); dg.to_csv(os.path.join(OUT, "dgp_grid_sensitivity.csv"), index=False)
    ck = pd.DataFrame(chk); ck.to_csv(os.path.join(OUT, "dgp_grid_calibration_check.csv"), index=False)
    return dg, ck


def seed_sensitivity(cals, macros):
    log("(d) seed sensitivity at base coefficients (50 seeds)")
    srows = []
    for sd_ in CAL_SEEDS:
        cal, macro = cals[sd_], macros[sd_]
        for conv in CONVENTIONS:
            for sname in SCEN_NAMES:
                lev = scenario_levels(macro, SCENARIOS[SCEN_NAMES.index(sname)])
                for sec in SECTORS:
                    s = summarise_curve(XGRID, evaluate(XGRID, cal.loc[sec], lev, theta_dict(), conv))
                    srows.append({"seed": sd_, "convention": conv, "scenario": sname, "sector": sec, "rf": lev["rf"],
                                  "macro_spread": lev["cs"], "earn_vol": cal.loc[sec, "earn_vol"],
                                  "rel_vol": cal.loc[sec, "rel_vol"], "median_roa": cal.loc[sec, "median_roa"],
                                  "actual_mean_dr": cal.loc[sec, "actual_mean_dr"],
                                  "x_wacc_min": s["x_wacc_min"], "x_value_max": s["x_value_max"],
                                  "legacy_optimum_analytic": 0.3 - 0.5 * lev["rf"]})
    ss = pd.DataFrame(srows); ss.to_csv(os.path.join(OUT, "seed_sensitivity.csv"), index=False)
    ssum = ss.groupby(["convention", "scenario", "sector"]).agg(
        xW_mean=("x_wacc_min", "mean"), xW_sd=("x_wacc_min", "std"), xW_min=("x_wacc_min", "min"),
        xW_max=("x_wacc_min", "max"), xV_mean=("x_value_max", "mean"), xV_sd=("x_value_max", "std"),
        xV_min=("x_value_max", "min"), xV_max=("x_value_max", "max"),
        rel_vol_mean=("rel_vol", "mean"), rel_vol_sd=("rel_vol", "std"),
        actual_dr_mean=("actual_mean_dr", "mean"), actual_dr_sd=("actual_mean_dr", "std"),
        legacy_opt_min=("legacy_optimum_analytic", "min"), legacy_opt_max=("legacy_optimum_analytic", "max")).reset_index()
    ssum.to_csv(os.path.join(OUT, "seed_sensitivity_summary.csv"), index=False)
    # rank stability across seeds (Base, zero beta): Spearman between every pair of seeds' sector rankings
    b = ss[(ss.convention == "zero") & (ss.scenario == "Base")]
    rank_rows = []
    for col in ["x_wacc_min", "x_value_max"]:
        piv = b.pivot_table(index="seed", columns="sector", values=col)[SECTORS]
        rhos = [stats.spearmanr(piv.iloc[i], piv.iloc[j]).statistic
                for i in range(len(piv)) for j in range(i + 1, len(piv))]
        rr = piv.rank(axis=1, method="average")
        rank_rows.append({"optimum": col, "n_seeds": len(piv), "n_pairs": len(rhos),
                          "pairwise_spearman_median": float(np.median(rhos)),
                          "pairwise_spearman_p05": q(rhos, .05), "pairwise_spearman_p95": q(rhos, .95),
                          "share_pairs_negative": float(np.mean(np.array(rhos) < 0)),
                          **{f"share_lowest_{s}": float((rr[s] <= 1).mean()) for s in SECTORS},
                          **{f"share_highest_{s}": float((rr[s] >= len(SECTORS)).mean()) for s in SECTORS}})
    rs = pd.DataFrame(rank_rows); rs.to_csv(os.path.join(OUT, "seed_rank_stability.csv"), index=False)
    return ss, ssum, rs


# ===========================================================================
# Figures
# ===========================================================================
SEC_COLOR = {"ConsDisc": "#E69F00", "Energy": "#56B4E9", "Financials": "#009E73", "Healthcare": "#000000",
             "Industrials": "#0072B2", "Materials": "#D55E00", "Tech": "#CC79A7", "Utilities": "#999999"}
SEC_LS = {"ConsDisc": "-", "Energy": "--", "Financials": "-.", "Healthcare": "-",
          "Industrials": "--", "Materials": "-.", "Tech": "-", "Utilities": "--"}


def figures(curves, det, mc, tor, sector_tab):
    plt = common.set_style()
    import matplotlib as mpl
    OI = common.OKABE_ITO
    base_c = curves[(curves.scenario == "Base") & (curves.convention == "zero")]
    bdet = det[(det.scenario == "Base") & (det.convention == "zero")].set_index("sector")

    # 1. U-shaped WACC curves (a) and firm value (b) by sector, Base scenario, zero debt beta
    fig, axes = plt.subplots(1, 2, figsize=(common.FIG_W_DOUBLE, 3.2))
    for sec in SECTORS:
        c = base_c[base_c.sector == sec]
        lw = 1.4 if sec == "Utilities" else 1.2
        axes[0].plot(c.x, c.wacc * 100, color=SEC_COLOR[sec], ls=SEC_LS[sec], lw=lw, label=SECTOR_LABEL[sec])
        axes[0].plot(bdet.loc[sec, "x_wacc_min"], bdet.loc[sec, "wacc_min"] * 100, "o", color=SEC_COLOR[sec],
                     ms=3.5, mec="black", mew=0.4)
        axes[1].plot(c.x, (c.value_rel - 1) * 100, color=SEC_COLOR[sec], ls=SEC_LS[sec], lw=lw)
        axes[1].plot(bdet.loc[sec, "x_value_max"], (bdet.loc[sec, "value_max"] - 1) * 100, "o",
                     color=SEC_COLOR[sec], ms=3.5, mec="black", mew=0.4)
    axes[0].set_xlabel("Debt ratio, D/V"); axes[0].set_ylabel("WACC (%)")
    axes[1].set_xlabel("Debt ratio, D/V"); axes[1].set_ylabel("Change in firm value, $V_L/V_U-1$ (%)")
    axes[1].axhline(0, color="grey", lw=0.5)
    lo = min(base_c.wacc.min() * 100 - 0.1, 7.0)
    axes[0].set_ylim(lo, min(base_c.wacc.max() * 100 + 0.1, 10.0))
    axes[1].set_ylim(max((base_c.value_rel.min() - 1) * 100, -6), (base_c.value_rel.max() - 1) * 100 + 0.5)
    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.16, 1.02, lab, transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.set_xlim(X_LO, X_HI)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               handlelength=2.6)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    common.savefig(fig, os.path.join(OUT, "fig_e5_wacc_value_curves")); plt.close(fig)

    # 2. WACC vs default probability (log scale); Pareto set coloured by debt ratio, dominated part grey
    fig, axes = plt.subplots(2, 4, figsize=(common.FIG_W_DOUBLE, 4.0), sharex=True, sharey=True)
    PD_MIN, PD_MAX = 1e-6, 0.5
    xs_hi = float(np.ceil(bdet["pareto_x_high"].max() * 20) / 20)
    norm = mpl.colors.Normalize(X_LO, xs_hi); cmap = mpl.colormaps["viridis"]
    trunc = {}
    for ax, sec in zip(axes.ravel(), SECTORS):
        call = base_c[base_c.sector == sec]
        nd_all = nondominated_2d(call.wacc.values, call.pd_T.values)
        call = call.assign(nd=nd_all)
        c = call[call.pd_T >= PD_MIN]
        x_first = float(c.x.min())
        trunc[sec] = x_first
        ax.plot(c.pd_T[~c.nd] * 100, c.wacc[~c.nd] * 100, color="#BBBBBB", lw=1.0)
        ax.scatter(c.pd_T[c.nd] * 100, c.wacc[c.nd] * 100, c=c.x[c.nd], cmap=cmap, norm=norm, s=5, lw=0)
        ax.plot(bdet.loc[sec, "pd_at_x_wacc"] * 100, bdet.loc[sec, "wacc_min"] * 100, marker="*", color=OI[5],
                ms=8, mec="black", mew=0.4, ls="none")
        ax.annotate(f"$x$={bdet.loc[sec, 'x_wacc_min']:.2f}",
                    (bdet.loc[sec, "pd_at_x_wacc"] * 100, bdet.loc[sec, "wacc_min"] * 100),
                    xytext=(0, -9), textcoords="offset points", fontsize=6.5, ha="center")
        first = c.iloc[0]
        lbl = f"$x$={x_first:.2f}" if x_first > X_LO + 1e-9 else f"$x$={X_LO:.2f}"
        ax.annotate(lbl, (first.pd_T * 100, first.wacc * 100), xytext=(3, 4), textcoords="offset points",
                    fontsize=6.5, ha="left")
        ax.set_title(SECTOR_LABEL[sec], fontsize=8, pad=2)
        ax.set_xscale("log"); ax.set_xlim(PD_MIN * 100, PD_MAX * 100)
        ax.set_ylim(7.5, 8.8)
        ax.tick_params(labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel(f"{THETA_BASE['T']:.0f}-year default prob. (%)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("WACC (%)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes([0.915, 0.15, 0.015, 0.7])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Debt ratio on Pareto set, D/V", fontsize=8); cb.ax.tick_params(labelsize=7)
    common.savefig(fig, os.path.join(OUT, "fig_e5_pareto_wacc_vs_pd")); plt.close(fig)
    pd.DataFrame([{"sector": s, "first_plotted_x": v, "pd_min_plotted": PD_MIN} for s, v in trunc.items()]
                 ).to_csv(os.path.join(OUT, "fig_pareto_truncation.csv"), index=False)

    # 3. Nested Monte Carlo distribution of x_W and x_V (zero debt beta)
    m = mc[(mc.mc_type == "nested") & (mc.convention == "zero")]
    fig, axes = plt.subplots(2, 1, figsize=(common.FIG_W_DOUBLE, 4.6), sharex=True)
    scen_color = {s: c for s, c in zip(SCEN_NAMES, [OI[7], OI[4], OI[5], OI[6], OI[2]])}
    w = 0.15
    for ax, col, ylab in zip(axes, ["x_wacc_min", "x_value_max"],
                             ["WACC-minimising D/V", "Value-maximising D/V"]):
        for i, sec in enumerate(SECTORS):
            for j, sname in enumerate(SCEN_NAMES):
                v = m[(m.sector == sec) & (m.scenario == sname)][col].values
                pos = i + (j - 2) * w
                bp = ax.boxplot(v, positions=[pos], widths=w * 0.85, whis=(5, 95), showfliers=False,
                                patch_artist=True, medianprops=dict(color="white" if sname == "Base" else "black", lw=0.8),
                                boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
                bp["boxes"][0].set_facecolor(scen_color[sname]); bp["boxes"][0].set_alpha(0.85)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, X_HI + 0.02)
        ax.axhline(X_LO, color="grey", lw=0.4, ls=":"); ax.axhline(X_HI, color="grey", lw=0.4, ls=":")
    axes[1].set_xticks(range(len(SECTORS)))
    axes[1].set_xticklabels([SECTOR_LABEL[s] for s in SECTORS], rotation=0, fontsize=7)
    handles = [mpl.patches.Patch(facecolor=scen_color[s], label=SCEN_LABEL[s]) for s in SCEN_NAMES]
    axes[0].legend(handles=handles, ncol=5, frameon=False, fontsize=7, loc="upper center",
                   bbox_to_anchor=(0.5, 1.16))
    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.07, 1.0, lab, transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    common.savefig(fig, os.path.join(OUT, "fig_e5_mc_optimum_distributions")); plt.close(fig)

    # 4. Tornado (average firm), rows = Base / Recession, columns = x_W / x_V
    pretty = {"tau": "Tax rate $\\tau$", "erp": "Equity risk premium", "beta_u": "Unlevered beta",
              "rf_shift": "Risk-free rate shift", "sigma_bar": "Asset volatility $\\bar\\sigma_A$",
              "s0": "Base spread $s_0$", "lam_cs": "Spread/expected-loss ratio", "alpha_scale": "Distress-cost scale",
              "recov_shift": "Recovery-rate shift", "T": "Debt horizon $T$", "mu_shift": "Asset drift shift",
              "g_to_mu": "GDP-to-drift loading"}
    fig, axes = plt.subplots(2, 2, figsize=(common.FIG_W_DOUBLE, 5.6))
    for r_i, sname in enumerate(["Base", "Recession"]):
        t = tor[(tor.target == "Average firm") & (tor.scenario == sname)]
        for c_i, (pre, xlab) in enumerate(zip(["xW", "xV"], ["WACC-minimising D/V", "Value-maximising D/V"])):
            ax = axes[r_i, c_i]
            tt = t.sort_values(f"{pre}_swing", ascending=True)
            base = tt[f"{pre}_base"].iloc[0]
            y = np.arange(len(tt))
            lo_v = tt[f"{pre}_at_low"].values; hi_v = tt[f"{pre}_at_high"].values
            ax.barh(y, lo_v - base, left=base, color=OI[4], height=0.6, label="Parameter at low end")
            ax.barh(y, hi_v - base, left=base, color=OI[5], height=0.6, label="Parameter at high end", alpha=0.9)
            ax.axvline(base, color="black", lw=0.7)
            ax.set_yticks(y); ax.set_yticklabels([pretty[p] for p in tt.parameter], fontsize=7)
            ax.set_xlabel(f"{xlab} ({SCEN_LABEL[sname]})")
            ax.set_xlim(max(0, min(lo_v.min(), hi_v.min()) - 0.03), min(0.9, max(lo_v.max(), hi_v.max()) + 0.03))
            ax.text(-0.5, 1.0, "(" + "abcd"[2 * r_i + c_i] + ")", transform=ax.transAxes, fontsize=9,
                    fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    common.savefig(fig, os.path.join(OUT, "fig_e5_tornado")); plt.close(fig)

    # 5. Sector figure: nested MC bands for x_W and x_V vs actual means (Base, zero debt beta)
    st = sector_tab.set_index("sector").loc[SECTORS]
    fig, ax = plt.subplots(figsize=(common.FIG_W_SINGLE, 4.0))
    y = np.arange(len(SECTORS))[::-1].astype(float)
    ax.hlines(y + 0.24, st.mc_nested_xW_p05, st.mc_nested_xW_p95, color=OI[4], lw=2.2)
    ax.plot(st.x_wacc_min, y + 0.24, "o", color=OI[4], mec="black", mew=0.4, ms=4,
            label="WACC-min. D/V: base value, MC 5-95%")
    ax.hlines(y + 0.08, st.mc_nested_xV_p05, st.mc_nested_xV_p95, color=OI[6], lw=2.2)
    ax.plot(st.x_value_max, y + 0.08, "^", color=OI[6], mec="black", mew=0.4, ms=4,
            label="Value-max. D/V: base value, MC 5-95%")
    ax.plot(st.actual_mean_dr_synthetic_ref, y - 0.10, "D", color=OI[0], mec="black", mew=0.4, ms=4,
            label="Mean D/A, reference synthetic panel")
    ax.plot(st.actual_mean_dr_real, y - 0.26, "s", color=OI[2], mec="black", mew=0.4, ms=3.5,
            label="Mean D/A, 46-firm real sample")
    ax.set_yticks(y); ax.set_yticklabels([SECTOR_LABEL[s] for s in SECTORS], fontsize=7)
    ax.set_xlabel("Debt ratio (model: D/V; observed: book D/A)"); ax.set_xlim(0, X_HI + 0.02)
    ax.set_ylim(-0.6, len(SECTORS) - 0.4)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=1,
              borderaxespad=0.2, handlelength=1.5)
    fig.tight_layout()
    common.savefig(fig, os.path.join(OUT, "fig_e5_sector_actual_vs_model")); plt.close(fig)


# ===========================================================================
def main():
    t0 = time.time()
    np.random.seed(SEED)
    df = common.generate_panel(seed=SEED)
    log(f"reference synthetic panel: {len(df)} rows")

    cnt, leg, rbar = run_legacy(df)

    firm_tang, tang = real_tangibility()
    firm_tang.to_csv(os.path.join(OUT, "real_tangibility_by_firm.csv"), index=False)

    log(f"calibrating on {len(CAL_SEEDS)} panel seeds")
    cals, macros = {}, {}
    for sd_ in CAL_SEEDS:
        d = df if sd_ == SEED else common.generate_panel(seed=sd_, add_extra_features=False)
        cals[sd_], macros[sd_] = calibrate(d, tang)
    cal42, macro42 = cals[SEED], macros[SEED]
    cal, macro = seed_average(cals, macros, tang)

    cal_out = cal.copy()
    cal_out["sigma_A_base"] = THETA_BASE["sigma_bar"] * cal["rel_vol"]
    cal_out["n_real_firms_tangibility"] = tang.loc[SECTORS, "n_firms"].values
    for col in ["earn_vol", "rel_vol", "median_roa", "actual_mean_dr"]:
        allv = pd.concat([c[col] for c in cals.values()], axis=1)
        cal_out[f"{col}_seed_sd"] = allv.std(axis=1).loc[SECTORS].values
        cal_out[f"{col}_seed42"] = cal42[col].loc[SECTORS].values
    cal_out["source_earn_vol"] = "mean over 50 synthetic panels (seeds 42-91) of the sector median over firms of the s.d. of year-on-year change in ROA"
    cal_out["source_median_roa"] = "mean over 50 synthetic panels of the sector median ROA (drift mu_s)"
    cal_out["source_tangibility"] = "46-firm Yahoo Finance sample: median over firms of median Net PPE / Total Assets"
    cal_out.to_csv(os.path.join(OUT, "calibration_sector_parameters.csv"))
    pd.DataFrame([{"seed": s, **macros[s]} for s in CAL_SEEDS]).to_csv(
        os.path.join(OUT, "calibration_macro_by_seed.csv"), index=False)
    pd.concat([cals[s].assign(seed=s).reset_index() for s in CAL_SEEDS]).to_csv(
        os.path.join(OUT, "calibration_sector_by_seed.csv"), index=False)
    pd.DataFrame([{"name": "rf_base", "value": macro["rf"], "source": "mean over 50 synthetic panels of mean interest_rate / 100"},
                  {"name": "macro_spread_base", "value": macro["cs"], "source": "mean over 50 synthetic panels of mean credit_spread / 100"},
                  {"name": "gdp_growth_base", "value": macro["g"], "source": "mean over 50 synthetic panels of mean gdp_growth / 100 (enters only as scenario shift)"},
                  {"name": "price_floor", "value": PRICE_FLOOR, "source": "price of default risk = max(lam_cs x k_cs, 1): priced spread never below expected loss"},
                  {"name": "alpha_min", "value": ALPHA_MIN, "source": "assumed; distress cost at full tangibility"},
                  {"name": "alpha_max", "value": ALPHA_MAX, "source": "assumed; distress cost at zero tangibility"},
                  {"name": "recovery_min", "value": RECOV_MIN, "source": "assumed; recovery at zero tangibility"},
                  {"name": "recovery_max", "value": RECOV_MAX, "source": "assumed; recovery at full tangibility"},
                  {"name": "x_grid", "value": f"{X_LO}:{X_STEP}:{X_HI} ({len(XGRID)} points)", "source": "feasible range of the original specification"},
                  {"name": "band", "value": BAND_BP, "source": "near-optimal band = WACC within 10 bp of minimum"},
                  {"name": "cal_seeds", "value": f"{CAL_SEEDS[0]}-{CAL_SEEDS[-1]} ({len(CAL_SEEDS)})", "source": "panel seeds for calibration uncertainty"},
                  ] + [{"name": t[0], "value": t[1], "source": f"MC range U[{t[2]}, {t[3]}]. {t[4]}"} for t in THETA]
                 ).to_csv(os.path.join(OUT, "model_parameters.csv"), index=False)
    sc_rows = []
    for s in SCENARIOS:
        lv = scenario_levels(macro, s)
        sc_rows.append({"scenario": s[0], "d_gdp_pp": s[1], "d_rate_pp": s[2], "d_spread_pp": s[3], **lv,
                        "drift_shift_base": THETA_BASE["g_to_mu"] * lv["dg"],
                        "price_of_risk_base": max(lv["k_cs"] * THETA_BASE["lam_cs"], PRICE_FLOOR),
                        "price_of_risk_mc_low": max(lv["k_cs"] * 1.5, PRICE_FLOOR),
                        "price_of_risk_mc_high": max(lv["k_cs"] * 4.0, PRICE_FLOOR),
                        "unfloored_price_base": lv["k_cs"] * THETA_BASE["lam_cs"]})
    pd.DataFrame(sc_rows).to_csv(os.path.join(OUT, "scenarios.csv"), index=False)

    log("(b,c) deterministic base-parameter results")
    det, curves = deterministic(cal, macro, "seed_average", keep_curves=True)
    det42, _ = deterministic(cal42, macro42, "seed42")
    det_all = pd.concat([det, det42], ignore_index=True)
    det_all.to_csv(os.path.join(OUT, "base_results_sector_scenario.csv"), index=False)
    curves.to_csv(os.path.join(OUT, "curves_sector_scenario.csv.gz"), index=False, compression="gzip")
    gc = new_model_grid_counts(cal, macro)
    gc.to_csv(os.path.join(OUT, "new_model_pareto_count_vs_grid.csv"), index=False)

    # algebra checks
    lev = scenario_levels(macro, SCENARIOS[0]); row = cal.loc["Tech"]; th = theta_dict()
    r = evaluate(XGRID, row, lev, th)
    rp = evaluate(XGRID, row, lev, th, "spread_prem")
    r_e_alt = r["r_u"] + (r["r_u"] - r["r_d"]) * (1 - th["tau"]) * XGRID / (1 - XGRID)
    wacc_alt = (1 - XGRID) * r_e_alt + XGRID * r["r_d"] * (1 - th["tau"])
    alg = {"max_abs_diff_wacc_rdHamada_vs_ru(1-tau x)": float(np.max(np.abs(wacc_alt - r["r_u"] * (1 - th["tau"] * XGRID)))),
           "wacc_rdHamada_monotone_decreasing": bool(np.all(np.diff(wacc_alt) < 0)),
           "max_abs_diff_wacc_zero_vs_closed_form": float(np.max(np.abs(r["wacc"] - (r["r_u"] * (1 - th["tau"] * XGRID)
                                                                                       + (1 - th["tau"]) * XGRID * r["spread"])))),
           "max_abs_diff_wacc_spreadprem_vs_closed_form": float(np.max(np.abs(rp["wacc"] - (rp["r_u"] * (1 - th["tau"] * XGRID)
                                                                                             + (1 - th["tau"]) * XGRID * rp["el"])))),
           "value_identical_across_conventions": bool(np.array_equal(r["value"], rp["value"]))}

    dpar, mc, summ, diffs, src, rk = monte_carlo(cal, macro, cals, macros)
    tor, avg = tornado(cal, macro)
    chan = channel_decomposition(cal, macro, avg)
    dg, ck = dgp_sensitivity(tang, cal42, macro42, cal, macro)
    ss, ssum, rstab = seed_sensitivity(cals, macros)

    # (e) sector table (Base scenario, zero debt beta, seed-averaged calibration; nested MC bands)
    real = common.load_real_panel()
    real_dr = real.groupby("sector")["debt_ratio"].agg(["mean", "count"])
    real_firms = real.groupby("sector")["ticker"].nunique() if "ticker" in real.columns else None
    b = det[(det.scenario == "Base") & (det.convention == "zero")].set_index("sector")
    bp = det[(det.scenario == "Base") & (det.convention == "spread_prem")].set_index("sector")
    b42 = det42[(det42.scenario == "Base") & (det42.convention == "zero")].set_index("sector")
    sel = lambda mt, cv: summ[(summ.scenario == "Base") & (summ.mc_type == mt) & (summ.convention == cv)].set_index("sector").loc[SECTORS]  # noqa: E731
    smn, smc, smp = sel("nested", "zero"), sel("coef_only", "zero"), sel("nested", "spread_prem")
    st = pd.DataFrame({
        "sector": SECTORS,
        "actual_mean_dr_synthetic_ref": cal42.loc[SECTORS, "actual_mean_dr"].values,
        "actual_sd_dr_synthetic_ref": df.groupby("sector")["debt_ratio"].std().loc[SECTORS].values,
        "actual_mean_dr_synthetic_50seeds": cal.loc[SECTORS, "actual_mean_dr"].values,
        "actual_mean_dr_real": real_dr.reindex(SECTORS)["mean"].values,
        "n_real_firm_years": real_dr.reindex(SECTORS)["count"].values,
        "n_real_firms": real_firms.reindex(SECTORS).values if real_firms is not None else np.nan,
        "pareto_x_low": b.loc[SECTORS, "pareto_x_low"].values, "pareto_x_high": b.loc[SECTORS, "pareto_x_high"].values,
        "band10bp_x_low": b.loc[SECTORS, "band10bp_x_low"].values, "band10bp_x_high": b.loc[SECTORS, "band10bp_x_high"].values,
        "x_wacc_min": b.loc[SECTORS, "x_wacc_min"].values, "x_value_max": b.loc[SECTORS, "x_value_max"].values,
        "x_wacc_min_spreadprem": bp.loc[SECTORS, "x_wacc_min"].values,
        "x_wacc_min_seed42": b42.loc[SECTORS, "x_wacc_min"].values,
        "x_value_max_seed42": b42.loc[SECTORS, "x_value_max"].values,
        "mc_nested_xW_median": smn.xW_median.values, "mc_nested_xW_p05": smn.xW_p05.values, "mc_nested_xW_p95": smn.xW_p95.values,
        "mc_nested_xV_median": smn.xV_median.values, "mc_nested_xV_p05": smn.xV_p05.values, "mc_nested_xV_p95": smn.xV_p95.values,
        "mc_coef_xW_p05": smc.xW_p05.values, "mc_coef_xW_p95": smc.xW_p95.values,
        "mc_coef_xV_p05": smc.xV_p05.values, "mc_coef_xV_p95": smc.xV_p95.values,
        "mc_nested_spreadprem_xW_median": smp.xW_median.values, "mc_nested_spreadprem_xW_p05": smp.xW_p05.values,
        "mc_nested_spreadprem_xW_p95": smp.xW_p95.values,
        "legacy_uniform_optimum": leg[leg.block == "sector"].set_index("name").loc[SECTORS, "optimal_debt_ratio"].values,
    })
    for nm, colm in [("synthetic_ref", "actual_mean_dr_synthetic_ref"), ("real", "actual_mean_dr_real")]:
        st[f"{nm}_inside_nested_xW_5_95"] = (st[colm] >= st.mc_nested_xW_p05) & (st[colm] <= st.mc_nested_xW_p95)
        st[f"{nm}_inside_nested_xV_5_95"] = (st[colm] >= st.mc_nested_xV_p05) & (st[colm] <= st.mc_nested_xV_p95)
        st[f"{nm}_inside_coef_xW_5_95"] = (st[colm] >= st.mc_coef_xW_p05) & (st[colm] <= st.mc_coef_xW_p95)
        st[f"gap_xW_minus_{nm}"] = st.x_wacc_min - st[colm]
        st[f"gap_xV_minus_{nm}"] = st.x_value_max - st[colm]
    st.to_csv(os.path.join(OUT, "sector_table_actual_vs_model.csv"), index=False)

    log("(f) figures")
    figures(curves, det, mc, tor, st)

    rho = {f"spearman_{o}_vs_{a}": float(stats.spearmanr(st[o], st[a]).statistic)
           for o in ["x_wacc_min", "x_value_max"]
           for a in ["actual_mean_dr_synthetic_ref", "actual_mean_dr_synthetic_50seeds", "actual_mean_dr_real"]}
    sg = summ[(summ.scenario == "Strong_Growth") & (summ.mc_type == "nested") & (summ.convention == "zero")]
    summary = {
        "runtime_minutes": (time.time() - t0) / 60,
        "panel_rows": len(df), "rf_base_seedavg": macro["rf"], "macro_spread_base_seedavg": macro["cs"],
        "rf_base_seed42": macro42["rf"], "macro_spread_base_seed42": macro42["cs"],
        "legacy": {
            "pareto_counts_full_panel": cnt[cnt.base_sample == "full_panel"][["grid_points", "n_pareto_3obj", "x_low_3obj", "x_high_3obj", "n_pareto_2obj_wacc_risk"]].to_dict("records"),
            "pareto_count_grid500_over_20_samples_min_max": [int(cnt[cnt.base_sample != "full_panel"].n_pareto_3obj.min()),
                                                             int(cnt[cnt.base_sample != "full_panel"].n_pareto_3obj.max())],
            "sector_optima": leg[leg.block == "sector"][["name", "optimal_debt_ratio", "analytic_optimum_0.3-0.5r"]].to_dict("records"),
            "scenario_optima": leg[leg.block == "scenario"][["name", "optimal_debt_ratio", "analytic_optimum_0.3-0.5r"]].to_dict("records"),
            "rbar": rbar},
        "algebra_check": alg,
        "base_results_seedavg_zero": b.reset_index()[["sector", "x_wacc_min", "wacc_min", "pd_at_x_wacc", "x_value_max",
                                                       "pareto_x_low", "pareto_x_high", "band10bp_x_low", "band10bp_x_high"]].to_dict("records"),
        "n_interior_cells": {f"{lab}_{cv}": {"xW": int((~d_.x_wacc_corner).sum()), "xV": int((~d_.x_value_corner).sum()), "cells": len(d_)}
                             for lab, dd in [("seedavg", det), ("seed42", det42)]
                             for cv, d_ in dd.groupby("convention")},
        "strong_growth_share_price_floor_binding_nested": float(sg.share_price_floor_binding.mean()),
        **rho,
        "dgp_grid_calibration_max_abs_diff": float(ck[["max_abs_diff_earn_vol", "max_abs_diff_median_roa", "max_abs_diff_rf", "max_abs_diff_cs"]].values.max()),
        "dgp_grid_xW_own_calibration_range_over_grid": float(dg.groupby("sector").x_wacc_min_own_calibration.agg(lambda s: s.max() - s.min()).max()),
        "n_mc": N_MC, "n_cal_seeds": len(CAL_SEEDS),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
