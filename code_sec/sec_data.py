"""
sec_data.py -- real-data firm-year panel from SEC XBRL filings (Datavidence sample).

Source file (long format, one row per filing x field):
    data/sec_datavidence/datavidence-financials-sample.csv.gz   (or $GENFIN_SEC_FILE; not redistributed)
    columns: cik, ticker, name, fiscal_year, period, period_start, period_end,
             statement, field, value, unit, empty_reason, ..., form, filed,
             is_original, is_latest, restated, ...

The file is treated as untrusted data: it is only parsed with pandas.read_csv,
nothing in it is evaluated.

Main entry point
----------------
    load_sec_panel(point_in_time=True, ...) -> pandas.DataFrame (one row per firm-year)

Design choices (all documented here)
-----------------------------------------------------------------------------
1. Annual data only: rows with period == "FY".

2. Vintage.
   point_in_time=True  (default): for every (cik, fiscal_year, field) keep the
       value from the FIRST filing that reported it (is_original == True).  This
       is what an analyst could have known at the time and avoids look-ahead from
       later restatements / re-classifications.  In the sample ~9% of FY values
       differ between the original and the latest vintage.
   point_in_time=False: use the most recent vintage (is_latest == True), i.e.
       restated / re-classified values ("as currently reported").
   Each (cik, fiscal_year, field) has exactly one original and one latest row.
   A small number of "original" values come from an 8-K or 10-Q (the first
   document in which that FY number appeared) rather than the 10-K.

3. Year index.  The source `fiscal_year` is the company's own label (e.g. Walmart
   "FY2026" ends 2026-01-31, Kroger "2025" ends 2026-01-31).  We add
   `fyear` using the Compustat convention: fyear = year(period_end) if the
   fiscal year ends in June-December, else year(period_end) - 1.  `fyear` is
   the panel year.  fyear 2026 is dropped by default (drop_incomplete_last_year)
   because by the data cut-off only the June-September year-end firms (MSFT, PG,
   COST, KOSS) had filed it; December year-end firms had not.

4. Firm identity / missing tickers.  Two CIKs carry ticker NaN:
     cik 34088   "Exxon Mobil Corporation"           -> ticker XOM
     cik 1001039 "WALT DISNEY CO/" (old registrant,  -> ticker DIS
                  FY2009-FY2018)
   Disney re-registered under cik 1744489 (FY2019-) after the 2019 21st Century
   Fox transaction.  The two Disney CIKs do not overlap in time and are spliced
   into one firm (firm_id "DIS").  With recover_missing_tickers=False these NaN-
   ticker rows are dropped instead.

5. Debt / leverage (targets).
     total_debt = long_term_debt + long_term_debt_current
       - If one component is missing and the other present, the missing one is
         treated as 0 (typical cause: the current portion is not tagged
         separately, or the balance sheet is unclassified and the single
         long-term-debt line already includes the current part).  Flag column
         `debt_component_imputed` = 1.
       - If both are missing, total_debt is NaN by default (debt_both_missing=
         "nan").  debt_both_missing="zero" sets it to 0 (plausible for genuinely
         debt-free years such as Apple FY2009-2012 or Koss, but it is an
         assumption the source does not confirm -- empty_reason = not_reported).
       - Caveat: the source has no short-term borrowings / commercial paper /
         lease-liability field, so total_debt is long-term debt incl. current
         maturities, not all interest-bearing debt.
     debt_ratio        = total_debt / total_assets          (book leverage)
     liabilities_ratio = total_liabilities / total_assets   (alternative)

6. Predictors -- LEAKAGE-FREE set (no debt, no interest expense, no equity-based
   leverage term in any numerator/denominator):
     size_assets   = log(total_assets)
     size_revenue  = log(total_revenue)          (NaN if revenue <= 0)
     tangibility   = net_ppe / total_assets
     profitability = operating_income / total_assets   (pre-interest)
         If operating_income is not reported (e.g. PFE, XOM, JNJ, GE some years)
         and ebit_fallback=True, EBIT is proxied by pretax_income + interest_expense
         (still pre-interest; interest is added back so the ratio carries no
         leverage information by construction).  `profitability_source` records
         "operating_income" / "pretax_plus_interest".  ebit_fallback=False leaves
         those rows NaN.
     cash_ratio    = cash_and_equivalents / total_assets
     capex_assets  = capex / total_assets         (capex is a positive outflow)
     da_assets     = depreciation_amortization / total_assets
     sales_growth  = revenue_t / revenue_{t-1} - 1, only if the previous row of
                     the same firm is fyear-1 (else NaN)
     div_payer     = 1 if dividends_paid > 0, else 0.  dividends_paid missing
                     (not_reported, e.g. AMZN, TSLA, pre-2024 META/GOOGL) is
                     treated as "no dividend reported" -> 0, flag `div_missing`.
     ocf_assets    = operating_cash_flow / total_assets
     ca_assets     = current_assets / total_assets   (liquidity without any
                     debt / current-liabilities term)
     eff_tax_rate  = income_tax_expense / pretax_income, NaN if pretax_income
                     <= 0, then clipped to [0, 1]
   Debt-linked variables (NOT for the leakage-free model, kept for reference):
     interest_coverage = EBIT / interest_expense (NaN if interest_expense <= 0)
     roa_net           = net_income / total_assets (after interest)
     current_ratio     = current_assets / current_liabilities (current
                         liabilities include current debt)

7. Winsorisation.  All ratio variables (predictors, debt-linked ratios, and the
   two leverage ratios) are winsorised at the pooled 1st / 99th percentiles of
   the RETURNED sample (after exclusions).  Unwinsorised copies are kept as
   `<col>_raw` for debt_ratio and liabilities_ratio.  Log-size variables and the
   dummy are not winsorised.  For out-of-sample ML evaluation the thresholds
   should be estimated on the training fold only; use winsorize(df, cols,
   ref=train_df) for that and call load_sec_panel(winsorize_ratios=False).

8. Sectors.  Tickers are mapped to the paper's 8 sector names (Tech, Healthcare,
   Financials, ConsDisc, Industrials, Energy, Utilities, Materials) via
   SECTOR_MAP below (GICS sector in the comment; Consumer Staples,
   Communication Services and Real Estate have no own bucket and are mapped to
   the nearest one, see MAPPED_TO_NEAREST).  `is_financial` flags banks,
   insurers, payment networks and REITs (GICS Financials + Real Estate);
   `is_bank_insurer` flags JPM, BRK-B, MET whose balance sheets (deposits,
   policy reserves) make leverage non-comparable.
   exclude_financials_utilities=True (standard practice) drops sector
   Financials (JPM, BRK-B, MET, MA, O) and Utilities (WTRG).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import sys as _sys
_sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "code")))
import config as _config  # noqa: E402
# Location of the vendor file; override with the GENFIN_SEC_FILE environment variable.
DEFAULT_PATH = _config.SEC_FILE

FIELDS = [
    "total_assets", "total_liabilities", "total_equity", "stockholders_equity",
    "long_term_debt", "long_term_debt_current", "current_assets",
    "current_liabilities", "cash_and_equivalents", "net_ppe", "total_revenue",
    "operating_income", "net_income", "interest_expense", "income_tax_expense",
    "pretax_income", "depreciation_amortization", "capex", "operating_cash_flow",
    "dividends_paid", "buybacks", "shares_outstanding",
]

# CIKs whose ticker is NaN in the source, identified from the `name` column.
MISSING_TICKER_CIK = {34088: "XOM", 1001039: "DIS"}
# CIK -> firm_id when several CIKs are the same economic firm (Disney re-registration 2019).
FIRM_ID_OVERRIDE = {1001039: "DIS", 1744489: "DIS"}

# ticker -> (paper sector, GICS sector as of 2024, note)
SECTOR_MAP = {
    "AAPL":  ("Tech",        "Information Technology", ""),
    "MSFT":  ("Tech",        "Information Technology", ""),
    "NVDA":  ("Tech",        "Information Technology", ""),
    "GOOGL": ("Tech",        "Communication Services", "Comm. Services -> Tech (classified IT before the 2018 GICS change)"),
    "META":  ("Tech",        "Communication Services", "Comm. Services -> Tech (classified IT before the 2018 GICS change)"),
    "JNJ":   ("Healthcare",  "Health Care", ""),
    "PFE":   ("Healthcare",  "Health Care", ""),
    "UNH":   ("Healthcare",  "Health Care", ""),
    "JPM":   ("Financials",  "Financials", "bank"),
    "BRK-B": ("Financials",  "Financials", "insurance conglomerate"),
    "MET":   ("Financials",  "Financials", "insurer"),
    "MA":    ("Financials",  "Financials", "payment network (GICS IT until 2023); not a bank/insurer"),
    "O":     ("Financials",  "Real Estate", "Real Estate (REIT) -> Financials (REITs were GICS Financials until 2016)"),
    "AMZN":  ("ConsDisc",    "Consumer Discretionary", ""),
    "HD":    ("ConsDisc",    "Consumer Discretionary", ""),
    "MCD":   ("ConsDisc",    "Consumer Discretionary", ""),
    "TSLA":  ("ConsDisc",    "Consumer Discretionary", ""),
    "KOSS":  ("ConsDisc",    "Consumer Discretionary", "micro-cap consumer electronics"),
    "DIS":   ("ConsDisc",    "Communication Services", "Comm. Services -> ConsDisc (classified ConsDisc before 2018)"),
    "KO":    ("ConsDisc",    "Consumer Staples", "Staples -> ConsDisc (nearest consumer bucket)"),
    "PG":    ("ConsDisc",    "Consumer Staples", "Staples -> ConsDisc (nearest consumer bucket)"),
    "WMT":   ("ConsDisc",    "Consumer Staples", "Staples -> ConsDisc (nearest consumer bucket)"),
    "COST":  ("ConsDisc",    "Consumer Staples", "Staples -> ConsDisc (nearest consumer bucket)"),
    "KR":    ("ConsDisc",    "Consumer Staples", "Staples -> ConsDisc (nearest consumer bucket)"),
    "PM":    ("ConsDisc",    "Consumer Staples", "Staples -> ConsDisc (nearest consumer bucket)"),
    "BA":    ("Industrials", "Industrials", ""),
    "GE":    ("Industrials", "Industrials", "conglomerate with large finance arm (GE Capital) until ~2015-2019"),
    "XOM":   ("Energy",      "Energy", ""),
    "WTRG":  ("Utilities",   "Utilities", "water / gas utility"),
}
MAPPED_TO_NEAREST = {t: v[2] for t, v in SECTOR_MAP.items() if "->" in v[2]}
BANK_INSURER = {"JPM", "BRK-B", "MET"}
PAPER_SECTORS = ["Tech", "Healthcare", "Financials", "ConsDisc", "Industrials",
                 "Energy", "Utilities", "Materials"]

LEAKAGE_FREE_PREDICTORS = [
    "size_assets", "size_revenue", "tangibility", "profitability", "cash_ratio",
    "capex_assets", "da_assets", "sales_growth", "div_payer", "ocf_assets",
    "ca_assets", "eff_tax_rate",
]
DEBT_LINKED = ["interest_coverage", "roa_net", "current_ratio"]
TARGETS = ["debt_ratio", "liabilities_ratio"]
RATIO_COLS = ["tangibility", "profitability", "cash_ratio", "capex_assets",
              "da_assets", "sales_growth", "ocf_assets", "ca_assets",
              "eff_tax_rate", "interest_coverage", "roa_net", "current_ratio",
              "debt_ratio", "liabilities_ratio"]


def _fyear(period_end: pd.Series) -> pd.Series:
    pe = pd.to_datetime(period_end, errors="coerce")
    return np.where(pe.dt.month >= 6, pe.dt.year, pe.dt.year - 1).astype("int64")


def winsorize(df: pd.DataFrame, cols, lower=0.01, upper=0.99, ref: pd.DataFrame | None = None):
    """Clip `cols` of df at the [lower, upper] quantiles estimated on `ref` (default df)."""
    ref = df if ref is None else ref
    out = df.copy()
    for c in cols:
        if c in out and out[c].notna().any():
            lo, hi = ref[c].quantile(lower), ref[c].quantile(upper)
            out[c] = out[c].clip(lo, hi)
    return out


def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.where(b != 0)


def load_sec_panel(point_in_time: bool = True,
                   path: str = DEFAULT_PATH,
                   recover_missing_tickers: bool = True,
                   drop_incomplete_last_year: bool = True,
                   exclude_financials_utilities: bool = False,
                   debt_both_missing: str = "nan",
                   ebit_fallback: bool = True,
                   winsorize_ratios: bool = True,
                   limits=(0.01, 0.99),
                   return_diagnostics: bool = False):
    """Build the firm-year panel. See module docstring for every choice."""
    raw = pd.read_csv(path, low_memory=False,
                      dtype={"ticker": "string", "name": "string", "field": "string",
                             "period": "string"})
    diag = {"raw_rows": len(raw)}
    fy = raw[raw["period"] == "FY"].copy()
    diag["fy_rows"] = len(fy)
    flag = "is_original" if point_in_time else "is_latest"
    fy = fy[fy[flag].astype(bool) & fy["field"].isin(FIELDS)]
    dup = fy.duplicated(["cik", "fiscal_year", "field"]).sum()
    diag["duplicate_keys_after_vintage_filter"] = int(dup)
    if dup:
        fy = fy.sort_values("filed").drop_duplicates(["cik", "fiscal_year", "field"],
                                                     keep="first" if point_in_time else "last")

    # firm identity
    ids = (raw[["cik", "ticker", "name"]].drop_duplicates("cik").set_index("cik"))
    diag["nan_ticker_ciks"] = {int(c): str(ids.loc[c, "name"])
                               for c in ids.index[ids["ticker"].isna()]}

    wide = fy.pivot_table(index=["cik", "fiscal_year"], columns="field",
                          values="value", aggfunc="first")
    wide = wide.reindex(columns=FIELDS)
    meta = (fy[fy["field"] == "total_assets"]
            .set_index(["cik", "fiscal_year"])[["period_end", "filed", "form"]]
            .rename(columns={"filed": "filed_total_assets", "form": "form_total_assets"}))
    pe_any = fy.groupby(["cik", "fiscal_year"])["period_end"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    wide = wide.join(meta).reset_index()
    wide["period_end"] = wide["period_end"].fillna(
        pd.Series(pe_any.reindex(pd.MultiIndex.from_frame(wide[["cik", "fiscal_year"]])).values,
                  index=wide.index))
    wide["fyear"] = _fyear(wide["period_end"])
    wide["ticker"] = wide["cik"].map(ids["ticker"]).astype("object")
    wide["name"] = wide["cik"].map(ids["name"]).astype("object")

    miss = wide["ticker"].isna()
    if recover_missing_tickers:
        wide.loc[miss, "ticker"] = wide.loc[miss, "cik"].map(MISSING_TICKER_CIK)
    wide = wide[wide["ticker"].notna()].copy()
    diag["rows_nan_ticker_recovered" if recover_missing_tickers else "rows_nan_ticker_dropped"] = int(miss.sum())
    wide["firm_id"] = wide["cik"].map(FIRM_ID_OVERRIDE).fillna(wide["ticker"])

    if wide.duplicated(["firm_id", "fyear"]).any():
        raise ValueError("duplicate firm_id-fyear after splicing")

    yr_counts = wide.groupby("fyear")["firm_id"].nunique()
    diag["firms_per_fyear_before_drop"] = yr_counts.to_dict()
    if drop_incomplete_last_year:
        last = wide["fyear"].max()
        if yr_counts[last] < 0.8 * yr_counts.median():
            diag["dropped_fyear"] = int(last)
            diag["dropped_fyear_firms"] = sorted(wide.loc[wide["fyear"] == last, "firm_id"])
            wide = wide[wide["fyear"] != last]

    # sectors
    wide["sector"] = wide["firm_id"].map(lambda t: SECTOR_MAP.get(t, ("Unmapped",))[0])
    wide["gics_sector"] = wide["firm_id"].map(lambda t: SECTOR_MAP.get(t, (None, None))[1])
    wide["is_financial"] = (wide["sector"] == "Financials").astype(int)
    wide["is_bank_insurer"] = wide["firm_id"].isin(BANK_INSURER).astype(int)
    if exclude_financials_utilities:
        wide = wide[~wide["sector"].isin(["Financials", "Utilities"])].copy()

    d = wide.sort_values(["firm_id", "fyear"]).reset_index(drop=True)
    ta = d["total_assets"].where(d["total_assets"] > 0)
    # debt
    ltd, ltdc = d["long_term_debt"], d["long_term_debt_current"]
    one_missing = ltd.isna() ^ ltdc.isna()
    d["total_debt"] = ltd.fillna(0) + ltdc.fillna(0)
    both = ltd.isna() & ltdc.isna()
    if debt_both_missing == "nan":
        d.loc[both, "total_debt"] = np.nan
    elif debt_both_missing != "zero":
        raise ValueError("debt_both_missing must be 'nan' or 'zero'")
    d["debt_component_imputed"] = one_missing.astype(int)
    d["debt_both_missing"] = both.astype(int)
    d["debt_ratio"] = d["total_debt"] / ta
    d["liabilities_ratio"] = d["total_liabilities"] / ta

    # predictors (leakage-free)
    d["size_assets"] = np.log(ta)
    d["size_revenue"] = np.log(d["total_revenue"].where(d["total_revenue"] > 0))
    d["tangibility"] = d["net_ppe"] / ta
    ebit = d["operating_income"].copy()
    src = pd.Series(np.where(ebit.notna(), "operating_income", None), index=d.index, dtype="object")
    if ebit_fallback:
        proxy = d["pretax_income"] + d["interest_expense"].fillna(0)
        proxy = proxy.where(d["pretax_income"].notna())
        fill = ebit.isna() & proxy.notna()
        ebit[fill] = proxy[fill]
        src[fill] = "pretax_plus_interest"
    d["ebit"] = ebit
    d["profitability_source"] = src
    d["profitability"] = ebit / ta
    d["cash_ratio"] = d["cash_and_equivalents"] / ta
    d["capex_assets"] = d["capex"] / ta
    d["da_assets"] = d["depreciation_amortization"] / ta
    prev_rev = d.groupby("firm_id")["total_revenue"].shift(1)
    prev_yr = d.groupby("firm_id")["fyear"].shift(1)
    d["sales_growth"] = (_safe_div(d["total_revenue"], prev_rev.where(prev_rev > 0)) - 1).where(prev_yr == d["fyear"] - 1)
    d["div_missing"] = d["dividends_paid"].isna().astype(int)
    d["div_payer"] = (d["dividends_paid"].fillna(0) > 0).astype(int)
    d["ocf_assets"] = d["operating_cash_flow"] / ta
    d["ca_assets"] = d["current_assets"] / ta
    etr = _safe_div(d["income_tax_expense"], d["pretax_income"].where(d["pretax_income"] > 0))
    d["eff_tax_rate"] = etr.clip(0, 1)
    # debt-linked (reference only)
    d["interest_coverage"] = _safe_div(ebit, d["interest_expense"].where(d["interest_expense"] > 0))
    d["roa_net"] = d["net_income"] / ta
    d["current_ratio"] = _safe_div(d["current_assets"], d["current_liabilities"])

    d["debt_ratio_raw"] = d["debt_ratio"]
    d["liabilities_ratio_raw"] = d["liabilities_ratio"]
    if winsorize_ratios:
        d = winsorize(d, RATIO_COLS, *limits)
    d["vintage"] = "original" if point_in_time else "latest"

    front = ["firm_id", "ticker", "cik", "name", "sector", "gics_sector", "is_financial",
             "is_bank_insurer", "fyear", "fiscal_year", "period_end", "filed_total_assets",
             "form_total_assets", "vintage"]
    cols = front + TARGETS + ["debt_ratio_raw", "liabilities_ratio_raw", "total_debt",
                              "debt_component_imputed", "debt_both_missing"] \
        + LEAKAGE_FREE_PREDICTORS + ["profitability_source", "div_missing"] + DEBT_LINKED \
        + ["ebit"] + FIELDS
    d = d[cols]
    diag["n_firms"] = int(d["firm_id"].nunique())
    diag["n_firm_years"] = int(len(d))
    return (d, diag) if return_diagnostics else d


if __name__ == "__main__":
    panel, info = load_sec_panel(return_diagnostics=True)
    out = os.path.join(HERE, "sec_panel.csv")
    panel.to_csv(out, index=False)
    print(info)
    print("saved", out, panel.shape)
