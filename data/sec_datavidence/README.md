# Datavidence SEC XBRL sample (not distributed)

E8 uses the vendor sample file `datavidence-financials-sample.csv.gz` (long format, one row per
filing x field; 173,783 rows covering 29 US firms) produced by Datavidence from SEC EDGAR XBRL
filings. We do not redistribute the raw vendor file; its use is governed by the vendor's terms.
To reproduce E8:

1. obtain the sample file from Datavidence (the data vendor) under their terms;
2. place it here as `data/sec_datavidence/datavidence-financials-sample.csv.gz`,
   or point `GENFIN_SEC_FILE` to it.

`code_sec/sec_data.py` documents every construction step (vintage, fiscal-year convention, debt
definition, sector mapping). The underlying 10-K figures are public SEC filings, so the panel
can in principle be rebuilt from EDGAR, but the code reads only the vendor's format.
Derived, aggregated results of E8 are in `results/reference_tables/SUPP_TABLES/S67-S70`.
