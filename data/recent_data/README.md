# Yahoo Finance statements (not distributed)

The 46-firm real sample is built from annual balance sheets, income statements and cash-flow
statements downloaded from Yahoo Finance with the `yfinance` package. Yahoo's terms of service
restrict redistribution of the data, so the raw files are not included in this repository.

Regenerate them with

    python data/download_recent_data.py

which writes, for each of the 46 tickers listed in the script, `<TICKER>_balance_sheet.csv`,
`<TICKER>_income_stmt.csv`, `<TICKER>_cashflow.csv`, plus `fundamentals_2022_2025.csv`, into this
folder. Alternatively, put the files anywhere and set `GENFIN_YAHOO_DIR`.

Yahoo Finance only serves the most recent four or five fiscal years, so a download made later than
the paper's (2026) snapshot will cover different years and the E3-E8 numbers will not match the
paper exactly. The paper's panel has 46 firms and 184 firm-years (fiscal years 2021-2025).
