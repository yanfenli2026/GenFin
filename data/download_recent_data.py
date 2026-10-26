"""Download annual financial statements from Yahoo Finance for the 46 S&P 100 tickers below.

Writes one balance-sheet, income-statement and cash-flow CSV per ticker to data/recent_data/;
common.load_real_panel builds the paper's real-firm sample (fiscal years 2021-2025) from these files.
The combined fundamentals_2022_2025.csv written at the end is the file used by the original
submission and is read only by the diagnosis sections of E3/E4 and generative/baselines/.
Yahoo Finance returns only the latest four to five annual statements, so a new download covers
later fiscal years than the paper (see README, section 4)."""
import yfinance as yf
import pandas as pd
import os
import time

TICKERS = [
    "MSFT","AAPL","AMZN","GOOGL","META","TSLA","NVDA","JPM","JNJ","V",
    "PG","XOM","MA","HD","CVX","LLY","ABBV","PFE","MRK","PEP",
    "KO","TMO","BAC","DIS","WMT","MCD","COST","CSCO","ADBE","CRM",
    "NFLX","INTC","TXN","QCOM","HON","LIN","MDT","UNH","NEE","IBM",
    "GE","MMM","CAT","BA","AXP","GS"
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "recent_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_fundamentals():
    records = []
    for i, ticker in enumerate(TICKERS):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            bs = stock.balance_sheet
            bs.to_csv(os.path.join(OUTPUT_DIR, f"{ticker}_balance_sheet.csv"))
            
            inc = stock.income_stmt
            inc.to_csv(os.path.join(OUTPUT_DIR, f"{ticker}_income_stmt.csv"))
            
            cf = stock.cashflow
            cf.to_csv(os.path.join(OUTPUT_DIR, f"{ticker}_cashflow.csv"))
            
            # Extract key metrics
            for date in bs.columns:
                try:
                    year = date.year
                    if year >= 2022:
                        assets = float(bs.loc["Total Assets", date]) if "Total Assets" in bs.index else None
                        equity = float(bs.loc["Total Equity Gross Minority Interest", date]) if "Total Equity Gross Minority Interest" in bs.index else None
                        debt = float(bs.loc["Total Debt", date]) if "Total Debt" in bs.index else None
                        cash = float(bs.loc["Cash And Cash Equivalents", date]) if "Cash And Cash Equivalents" in bs.index else None
                        
                        ni = float(inc.loc["Net Income", date]) if inc is not None and "Net Income" in inc.index else None
                        revenue = float(inc.loc["Total Revenue", date]) if inc is not None and "Total Revenue" in inc.index else None
                        ebit = float(inc.loc["EBIT", date]) if inc is not None and "EBIT" in inc.index else None
                        
                        mkt_cap_val = info.get("marketCap", None)
                        
                        if assets and equity:
                            records.append({
                                "ticker": ticker,
                                "year": year,
                                "total_assets": assets,
                                "total_equity": equity,
                                "total_debt": debt if debt else max(assets - equity, 0),
                                "cash_holdings": cash if cash else 0,
                                "net_income": ni if ni else 0,
                                "revenue": revenue if revenue else 0,
                                "ebit": ebit if ebit else (ni * 1.4 if ni else 0),
                                "market_cap": mkt_cap_val if mkt_cap_val else 0,
                            })
                except Exception as e:
                    pass
            
            print(f"[{i+1}/{len(TICKERS)}] {ticker}: {len(records)} records")
            time.sleep(0.5)  # rate limit
            
        except Exception as e:
            print(f"[{i+1}/{len(TICKERS)}] {ticker}: FAILED - {str(e)[:60]}")
    
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["ticker", "year"])
    
    # Compute derived ratios
    df["debt_ratio"] = df["total_debt"] / df["total_assets"].clip(lower=1)
    df["roa"] = df["net_income"] / df["total_assets"].clip(lower=1)
    df["roe"] = df["net_income"] / df["total_equity"].clip(lower=1)
    df["cash_ratio"] = df["cash_holdings"] / df["total_assets"].clip(lower=1)
    df["tobin_q"] = (df["market_cap"] + df["total_debt"]) / df["total_assets"].clip(lower=1)
    df["wacc"] = 0.09 * (df["total_equity"] / df["total_assets"].clip(lower=1)) + 0.0316 * (df["total_debt"] / df["total_assets"].clip(lower=1))
    df["interest_coverage"] = df["ebit"] / (df["total_debt"] * 0.035 + 1)
    df["altman_z_score"] = (
        1.2 * (df["total_assets"] - df["total_debt"] - df["cash_holdings"]) / df["total_assets"].clip(lower=1)
        + 1.4 * df["net_income"] / df["total_assets"].clip(lower=1)
        + 3.3 * df["ebit"] / df["total_assets"].clip(lower=1)
        + 0.6 * df["market_cap"] / df["total_debt"].clip(lower=1)
        + 1.0 * df["revenue"] / df["total_assets"].clip(lower=1)
    ).clip(-10, 20)
    df["default_risk_score"] = (0.01 + 0.15 * df["debt_ratio"]).clip(0, 0.5)
    
    # Save
    csv_path = os.path.join(OUTPUT_DIR, "fundamentals_2022_2025.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} records to {csv_path}")
    print(f"\nYear distribution:")
    print(df["year"].value_counts().sort_index())
    print(f"\nTickers: {sorted(df['ticker'].unique())}")
    print(f"\nMean debt_ratio: {df['debt_ratio'].mean():.3f} ± {df['debt_ratio'].std():.3f}")
    
    return df

if __name__ == "__main__":
    download_fundamentals()