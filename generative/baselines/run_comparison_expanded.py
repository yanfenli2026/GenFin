"""Run generative baseline comparison on EXPANDED Yahoo Finance data (2022-2026)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import pandas as pd
import time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_comparison import (
    ks_2samp, wasserstein_distance, js_divergence, evaluate_synthetic,
    run_constrained_baseline, run_ctgan_baseline, run_tvae_baseline
)

def main():
    print("=" * 70)
    print("GENFIN-HYBRID: EXPANDED GENERATIVE COMPARISON (YAHOO FINANCE)")
    print("=" * 70)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "data", "recent_data")
    df = pd.read_csv(os.path.join(data_dir, "fundamentals_2022_2025.csv"))
    print(f"Loaded: {len(df)} records, {df['ticker'].nunique()} tickers, "
          f"years {df['year'].min()}-{df['year'].max()}")
    print(f"Mean DR={df['debt_ratio'].mean():.3f} ± {df['debt_ratio'].std():.3f}")
    print(f"Mean WACC={df['wacc'].mean():.4f}")
    print(f"Mean Altman Z={df['altman_z_score'].mean():.2f}")
    print(f"Tickers: {sorted(df['ticker'].unique())[:20]}...")

    feature_cols = ["debt_ratio", "total_assets", "total_debt", "total_equity",
                    "roa", "roe", "interest_coverage", "wacc",
                    "default_risk_score", "altman_z_score"]
    data = df[feature_cols].dropna()
    for col in ["total_assets", "total_debt", "total_equity"]:
        data[col] = np.log1p(data[col])
    X = data.values
    print(f"\nFeature matrix: {X.shape}")
    n = len(X)

    results = []

    # 1. Constrained Baseline
    print("\n" + "=" * 60)
    print("Constrained Noise-Based Baseline")
    print("=" * 60)
    t0 = time.time()
    synth = run_constrained_baseline(X, n)
    t1 = time.time()
    m = evaluate_synthetic(X, synth)
    m["Method"] = "Constrained Baseline"
    m["Time (s)"] = round(t1 - t0, 3)
    m["CVR"] = 0.0
    results.append(m)
    print(f"  KS={m['KS_statistic']:.4f}  Wass={m['Wasserstein']:.4f}  "
          f"JS={m['JS_divergence']:.4f}  Corr={m['correlation_preservation']:.4f}")

    # 2. CTGAN
    print("\n" + "=" * 60)
    print("CTGAN Baseline")
    print("=" * 60)
    try:
        synth_ctgan = run_ctgan_baseline(X, n)
        t2 = time.time()
        m = evaluate_synthetic(X, synth_ctgan)
        m["Method"] = "CTGAN"
        m["Time (s)"] = round(t2 - t1, 3)
        results.append(m)
        print(f"  KS={m['KS_statistic']:.4f}  Wass={m['Wasserstein']:.4f}  "
              f"Corr={m['correlation_preservation']:.4f}")
    except Exception as e:
        print(f"  CTGAN failed: {e}")

    # 3. TVAE
    print("\n" + "=" * 60)
    print("TVAE Baseline")
    print("=" * 60)
    try:
        synth_tvae = run_tvae_baseline(X, n)
        t3 = time.time()
        m = evaluate_synthetic(X, synth_tvae)
        m["Method"] = "TVAE"
        m["Time (s)"] = round(t3 - t2, 3)
        results.append(m)
        print(f"  KS={m['KS_statistic']:.4f}  Wass={m['Wasserstein']:.4f}  "
              f"Corr={m['correlation_preservation']:.4f}")
    except Exception as e:
        print(f"  TVAE failed: {e}")

    dfr = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS — Expanded Real Data (46 firms, 2022-2026)")
    print("=" * 70)
    print(dfr.to_string(index=False))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "outputs")
    os.makedirs(out_dir, exist_ok=True)
    dfr.to_csv(os.path.join(out_dir, "generative_comparison_EXPANDED.csv"), index=False)
    dfr.to_json(os.path.join(out_dir, "generative_comparison_EXPANDED.json"), orient="records", indent=2)
    print(f"\nSaved to {out_dir}")

    # Also save descriptive stats
    stats = df[feature_cols].describe()
    stats.to_csv(os.path.join(out_dir, "descriptive_stats_EXPANDED.csv"))
    print("Descriptive stats saved.")

if __name__ == "__main__":
    main()