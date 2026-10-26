"""
E4 - Re-run and verify the generative-model comparison.

One command:
    python code/e4_generative.py
Optional: --quick (few epochs / small grids, for debugging only; writes to e4_generative_quick).

Sections
  1. Consistency check of the earlier generative comparison: re-runs the original scripts' functions (read-only import of
     generative/baselines/run_comparison.py) and the original pipeline's run_synthetic_generator() logic.
  2. One consistent comparison on the reference synthetic panel (seed 42), 17 numeric columns + sector:
     B1 noise baseline (pipeline mechanism + A = D + E projection), B2 Gaussian baseline (comparison-script
     mechanism, projection done in levels), CTGAN and TVAE (ctgan 0.12.1; 300 and 50 epochs; seeds 42-44),
     and CTGAN/TVAE outputs after the same post-hoc projection.
  3. Metrics: KS (mean/median/max), standardised Wasserstein, JSD (base 2), sector TVD, correlation MAD
     and Frobenius norm, CVR (identity / non-negativity / bounds), ratio consistency, discriminator AUC
     (grouped 5-fold), distance to closest record.
  4. Generator sensitivity: DGP grid (lambda x sigma_zeta), baseline noise scale, reference-panel seed.
  5. Figures (PNG 300 dpi + PDF).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ROOT = common.ROOT
BASELINE_DIR = os.path.join(ROOT, "generative", "baselines")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM = ["total_assets", "total_debt", "total_equity", "revenue", "net_income", "cash_holdings",
       "interest_expense", "market_cap", "debt_ratio", "roa", "roe", "cash_ratio", "interest_coverage",
       "tobin_q", "wacc", "default_risk_score", "altman_zscore"]
LEVELS_POS = ["total_assets", "total_debt", "total_equity", "revenue", "cash_holdings",
              "interest_expense", "market_cap"]          # strictly positive in the reference panel
LEVELS_SIGNED = ["net_income"]
CAT = "sector"
GEN_SEEDS = [42, 43, 44]
PANEL_SEED = 42
N_GEN = 4440
CVR_TOL = 1e-3            # relative tolerance for A = D + E, |A - D - E| <= tol * |A|
CVR_TOL_LOOSE = 1e-2
RATIO_TOL = 0.01          # |debt_ratio - D/A| <= 0.01
JSD_BINS = 50             # equal-width bins on the reference 1st-99th percentile + 2 open tail bins
LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.6]
SIGMAS = [0.005, 0.01, 0.02, 0.03, 0.05]
NOISE_SCALES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
PANEL_SEEDS = [42, 43, 44, 45, 46]
CTGAN_KW = dict(embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4, discriminator_decay=1e-6,
                batch_size=500, discriminator_steps=1, log_frequency=True, pac=10)
TVAE_KW = dict(embedding_dim=128, compress_dims=(128, 128), decompress_dims=(128, 128), l2scale=1e-5,
               batch_size=500, loss_factor=2)
DISC_KW = dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=20,
               subsample=1.0, colsample_bytree=1.0, n_jobs=4, verbose=-1)
GEN_LABEL = {"B1_noise": "Noise baseline (B1)", "B2_gauss": "Gaussian baseline (B2)",
             "B1_noise_log": "Noise baseline, log levels (B1-log)",
             "CTGAN": "CTGAN", "TVAE": "TVAE",
             "CTGAN_proj": "CTGAN + projection", "TVAE_proj": "TVAE + projection"}
GEN_COLOR = {"B1_noise": "#E69F00", "B2_gauss": "#56B4E9", "B1_noise_log": "#CC79A7", "CTGAN": "#009E73",
             "TVAE": "#0072B2", "CTGAN_proj": "#009E73", "TVAE_proj": "#0072B2"}
# columns with point masses in the reference panel (share of rows at a single value):
# interest_coverage = 0 (25.1%), altman_zscore = 20 cap (17.5%), default_risk_score clipped at 0.001/0.999
POINT_MASS = {"interest_coverage": [0.0], "altman_zscore": [20.0], "default_risk_score": [0.001, 0.999]}
NO_PM_COLS = [c for c in NUM if c not in POINT_MASS]
REF_COLOR = "#999999"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _finite(v):
    v = np.asarray(v, dtype=float)
    return v[np.isfinite(v)]


def jsd_base2(ref, syn, bins=JSD_BINS):
    """Jensen-Shannon divergence (base 2, in [0, 1]) on shared bins: `bins` equal-width bins between the
    reference 1st and 99th percentiles, plus one open bin below and one above (bins + 2 in total)."""
    ref, syn = _finite(ref), _finite(syn)
    lo, hi = np.percentile(ref, [1, 99])
    if hi <= lo:
        hi = lo + 1e-12
    edges = np.concatenate([[-np.inf], np.linspace(lo, hi, bins + 1), [np.inf]])
    p = np.histogram(ref, edges)[0].astype(float)
    q = np.histogram(syn, edges)[0].astype(float)
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q, base=2) ** 2)


def corr_metrics(ref_df, syn_df, cols, method="pearson", return_nan=False):
    """MAD = mean |r_syn - r_ref| over the upper triangle (bounded [0, 2]); Frobenius = ||R_syn - R_ref||_F
    over the full matrix. Undefined correlations (constant generated column) are NOT replaced by zero:
    they are counted (n_nan, upper triangle) and both summaries are computed over the finite entries only."""
    cr = ref_df[cols].corr(method=method).values
    cs = syn_df[cols].corr(method=method).values
    d = cs - cr
    iu = np.triu_indices(len(cols), 1)
    du = d[iu]
    n_nan = int((~np.isfinite(du)).sum())
    mad = float(np.mean(np.abs(du[np.isfinite(du)]))) if np.isfinite(du).any() else np.nan
    fro = float(np.sqrt(np.nansum(d ** 2)))
    if return_nan:
        return mad, fro, d, n_nan
    return mad, fro, d


def violation_breakdown(df):
    """Per-column components of CVR (shares of rows): negative level columns, ratio columns outside [0, 1],
    D > A, and |A - (D + E)| > CVR_TOL |A|."""
    out = {f"neg_{c}": float((df[c] < 0).mean()) for c in LEVELS_POS}
    for c in ["debt_ratio", "cash_ratio", "default_risk_score"]:
        out[f"oob_{c}"] = float(((df[c] < 0) | (df[c] > 1)).mean())
    out["debt_gt_assets"] = float((df["total_debt"] > df["total_assets"]).mean())
    A = df["total_assets"]
    out["identity_violation"] = float(((A - df["total_debt"] - df["total_equity"]).abs() > CVR_TOL * A.abs()).mean())
    return out


def point_mass_shares(df):
    out = {}
    for c, vals in POINT_MASS.items():
        for v in vals:
            out[f"pm_{c}_at_{v:g}"] = float(np.isclose(df[c].astype(float), v, atol=1e-9).mean())
    return out


def project_identity(df):
    """Projection used by the original comparison script, applied in levels:
    diff = A - (D + E);  A -= diff/3;  D += diff/3;  E += diff/3   (Euclidean projection onto A = D + E)."""
    out = df.copy()
    diff = out["total_assets"] - (out["total_debt"] + out["total_equity"])
    out["total_assets"] -= diff / 3
    out["total_debt"] += diff / 3
    out["total_equity"] += diff / 3
    return out


def cvr(df, tol=CVR_TOL):
    """Constraint-violation shares. A row violates
       identity  if |A - (D + E)| > tol * |A|
       nonneg    if A <= 0 or any of D, cash, revenue, interest_expense, market_cap < 0
       bounds    if debt_ratio, cash_ratio or default_risk_score outside [0, 1], or D > A
       any       if any of the three."""
    A, D, E = df["total_assets"], df["total_debt"], df["total_equity"]
    ident = (A - (D + E)).abs() > tol * A.abs()
    nonneg = (A <= 0) | (D < 0) | (df["cash_holdings"] < 0) | (df["revenue"] < 0) | \
             (df["interest_expense"] < 0) | (df["market_cap"] < 0)
    bounds = pd.Series(False, index=df.index)
    for c in ["debt_ratio", "cash_ratio", "default_risk_score"]:
        bounds |= (df[c] < 0) | (df[c] > 1)
    bounds |= D > A
    anyv = ident | nonneg | bounds
    ratio_bad = (df["debt_ratio"] - D / A.where(A != 0)).abs() > RATIO_TOL
    ratio_bad |= A.eq(0)
    return dict(CVR_identity=float(ident.mean()), CVR_nonneg=float(nonneg.mean()),
                CVR_bounds=float(bounds.mean()), CVR=float(anyv.mean()),
                ratio_inconsistency=float(ratio_bad.mean()))


RATIO_COLS = ["debt_ratio", "roa", "roe", "cash_ratio", "interest_coverage", "tobin_q", "wacc",
              "default_risk_score", "altman_zscore"]


def disc_auc(ref, syn, ref_groups, syn_groups, seed, n_splits=5, cols=None, model="lgbm"):
    """Classifier two-sample test: a classifier separates reference (0) from generated (1) rows;
    GroupKFold(5) with reference rows grouped by firm and generated rows by source firm (B1) or by a seeded
    random firm label (other generators). Returns mean and s.d. of the 5 out-of-fold ROC AUCs.
    model = "lgbm": LightGBM (DISC_KW) on raw values + sector code (tree splits are invariant to monotone
    per-feature transforms, so a rank-transformed version gives the same partitions).
    model = "logit": L2 logistic regression (C = 1, lbfgs, max_iter 2000) on features mapped through the
    training-fold empirical CDF to N(0, 1) (QuantileTransformer, 1000 quantiles) + sector one-hot; a weak,
    smooth discriminator that is not driven by single point masses or tail clipping."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    sec_codes = {s: i for i, s in enumerate(sorted(common.SECTORS))}
    cols = cols or NUM
    X = pd.concat([ref[cols], syn[cols]], ignore_index=True).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    sec = np.concatenate([ref[CAT].map(sec_codes).values, syn[CAT].map(sec_codes).values])
    y = np.r_[np.zeros(len(ref)), np.ones(len(syn))]
    g = np.r_[np.asarray(ref_groups), np.asarray(syn_groups)]
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        if model == "lgbm":
            from lightgbm import LGBMClassifier
            Xl = X.copy()
            Xl["sector_code"] = sec
            clf = LGBMClassifier(random_state=seed, **DISC_KW)
            clf.fit(Xl.iloc[tr], y[tr], categorical_feature=["sector_code"])
            p = clf.predict_proba(Xl.iloc[te])[:, 1]
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import QuantileTransformer
            qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0)
            Ztr = qt.fit_transform(X.values[tr]); Zte = qt.transform(X.values[te])
            oh = np.eye(len(sec_codes))[sec]
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(np.c_[Ztr, oh[tr]], y[tr])
            p = clf.predict_proba(np.c_[Zte, oh[te]])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


class DCR:
    """Distance to closest record (Euclidean) after mapping every NUM column through the reference
    empirical CDF (QuantileTransformer, uniform output in [0, 1]; values outside the reference range map to
    0 or 1). Reference baseline: each reference row's nearest reference row from a different firm."""

    def __init__(self, ref):
        from sklearn.neighbors import NearestNeighbors
        self.mu, self.sd = None, None
        Z = self._z(ref, fit=True)
        self.nn = NearestNeighbors(n_neighbors=1).fit(Z)
        self.Zref = Z
        # reference -> nearest other-firm reference row
        nn20 = NearestNeighbors(n_neighbors=40).fit(Z)
        dist, idx = nn20.kneighbors(Z)
        firm = ref["firm_id"].values
        other = firm[idx] != firm[:, None]
        first = np.argmax(other, axis=1)
        d = dist[np.arange(len(Z)), first]
        d[~other.any(axis=1)] = np.nan
        self.ref_median = float(np.nanmedian(d))
        self.ref_p05 = float(np.nanpercentile(d, 5))

    def _z(self, df, fit=False):
        from sklearn.preprocessing import QuantileTransformer
        X = np.nan_to_num(df[NUM].astype(float).values)
        if fit:
            self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform", random_state=0).fit(X)
        return self.qt.transform(X)

    def __call__(self, syn):
        d = self.nn.kneighbors(self._z(syn))[0][:, 0]
        return dict(DCR_median=float(np.median(d)), DCR_ref_median=self.ref_median,
                    DCR_share_below_ref_p05=float(np.mean(d < self.ref_p05)))


def evaluate(ref, syn, ref_groups, syn_groups, seed, dcr=None, with_auc=True, full_auc=False):
    mu, sd = ref[NUM].mean(), ref[NUM].std()
    ks, ws, js, nonfin = {}, {}, {}, 0
    for c in NUM:
        s = syn[c].astype(float).values
        nonfin += int((~np.isfinite(s)).sum())
        s = _finite(s)
        r = ref[c].astype(float).values
        ks[c] = float(stats.ks_2samp(r, s).statistic)
        ws[c] = float(stats.wasserstein_distance((r - mu[c]) / sd[c], (s - mu[c]) / sd[c]))
        js[c] = jsd_base2(r, s)
    pr = ref[CAT].value_counts(normalize=True)
    ps = syn[CAT].value_counts(normalize=True).reindex(pr.index).fillna(0)
    out = dict(KS_mean=np.mean(list(ks.values())), KS_median=np.median(list(ks.values())),
               KS_max=np.max(list(ks.values())), KS_argmax=max(ks, key=ks.get),
               W_std_mean=np.mean(list(ws.values())), JSD_mean=np.mean(list(js.values())),
               sector_TVD=float(0.5 * np.abs(pr - ps).sum()), n_nonfinite=nonfin)
    out["corr_MAD"], out["corr_Frobenius"], _, out["corr_n_nan"] = corr_metrics(ref, syn, NUM, "pearson", True)
    out["spearman_MAD"], out["spearman_Frobenius"], _, out["spearman_n_nan"] = corr_metrics(ref, syn, NUM,
                                                                                            "spearman", True)
    out["n_constant_cols"] = int(sum(syn[c].astype(float).std() == 0 for c in NUM))
    out.update(cvr(syn))
    out["CVR_tol1pct"] = cvr(syn, CVR_TOL_LOOSE)["CVR"]
    out.update(point_mass_shares(syn))
    if with_auc:
        out["disc_AUC"], out["disc_AUC_sd"] = disc_auc(ref, syn, ref_groups, syn_groups, seed)
    if with_auc and full_auc:
        out["disc_AUC_ratios"], out["disc_AUC_ratios_sd"] = disc_auc(ref, syn, ref_groups, syn_groups, seed,
                                                                     cols=RATIO_COLS)
        out["disc_AUC_nopm"], out["disc_AUC_nopm_sd"] = disc_auc(ref, syn, ref_groups, syn_groups, seed,
                                                                 cols=NO_PM_COLS)
        out["disc_AUC_logit"], out["disc_AUC_logit_sd"] = disc_auc(ref, syn, ref_groups, syn_groups, seed,
                                                                   model="logit")
    if dcr is not None:
        out.update(dcr(syn))
    return out, ks, ws, js


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def gen_b1_noise(ref, n, seed, groups, noise_scale=1.0, project=True):
    """Noise baseline, mechanism of run_pipeline_enhanced.run_synthetic_generator(): z-score columns with
    the reference mean/s.d., take reference rows, add N(0, s_j * noise_scale) noise with
    s_j = linspace(0.05, 0.15, n_features) in NUM order, clip to [-3, 3], invert the z-score.
    Differences from the original: rows are drawn with replacement (n rows instead of the first 800),
    all 17 NUM columns are generated, and the A = D + E projection is applied in levels."""
    rs = np.random.RandomState(seed)
    X = ref[NUM].astype(float)
    mu, sd = X.mean(), X.std()
    Z = ((X - mu) / sd).values
    idx = rs.randint(0, len(ref), n)
    s = np.linspace(0.05, 0.15, len(NUM)) * noise_scale
    Zs = np.clip(Z[idx] + rs.normal(0, 1, (n, len(NUM))) * s, -3, 3)
    out = pd.DataFrame(Zs * sd.values + mu.values, columns=NUM)
    out[CAT] = ref[CAT].values[idx]
    if project:
        out = project_identity(out)
    return out, np.asarray(groups)[idx]


def gen_b1_noise_log(ref, n, seed, groups, noise_scale=1.0, project=True):
    """Sensitivity variant of B1 (not tuned): identical noise scales, row draws and [-3, 3] clipping, but the
    positive level columns (LEVELS_POS) are log-transformed and net_income is signed-log transformed
    (sign(x) log(1 + |x|)) before z-scoring; inverse transforms afterwards. Keeps levels positive, so the
    non-negativity part of CVR no longer reflects the z-scoring of heavy-tailed levels."""
    rs = np.random.RandomState(seed)
    X = ref[NUM].astype(float).copy()
    for c in LEVELS_POS:
        X[c] = np.log(X[c])
    for c in LEVELS_SIGNED:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    mu, sd = X.mean(), X.std()
    Z = ((X - mu) / sd).values
    idx = rs.randint(0, len(ref), n)
    s = np.linspace(0.05, 0.15, len(NUM)) * noise_scale
    Zs = np.clip(Z[idx] + rs.normal(0, 1, (n, len(NUM))) * s, -3, 3)
    out = pd.DataFrame(Zs * sd.values + mu.values, columns=NUM)
    for c in LEVELS_POS:
        out[c] = np.exp(out[c])
    for c in LEVELS_SIGNED:
        out[c] = np.sign(out[c]) * np.expm1(np.abs(out[c]))
    out[CAT] = ref[CAT].values[idx]
    if project:
        out = project_identity(out)
    return out, np.asarray(groups)[idx]


def gen_b2_gauss(ref, n, seed, groups, project=True):
    """Gaussian baseline, mechanism of generative/baselines/run_comparison.run_constrained_baseline():
    log of positive level columns, winsorise at the 1st/99th percentiles, fit a multivariate normal
    (on z-scores, covariance + 1e-4 I), sample, back-transform, then project onto A = D + E in levels
    (the original projected log columns). Sector drawn independently from its empirical frequencies."""
    rs = np.random.RandomState(seed)
    X = ref[NUM].astype(float).copy()
    for c in LEVELS_POS:
        X[c] = np.log(X[c])
    for c in NUM:
        lo, hi = np.percentile(X[c], [1, 99])
        X[c] = X[c].clip(lo, hi)
    mu, sd = X.mean(), X.std()
    Z = ((X - mu) / sd).values
    cov = np.cov(Z.T) + np.eye(len(NUM)) * 1e-4
    S = rs.multivariate_normal(np.zeros(len(NUM)), cov, n)
    out = pd.DataFrame(S * sd.values + mu.values, columns=NUM)
    for c in LEVELS_POS:
        out[c] = np.exp(out[c])
    freq = ref[CAT].value_counts(normalize=True).sort_index()
    out[CAT] = rs.choice(freq.index.values, n, p=freq.values)
    if project:
        out = project_identity(out)
    return out, rs.choice(np.unique(groups), n)


class _CachedTransformer:
    """Pre-fitted ctgan DataTransformer shared by all CTGAN/TVAE runs (mode-specific normalisation fitted
    once on the reference panel); fit() is a no-op and transform() of the training frame returns the
    cached matrix."""

    def __init__(self, tr, df_ref, Z):
        self._tr, self._df, self._Z = tr, df_ref, Z

    def fit(self, *a, **k):
        return None

    def transform(self, df):
        if df is self._df:
            return self._Z.copy()
        return self._tr.transform(df)

    def __getattr__(self, name):
        return getattr(self._tr, name)


def fit_shared_transformer(train_df):
    from ctgan.data_transformer import DataTransformer
    np.random.seed(PANEL_SEED)
    t0 = time.perf_counter()
    tr = DataTransformer()          # max_clusters=10, weight_threshold=0.005
    tr.fit(train_df, [CAT])
    Z = tr.transform(train_df)
    secs = time.perf_counter() - t0
    info = [{"column": c.column_name, "type": c.column_type, "output_dimensions": c.output_dimensions}
            for c in tr._column_transform_info_list]
    return _CachedTransformer(tr, train_df, Z), secs, pd.DataFrame(info)


def _deep_job(kind, epochs, seed, n, train_df, tr, Z):
    """Worker for one CTGAN/TVAE fit (separate process, 1 torch thread)."""
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    cached = _CachedTransformer(tr, train_df, Z)
    return (kind, epochs, seed) + train_deep(kind, train_df, cached, epochs, seed, n)


def train_deep(kind, train_df, cached, epochs, seed, n):
    import torch
    import ctgan.synthesizers.ctgan as ctmod
    import ctgan.synthesizers.tvae as tvmod
    from ctgan import CTGAN, TVAE
    ctmod.DataTransformer = lambda *a, **k: cached
    tvmod.DataTransformer = lambda *a, **k: cached
    torch.manual_seed(seed)
    np.random.seed(seed)
    if kind == "CTGAN":
        m = CTGAN(epochs=epochs, enable_gpu=False, verbose=False, **CTGAN_KW)
    else:
        m = TVAE(epochs=epochs, enable_gpu=False, verbose=False, **TVAE_KW)
    m.set_random_state(seed)
    t0 = time.perf_counter()
    m.fit(train_df, discrete_columns=[CAT])
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    syn = m.sample(n)
    t_sample = time.perf_counter() - t0
    lv = m.loss_values.copy()
    if kind == "CTGAN":
        lv = lv.rename(columns={"Distriminator Loss": "discriminator_loss", "Discriminator Loss": "discriminator_loss",
                                "Generator Loss": "generator_loss",
                                "Epoch": "epoch"})
        lv = lv[["epoch", "generator_loss", "discriminator_loss"]]
    else:
        lv = lv.groupby("Epoch", as_index=False)["Loss"].mean().rename(columns={"Epoch": "epoch", "Loss": "loss"})
    syn = syn[NUM + [CAT]].reset_index(drop=True)
    for c in NUM:
        syn[c] = syn[c].astype(float)
    return syn, lv, t_fit, t_sample


# ---------------------------------------------------------------------------
# Section 1: consistency check of the earlier generative comparison
# ---------------------------------------------------------------------------
def diagnose_original(out_dir, quick=False):
    log("Section 1: diagnosis of the original generative comparison")
    # read-only import of the original module (main() is not called). The baselines folder also holds a
    # file named ctgan.py that would shadow the ctgan package, so sys.path and sys.modules are restored.
    import importlib.util
    saved_path = list(sys.path)
    spec = importlib.util.spec_from_file_location("orig_run_comparison", os.path.join(BASELINE_DIR, "run_comparison.py"))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    sys.path[:] = saved_path
    if "ctgan" in sys.modules and BASELINE_DIR.lower() in str(getattr(sys.modules["ctgan"], "__file__", "")).lower():
        del sys.modules["ctgan"]
    rows, buf = [], io.StringIO()

    # (a) generative/baselines/run_comparison.py as it ran: load_financial_data() looks for
    #     <root>/outputs/...ENHANCED.xlsx sheet "Dataset"; that sheet does not exist in the workbook, so the
    #     script falls back to its own internal generator (a different DGP from the reference panel).
    xl = os.path.join(ROOT, "outputs_original", "GenFin_AI_Capital_Structure_Analysis_ENHANCED.xlsx")
    sheets = pd.ExcelFile(xl).sheet_names
    with contextlib.redirect_stdout(buf):
        df_fb = rc.load_financial_data()
    branch = "fallback internal generator" if "Generating synthetic financial dataset" in buf.getvalue() else "excel"
    X, names = rc.prepare_features(df_fb)
    real = X[:4440]
    t0 = time.perf_counter()
    s_base = rc.run_constrained_baseline(real.copy(), 4440)
    tb = time.perf_counter() - t0
    with contextlib.redirect_stdout(buf):
        t0 = time.perf_counter(); s_ct = rc.run_ctgan_baseline(real, 4440); tc = time.perf_counter() - t0
        t0 = time.perf_counter(); s_tv = rc.run_tvae_baseline(real, 4440); tt = time.perf_counter() - t0
    orig = pd.read_csv(os.path.join(ROOT, "outputs_original", "generative_baseline_comparison.csv"))
    orig.columns = ["Method", "KS", "Wasserstein", "JS", "Corr", "CVR", "Time"]
    iu = np.triu_indices(real.shape[1], 1)
    for meth, s, tsec in [("Constrained Baseline", s_base, tb), ("CTGAN", s_ct, tc), ("TVAE", s_tv, tt)]:
        m = rc.evaluate_synthetic(real, s)
        if meth == "Constrained Baseline":
            m_cvr_computed = m["CVR"]
            m["CVR"] = 0.0      # main() overwrote the computed value with 0.0
        else:
            m_cvr_computed = m["CVR"]
        dcorr = np.corrcoef(s.T) - np.corrcoef(real.T)
        o = orig.loc[orig.Method == meth].iloc[0]
        rows.append(dict(source="run_comparison.py on its fallback dataset (4,440 x 10)", method=meth,
                         KS_rerun=m["KS_statistic"], KS_file=o.KS, W_rerun=m["Wasserstein"], W_file=o.Wasserstein,
                         JS_rerun=m["JS_divergence"], JS_file=o.JS,
                         corr_frobenius_rerun=m["correlation_preservation"], corr_file=o.Corr,
                         corr_mean_abs_upper_rerun=float(np.nanmean(np.abs(dcorr[iu]))),
                         CVR_reported_rerun=m["CVR"], CVR_computed_before_override=m_cvr_computed, CVR_file=o.CVR,
                         time_rerun_s=tsec, time_file_s=o.Time))
    # output statistics of the original "CTGAN"/"TVAE" (shows the outputs are near zero for every column)
    st = []
    for meth, s in [("reference(fallback, transformed)", real), ("CTGAN(original numpy)", s_ct),
                    ("TVAE(original numpy)", s_tv), ("Constrained Baseline(original)", s_base)]:
        for j, c in enumerate(names):
            st.append(dict(method=meth, feature=c, mean=float(np.mean(s[:, j])), sd=float(np.std(s[:, j])),
                           min=float(np.min(s[:, j])), max=float(np.max(s[:, j]))))
    pd.DataFrame(st).to_csv(os.path.join(out_dir, "diag_original_output_stats.csv"), index=False)

    # (b) run_comparison_expanded.py on the 46-firm Yahoo CSV (source of KS 0.985 / 0.987 in Table X)
    fx = os.path.join(common.REAL_DATA_DIR, "fundamentals_2022_2025.csv")
    d = pd.read_csv(fx)
    fc = ["debt_ratio", "total_assets", "total_debt", "total_equity", "roa", "roe", "interest_coverage",
          "wacc", "default_risk_score", "altman_z_score"]
    dd = d[fc].dropna().copy()
    for c in ["total_assets", "total_debt", "total_equity"]:
        dd[c] = np.log1p(dd[c])
    XE = dd.values
    orig_e = pd.read_csv(os.path.join(ROOT, "outputs_original", "generative_comparison_EXPANDED.csv"))
    with contextlib.redirect_stdout(buf):
        sE = {"Constrained Baseline": rc.run_constrained_baseline(XE.copy(), len(XE)),
              "CTGAN": rc.run_ctgan_baseline(XE, len(XE)), "TVAE": rc.run_tvae_baseline(XE, len(XE))}
    for meth, s in sE.items():
        m = rc.evaluate_synthetic(XE, s)
        o = orig_e.loc[orig_e.Method == meth].iloc[0]
        rows.append(dict(source=f"run_comparison_expanded.py on fundamentals_2022_2025.csv ({len(XE)} x 10, real firms)",
                         method=meth, KS_rerun=m["KS_statistic"], KS_file=o.KS_statistic,
                         W_rerun=m["Wasserstein"], W_file=o.Wasserstein, JS_rerun=m["JS_divergence"],
                         JS_file=o.JS_divergence, corr_frobenius_rerun=m["correlation_preservation"],
                         corr_file=o.correlation_preservation,
                         CVR_reported_rerun=0.0 if meth == "Constrained Baseline" else m["CVR"],
                         CVR_computed_before_override=m["CVR"], CVR_file=o.CVR, time_file_s=o["Time (s)"]))

    # (c) run_pipeline_enhanced.run_synthetic_generator() on the reference panel (source of KS 0.135,
    #     W 0.260, JS 1.557, CVR 0). The original used the global numpy stream at that point of the pipeline,
    #     which cannot be recovered; the computation is repeated with 20 seeds.
    ref = common.generate_panel(PANEL_SEED)
    gcols = common.FEATURES_ORIGINAL[:14]
    Xs = (ref[gcols] - ref[gcols].mean()) / ref[gcols].std()
    nf = len(gcols)
    pip_rows, pip_ks = [], []
    for sd in range(20 if not quick else 3):
        rs = np.random.RandomState(sd)
        x = Xs.values[:800].copy()
        lv = np.linspace(0.05, 0.15, nf)
        for j in range(nf):
            x[:, j] += rs.normal(0, lv[j], 800)
        x = np.clip(x, -3, 3)
        ks8 = [stats.ks_2samp(Xs.values[:, j], x[:, j])[0] for j in range(8)]
        ws8 = [stats.wasserstein_distance(Xs.values[:, j], x[:, j]) for j in range(8)]
        js8 = []
        for j in range(8):
            h1, _ = np.histogram(Xs.values[:, j], bins=30, density=True)
            h2, _ = np.histogram(x[:, j], bins=30, density=True)
            mm = (h1 + h2) / 2
            js8.append((np.sum(h1 * np.log((h1 + 1e-10) / (mm + 1e-10))) +
                        np.sum(h2 * np.log((h2 + 1e-10) / (mm + 1e-10)))) / 2)
        cvr_orig = 1.0 - ((np.abs(x) < 4).all(axis=1).mean())
        ks14 = [stats.ks_2samp(Xs.values[:, j], x[:, j])[0] for j in range(nf)]
        js14_fixed = [jsd_base2(Xs.values[:, j], x[:, j]) for j in range(nf)]
        # like-for-like correction of the original JS: same first 8 features and 30 bins, but shared bin
        # edges (reference range), probabilities instead of densities, base 2
        js8_fixed = []
        for j in range(8):
            e = np.histogram_bin_edges(Xs.values[:, j], bins=30)
            e[0], e[-1] = -np.inf, np.inf
            p_ = np.histogram(Xs.values[:, j], e)[0].astype(float); q_ = np.histogram(x[:, j], e)[0].astype(float)
            js8_fixed.append(float(jensenshannon(p_ / p_.sum(), q_ / q_.sum(), base=2) ** 2))
        pip_rows.append(dict(seed=sd, KS_mean_first8_original=np.mean(ks8), W_mean_first8_original=np.mean(ws8),
                             JS_mean_first8_original_formula=np.mean(js8), CVR_original_formula=cvr_orig,
                             JSD_base2_shared30bins_first8=np.mean(js8_fixed),
                             KS_mean_all14=np.mean(ks14), JSD_base2_shared_bins_all14=np.mean(js14_fixed),
                             share_rows_copied_from_reference_first800=800 / len(ref),
                             sectors_in_first800=";".join(f"{k}:{v}" for k, v in
                                                          ref["sector"].iloc[:800].value_counts().items())))
        pip_ks.append(dict(zip(gcols, ks14)))
    pip = pd.DataFrame(pip_rows)
    pip.to_csv(os.path.join(out_dir, "diag_pipeline_noise_generator_rerun.csv"), index=False)
    pk = pd.DataFrame(pip_ks).agg(["mean", "std"]).T.reset_index().rename(columns={"index": "feature"})
    pk.to_csv(os.path.join(out_dir, "diag_pipeline_noise_generator_per_feature_ks.csv"), index=False)

    diag = pd.DataFrame(rows)
    diag.to_csv(os.path.join(out_dir, "diag_original_reproduction.csv"), index=False)

    # provenance of each Table X cell
    prov = [
        ("Constrained Baseline", "KS", 0.135, "Excel sheet Synth_Data_Eval (run_synthetic_generator; 800 noisy copies of reference rows, KS over first 8 standardised features)"),
        ("Constrained Baseline", "Wasserstein", 0.260, "Excel sheet Synth_Data_Eval"),
        ("Constrained Baseline", "JS", 1.557, "Excel sheet Synth_Data_Eval (natural-log formula on histograms with different bin edges for the two samples; not a valid JSD, which is <= ln 2 = 0.693)"),
        ("Constrained Baseline", "CVR", 0.0, "Excel sheet Synth_Data_Eval (share of rows with |z| >= 4 after clipping to [-3, 3]: 0 by construction)"),
        ("Constrained Baseline", "Time", 0.003, "generative_comparison_EXPANDED.csv (real 46-firm data run)"),
        ("CTGAN", "KS", 0.985, "generative_comparison_EXPANDED.csv (46-firm Yahoo data, 184 rows; not the synthetic panel)"),
        ("TVAE", "KS", 0.987, "generative_comparison_EXPANDED.csv (46-firm Yahoo data, 184 rows)"),
        ("CTGAN", "Wasserstein", 2.881, "generative_baseline_comparison.csv (run_comparison.py fallback dataset; baseline KS in that run = 0.774)"),
        ("TVAE", "Wasserstein", 2.928, "generative_baseline_comparison.csv"),
        ("CTGAN", "Corr. pres.", 3.278, "generative_baseline_comparison.csv; Frobenius norm of the 10 x 10 correlation difference, not a mean absolute difference"),
        ("TVAE", "Corr. pres.", 3.183, "generative_baseline_comparison.csv; Frobenius norm"),
        ("CTGAN", "CVR", 0.016, "generative_baseline_comparison.csv; ratio mean|x0-(x1+x2)|_syn / mean|x0-(x1+x2)|_ref with x0 = debt_ratio, x1 = log1p(assets), x2 = log1p(debt)"),
        ("TVAE", "CVR", 0.001, "generative_baseline_comparison.csv; same ratio"),
        ("CTGAN", "Time", 9.393, "generative_baseline_comparison.csv"),
        ("TVAE", "Time", 4.730, "generative_baseline_comparison.csv"),
    ]
    pd.DataFrame(prov, columns=["row", "column", "value_in_paper", "source_and_definition"]).to_csv(
        os.path.join(out_dir, "diag_table_x_provenance.csv"), index=False)
    info = dict(excel_has_Dataset_sheet="Dataset" in sheets, run_comparison_data_branch=branch,
                fallback_shape=list(df_fb.shape), fallback_features=names, expanded_rows=int(len(XE)))
    log(f"  run_comparison data branch: {branch}; diag rows: {len(diag)}")
    return diag, pip, info


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FEAT_LABEL = {"debt_ratio": "Debt ratio (D/A)", "roa": "ROA (NI/A)", "cash_ratio": "Cash ratio (cash/A)",
              "wacc": "WACC (fraction)", "interest_coverage": "Interest coverage (EBIT/interest, x)",
              "tobin_q": "Tobin's Q (ratio)", "total_assets": "log10 total assets (model units)",
              "default_risk_score": "Default-risk score (0-1)", "roe": "ROE (NI/E)",
              "altman_zscore": "Altman Z (score)"}
SHORT = {"total_assets": "A", "total_debt": "D", "total_equity": "E", "revenue": "Rev", "net_income": "NI",
         "cash_holdings": "Cash", "interest_expense": "IntExp", "market_cap": "MCap", "debt_ratio": "DR",
         "roa": "ROA", "roe": "ROE", "cash_ratio": "CashR", "interest_coverage": "IntCov", "tobin_q": "Q",
         "wacc": "WACC", "default_risk_score": "DRisk", "altman_zscore": "Z"}


def fig_distributions(ref, samples, path):
    plt = common.set_style()
    feats = ["debt_ratio", "roa", "cash_ratio", "wacc", "interest_coverage", "tobin_q", "total_assets",
             "default_risk_score"]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6))
    for ax, f in zip(axes.ravel(), feats):
        tf = (lambda v: np.log10(v[v > 0])) if f == "total_assets" else (lambda v: v)
        r = tf(ref[f].values.astype(float))
        lo, hi = np.percentile(r, [0.5, 99.5])
        pad = 0.15 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, 41)
        ax.hist(r, bins=edges, density=True, color=REF_COLOR, alpha=0.55, label="Reference synthetic panel")
        for k, s in samples.items():
            v = tf(s[f].values.astype(float))
            v = v[np.isfinite(v)]
            ax.hist(np.clip(v, edges[0], edges[-1]), bins=edges, density=True, histtype="step",
                    lw=1.0, color=GEN_COLOR[k], label=GEN_LABEL[k])
        ax.set_xlabel(FEAT_LABEL[f], fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    common.savefig(fig, path)
    plt.close(fig)


def _pca_space(ref):
    X = ref[NUM].astype(float).copy()
    for c in LEVELS_POS + LEVELS_SIGNED:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    return X.mean(), X.std()


def _pca_transform(df, mu, sd):
    X = df[NUM].astype(float).copy()
    for c in LEVELS_POS + LEVELS_SIGNED:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    return ((X - mu) / sd).values


def fig_pca(ref, samples, path):
    from sklearn.decomposition import PCA
    plt = common.set_style()
    mu, sd = _pca_space(ref)
    Zr = _pca_transform(ref, mu, sd)
    pca = PCA(2, random_state=0).fit(Zr)
    Pr = pca.transform(Zr)
    ev = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, len(samples), figsize=(7.0, 2.1), sharex=True, sharey=True)
    xl = np.percentile(Pr[:, 0], [0.2, 99.8]); yl = np.percentile(Pr[:, 1], [0.2, 99.8])
    padx, pady = 0.4 * (xl[1] - xl[0]), 0.4 * (yl[1] - yl[0])
    for ax, (k, s) in zip(axes, samples.items()):
        Ps = pca.transform(np.nan_to_num(_pca_transform(s, mu, sd)))
        ax.scatter(Pr[:, 0], Pr[:, 1], s=1.5, c=REF_COLOR, alpha=0.35, lw=0, label="Reference synthetic panel",
                   rasterized=True)
        ax.scatter(Ps[:, 0], Ps[:, 1], s=1.5, c=GEN_COLOR[k], alpha=0.35, lw=0, label="Generated", rasterized=True)
        ax.text(0.03, 0.97, GEN_LABEL[k], transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlim(xl[0] - padx, xl[1] + padx); ax.set_ylim(yl[0] - pady, yl[1] + pady)
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.0f}% var.)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(f"PC2 ({100 * ev[1]:.0f}% var.)", fontsize=8)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=REF_COLOR, ms=3, label="Reference synthetic panel"),
          Line2D([], [], marker="o", ls="", color="#444444", ms=3, label="Generated (colour by generator)")]
    fig.legend(handles=hs, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    common.savefig(fig, path)
    plt.close(fig)
    return ev


def fig_losses(losses, path):
    """Loss trajectories of the longest run of each model (CTGAN 300 epochs; TVAE 600-epoch diagnostic run
    if present, else 300). Vertical lines: 50 and 300 epochs (the two reported training lengths)."""
    plt = common.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    ls = {42: "-", 43: "--", 44: ":"}
    longest = {k: max(e for (kk, e, _) in losses if kk == k) for k in {kk for (kk, _, _) in losses}}
    for (kind, ep, seed), lv in losses.items():
        if ep != longest[kind]:
            continue
        if kind == "CTGAN":
            axes[0].plot(lv.epoch + 1, lv.generator_loss, ls[seed], color=GEN_COLOR["CTGAN"], lw=0.8,
                         label=f"seed {seed}")
            axes[1].plot(lv.epoch + 1, lv.discriminator_loss, ls[seed], color=GEN_COLOR["CTGAN"], lw=0.8,
                         label=f"seed {seed}")
        else:
            axes[2].plot(lv.epoch + 1, lv.loss, ls[seed], color=GEN_COLOR["TVAE"], lw=0.8, label=f"seed {seed}")
    for ax, yl in zip(axes, ["CTGAN generator loss", "CTGAN critic loss",
                             "TVAE loss (recon. + KL)"]):
        for v in (50, 300):
            ax.axvline(v, color="#000000", lw=0.6, alpha=0.4)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    common.savefig(fig, path)
    plt.close(fig)


def fig_corr_diff(ref, samples, path):
    plt = common.set_style()
    keys = list(samples)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
    lab = [SHORT[c] for c in NUM]
    im = None
    for ax, k in zip(axes.ravel(), keys):
        mad, fro, d = corr_metrics(ref, samples[k], NUM)
        im = ax.imshow(np.ma.masked_invalid(d), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(NUM))); ax.set_yticks(range(len(NUM)))
        ax.set_xticklabels(lab, rotation=90, fontsize=6); ax.set_yticklabels(lab, fontsize=6)
        ax.set_xlabel(f"{GEN_LABEL[k]}: MAD = {mad:.3f}, Frobenius = {fro:.2f}", fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Pearson r (generated) - r (reference)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    common.savefig(fig, path)
    plt.close(fig)


def fig_sens_dgp(grid, path):
    """Rows: mean KS, mean standardised Wasserstein, CVR; columns: B1, B2. x = sigma_zeta, one line per lambda.
    Each row shares its y-axis across the two generators."""
    plt = common.set_style()
    lam_col = dict(zip(LAMBDAS, ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00"]))
    rows = [("KS_mean", "Mean KS"), ("W_std_mean", "Mean Wasserstein (z units)"), ("CVR", "CVR (share of rows)")]
    gens = ["B1_noise", "B2_gauss"]
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 5.4), sharey="row", sharex=True)
    for i, (met, lab) in enumerate(rows):
        for j, g in enumerate(gens):
            ax = axes[i, j]
            sub = grid[grid.generator == g]
            for lam in sorted(sub.lam.unique()):
                t = sub[sub.lam == lam].sort_values("sigma_zeta")
                ax.plot(t.sigma_zeta, t[met], marker="o", ms=2.5, lw=0.9, color=lam_col.get(lam, "#000000"),
                        label=rf"$\lambda$ = {lam:g}")
            ax.set_xscale("log")
            ax.set_xticks(SIGMAS); ax.set_xticklabels([f"{v:g}" for v in SIGMAS])
            ax.minorticks_off()
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            if i == 2:
                ax.set_xlabel(r"$\sigma_\zeta$ (s.d. of adjustment shock)", fontsize=8)
            if i == 0:
                ax.text(0.02, 0.95, GEN_LABEL[g], transform=ax.transAxes, va="top", fontsize=8)
    lo = min(0.0, grid["KS_mean"].min()); axes[0, 0].set_ylim(lo, grid["KS_mean"].max() * 1.25)
    axes[1, 0].set_ylim(0, grid["W_std_mean"].max() * 1.25)
    axes[2, 0].set_ylim(0, max(0.05, grid["CVR"].max() * 1.15))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    common.savefig(fig, path)
    plt.close(fig)


def fig_sens_noise(ns, path):
    plt = common.set_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1))
    for ax, (met, lab) in zip(axes, [("KS_mean", "Mean KS"), ("W_std_mean", "Mean Wasserstein (z units)"),
                                     ("CVR", "CVR (share of rows)"),
                                     ("DCR_median", "Median DCR (quantile units)")]):
        for gname in ["B1_noise", "B1_noise_log"]:
            agg = ns[ns.generator == gname].groupby("noise_scale")[met].agg(["mean", "std"])
            ax.errorbar(agg.index.values, agg["mean"], yerr=agg["std"], color=GEN_COLOR[gname], marker="o",
                        ms=3, lw=1, capsize=2, label=GEN_LABEL[gname])
        if met == "DCR_median":
            ax.axhline(ns["DCR_ref_median"].iloc[0], color="#000000", ls="--", lw=0.7)
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xticks([0, 0.25, 1, 4]); ax.set_xticklabels(["0", "0.25", "1", "4"])
        ax.set_xlabel("Noise scale k (x original)", fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    common.savefig(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fmt_pm(m, s, d=3):
    return f"{m:.{d}f} +- {s:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    out_dir = os.path.join(common.OUT_MAIN, "e4_generative" + ("_quick" if quick else ""))
    fig_dir = os.path.join(out_dir, "figures")
    smp_dir = os.path.join(out_dir, "samples")
    for d in (out_dir, fig_dir, smp_dir):
        os.makedirs(d, exist_ok=True)
    import torch
    torch.set_num_threads(4)
    T0 = time.perf_counter()
    epochs_list = [300, 50] if not quick else [6, 2]

    # ---- Section 1
    diag, pip, diag_info = diagnose_original(out_dir, quick)

    # ---- Section 2: reference panel
    ref = common.generate_panel(PANEL_SEED)
    ref_gen = ref[NUM + [CAT]].reset_index(drop=True)
    ref_groups = ref["firm_id"].values
    log(f"Reference synthetic panel: {ref.shape}, generation columns: {len(NUM)} numeric + sector")
    ref_cvr = cvr(ref_gen)
    dcr = DCR(ref)
    pd.DataFrame([{"column": c, "role": "numeric (continuous)", "log_in_B2": c in LEVELS_POS,
                   "ref_mean": ref[c].mean(), "ref_sd": ref[c].std(), "ref_min": ref[c].min(), "ref_max": ref[c].max()}
                  for c in NUM] + [{"column": CAT, "role": "categorical (8 levels)"}]).to_csv(
        os.path.join(out_dir, "generation_columns.csv"), index=False)

    runs, perks, losses, samples_first = [], [], {}, {}
    per_ws, per_js, viol = [], [], []

    def record(gen, epochs, seed, syn, groups, t_fit, t_sample, extra=None):
        m, ks, ws, js = evaluate(ref_gen, syn, ref_groups, groups, seed, dcr=dcr, full_auc=True)
        row = dict(generator=gen, epochs=epochs, seed=seed, n_rows=len(syn), fit_time_s=t_fit,
                   sample_time_s=t_sample, **m)
        if extra:
            row.update(extra)
        runs.append(row)
        perks.append(dict(generator=gen, epochs=epochs, seed=seed, **ks))
        viol.append(dict(generator=gen, epochs=epochs, seed=seed, **violation_breakdown(syn)))
        per_ws.append(dict(generator=gen, epochs=epochs, seed=seed, **ws))
        per_js.append(dict(generator=gen, epochs=epochs, seed=seed, **js))
        log(f"  {gen:11s} ep={epochs} seed={seed}: KS={m['KS_mean']:.3f} W={m['W_std_mean']:.3f} "
            f"JSD={m['JSD_mean']:.3f} corrMAD={m['corr_MAD']:.3f} CVR={m['CVR']:.3f} AUC={m['disc_AUC']:.3f}")

    # reference against itself (sanity row): split firms in half
    firms = np.unique(ref_groups)
    rsx = np.random.RandomState(0)
    half = set(rsx.choice(firms, len(firms) // 2, replace=False))
    msk = np.isin(ref_groups, list(half))
    m_self, _, _, _ = evaluate(ref_gen[msk].reset_index(drop=True), ref_gen[~msk].reset_index(drop=True),
                               ref_groups[msk], ref_groups[~msk], PANEL_SEED, dcr=None)
    m_self.update(dict(generator="Reference half vs half (firms split 148/148)", epochs=np.nan, seed=0))

    # baselines
    log("Section 2: baselines")
    for seed in GEN_SEEDS:
        t0 = time.perf_counter(); syn, g = gen_b1_noise(ref_gen, N_GEN, seed, ref_groups); t1 = time.perf_counter() - t0
        record("B1_noise", 0, seed, syn, g, t1, 0.0)
        if seed == GEN_SEEDS[0]:
            samples_first["B1_noise"] = syn
            syn.to_csv(os.path.join(smp_dir, f"B1_noise_seed{seed}.csv"), index=False)
        t0 = time.perf_counter(); syn, g = gen_b2_gauss(ref_gen, N_GEN, seed, ref_groups); t1 = time.perf_counter() - t0
        record("B2_gauss", 0, seed, syn, g, t1, 0.0)
        if seed == GEN_SEEDS[0]:
            samples_first["B2_gauss"] = syn
            syn.to_csv(os.path.join(smp_dir, f"B2_gauss_seed{seed}.csv"), index=False)
        t0 = time.perf_counter(); syn, g = gen_b1_noise_log(ref_gen, N_GEN, seed, ref_groups)
        t1 = time.perf_counter() - t0
        record("B1_noise_log", 0, seed, syn, g, t1, 0.0)
        if seed == GEN_SEEDS[0]:
            samples_first["B1_noise_log"] = syn
            syn.to_csv(os.path.join(smp_dir, f"B1_noise_log_seed{seed}.csv"), index=False)
    # unprojected baselines (to isolate the effect of the projection)
    for seed in GEN_SEEDS:
        syn, g = gen_b1_noise(ref_gen, N_GEN, seed, ref_groups, project=False)
        record("B1_noise_noproj", 0, seed, syn, g, np.nan, 0.0)
        syn, g = gen_b2_gauss(ref_gen, N_GEN, seed, ref_groups, project=False)
        record("B2_gauss_noproj", 0, seed, syn, g, np.nan, 0.0)

    # deep generators
    log("Section 2: CTGAN / TVAE (shared DataTransformer)")
    cached, t_transformer, tinfo = fit_shared_transformer(ref_gen)
    tinfo.to_csv(os.path.join(out_dir, "ctgan_transformer_columns.csv"), index=False)
    log(f"  DataTransformer fit+transform: {t_transformer:.1f} s; output width {cached.output_dimensions}")
    pd.DataFrame(runs).to_csv(os.path.join(out_dir, "generative_runs_partial.csv"), index=False)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(kind, ep, seed) for kind in ["CTGAN", "TVAE"] for ep in epochs_list for seed in GEN_SEEDS]
    # convergence diagnostic (TVAE loss still falling at 300 epochs): TVAE at 2 x the main epoch count
    extra_jobs = [("TVAE", 2 * epochs_list[0], seed) for seed in GEN_SEEDS]
    jobs += extra_jobs
    jobs.sort(key=lambda j: -(j[1] * (3 if j[0] == "CTGAN" else 1)))     # longest first
    # checkpoint: every finished fit is stored in deep_cache/ (pickle of sample, loss table and timings), so an
    # interrupted run resumes without refitting. Delete deep_cache/ to force a clean refit.
    cache_dir = os.path.join(out_dir, "deep_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cpath = lambda k, e, s_: os.path.join(cache_dir, f"{k}_ep{e}_seed{s_}.pkl")
    results = {}
    for k, e, s_ in jobs:
        if os.path.exists(cpath(k, e, s_)):
            results[(k, e, s_)] = pd.read_pickle(cpath(k, e, s_))
            log(f"  loaded cached {k} ep={e} seed={s_} (fit {results[(k, e, s_)][2]:.0f} s)")
    todo = [j for j in jobs if j not in results]
    t0 = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(_deep_job, k, e, s_, N_GEN, ref_gen, cached._tr, cached._Z): (k, e, s_)
                    for k, e, s_ in todo}
            for f in as_completed(futs):
                kind, ep, seed, syn, lv, tf, ts = f.result()
                results[(kind, ep, seed)] = (syn, lv, tf, ts)
                pd.to_pickle((syn, lv, tf, ts), cpath(kind, ep, seed))
                log(f"  trained {kind} ep={ep} seed={seed}: {tf:.0f} s")
    t_deep_wall = time.perf_counter() - t0
    os.makedirs(os.path.join(out_dir, "loss_curves"), exist_ok=True)
    for kind, ep in [(k_, e_) for k_ in ["CTGAN", "TVAE"] for e_ in epochs_list] + [("TVAE", 2 * epochs_list[0])]:
        if True:
            for seed in GEN_SEEDS:
                syn, lv, tf, ts = results[(kind, ep, seed)]
                groups = np.random.RandomState(seed).choice(firms, N_GEN)
                losses[(kind, ep, seed)] = lv
                lv.assign(model=kind, epochs=ep, seed=seed).to_csv(
                    os.path.join(out_dir, "loss_curves", f"{kind}_ep{ep}_seed{seed}.csv"), index=False)
                record(kind, ep, seed, syn, groups, tf, ts)
                syn.to_csv(os.path.join(smp_dir, f"{kind}_ep{ep}_seed{seed}.csv"), index=False)
                if ep == epochs_list[0]:
                    synp = project_identity(syn)
                    record(f"{kind}_proj", ep, seed, synp, groups, tf, ts)
                    if seed == GEN_SEEDS[0]:
                        samples_first[kind] = syn
                        samples_first[f"{kind}_proj"] = synp

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(os.path.join(out_dir, "generative_runs.csv"), index=False)
    pd.DataFrame(perks).to_csv(os.path.join(out_dir, "per_feature_ks_runs.csv"), index=False)
    pd.DataFrame(per_ws).to_csv(os.path.join(out_dir, "per_feature_wasserstein_runs.csv"), index=False)
    pd.DataFrame(per_js).to_csv(os.path.join(out_dir, "per_feature_jsd_runs.csv"), index=False)
    vdf = pd.DataFrame(viol)
    vdf.to_csv(os.path.join(out_dir, "cvr_by_column_runs.csv"), index=False)
    vdf.drop(columns="seed").groupby(["generator", "epochs"], sort=False).agg(["mean", "std"]).round(4).to_csv(
        os.path.join(out_dir, "cvr_by_column_summary.csv"))

    metrics = ["KS_mean", "KS_median", "KS_max", "W_std_mean", "JSD_mean", "sector_TVD", "corr_MAD",
               "corr_Frobenius", "spearman_MAD", "CVR", "CVR_identity", "CVR_nonneg", "CVR_bounds",
               "CVR_tol1pct", "ratio_inconsistency", "disc_AUC", "disc_AUC_ratios", "disc_AUC_nopm",
               "disc_AUC_logit", "DCR_median", "DCR_share_below_ref_p05", "corr_n_nan", "n_constant_cols",
               "n_nonfinite", "fit_time_s", "sample_time_s"] + \
              [c for c in runs[0] if c.startswith("pm_")]
    summ = runs_df.groupby(["generator", "epochs"], sort=False)[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ["n_seeds"] = runs_df.groupby(["generator", "epochs"], sort=False).size().values
    summ.to_csv(os.path.join(out_dir, "generative_summary.csv"), index=False)
    pk = pd.DataFrame(perks).groupby(["generator", "epochs"], sort=False)[NUM].agg(["mean", "std"])
    pk_t = pd.DataFrame({f"{g}|ep{e}": [fmt_pm(pk.loc[(g, e), (c, "mean")], pk.loc[(g, e), (c, "std")])
                                         for c in NUM] for g, e in pk.index}, index=NUM)
    pk_t.index.name = "feature"
    pk_t.to_csv(os.path.join(out_dir, "per_feature_ks_table.csv"))
    pd.DataFrame([m_self]).to_csv(os.path.join(out_dir, "reference_self_comparison.csv"), index=False)
    pd.DataFrame([dict(**ref_cvr, **point_mass_shares(ref_gen))]).to_csv(
        os.path.join(out_dir, "reference_cvr.csv"), index=False)

    # ---- Section 4: sensitivity
    log("Section 4: sensitivity")
    grid = []
    lams = LAMBDAS if not quick else LAMBDAS[:2]
    sigs = SIGMAS if not quick else SIGMAS[:2]
    for lam in lams:
        for sz in sigs:
            p = common.generate_panel(PANEL_SEED, lam=lam, sigma_zeta=sz)
            pg = p[NUM + [CAT]].reset_index(drop=True)
            for gname, fn in [("B1_noise", gen_b1_noise), ("B2_gauss", gen_b2_gauss)]:
                syn, g = fn(pg, N_GEN, PANEL_SEED, p["firm_id"].values)
                m, _, _, _ = evaluate(pg, syn, p["firm_id"].values, g, PANEL_SEED)
                grid.append(dict(lam=lam, sigma_zeta=sz, sigma_zeta_sq=sz ** 2, generator=gname,
                                 ref_dr_mean=p["debt_ratio"].mean(), ref_dr_sd=p["debt_ratio"].std(), **m))
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(out_dir, "sensitivity_dgp_grid.csv"), index=False)
    log(f"  DGP grid done ({len(grid)} rows)")

    ns = []
    for k in (NOISE_SCALES if not quick else [0.0, 1.0]):
        for seed in GEN_SEEDS:
            for gname, fn in [("B1_noise", gen_b1_noise), ("B1_noise_log", gen_b1_noise_log)]:
                syn, g = fn(ref_gen, N_GEN, seed, ref_groups, noise_scale=k)
                m, _, _, _ = evaluate(ref_gen, syn, ref_groups, g, seed, dcr=dcr)
                ns.append(dict(generator=gname, noise_scale=k, seed=seed, **m))
    ns = pd.DataFrame(ns)
    ns.to_csv(os.path.join(out_dir, "sensitivity_noise_scale.csv"), index=False)

    ps = []
    for psd in (PANEL_SEEDS if not quick else PANEL_SEEDS[:2]):
        p = common.generate_panel(psd)
        pg = p[NUM + [CAT]].reset_index(drop=True)
        for gname, fn in [("B1_noise", gen_b1_noise), ("B2_gauss", gen_b2_gauss)]:
            syn, g = fn(pg, N_GEN, 42, p["firm_id"].values)
            m, _, _, _ = evaluate(pg, syn, p["firm_id"].values, g, 42)
            ps.append(dict(panel_seed=psd, generator=gname, **m))
    ps = pd.DataFrame(ps)
    ps.to_csv(os.path.join(out_dir, "sensitivity_panel_seed.csv"), index=False)

    # ---- Section 5: figures
    log("Section 5: figures")
    four = {k: samples_first[k] for k in ["B1_noise", "B2_gauss", "CTGAN", "TVAE"]}
    fig_distributions(ref_gen, four, os.path.join(fig_dir, "fig_e4_distributions"))
    ev = fig_pca(ref_gen, four, os.path.join(fig_dir, "fig_e4_pca"))
    fig_losses(losses, os.path.join(fig_dir, "fig_e4_loss_curves"))
    fig_corr_diff(ref_gen, four, os.path.join(fig_dir, "fig_e4_corr_diff"))
    fig_sens_dgp(grid, os.path.join(fig_dir, "fig_e4_sensitivity_dgp"))
    fig_sens_noise(ns, os.path.join(fig_dir, "fig_e4_sensitivity_noise"))

    # loss diagnostics: last-50-epoch mean and change vs previous 50 epochs
    ld = []
    for (kind, ep, seed), lv in losses.items():
        cols = ["generator_loss", "discriminator_loss"] if kind == "CTGAN" else ["loss"]
        for c in cols:
            v = lv[c].values.astype(float)
            w = min(50, max(1, len(v) // 3))
            ld.append(dict(model=kind, epochs=ep, seed=seed, series=c, first_epoch=v[0], last_epoch=v[-1],
                           mean_last_window=v[-w:].mean(), sd_last_window=v[-w:].std(),
                           mean_prev_window=v[-2 * w:-w].mean() if len(v) >= 2 * w else np.nan, window=w))
    pd.DataFrame(ld).to_csv(os.path.join(out_dir, "loss_diagnostics.csv"), index=False)

    meta = dict(runtime_minutes=(time.perf_counter() - T0) / 60, quick=quick, panel_seed=PANEL_SEED,
                gen_seeds=GEN_SEEDS, n_generated=N_GEN, epochs=epochs_list, numeric_columns=NUM,
                ctgan_kwargs={k: list(v) if isinstance(v, tuple) else v for k, v in CTGAN_KW.items()},
                tvae_kwargs={k: list(v) if isinstance(v, tuple) else v for k, v in TVAE_KW.items()},
                tvae_learning_rate="1e-3 (torch Adam default; fixed inside ctgan.TVAE)",
                ctgan_version=__import__("ctgan").__version__, torch_version=torch.__version__,
                device="cpu (torch build without CUDA)",
                transformer_seconds=t_transformer, deep_training_wall_seconds_this_invocation=t_deep_wall,
                deep_jobs_trained_this_invocation=len(todo), deep_jobs_loaded_from_cache=len(jobs) - len(todo),
                deep_training_parallelism="4 worker processes x 1 torch thread", transformer_output_dim=int(cached.output_dimensions),
                pca_explained_variance=list(map(float, ev)), discriminator=DISC_KW,
                cvr_tol=CVR_TOL, ratio_tol=RATIO_TOL, jsd_bins=JSD_BINS, diagnosis=diag_info,
                reference_cvr=ref_cvr)
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"Done in {meta['runtime_minutes']:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
