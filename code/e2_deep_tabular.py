"""
e2_deep_tabular.py -- modern tabular deep-learning baselines for debt_ratio.

Models (all evaluated on identical outer splits):
  LightGBM (tuned)       small grid, early stopping on the inner validation firms
  LightGBM (paper cfg)   n_estimators=100, max_depth=5, learning_rate=0.1 (run_pipeline_enhanced.py), full outer train
  FT-Transformer         Gorishniy et al. (2021), own PyTorch implementation, small grid, early stopping
  TabNet                 Arik & Pfister (2021), pytorch-tabnet, small grid, early stopping
  MLP (tuned)            PyTorch MLP, random search over width/depth/dropout/lr, early stopping
  MLP (paper cfg)        sklearn MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000) as in run_pipeline_enhanced.py

Protocols on the reference synthetic panel (common.generate_panel(seed=42)):
  grouped5    common.grouped_kfold(df, 5, seed=42)   (firm-grouped, sector-balanced)
  temporal    common.temporal_split(df)              (train 2010-2020, test 2021-2024)
  unseen      common.unseen_firm_future_split(df)    (held-out firms in 2021-2024)

Inner validation (tuning and early stopping): one firm-grouped, sector-balanced 20% split of the outer
training rows, common.grouped_kfold(train_rows, 5, seed=42)[0].  Hyperparameters are chosen per outer
split, per feature set, per model, by inner-validation MSE with initialisation seed 0; the chosen
configuration is then refitted with seeds 1 and 2.  Outer test rows are never used for any choice.

Run:  python e2_deep_tabular.py            (full run)
      ... e2_deep_tabular.py --pilot                                         (fold 0, strict, timing check)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

OUT = os.path.join(common.OUT_MAIN, "e2_deep_tabular")
SEEDS = [0, 1, 2]
CAT_COLS = ["sector_code", "size_code"]
TORCH_THREADS = 2
LGB_THREADS = 2
BATCH = 256
MAX_EPOCHS = 300
PATIENCE = 20

FEATURE_SETS = {"strict": common.FEATURES_STRICT, "original": common.FEATURES_ORIGINAL}

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
GRID_LGB = [dict(num_leaves=nl, learning_rate=lr, min_child_samples=mcs)
            for nl in (15, 31) for lr in (0.03, 0.1) for mcs in (20, 50)]
# Reduced for runtime (CPU-only torch build, machine shared with other jobs):
#  - pilot FT grid had 4 configs (also d=128 lr=3e-4 and d=192 lr=1e-4); a first full run with 3 configs
#    (incl. d=192) needed ~1,100 s per FT job on the original features (projected > 60 min), so d=192 was dropped
#    and FT max epochs set to 200;
#  - pilot TabNet grid had 3 configs (also n_d=n_a=8, n_steps=3, lr=5e-3, worst inner-validation MSE in the pilot).
GRID_FTT = [dict(d_token=64, lr=3e-4), dict(d_token=128, lr=1e-4)]
FTT_MAX_EPOCHS = 200
GRID_TABNET = [dict(n_d=8, n_a=8, n_steps=3, gamma=1.3, lr=2e-2),
               dict(n_d=16, n_a=16, n_steps=4, gamma=1.5, lr=2e-2)]
_MLP_FULL = [dict(width=w, depth=d, dropout=p, lr=lr)
             for w in (64, 128, 256) for d in (1, 2, 3) for p in (0.0, 0.1, 0.3) for lr in (1e-3, 3e-4)]
_rs = np.random.RandomState(42)
GRID_MLP = [_MLP_FULL[i] for i in sorted(_rs.choice(len(_MLP_FULL), 10, replace=False))]

FTT_FIXED = dict(n_blocks=3, n_heads=8, attention_dropout=0.2, ffn_dropout=0.1, residual_dropout=0.0,
                 ffn_factor=4 / 3, weight_decay=1e-5, batch_size=BATCH, max_epochs=FTT_MAX_EPOCHS, patience=PATIENCE,
                 optimizer="AdamW", prenorm=True, activation="ReGLU", cls_token="appended last")
MLP_FIXED = dict(activation="ReLU", weight_decay=1e-4, batch_size=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 optimizer="AdamW")
TABNET_FIXED = dict(cat_emb_dim=2, mask_type="sparsemax", lambda_sparse=1e-3, batch_size=BATCH, virtual_batch_size=128,
                    max_epochs=200, patience=PATIENCE, optimizer="Adam", scheduler="StepLR(step_size=50, gamma=0.9)")
LGB_FIXED = dict(n_estimators_max=5000, early_stopping_rounds=100, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, n_jobs=LGB_THREADS)

MODELS = ["LightGBM", "LightGBM_paper", "FT-Transformer", "TabNet", "MLP", "MLP_paper"]
GRIDS = {"LightGBM": GRID_LGB, "FT-Transformer": GRID_FTT, "TabNet": GRID_TABNET, "MLP": GRID_MLP,
         "LightGBM_paper": [dict(n_estimators=100, max_depth=5, learning_rate=0.1)],
         "MLP_paper": [dict(hidden_layer_sizes=(64, 32), max_iter=1000)]}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(r2=1 - ss_res / ss_tot, rmse=float(np.sqrt(np.mean((y - p) ** 2))), mae=float(np.mean(np.abs(y - p))))


def inner_split(df, tr_idx):
    """Firm-grouped, sector-balanced 20% validation split inside the outer training rows."""
    sub = df.iloc[tr_idx].reset_index(drop=True)
    itr, iva = common.grouped_kfold(sub, n_splits=5, seed=42)[0]
    return tr_idx[itr], tr_idx[iva]


class Prepared:
    """Scalers are fitted on the inner-training rows only."""

    def __init__(self, df, feats, itr, iva, te, full_tr):
        self.feats = list(feats)
        self.cat = [c for c in CAT_COLS if c in feats]
        # sector dummies duplicate sector_code; token/embedding models use the categorical code instead
        self.num_tok = [f for f in feats if f not in self.cat and not f.startswith("sector_")]
        self.cards = [int(df[c].max()) + 1 for c in self.cat]
        y = df[common.TARGET].values.astype(np.float64)
        self.y = {"itr": y[itr], "iva": y[iva], "te": y[te], "full": y[full_tr]}
        self.y_mu, self.y_sd = self.y["itr"].mean(), self.y["itr"].std()
        Xn = df[self.num_tok].values.astype(np.float64)
        mu, sd = Xn[itr].mean(0), Xn[itr].std(0)
        sd[sd == 0] = 1.0
        Xc = df[self.cat].values.astype(np.int64)
        self.tok = {k: ((Xn[ix] - mu) / sd, Xc[ix]) for k, ix in (("itr", itr), ("iva", iva), ("te", te))}
        Xa = df[self.feats].values.astype(np.float64)
        self.raw = {k: Xa[ix] for k, ix in (("itr", itr), ("iva", iva), ("te", te), ("full", full_tr))}
        mu2, sd2 = Xa[itr].mean(0), Xa[itr].std(0)
        sd2[sd2 == 0] = 1.0
        self.std = {k: (Xa[ix] - mu2) / sd2 for k, ix in (("itr", itr), ("iva", iva), ("te", te))}
        mu3, sd3 = Xa[full_tr].mean(0), Xa[full_tr].std(0)
        sd3[sd3 == 0] = 1.0
        self.std_full = {"full": (Xa[full_tr] - mu3) / sd3, "te": (Xa[te] - mu3) / sd3}

    def ys(self, k):
        return (self.y[k] - self.y_mu) / self.y_sd

    def unscale(self, p):
        return np.asarray(p).ravel() * self.y_sd + self.y_mu


# ---------------------------------------------------------------------------
# PyTorch models
# ---------------------------------------------------------------------------
def _torch():
    import torch
    torch.set_num_threads(TORCH_THREADS)
    return torch


def build_ftt(n_num, cards, d_token, n_blocks, n_heads, attention_dropout, ffn_dropout, residual_dropout,
              ffn_factor, **_):
    torch = _torch()
    nn = torch.nn

    class ReGLU(nn.Module):
        def forward(self, x):
            a, b = x.chunk(2, dim=-1)
            return a * torch.relu(b)

    class Block(nn.Module):
        def __init__(self, first):
            super().__init__()
            self.ln1 = nn.Identity() if first else nn.LayerNorm(d_token)  # first block: no attention prenorm (paper)
            self.att = nn.MultiheadAttention(d_token, n_heads, dropout=attention_dropout, batch_first=True)
            self.ln2 = nn.LayerNorm(d_token)
            h = int(d_token * ffn_factor)
            self.ffn = nn.Sequential(nn.Linear(d_token, 2 * h), ReGLU(), nn.Dropout(ffn_dropout), nn.Linear(h, d_token))
            self.drop = nn.Dropout(residual_dropout)

        def forward(self, x):
            z = self.ln1(x)
            z, _ = self.att(z, z, z, need_weights=False)
            x = x + self.drop(z)
            x = x + self.drop(self.ffn(self.ln2(x)))
            return x

    class FTT(nn.Module):
        def __init__(self):
            super().__init__()
            s = 1 / np.sqrt(d_token)
            self.num_w = nn.Parameter(torch.empty(n_num, d_token).uniform_(-s, s))
            self.num_b = nn.Parameter(torch.empty(n_num, d_token).uniform_(-s, s))
            self.n_cat = len(cards)
            if self.n_cat:
                self.register_buffer("offs", torch.tensor(np.concatenate([[0], np.cumsum(cards)[:-1]]), dtype=torch.long))
                self.cat_emb = nn.Embedding(int(sum(cards)), d_token)
                nn.init.uniform_(self.cat_emb.weight, -s, s)
                self.cat_b = nn.Parameter(torch.empty(self.n_cat, d_token).uniform_(-s, s))
            self.cls = nn.Parameter(torch.empty(1, 1, d_token).uniform_(-s, s))
            self.blocks = nn.ModuleList([Block(i == 0) for i in range(n_blocks)])
            self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

        def forward(self, xn, xc):
            t = xn.unsqueeze(-1) * self.num_w + self.num_b
            if self.n_cat:
                t = torch.cat([t, self.cat_emb(xc + self.offs) + self.cat_b], dim=1)
            t = torch.cat([t, self.cls.expand(t.shape[0], 1, -1)], dim=1)
            for b in self.blocks:
                t = b(t)
            return self.head(t[:, -1]).squeeze(-1)

    return FTT()


def build_mlp(n_in, width, depth, dropout, **_):
    torch = _torch()
    nn = torch.nn
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
        d = width
    layers += [nn.Linear(d, 1)]
    net = nn.Sequential(*layers)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, xn, xc=None):
            return self.net(xn).squeeze(-1)

    return Wrap()


def train_torch(model, Xtr, ytr, Xva, yva, lr, weight_decay, seed, batch_size=BATCH, max_epochs=MAX_EPOCHS,
                patience=PATIENCE):
    """Xtr / Xva are tuples (x_num float array, x_cat int array or None). MSE on standardised y.
    Returns best-epoch model (state restored), best_epoch (1-based), epochs_run."""
    torch = _torch()
    import copy
    tn = [torch.tensor(Xtr[0], dtype=torch.float32), torch.tensor(Xtr[1], dtype=torch.long) if Xtr[1] is not None else None]
    vn = [torch.tensor(Xva[0], dtype=torch.float32), torch.tensor(Xva[1], dtype=torch.long) if Xva[1] is not None else None]
    yt = torch.tensor(ytr, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    n = len(yt)
    best, best_ep, best_state, ep = np.inf, 0, None, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            if len(b) < 2:
                continue
            opt.zero_grad()
            out = model(tn[0][b], tn[1][b] if tn[1] is not None else None)
            loss = torch.mean((out - yt[b]) ** 2)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = torch.mean((model(vn[0], vn[1]) - yv) ** 2).item()
        if vl < best - 1e-7:
            best, best_ep, best_state = vl, ep, copy.deepcopy(model.state_dict())
        elif ep - best_ep >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_ep, ep


def predict_torch(model, X):
    torch = _torch()
    model.eval()
    with torch.no_grad():
        xn = torch.tensor(X[0], dtype=torch.float32)
        xc = torch.tensor(X[1], dtype=torch.long) if X[1] is not None else None
        return model(xn, xc).numpy()


def seed_all(seed):
    import random
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# One fit = (model, config, seed) on one prepared split.  Returns dict with test predictions.
# ---------------------------------------------------------------------------
def fit_one(model_name, cfg, seed, P: Prepared):
    t0 = time.perf_counter()
    info = dict(best_epoch=np.nan, epochs_run=np.nan, n_params=np.nan, val_mse=np.nan)
    if model_name == "LightGBM":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=LGB_FIXED["n_estimators_max"], learning_rate=cfg["learning_rate"],
                              num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child_samples"],
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, n_jobs=LGB_THREADS, verbose=-1)
        m.fit(P.raw["itr"], P.y["itr"], eval_set=[(P.raw["iva"], P.y["iva"])], eval_metric="l2",
              callbacks=[lgb.early_stopping(LGB_FIXED["early_stopping_rounds"], verbose=False)])
        pv, pt = m.predict(P.raw["iva"]), m.predict(P.raw["te"])
        info.update(best_epoch=m.best_iteration_, epochs_run=m.booster_.current_iteration())
    elif model_name == "LightGBM_paper":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=seed,
                              n_jobs=LGB_THREADS, verbose=-1)
        m.fit(P.raw["full"], P.y["full"])
        pv, pt = None, m.predict(P.raw["te"])
        info.update(epochs_run=100)
    elif model_name == "MLP_paper":
        from sklearn.neural_network import MLPRegressor
        m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(P.std_full["full"], P.y["full"])
        pv, pt = None, m.predict(P.std_full["te"])
        info.update(epochs_run=m.n_iter_)
    elif model_name == "FT-Transformer":
        seed_all(seed)
        net = build_ftt(len(P.num_tok), P.cards, **cfg, **FTT_FIXED)
        Xtr, Xva, Xte = (P.tok[k] if P.cat else (P.tok[k][0], None) for k in ("itr", "iva", "te"))
        net, be, er = train_torch(net, Xtr, P.ys("itr"), Xva, P.ys("iva"), cfg["lr"], FTT_FIXED["weight_decay"], seed,
                                  max_epochs=FTT_MAX_EPOCHS)
        pv, pt = P.unscale(predict_torch(net, Xva)), P.unscale(predict_torch(net, Xte))
        info.update(best_epoch=be, epochs_run=er, n_params=sum(p.numel() for p in net.parameters()))
    elif model_name == "MLP":
        seed_all(seed)
        net = build_mlp(P.std["itr"].shape[1], **cfg)
        net, be, er = train_torch(net, (P.std["itr"], None), P.ys("itr"), (P.std["iva"], None), P.ys("iva"),
                                  cfg["lr"], MLP_FIXED["weight_decay"], seed)
        pv, pt = P.unscale(predict_torch(net, (P.std["iva"], None))), P.unscale(predict_torch(net, (P.std["te"], None)))
        info.update(best_epoch=be, epochs_run=er, n_params=sum(p.numel() for p in net.parameters()))
    elif model_name == "TabNet":
        _torch()
        import torch
        from pytorch_tabnet.tab_model import TabNetRegressor
        seed_all(seed)
        nc = len(P.num_tok)
        Xs = {k: np.hstack([P.tok[k][0], P.tok[k][1].astype(np.float64)]) for k in ("itr", "iva", "te")}
        m = TabNetRegressor(n_d=cfg["n_d"], n_a=cfg["n_a"], n_steps=cfg["n_steps"], gamma=cfg["gamma"],
                            cat_idxs=list(range(nc, nc + len(P.cat))), cat_dims=P.cards,
                            cat_emb_dim=TABNET_FIXED["cat_emb_dim"], mask_type=TABNET_FIXED["mask_type"],
                            lambda_sparse=TABNET_FIXED["lambda_sparse"],
                            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=cfg["lr"]),
                            scheduler_fn=torch.optim.lr_scheduler.StepLR,
                            scheduler_params=dict(step_size=50, gamma=0.9),
                            seed=seed, device_name="cpu", verbose=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xs["itr"], P.ys("itr").reshape(-1, 1), eval_set=[(Xs["iva"], P.ys("iva").reshape(-1, 1))],
                  eval_name=["val"], eval_metric=["mse"], max_epochs=TABNET_FIXED["max_epochs"],
                  patience=PATIENCE, batch_size=BATCH, virtual_batch_size=128, drop_last=False)
        pv, pt = P.unscale(m.predict(Xs["iva"])), P.unscale(m.predict(Xs["te"]))
        info.update(best_epoch=int(m.best_epoch) + 1, epochs_run=len(m.history["loss"]),
                    n_params=sum(p.numel() for p in m.network.parameters()))
    else:
        raise ValueError(model_name)
    info["train_time_s"] = time.perf_counter() - t0
    if pv is not None:
        info["val_mse"] = float(np.mean((pv - P.y["iva"]) ** 2))
    return info, pt


# ---------------------------------------------------------------------------
# Worker: tune + refit one model on one outer split / feature set
# ---------------------------------------------------------------------------
_DF = None


def _init_worker():
    global _DF
    warnings.filterwarnings("ignore")
    _DF = common.generate_panel(seed=42)
    _torch()


def get_splits(df, pilot=False):
    out = [("grouped5", k, tr, te) for k, (tr, te) in enumerate(common.grouped_kfold(df, n_splits=5, seed=42))]
    if pilot:
        return out[:1]
    tr, te = common.temporal_split(df)
    out.append(("temporal", 0, tr, te))
    tr, te = common.unseen_firm_future_split(df)
    out.append(("unseen", 0, tr, te))
    return out


def _cache_path(job):
    import hashlib
    protocol, split_id, fset, model_name, pilot = job
    key = json.dumps([GRIDS[model_name], FTT_FIXED if model_name == "FT-Transformer" else None, SEEDS, PATIENCE,
                      MAX_EPOCHS, BATCH], sort_keys=True, default=str)
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    d = os.path.join(OUT + ("_pilot" if pilot else ""), "cache")
    return os.path.join(d, f"{protocol}_{split_id}_{fset}_{model_name}_{h}.pkl")


def run_job(job):
    import pickle
    cp = _cache_path(job)
    if os.path.exists(cp):
        with open(cp, "rb") as fh:
            return pickle.load(fh)
    res = _run_job(job)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    with open(cp + ".tmp", "wb") as fh:
        pickle.dump(res, fh)
    os.replace(cp + ".tmp", cp)
    return res


def _run_job(job):
    protocol, split_id, fset, model_name, pilot = job
    df = _DF
    sp = {(p, k): (tr, te) for p, k, tr, te in get_splits(df, pilot)}
    tr, te = sp[(protocol, split_id)]
    itr, iva = inner_split(df, tr)
    P = Prepared(df, FEATURE_SETS[fset], itr, iva, te, tr)
    base = dict(protocol=protocol, split=split_id, feature_set=fset, model=model_name,
                n_train_outer=len(tr), n_train_inner=len(itr), n_val=len(iva), n_test=len(te))
    tune_rows, run_rows, preds = [], [], {}
    grid = GRIDS[model_name]
    seed0 = {}
    for ci, cfg in enumerate(grid):
        info, pt = fit_one(model_name, cfg, SEEDS[0], P)
        seed0[ci] = (info, pt)
        tune_rows.append({**base, "config_id": ci, "config": json.dumps(cfg), **info})
    best_ci = min(seed0, key=lambda c: seed0[c][0]["val_mse"]) if len(grid) > 1 else 0
    for s in SEEDS:
        if s == SEEDS[0]:
            info, pt = seed0[best_ci]
        else:
            info, pt = fit_one(model_name, grid[best_ci], s, P)
        run_rows.append({**base, "seed": s, "config_id": best_ci, "config": json.dumps(grid[best_ci]),
                         **metrics(P.y["te"], pt), **info})
        preds[s] = pt
    pred_df = pd.DataFrame({"row": te, "y_true": P.y["te"], **{f"pred_seed{s}": preds[s] for s in SEEDS}})
    for k, v in base.items():
        pred_df[k] = v
    return tune_rows, run_rows, pred_df


# ---------------------------------------------------------------------------
# Aggregation, tests, figure
# ---------------------------------------------------------------------------
LABEL = {"LightGBM": "LightGBM (tuned)", "LightGBM_paper": "LightGBM (paper config)",
         "FT-Transformer": "FT-Transformer", "TabNet": "TabNet", "MLP": "MLP (tuned)",
         "MLP_paper": "MLP (paper config)"}
PROTO_LABEL = {"grouped5": "Grouped 5-fold\n(unseen firms)", "temporal": "Temporal\n(2021-2024)",
               "unseen": "Unseen firms,\nfuture years"}


def summarise(runs):
    rows = []
    for (p, fs, m), g in runs.groupby(["protocol", "feature_set", "model"], sort=False):
        r = dict(protocol=p, feature_set=fs, model=m, n_runs=len(g), n_splits=g["split"].nunique())
        for met in ("r2", "rmse", "mae"):
            r[f"{met}_mean"] = g[met].mean()
            r[f"{met}_sd"] = g[met].std(ddof=1)
            # sd across splits of the seed-averaged metric, and sd across seeds of the split-averaged metric
            r[f"{met}_sd_folds"] = g.groupby("split")[met].mean().std(ddof=1) if g["split"].nunique() > 1 else np.nan
            r[f"{met}_sd_seeds"] = g.groupby("seed")[met].mean().std(ddof=1)
        for c in ("best_epoch", "epochs_run", "train_time_s", "n_params"):
            r[f"{c}_mean"] = g[c].mean()
        r["train_time_s_total"] = g["train_time_s"].sum()
        r["configs_chosen"] = "; ".join(sorted(g["config"].unique()))
        rows.append(r)
    return pd.DataFrame(rows)


def pairwise_tests(runs, df):
    """LightGBM (tuned) minus each other model, grouped 5-fold, fold-level seed-averaged differences,
    Nadeau-Bengio corrected t-test (J = 5, n_test / n_train from the folds)."""
    out = []
    g5 = runs[runs.protocol == "grouped5"]
    n_tr = g5["n_train_outer"].mean()
    n_te = g5["n_test"].mean()
    for fs in g5.feature_set.unique():
        sub = g5[g5.feature_set == fs]
        ref = sub[sub.model == "LightGBM"].groupby("split")[["r2", "rmse"]].mean()
        for m in sub.model.unique():
            if m == "LightGBM":
                continue
            oth = sub[sub.model == m].groupby("split")[["r2", "rmse"]].mean()
            d = (ref["r2"] - oth["r2"]).values
            t, pv = common.corrected_resampled_ttest(d, int(n_tr), int(n_te))
            out.append(dict(feature_set=fs, comparison=f"LightGBM (tuned) - {LABEL[m]}", metric="r2",
                            mean_diff=d.mean(), sd_diff=d.std(ddof=1), folds_lgbm_better=int((d > 0).sum()),
                            n_folds=len(d), t_corrected=t, p_two_sided=pv))
    return pd.DataFrame(out)


def make_figure(summary, models=("LightGBM", "FT-Transformer", "TabNet", "MLP"), name="fig_e2_r2_comparison",
                ncol=4):
    plt = common.set_style()
    models = list(models)
    colors = {"LightGBM": common.OKABE_ITO[4], "FT-Transformer": common.OKABE_ITO[5], "TabNet": common.OKABE_ITO[2],
              "MLP": common.OKABE_ITO[0], "LightGBM_paper": common.OKABE_ITO[4], "MLP_paper": common.OKABE_ITO[0]}
    markers = {"LightGBM": "o", "FT-Transformer": "s", "TabNet": "^", "MLP": "D", "LightGBM_paper": "o", "MLP_paper": "D"}
    protos = ["grouped5", "temporal", "unseen"]
    fsets = [("strict", "Strict (leakage-free) features"), ("original", "Original 21 features")]
    fig, axes = plt.subplots(1, 2, figsize=(common.FIG_W_DOUBLE, 2.9), sharey=False)
    off = np.linspace(-0.27, 0.27, len(models))
    for ax, (fs, lab), tag in zip(axes, fsets, ("(a)", "(b)")):
        s = summary[summary.feature_set == fs]
        for j, m in enumerate(models):
            xs, ys, es = [], [], []
            for i, p in enumerate(protos):
                r = s[(s.protocol == p) & (s.model == m)]
                if len(r):
                    xs.append(i + off[j]); ys.append(r["r2_mean"].iloc[0]); es.append(r["r2_sd"].iloc[0])
            hollow = m.endswith("_paper")
            ax.errorbar(xs, ys, yerr=es, fmt=markers[m], ms=4.5, capsize=2, lw=0.8, color=colors[m],
                        mfc="white" if hollow else colors[m], mec=colors[m], label=LABEL[m])
        ax.set_xticks(range(len(protos)))
        ax.set_xticklabels([PROTO_LABEL[p] for p in protos])
        ax.set_ylabel(r"Test $R^2$ (debt ratio)")
        ax.axhline(0, color="0.6", lw=0.6, ls=":")
        ax.set_title(f"{tag} {lab}", loc="left", fontsize=8.5)
        ax.set_xlim(-0.6, len(protos) - 0.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.10 if ncol >= 4 else 0.14, 1, 1))
    common.savefig(fig, os.path.join(OUT, name))
    plt.close(fig)


def make_figures(summary):
    make_figure(summary)
    make_figure(summary, models=MODELS_FIG_ALL, name="fig_e2_r2_comparison_with_paper_configs", ncol=3)


MODELS_FIG_ALL = ("LightGBM", "FT-Transformer", "TabNet", "MLP", "LightGBM_paper", "MLP_paper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="fold 0 of grouped5, strict features only")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from summary.csv")
    args = ap.parse_args()
    if args.figures_only:
        make_figures(pd.read_csv(os.path.join(OUT, "summary.csv")))
        return
    out = OUT + ("_pilot" if args.pilot else "")
    os.makedirs(out, exist_ok=True)
    t_start = time.perf_counter()
    df = common.generate_panel(seed=42)
    splits = get_splits(df, args.pilot)
    fsets = ["strict"] if args.pilot else list(FEATURE_SETS)
    # slow jobs first for better load balancing
    order = {"FT-Transformer": 0, "TabNet": 1, "MLP": 2, "LightGBM": 3, "MLP_paper": 4, "LightGBM_paper": 5}
    jobs = [(p, k, fs, m, args.pilot) for (p, k, _, _) in splits for fs in fsets for m in MODELS]
    jobs.sort(key=lambda j: (order[j[3]], j[2] != "original"))
    tune, runs, preds = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            tr_, rr_, pr_ = f.result()
            tune += tr_; runs += rr_; preds.append(pr_)
            r2s = [r["r2"] for r in rr_]
            print(f"[{i}/{len(jobs)}] {j[0]} split{j[1]} {j[2]} {j[3]}: R2 {np.mean(r2s):.4f} "
                  f"({time.perf_counter() - t_start:.0f}s)", flush=True)
    tune = pd.DataFrame(tune).sort_values(["feature_set", "protocol", "split", "model", "config_id"])
    runs = pd.DataFrame(runs).sort_values(["feature_set", "protocol", "split", "model", "seed"])
    preds = pd.concat(preds, ignore_index=True)
    tune.to_csv(os.path.join(out, "tuning_log.csv"), index=False)
    runs.to_csv(os.path.join(out, "runs.csv"), index=False)
    preds.to_csv(os.path.join(out, "test_predictions.csv.gz"), index=False, compression="gzip")
    summary = summarise(runs)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    if not args.pilot:
        pairwise_tests(runs, df).to_csv(os.path.join(out, "pairwise_tests_grouped5.csv"), index=False)
        make_figures(summary)
    import torch, lightgbm, sklearn
    from importlib.metadata import version
    meta = dict(
        python=sys.version, torch=torch.__version__, cuda_available=torch.cuda.is_available(),
        device="cpu", torch_threads_per_worker=TORCH_THREADS, lightgbm=lightgbm.__version__,
        sklearn=sklearn.__version__, pytorch_tabnet=version("pytorch-tabnet"), workers=args.workers,
        seeds=SEEDS, protocols=sorted({s[0] for s in splits}), feature_sets={k: FEATURE_SETS[k] for k in fsets},
        grids={k: GRIDS[k] for k in GRIDS}, fixed=dict(ftt=FTT_FIXED, mlp=MLP_FIXED, tabnet=TABNET_FIXED, lgb=LGB_FIXED),
        mlp_full_grid_size=len(_MLP_FULL), runtime_minutes=(time.perf_counter() - t_start) / 60)
    with open(os.path.join(out, "run_config.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    pd.set_option("display.width", 200)
    print(summary[["protocol", "feature_set", "model", "r2_mean", "r2_sd", "rmse_mean", "mae_mean",
                   "best_epoch_mean", "train_time_s_mean"]].to_string(index=False))
    print(f"total {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
