"""Probability models (Owen, Nov 12): P(at least one bike of each type at t + h), trained with binary cross-entropy.

Same data as the count search (data-v1 with the weather fix, score.TEAM, SHA-1 263bb791...), same split (train to
Oct 18, val Oct 19-23; the test rows are never read here: score.load is called with ["train", "val"]), same 20
inputs and station embedding (33_train_mlp.team_features), same horizons (one model for h = 2-15 min, h is an input).
Only the output and the loss change:
  output  2 logits; p = sigmoid(logit) for classic and e-bike. Label y_{x} = 1[{x}_future >= 1] (table column).
  loss    binary cross-entropy (BCEWithLogitsLoss), mean over rows and both targets.
  training  Adam (lr 1e-3, batch 4096), one pass over the train rows per epoch, up to 30 epochs, keep the epoch with
          the lowest val cross-entropy, stop after 3 epochs without improvement (the "full" budget of the count search).

Models (a small search, not the full grid; names and configs from 37_model_search.GRIDS):
  mlp_2x64_e8         the Nov 10 base MLP (2 x 64)
  mlp_2x256_do0.1     the count search winner (2 x 256, dropout 0.1)
  mlp_2x256_do0.3     2 x 256, dropout 0.3 (best count val skill, within noise of 0.1)
  gru64_3x256_do0.1   the GRU over the 3 lag counts with the 3 x 256 head and dropout 0.1 (the GRU of Step 5)
  lgbm                LightGBM binary classifier per target (references; inputs + station_idx categorical,
                      early stopping on val log loss, patience 50)
  logreg              logistic regression per target on the 20 standardised inputs (sklearn, L2, C = 1; no station)
Neural nets: seeds 0-2. LightGBM: seeds 0-2 (row and feature bagging). Logistic regression is deterministic: 1 run.

Selection rule (fixed Nov 12, before any run): among the four neural nets, the model with the lowest val Brier score
averaged over the two targets (all rows), mean over seeds 0-2. If a model with fewer parameters is within noise of it
(gap < 2 x sqrt(sd_a^2 / 3 + sd_b^2 / 3), seed sd), the smallest such model is chosen. LightGBM and logistic
regression are references and are not chosen. Only the chosen model is scored on test (48_prob_test.py).

Writes
  reports/tables/prob_runs.csv     one row per run: config, epochs, time, val Brier / BSS vs each reference / log loss
  reports/tables/prob_curves.csv   train and val cross-entropy per epoch
  DATA_DIR/models/prob/<run_id>.npy   val probabilities [n_val, 2] (float32); .pt weights, input stats, station index
                                      (LightGBM: <run_id>_<target>.txt boosters; logreg: .pkl). Not in git.
Checks added after the six reviews (Nov 12; val only, separate logs, the search logs and the choice unchanged):
  --patience N        a net with another early-stopping patience -> prob_extra_runs.csv, prob_extra_curves.csv
  --lgbm-ablation S   LightGBM on the input sets in ABLATION -> prob_ablation.csv
Usage (from src/):
  python 46_prob_search.py --run mlp_2x256_do0.1 --seed 0 1 2
  python 46_prob_search.py --lgbm --seed 0 1 2
  python 46_prob_search.py --logreg
"""
import argparse
import importlib
import json
import pickle
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import DATA_DIR
from score import TARGETS, TEAM, load
from score_prob import REF_COLUMNS, fit_references, labels, prob_table

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
ms = importlib.import_module("37_model_search")
mlp33 = ms.mlp33
TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
RUNS = TABLES / "prob_runs.csv"
CURVES = TABLES / "prob_curves.csv"
SAVED = DATA_DIR / "models" / "prob"
NETS = ["mlp_2x64_e8", "mlp_2x256_do0.1", "mlp_2x256_do0.3", "gru64_3x256_do0.1"]
MAX_EPOCHS, PATIENCE = ms.BUDGETS["full"]
EXTRA_RUNS = TABLES / "prob_extra_runs.csv"      # checks after the reviews (Nov 12): --patience
EXTRA_CURVES = TABLES / "prob_extra_curves.csv"
ABLATION_LOG = TABLES / "prob_ablation.csv"      # --lgbm-ablation
_IN = mlp33.TEAM_INPUTS
_LAGS = [c for c in _IN if c.endswith(("_tm10", "_tm30"))]
_COUNTS_NOW = ["classic_now", "ebike_now", "docks_now"]
ABLATION = {  # LightGBM input sets (after the reviews): what does the gain over calibrated persistence need?
    "all": [*_IN, "station_idx"],
    "no_station_idx": list(_IN),
    "no_station_info": [c for c in _IN if c not in ("lat", "lon", "capacity")],
    "no_lags": [c for c in _IN if c not in _LAGS] + ["station_idx"],
    "counts_now_h": ["h", *_COUNTS_NOW],
    "counts_lags_h": ["h", *_COUNTS_NOW, *_LAGS],
}
COLUMNS = list(dict.fromkeys([*mlp33.TEAM_COLUMNS, *REF_COLUMNS, "station_idx"]))


def append(path, rows):
    """Append rows under a lock; a new column (the logs mix nets, LightGBM and logreg) widens the file."""
    import fcntl
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        df = pd.DataFrame(rows)
        if path.exists():
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df.to_csv(path, index=False)


def data():
    """Train and val tensors (inputs, station index, labels), the val frame, and the references fit on train."""
    df = load(["train", "val"], columns=COLUMNS, ds=TEAM)
    tr, va = df[df.split == "train"].reset_index(drop=True), df[df.split == "val"].reset_index(drop=True)
    del df
    refs = fit_references(tr)
    _, stats = mlp33.features(tr, ds=TEAM)
    station_index = {sid: i for i, sid in enumerate(sorted(tr.station_id.unique()))}
    xt, st, _, _ = mlp33.tensors(tr, stats, station_index, TEAM)
    xv, sv, _, _ = mlp33.tensors(va, stats, station_index, TEAM)
    yt, yv = torch.from_numpy(labels(tr).astype(np.float32)), torch.from_numpy(labels(va).astype(np.float32))
    return dict(xt=xt, st=st, yt=yt, xv=xv, sv=sv, yv=yv, va=va, tr=tr, refs=refs, stats=stats,
                station_index=station_index)


@torch.no_grad()
def predict(model, x, s, batch=1 << 16):
    """Probabilities [n, 2]."""
    model.eval()
    return torch.cat([torch.sigmoid(model(x[i:i + batch], s[i:i + batch])) for i in range(0, len(x), batch)])


def bce(p, y, eps=1e-7):
    p = p.clamp(eps, 1 - eps)
    return -(y * p.log() + (1 - y) * (1 - p).log()).mean().item()


def summary(va, p, refs):
    """val_{brier,log_loss,rel,bss_pers,bss_calpers,bss_clim}_{target}_{subset} of the model's rows."""
    t = prob_table(va, {"m": p}, refs)
    t = t[t.model == "m"]
    out = {}
    for r in t.itertuples():
        for k in ["brier", "log_loss", "rel", "bss_pers", "bss_calpers", "bss_clim"]:
            out[f"{k}_{r.target}_{r.subset}"] = getattr(r, k)
    return out


def train(name, seed, d, patience=None):
    """patience=None: the logged search (PATIENCE, logs RUNS / CURVES). An int: a check run after the reviews,
    logged to EXTRA_RUNS / EXTRA_CURVES as p_<name>_pat<N>_seed<s>; the search logs are not touched."""
    cfg = ms.config_by_name(name)
    xt, st, yt, xv, sv, yv = (d[k] for k in ["xt", "st", "yt", "xv", "sv", "yv"])
    n_st = len(d["station_index"])
    run_id = f"p_{name}_seed{seed}" if patience is None else f"p_{name}_pat{patience}_seed{seed}"
    pat = PATIENCE if patience is None else patience
    ms.warm_up(cfg, xt.shape[1], n_st, "cpu")
    torch.manual_seed(seed)
    model = ms.Net(cfg, xt.shape[1], n_st)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_epoch, best_state, curve = np.inf, 0, None, []
    t_start = time.time()
    print(f"{run_id}: {n_params:,} params, {torch.get_num_threads()} threads", flush=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for idx in torch.randperm(len(xt)).split(cfg.batch):
            loss = loss_fn(model(xt[idx], st[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sec = time.time() - t0
        val = bce(predict(model, xv, sv), yv)
        curve.append({"run_id": run_id, "epoch": epoch, "train_bce": total / len(xt), "val_bce": val, "epoch_s": sec})
        print(f"  epoch {epoch:2d}  train {total / len(xt):.5f}  val {val:.5f}  ({sec:.0f} s)", flush=True)
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= pat:
            break
    wall = time.time() - t_start
    model.load_state_dict(best_state)
    p = predict(model, xv, sv).numpy()
    np.save(SAVED / f"{run_id}.npy", p.astype(np.float32))
    torch.save({"state": best_state, "cfg": asdict(cfg), "name": name, "best_epoch": best_epoch,
                "stats": d["stats"], "station_index": d["station_index"]}, SAVED / f"{run_id}.pt")
    row = {"run_id": run_id, "name": name, "arch": cfg.arch, "depth": cfg.depth, "width": cfg.width, "emb": cfg.emb,
           "gru": cfg.gru, "dropout": cfg.dropout, "lr": cfg.lr, "batch": cfg.batch, "seed": seed,
           "n_params": n_params, "patience": pat, "best_epoch": best_epoch, "epochs_run": len(curve), "val_bce_best": best,
           "threads": torch.get_num_threads(), "wall_s": round(wall, 1),
           "s_per_epoch": round(np.mean([c["epoch_s"] for c in curve]), 1), **summary(d["va"], p, d["refs"]),
           "finished": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    append(RUNS if patience is None else EXTRA_RUNS, [row])
    append(CURVES if patience is None else EXTRA_CURVES, curve)
    print(f"  -> Brier classic {row['brier_classic_all']:.5f} ebike {row['brier_ebike_all']:.5f}; BSS vs calpers "
          f"{row['bss_calpers_classic_all']:.4f} / {row['bss_calpers_ebike_all']:.4f} (epoch {best_epoch}, "
          f"{wall:.0f} s)", flush=True)


def lgbm(seed, d, rounds=3000, lr=0.1, leaves=127, ablation=None):
    """LightGBM binary classifier per target, the count search's LightGBM settings, early stop on val log loss.
    ablation = a key of ABLATION: the same model on that input set only, logged to ABLATION_LOG (after the reviews;
    the search log is not touched)."""
    import lightgbm as lgb
    tr, va = d["tr"], d["va"]
    feats = [*mlp33.TEAM_INPUTS, "station_idx"] if ablation is None else ABLATION[ablation]
    run_id = f"p_lgbm_seed{seed}" if ablation is None else f"p_lgbm_{ablation}_seed{seed}"
    t_start = time.time()
    p, iters = np.zeros((len(va), len(TARGETS))), []
    for i, x in enumerate(TARGETS):
        cat = ["station_idx"] if "station_idx" in feats else []
        dtr = lgb.Dataset(tr[feats], tr[f"y_{x}"], categorical_feature=cat, free_raw_data=False)
        dva = lgb.Dataset(va[feats], va[f"y_{x}"], reference=dtr)
        params = {"objective": "binary", "learning_rate": lr, "num_leaves": leaves, "min_data_in_leaf": 200,
                  "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed,
                  "num_threads": torch.get_num_threads(), "verbose": -1, "max_cat_to_onehot": 4,
                  "cat_smooth": 10, "max_bin": 255}
        booster = lgb.train(params, dtr, rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
        iters.append(booster.best_iteration)
        p[:, i] = booster.predict(va[feats], num_iteration=booster.best_iteration)
        if ablation is None:
            booster.save_model(str(SAVED / f"{run_id}_{x}.txt"), num_iteration=booster.best_iteration)
    wall = time.time() - t_start
    np.save(SAVED / f"{run_id}.npy", p.astype(np.float32))
    row = {"run_id": run_id, "name": "lgbm" if ablation is None else f"lgbm_{ablation}", "arch": "lgbm",
           "inputs": " ".join(feats), "lr": lr, "seed": seed, "best_epoch": json.dumps(iters),
           "threads": torch.get_num_threads(), "wall_s": round(wall, 1), **summary(va, p, d["refs"]),
           "finished": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    if ablation is None:
        row.pop("inputs")
    append(RUNS if ablation is None else ABLATION_LOG, [row])
    print(f"{run_id}: Brier {row['brier_classic_all']:.5f} / {row['brier_ebike_all']:.5f} (iters {iters}, "
          f"{wall:.0f} s)", flush=True)


def logreg(d):
    """Logistic regression per target on the 20 standardised inputs (no station), L2 with C = 1, lbfgs."""
    from sklearn.linear_model import LogisticRegression
    xt, yt, xv = d["xt"].numpy(), d["yt"].numpy(), d["xv"].numpy()
    run_id = "p_logreg_seed0"
    t_start = time.time()
    p, models = np.zeros((len(xv), len(TARGETS))), []
    for i in range(len(TARGETS)):
        m = LogisticRegression(C=1.0, max_iter=500).fit(xt, yt[:, i])
        p[:, i] = m.predict_proba(xv)[:, 1]
        models.append(m)
    wall = time.time() - t_start
    np.save(SAVED / f"{run_id}.npy", p.astype(np.float32))
    with open(SAVED / f"{run_id}.pkl", "wb") as f:
        pickle.dump({"models": models, "stats": d["stats"]}, f)
    row = {"run_id": run_id, "name": "logreg", "arch": "logreg", "seed": 0, "n_params": 2 * (xt.shape[1] + 1),
           "best_epoch": json.dumps([int(m.n_iter_[0]) for m in models]), "wall_s": round(wall, 1),
           **summary(d["va"], p, d["refs"]), "finished": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    append(RUNS, [row])
    print(f"{run_id}: Brier {row['brier_classic_all']:.5f} / {row['brier_ebike_all']:.5f} ({wall:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", choices=NETS)
    ap.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lgbm", action="store_true")
    ap.add_argument("--logreg", action="store_true")
    ap.add_argument("--patience", type=int, default=None,
                    help="check run with another patience; logged to prob_extra_runs.csv, not the search log")
    ap.add_argument("--lgbm-ablation", nargs="+", choices=list(ABLATION),
                    help="LightGBM on an input subset; logged to prob_ablation.csv, not the search log")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on 200,000 train rows; logs and files go to a scratch folder, not the real ones")
    args = ap.parse_args()
    if args.smoke:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="prob_smoke_"))
        RUNS, CURVES, SAVED, MAX_EPOCHS = tmp / "runs.csv", tmp / "curves.csv", tmp, 1
        EXTRA_RUNS, EXTRA_CURVES, ABLATION_LOG = tmp / "extra.csv", tmp / "extra_curves.csv", tmp / "ablation.csv"
        print(f"smoke run: writing to {tmp}")
    SAVED.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    d = data()
    if args.smoke:
        keep = torch.randperm(len(d["xt"]), generator=torch.Generator().manual_seed(0))[:200_000]
        for k in ["xt", "st", "yt"]:
            d[k] = d[k][keep]
        d["tr"] = d["tr"].iloc[keep.numpy()].reset_index(drop=True)
    if args.logreg:
        logreg(d)
    if args.lgbm:
        for seed in args.seed:
            lgbm(seed, d)
    if args.lgbm_ablation:
        for name in args.lgbm_ablation:
            for seed in args.seed:
                lgbm(seed, d, ablation=name)
    if args.run:
        del d["tr"]
        for name in args.run:
            for seed in args.seed:
                train(name, seed, d, args.patience)
