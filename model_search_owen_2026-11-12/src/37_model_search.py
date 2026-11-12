"""Model search on the team table (data-v1, weather fix; score.TEAM): architectures, then hyperparameters.

Selection uses the train split (to Oct 18) and the val split (Oct 19-23) only; the test rows are never read
(score.load is called with ["train", "val"]). Every run appends one row to reports/tables/model_search_runs.csv,
its loss curve to model_search_curves.csv and its val skill per horizon to model_search_by_horizon.csv.

Shared with 33_train_mlp.py (so a run with the default config is the Nov 10 MLP): the inputs (team_features: the
table's 20 documented numeric inputs, standardised with the train mean / std), the station embedding (unseen station -> one
extra row), the output (change of each count; forecast = count now + change, clipped at 0 at evaluation), the loss
(MSE of the count), Adam, one pass over the train rows per epoch, keep the epoch with the lowest val MSE.

Models
  mlp   station embedding (emb) + inputs -> `depth` hidden layers of `width` units (ReLU, optional dropout) -> 2
  gru   the counts at t - 30, t - 10 and t (classic, ebike, docks; 3 steps, plus a one-hot of the step) go through a
        GRU of `gru` units; its last state is joined with the other inputs and the embedding, then the same MLP head.
  lgbm  LightGBM, one regressor per target on the change of the count, inputs = the 20 inputs + station_idx as a
        categorical feature; early stopping on val MSE (a non-deep reference; separate function).

Budgets (--budget)
  short   4 epochs, no early stop, the best of the 4 val epochs kept: ranks candidates at a fixed cost (Steps 1-2)
  full    up to 30 epochs, patience 3 (the Nov 10 rule had 20 / 3)

Usage (from src/):
  python 37_model_search.py --step 1            run the Step-1 grid (see GRIDS) with seed 0, short budget
  python 37_model_search.py --run NAME --seed 1 --budget full     one named config
  python 37_model_search.py --lgbm              LightGBM reference
  add --save to any of these to retrain runs already in the log and save their val predictions (39_val_noise.py)
Device: CPU (--device mps to try the Mac GPU). Threads: --threads (default 6).
"""
import argparse
import fcntl
import importlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import warnings

from common import DATA_DIR
from score import TARGETS, TEAM, cols, load, score_table

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
mlp33 = importlib.import_module("33_train_mlp")
TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
RUNS = TABLES / "model_search_runs.csv"
CURVES = TABLES / "model_search_curves.csv"
BY_H = TABLES / "model_search_by_horizon.csv"
INPUTS = mlp33.TEAM_INPUTS
SEQ = [[f"{c}_tm30" for c in mlp33.COUNTS], [f"{c}_tm10" for c in mlp33.COUNTS], [f"{c}_now" for c in mlp33.COUNTS]]
SEQ_IDX = [[INPUTS.index(c) for c in step] for step in SEQ]
CTX_IDX = [i for i, c in enumerate(INPUTS) if all(c not in step for step in SEQ)]
SAVED = DATA_DIR / "models" / "search"  # --save: val predictions [n_val, 2] (.npy) and weights (.pt); not in git
SAVE = False  # set by --save
BUDGETS = {"short": (4, None), "full": (30, 3)}  # max epochs, patience (None: no early stop)


@dataclass(frozen=True)
class Config:
    arch: str = "mlp"
    depth: int = 2
    width: int = 64
    emb: int = 8
    gru: int = 0
    lr: float = 1e-3
    batch: int = 4096
    dropout: float = 0.0
    wd: float = 0.0  # AdamW decoupled weight decay when > 0; plain Adam when 0 (the Nov 10 setting)


BASE = Config()
GRIDS = {  # step -> name -> config; every step changes one thing at a time from the base or the step winner
    1: {"mlp_2x64_e8": BASE,
        "mlp_2x128_e8": replace(BASE, width=128),
        "mlp_2x256_e8": replace(BASE, width=256),
        "mlp_3x128_e8": replace(BASE, depth=3, width=128),
        "mlp_3x256_e8": replace(BASE, depth=3, width=256),
        "mlp_4x256_e8": replace(BASE, depth=4, width=256),
        "mlp_2x64_e32": replace(BASE, emb=32),
        "mlp_3x256_e32": replace(BASE, depth=3, width=256, emb=32),
        "gru64_2x64_e8": replace(BASE, arch="gru", gru=64),
        "gru64_3x256_e8": replace(BASE, arch="gru", gru=64, depth=3, width=256)},
}
W = GRIDS[1]["mlp_2x256_e8"]  # Step-1 pick: the smallest MLP within noise of the best at the short budget
GRIDS[2] = {"mlp_2x256_lr3e-4": replace(W, lr=3e-4),
            "mlp_2x256_lr3e-3": replace(W, lr=3e-3),
            "mlp_2x256_b1024": replace(W, batch=1024),
            "mlp_2x256_b16384": replace(W, batch=16384),
            "mlp_2x256_do0.1": replace(W, dropout=0.1),
            "mlp_2x256_wd1e-4": replace(W, wd=1e-4),
            "mlp_2x256_e32": replace(W, emb=32)}
GRIDS[4] = {"mlp_2x64_do0.1": replace(BASE, dropout=0.1),  # Step 4 (full budget): does dropout need the width?
            "mlp_2x256_do0.2": replace(W, dropout=0.2)}    # and is 0.1 enough?
GRIDS[5] = {  # Step 5 (full budget, after the Nov 11 reviews): models that lost without dropout, with it; bracket 0.2
    "mlp_3x256_do0.1": replace(GRIDS[1]["mlp_3x256_e8"], dropout=0.1),
    "gru64_3x256_do0.1": replace(GRIDS[1]["gru64_3x256_e8"], dropout=0.1),
    "mlp_2x256_do0.3": replace(W, dropout=0.3),
    "mlp_2x256_do0.5": replace(W, dropout=0.5)}  # added after 0.3 still rose: brackets the dropout trend


class Net(nn.Module):
    def __init__(self, cfg, n_inputs, n_stations):
        super().__init__()
        self.cfg = cfg
        self.station = nn.Embedding(n_stations + 1, cfg.emb)
        if cfg.arch == "gru":
            self.rnn = nn.GRU(len(SEQ[0]) + len(SEQ), cfg.gru, batch_first=True)
            self.register_buffer("seq_idx", torch.tensor(SEQ_IDX))
            self.register_buffer("ctx_idx", torch.tensor(CTX_IDX))
            self.register_buffer("step_onehot", torch.eye(len(SEQ)))
            n_in = len(CTX_IDX) + cfg.gru + cfg.emb
        else:
            n_in = n_inputs + cfg.emb
        layers = []
        for _ in range(cfg.depth):
            layers += [nn.Linear(n_in, cfg.width), nn.ReLU()]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            n_in = cfg.width
        layers.append(nn.Linear(n_in, len(TARGETS)))
        self.net = nn.Sequential(*layers)

    def forward(self, x, station):
        e = self.station(station)
        if self.cfg.arch == "gru":
            seq = x[:, self.seq_idx]  # [n, 3 steps, 3 counts]
            seq = torch.cat([seq, self.step_onehot.expand(len(x), -1, -1)], dim=2)
            _, hT = self.rnn(seq)
            return self.net(torch.cat([x[:, self.ctx_idx], hT[0], e], dim=1))
        return self.net(torch.cat([x, e], dim=1))


def data():
    """Train and val tensors (built once per process) and the val frame for scoring."""
    df = load(["train", "val"], columns=[*mlp33.TEAM_COLUMNS, "h_target"], ds=TEAM)
    tr, va = df[df.split == "train"].reset_index(drop=True), df[df.split == "val"].reset_index(drop=True)
    _, stats = mlp33.features(tr, ds=TEAM)
    station_index = {sid: i for i, sid in enumerate(sorted(tr.station_id.unique()))}
    t = mlp33.tensors(tr, stats, station_index, TEAM)
    v = mlp33.tensors(va, stats, station_index, TEAM)
    return t, v, va, len(station_index), tr.t_local.str[:10].to_numpy()  # last: the local day of each train row


@torch.no_grad()
def predict(model, x, s, now, device, batch=1 << 16):
    model.eval()
    out = [now[i:i + batch] + model(x[i:i + batch].to(device), s[i:i + batch].to(device)).cpu()
           for i in range(0, len(x), batch)]
    return torch.cat(out).clamp(min=0)


def by_horizon(va, pred, run_id):
    rows = []
    for h, idx in va.groupby("h_target").indices.items():
        t = score_table(va.iloc[idx], {"m": pred[idx]}, TEAM).query("subset == 'all' and model == 'm'")
        rows += [{"run_id": run_id, "target": r.target, "h_target": h, "rows": r.rows, "skill": r.skill}
                 for r in t.itertuples()]
    return rows


def append(path, rows):
    """Append rows under a lock, so parallel runs can share one log."""
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        df = pd.DataFrame(rows)
        if path.exists():  # align to the header already in the file (lgbm rows lack some columns)
            df = df.reindex(columns=pd.read_csv(path, nrows=0).columns)
        df.to_csv(path, mode="a", header=not path.exists(), index=False)


def summary(table):
    """skill_{target}_{subset} and mse_{target}_{subset} of the model rows of a score_table."""
    t = table[table.model != "persistence"]
    out = {f"skill_{r.target}_{r.subset}": r.skill for r in t.itertuples()}
    out |= {f"mse_{r.target}_{r.subset}": r.mse for r in t.itertuples()}
    return out


def warm_up(cfg, n_inputs, n_stations, device):
    """One discarded forward/backward pass, before the seed is set (GPU warm-up rule; harmless on CPU)."""
    m = Net(cfg, n_inputs, n_stations).to(device)
    x = torch.zeros(cfg.batch, n_inputs, device=device)
    s = torch.zeros(cfg.batch, dtype=torch.long, device=device)
    m(x, s).sum().backward()
    if device == "mps":
        torch.mps.synchronize()


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def train(name, cfg, seed, budget, step, d, device, train_days=None):
    """train_days = k: fit on the last k train days only (--train-days, Step 6). An epoch then still makes as many
    updates as one pass over all train rows (the subset is cycled), so only the amount of data changes."""
    (xt, st, nt, yt), (xv, sv, nv, yv), va, n_st, tr_day = d
    n_updates = -(-len(xt) // cfg.batch)  # batches per epoch, as with all train rows
    if train_days is not None:
        keep = tr_day >= sorted(set(tr_day))[-train_days]
        xt, st, nt, yt = xt[keep], st[keep], nt[keep], yt[keep]
        name = f"{name}_days{train_days}"
        print(f"{name}: last {train_days} train days, {len(xt):,} rows (from {sorted(set(tr_day))[-train_days]})",
              flush=True)
    max_epochs, patience = BUDGETS[budget]
    run_id = f"s{step}_{name}_{budget}_seed{seed}"
    warm_up(cfg, xt.shape[1], n_st, device)
    torch.manual_seed(seed)
    model = Net(cfg, xt.shape[1], n_st).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = (torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd) if cfg.wd > 0
           else torch.optim.Adam(model.parameters(), lr=cfg.lr))
    loss_fn = nn.MSELoss()
    best, best_epoch, best_state, curve = np.inf, 0, None, []
    t_start = time.time()
    print(f"{run_id}: {n_params:,} params, {device}", flush=True)
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        if train_days is None:  # unchanged from the original runs, so they reproduce exactly
            batches = torch.randperm(len(xt)).split(cfg.batch)
        else:
            perm = torch.cat([torch.randperm(len(xt)) for _ in range(-(-n_updates * cfg.batch // len(xt)))])
            batches = perm[:n_updates * cfg.batch].split(cfg.batch)
        seen = 0
        for idx in batches:
            seen += len(idx)
            xb, sb = xt[idx].to(device), st[idx].to(device)
            pred = nt[idx].to(device) + model(xb, sb)
            loss = loss_fn(pred, yt[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sync(device)
        sec = time.time() - t0
        val = ((predict(model, xv, sv, nv, device) - yv) ** 2).mean().item()
        curve.append({"run_id": run_id, "epoch": epoch, "train_mse": total / seen, "val_mse": val,
                      "epoch_s": sec})
        print(f"  epoch {epoch:2d}  train {total / seen:.4f}  val {val:.4f}  ({sec:.0f} s)", flush=True)
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif patience is not None and epoch - best_epoch >= patience:
            break
    wall = time.time() - t_start
    model.load_state_dict(best_state)
    pred = predict(model, xv, sv, nv, device).numpy()
    table = score_table(va, {"m": pred}, TEAM)
    row = {"run_id": run_id, "step": step, "name": name, **asdict(cfg), "seed": seed, "budget": budget,
           "max_epochs": max_epochs, "patience": patience, "device": device, "threads": torch.get_num_threads(),
           "n_params": n_params, "best_epoch": best_epoch, "epochs_run": len(curve), "wall_s": round(wall, 1),
           "s_per_epoch": round(np.mean([c["epoch_s"] for c in curve]), 1), **summary(table),
           "finished": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    if SAVE:
        torch.save({"state": best_state, "cfg": asdict(cfg), "best_epoch": best_epoch}, SAVED / f"{run_id}.pt")
        save_check(run_id, pred, row)
    else:
        append(RUNS, [row])
        append(CURVES, curve)
        append(BY_H, by_horizon(va, pred, run_id))
    print(f"  -> skill classic {row['skill_classic_all']:.4f}  ebike {row['skill_ebike_all']:.4f}  "
          f"(best epoch {best_epoch}, {wall:.0f} s)", flush=True)


def lgbm(seed=0, rounds=2000, lr=0.1, leaves=127):
    """LightGBM reference: one regressor per target on the change, early stopping on val MSE (patience 50)."""
    import lightgbm as lgb
    df = load(["train", "val"], columns=[*mlp33.TEAM_COLUMNS, "h_target", "station_idx"], ds=TEAM)
    tr, va = df[df.split == "train"].reset_index(drop=True), df[df.split == "val"].reset_index(drop=True)
    feats = [*INPUTS, "station_idx"]
    run_id = f"s1_lgbm_l{leaves}_lr{lr}_seed{seed}"
    t_start = time.time()
    pred, best_iters = np.zeros((len(va), len(TARGETS))), []
    for i, (now, fut) in enumerate(zip(cols(TEAM, "now"), cols(TEAM, "truth"))):
        dtr = lgb.Dataset(tr[feats], tr[fut] - tr[now], categorical_feature=["station_idx"], free_raw_data=False)
        dva = lgb.Dataset(va[feats], va[fut] - va[now], reference=dtr)
        params = {"objective": "regression", "learning_rate": lr, "num_leaves": leaves, "min_data_in_leaf": 200,
                  "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed,
                  "num_threads": torch.get_num_threads(), "verbose": -1, "max_cat_to_onehot": 4,
                  "cat_smooth": 10, "max_bin": 255}
        booster = lgb.train(params, dtr, rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        best_iters.append(booster.best_iteration)
        pred[:, i] = np.clip(va[now].to_numpy() + booster.predict(va[feats], num_iteration=booster.best_iteration),
                             0, None)
    wall = time.time() - t_start
    table = score_table(va, {"m": pred}, TEAM)
    row = {"run_id": run_id, "step": 1, "name": f"lgbm_l{leaves}", "arch": "lgbm", "lr": lr, "seed": seed,
           "budget": f"early stop 50, max {rounds}", "device": "cpu", "threads": torch.get_num_threads(),
           "n_params": None, "best_epoch": json.dumps(best_iters), "epochs_run": None, "wall_s": round(wall, 1),
           **summary(table), "finished": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    if SAVE:
        save_check(run_id, pred, row)
    else:
        append(RUNS, [row])
        append(BY_H, by_horizon(va, pred, run_id))
    print(f"{run_id}: skill classic {row['skill_classic_all']:.4f}  ebike {row['skill_ebike_all']:.4f}  "
          f"(best iters {best_iters}, {wall:.0f} s)", flush=True)


def save_check(run_id, pred, row):
    """--save: a retrain of a logged run. Save its val predictions, and stop if its val skill differs from the log."""
    logged = pd.read_csv(RUNS).set_index("run_id").loc[run_id]
    for c in [c for c in row if c.startswith("skill_")]:
        assert abs(row[c] - logged[c]) < 1e-6, f"{run_id}: {c} {row[c]:.6f} != logged {logged[c]:.6f}"
    np.save(SAVED / f"{run_id}.npy", pred.astype(np.float32))
    print(f"  saved {run_id}.npy; val skill equals the log", flush=True)


def config_by_name(name):
    for grid in GRIDS.values():
        if name in grid:
            return grid[name]
    raise KeyError(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, help="run every config of this step's grid")
    ap.add_argument("--run", nargs="+", help="named configs (from GRIDS)")
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--budget", choices=sorted(BUDGETS), default="short")
    ap.add_argument("--tag", type=int, default=None, help="step number written to the log for --run")
    ap.add_argument("--lgbm", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--train-days", type=int, default=None,
                    help="fit on the last N train days only (equal updates per epoch); logged as NAME_daysN")
    ap.add_argument("--save", action="store_true",
                    help="retrain logged runs: save val predictions and weights to SAVED, check the skill, log nothing")
    args = ap.parse_args()
    SAVE = args.save
    SAVED.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.lgbm:
        for seed in args.seed:
            lgbm(seed)
    else:
        names = list(GRIDS[args.step]) if args.step else args.run
        step = args.step if args.step else args.tag
        d = data()
        for name in names:
            for seed in args.seed:
                train(name, config_by_name(name), seed, args.budget, step, d, args.device, args.train_days)
