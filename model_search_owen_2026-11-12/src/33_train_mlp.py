"""First model end to end (NEXT item 4): a small MLP that predicts each bike type's count 15 min ahead.

Team direction (Oct 28): inputs are the current counts plus weather and more, one model for all stations, MSE loss.
Data and scoring: score.py (prototype_h15_v3.parquet, cut at Oct 28 08:31 UTC; train <= Oct 21, val Oct 22-28, test Oct 29 - Nov 4).

  inputs   counts now (classic, ebike, docks), capacity, local hour and day of week (sin/cos), weekend,
           weather (temp, precip, wind, humidity, visibility; missing -> train mean plus a flag), ebike_system,
           and a learned 8-number embedding of the station (stations not in train share one "unknown" row)
  output   the change of each count over 15 min; the forecast is count now + change, so an output of 0 is
           persistence. The loss is the MSE of the forecast count, the same as the MSE of the change.
  training Adam, batch 4096, one pass over the train rows per epoch; keep the epoch with the lowest val MSE and
           stop after 3 epochs without improvement.

Team table (--data team, from Nov 10; score.TEAM): the table of the team repo's build_dataset.ipynb at main 969473f.
  Its rows have 3 random horizons h (2-15 min) per anchor, so h is one more input and one model covers every
  horizon; scores are over all horizons together. Inputs are the table's documented inputs (docs/training_table_
  columns.md): h, classic/ebike/docks now, renting/returning now, the counts 10 and 30 min before t, capacity, lat,
  lon, hour sin/cos, weekend, temp_c, rel_humidity (no missing values), plus the same station embedding. Output and
  training are as above. Files: mlp_team.pt, mlp_team_val.csv, mlp_team_curve.csv (seed N: mlp_team_seedN_*).

Usage (from src/):
  python 33_train_mlp.py            train, save the model to DATA_DIR/models, print the val table
  python 33_train_mlp.py --seed 1   the same with another seed; seeds other than 0 save mlp_h15_seed1.pt and
                                    mlp_h15_seed1_val.csv / _curve.csv, so the default files stay as they are
  python 33_train_mlp.py --test     load the saved model and score the test split once (score.score_test)
  python 33_train_mlp.py --data team [--seed N] [--test]   the same on the team table
Outputs: reports/tables/mlp_h15_val.csv (val scores, persistence beside), mlp_h15_curve.csv (loss per epoch),
and on --test a row block in reports/tables/test_scores.csv.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import DATA_DIR
from score import DATASETS, H, TARGETS, V3, cols, load, score_table, score_test

NAMES = {"v3": "mlp_h15", "team": "mlp_team"}
NAME = NAMES["v3"]
MODELS = DATA_DIR / "models"
TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
WEATHER = ["temp_c", "precip_mm", "wind_speed_kmh", "rel_humidity_pct", "visibility_km"]
COLUMNS = ["station_id", "classic", "ebike", "docks", "capacity", "hour", "dow", *WEATHER, "ebike_system",
           *(f"{x}_h{H}" for x in TARGETS), *(f"low_{x}" for x in TARGETS)]
COUNTS = ["classic", "ebike", "docks"]
TEAM_INPUTS = ["h", *(f"{c}_now" for c in COUNTS), "renting_now", "returning_now", *(f"{c}_tm10" for c in COUNTS),
               *(f"{c}_tm30" for c in COUNTS), "capacity", "lat", "lon", "hour_sin", "hour_cos", "is_weekend",
               "temp_c", "rel_humidity"]
TEAM_COLUMNS = ["station_id", *TEAM_INPUTS, *(f"{x}_future" for x in TARGETS), *(f"m_{x}" for x in TARGETS)]
BATCH = 4096
MAX_EPOCHS = 20
PATIENCE = 3


def features(df, stats=None, ds=V3):
    """Numeric input matrix [n, k] (float32), standardised with the train mean / std in `stats`."""
    if ds is not V3:
        return team_features(df, stats)
    f = pd.DataFrame({"classic": df.classic, "ebike": df.ebike, "docks": df.docks,
                      "capacity": df.capacity.fillna(df.classic + df.ebike + df.docks)})  # 178 rows have none
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * df.hour / 24), np.cos(2 * np.pi * df.hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * df.dow / 7), np.cos(2 * np.pi * df.dow / 7)
    f["weekend"] = df.dow >= 5
    f["wx_missing"] = df.temp_c.isna()
    for c in WEATHER:
        f[c] = df[c]
    f["ebike_system"] = df.ebike_system
    f = f.astype(np.float32)
    if stats is None:  # train: compute the stats once
        stats = {"mean": f.mean(), "std": f.std().replace(0, 1)}
    f = f.fillna(stats["mean"])  # missing weather -> train mean (the flag says it was missing)
    return ((f - stats["mean"]) / stats["std"]).to_numpy(np.float32), stats


def team_features(df, stats=None):
    """features() for the team table: its documented input columns, standardised with the train mean / std."""
    f = df[TEAM_INPUTS].astype(np.float32)
    if stats is None:
        stats = {"mean": f.mean(), "std": f.std().replace(0, 1)}
    return ((f - stats["mean"]) / stats["std"]).to_numpy(np.float32), stats


def columns(ds):
    return COLUMNS if ds is V3 else TEAM_COLUMNS


class MLP(nn.Module):
    def __init__(self, n_inputs, n_stations, emb=8, hidden=64):
        super().__init__()
        self.station = nn.Embedding(n_stations + 1, emb)  # last row: station not seen in train
        self.net = nn.Sequential(nn.Linear(n_inputs + emb, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, len(TARGETS)))

    def forward(self, x, station):
        return self.net(torch.cat([x, self.station(station)], dim=1))  # change of each count


def tensors(df, stats, station_index, ds=V3):
    x, _ = features(df, stats, ds)
    s = df.station_id.map(station_index).fillna(len(station_index)).to_numpy(np.int64)
    now = df[cols(ds, "now")].to_numpy(np.float32)
    y = df[cols(ds, "truth")].to_numpy(np.float32)
    return torch.from_numpy(x), torch.from_numpy(s), torch.from_numpy(now), torch.from_numpy(y)


@torch.no_grad()
def predict(model, x, s, now, batch=1 << 16):
    model.eval()
    out = [now[i:i + batch] + model(x[i:i + batch], s[i:i + batch]) for i in range(0, len(x), batch)]
    return torch.cat(out).clamp(min=0)  # a count cannot be negative


def run_name(seed, ds=V3):
    """File stem for a seed: seed 0 keeps the original names (mlp_h15.pt, mlp_h15_val.csv)."""
    name = NAMES[ds.key]
    return name if seed == 0 else f"{name}_seed{seed}"


def train(seed=0, ds=V3):
    torch.manual_seed(seed)
    name, stem = NAMES[ds.key], run_name(seed, ds)
    model_path = MODELS / f"{stem}.pt"
    df = load(["train", "val"], columns=columns(ds), ds=ds)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    _, stats = features(tr, ds=ds)
    station_index = {sid: i for i, sid in enumerate(sorted(tr.station_id.unique()))}
    xt, st, nt, yt = tensors(tr, stats, station_index, ds)
    xv, sv, nv, yv = tensors(va, stats, station_index, ds)
    print(f"train {len(tr):,} rows, val {len(va):,} rows, {xt.shape[1]} inputs, {len(station_index)} stations")

    model = MLP(xt.shape[1], len(station_index))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best, best_epoch, curve = np.inf, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for idx in torch.randperm(len(xt)).split(BATCH):
            pred = nt[idx] + model(xt[idx], st[idx])
            loss = loss_fn(pred, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val = ((predict(model, xv, sv, nv) - yv) ** 2).mean().item()
        curve.append({"epoch": epoch, "train_mse": total / len(xt), "val_mse": val})
        print(f"epoch {epoch:2d}  train MSE {total / len(xt):.4f}  val MSE {val:.4f}  ({time.time() - t0:.0f} s)",
              flush=True)
        if val < best:
            best, best_epoch = val, epoch
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "stats": stats, "station_index": station_index,
                        "n_inputs": xt.shape[1], "epoch": epoch, "seed": seed, "data": ds.key}, model_path)
        elif epoch - best_epoch >= PATIENCE:
            break
    print(f"best epoch {best_epoch} (val MSE {best:.4f}) -> {model_path}")
    pd.DataFrame(curve).to_csv(TABLES / f"{stem}_curve.csv", index=False)

    model, ckpt = load_model(model_path)
    table = score_table(va, {name: predict(model, xv, sv, nv).numpy()}, ds)
    table.to_csv(TABLES / f"{stem}_val.csv", index=False)
    print(table.to_string(index=False, float_format="{:.4f}".format))


def load_model(path=MODELS / f"{NAME}.pt"):
    ckpt = torch.load(path, weights_only=False)
    model = MLP(ckpt["n_inputs"], len(ckpt["station_index"]))
    model.load_state_dict(ckpt["state"])
    return model, ckpt


def test(ds=V3):
    name = NAMES[ds.key]
    model, ckpt = load_model(MODELS / f"{name}.pt")
    te = load(["test"], columns=columns(ds), ds=ds)
    x, s, now, _ = tensors(te, ckpt["stats"], ckpt["station_index"], ds)
    last = f"{te.ts_local.max():%Y-%m-%d %H:%M %Z}" if ds is V3 else te[ds.time].max()
    t = score_test(name, predict(model, x, s, now).numpy(), te, ds=ds,
                   note=f"best val epoch {ckpt['epoch']}; test rows to {last}")
    print(t[["target", "subset", "rows", "mse", "mse_persistence", "skill"]].to_string(
        index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="score the saved model on the test split (once)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for the weights and the batch order")
    ap.add_argument("--data", choices=sorted(DATASETS), default="v3", help="data file (score.DATASETS)")
    args = ap.parse_args()
    ds = DATASETS[args.data]
    test(ds) if args.test else train(args.seed, ds)
