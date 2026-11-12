"""Cheap baselines beside persistence (NEXT item 4), scored with score.py on the val split of prototype_h15_v3.

Each baseline predicts the 15-min change of each bike type; the forecast is count now + change, clipped at 0, as in
the MLP (33_train_mlp.py). Everything is fitted on the train rows only.

  station_hour_mean          mean 15-min change per (station, local hour) on train. Day of week is NOT included.
                             A (station, hour) not seen in train gets change 0 (persistence).
  station_hour_weekend_mean  the same per (station, local hour, weekend yes/no); weekend = Sat or Sun local.
  linear                     ridge regression (alpha 1, on standardised inputs) on the MLP's inputs without the station
                             embedding: features() of 33_train_mlp.py (counts, capacity, hour and dow sin/cos, weekend,
                             weather + missing flag, ebike_system), plus an intercept.
  linear_station             the same plus a one-hot station (one column per train station; unseen stations get the
                             intercept only). Solved from the normal equations built from per-station sums, so the
                             one-hot matrix is never stored.

Team table (--data team, from Nov 10; score.TEAM): each row has its own horizon h (2-15 min), so every baseline is
fitted to the change per minute, (count at the horizon - count now) / h, and forecasts count now + h x that rate,
clipped at 0. The local hour comes from t_local, weekend is the table's is_weekend, and the ridge inputs are the MLP's
team inputs (33_train_mlp.team_features, which include h). Everything else as above.

Usage (from src/):  python 34_baselines.py [--data team]
Output: reports/tables/baselines_h15_val.csv (same layout as mlp_h15_val.csv: persistence first, then each baseline);
with --data team: baselines_team_val.csv.
"""
import argparse
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from score import DATASETS, H, TARGETS, V3, cols, load, score_table

mlp = importlib.import_module("33_train_mlp")  # reuse its column list and features(), so the inputs are the same

TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
CHANGE = [f"d_{x}" for x in TARGETS]
ALPHA = 1.0  # ridge penalty; tiny next to 32.5 M rows, it only keeps the one-hot system well posed
CHUNK = 1 << 21  # rows per float64 chunk when building X'X


def add_change(df, ds=V3):
    """The fitted target: the change over H on v3; the change per minute (divided by the row's h) otherwise."""
    per = 1.0 if ds is V3 else df.h.to_numpy(np.float64)
    for now, fut, d in zip(cols(ds, "now"), cols(ds, "truth"), CHANGE):
        df[d] = ((df[fut] - df[now]) / per).astype(np.float32)


def forecast(df, change, ds=V3):
    """Count now + change (v3) or count now + h x change per minute (team), clipped at 0. [n, len(TARGETS)]"""
    if ds is not V3:
        change = change * df.h.to_numpy(np.float64)[:, None]
    return np.clip(df[cols(ds, "now")].to_numpy(np.float64) + change, 0, None)


def group_mean(tr, va, keys):
    """Mean change per group on train, looked up for each val row; groups not seen in train -> 0."""
    means = tr.groupby(keys, observed=True)[CHANGE].mean()
    looked_up = va[keys].join(means, on=keys)[CHANGE]
    print(f"  {' x '.join(keys)}: {len(means):,} groups, {looked_up[CHANGE[0]].notna().mean():.2%} of val rows "
          "in a seen group")
    return looked_up.fillna(0).to_numpy(np.float64)


def with_intercept(x):
    return np.hstack([x, np.ones((len(x), 1), x.dtype)])


def ridge(tr, va, stats, with_station, ds=V3):
    """Ridge on [features, 1, (one-hot station)], solved from the normal equations in float64."""
    x = with_intercept(mlp.features(tr, stats, ds)[0])
    y = tr[CHANGE].to_numpy(np.float64)
    k = x.shape[1]
    xtx, xty = np.zeros((k, k)), np.zeros((k, len(CHANGE)))
    for i in range(0, len(x), CHUNK):  # in float64, a chunk at a time, to keep memory modest
        xc = x[i:i + CHUNK].astype(np.float64)
        xtx += xc.T @ xc
        xty += xc.T @ y[i:i + CHUNK]
    if with_station:
        stations = np.sort(tr.station_id.unique())
        s = pd.Index(stations).get_indexer(tr.station_id)
        # blocks of the full normal equations for [x, onehot]: onehot'onehot is diagonal (row counts per station),
        # onehot'x and onehot'y are per-station sums
        n_s = np.bincount(s, minlength=len(stations)).astype(np.float64)
        sx = np.stack([np.bincount(s, weights=x[:, j], minlength=len(stations)) for j in range(k)], axis=1)
        sy = np.stack([np.bincount(s, weights=y[:, j], minlength=len(stations)) for j in range(y.shape[1])], axis=1)
        xtx = np.block([[xtx, sx.T], [sx, np.diag(n_s)]])
        xty = np.vstack([xty, sy])
    del x
    w = np.linalg.solve(xtx + ALPHA * np.eye(len(xtx)), xty)  # intercept is penalised too; negligible here

    xv = with_intercept(mlp.features(va, stats, ds)[0])
    change = xv.astype(np.float64) @ w[:k]
    if with_station:
        sv = pd.Index(stations).get_indexer(va.station_id)  # -1 for a station not in train
        change += np.where(sv[:, None] >= 0, w[k:][np.maximum(sv, 0)], 0)
        print(f"  one-hot: {len(stations):,} train stations, {(sv < 0).mean():.3%} of val rows unseen")
    return change


def main(ds=V3):
    t0 = time.time()
    df = load(["train", "val"], columns=mlp.columns(ds), ds=ds)
    if ds is V3:
        df["weekend"] = df.dow >= 5
    else:
        df["hour"] = df.t_local.str[11:13].astype(int)  # local hour of t (t_local is "YYYY-MM-DDTHH:MM:SS-04:00")
        df["weekend"] = df.is_weekend.astype(bool)
    add_change(df, ds)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    print(f"train {len(tr):,} rows, val {len(va):,} rows ({time.time() - t0:.0f} s)")

    preds = {}
    preds["station_hour_mean"] = forecast(va, group_mean(tr, va, ["station_id", "hour"]), ds)
    preds["station_hour_weekend_mean"] = forecast(va, group_mean(tr, va, ["station_id", "hour", "weekend"]), ds)
    _, stats = mlp.features(tr, ds=ds)
    preds["linear"] = forecast(va, ridge(tr, va, stats, with_station=False, ds=ds), ds)
    print(f"  linear done ({time.time() - t0:.0f} s)")
    preds["linear_station"] = forecast(va, ridge(tr, va, stats, with_station=True, ds=ds), ds)
    print(f"  linear_station done ({time.time() - t0:.0f} s)")

    table = score_table(va, preds, ds)
    table.to_csv(TABLES / ("baselines_h15_val.csv" if ds is V3 else f"baselines_{ds.key}_val.csv"), index=False)
    print(table.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=sorted(DATASETS), default="v3", help="data file (score.DATASETS)")
    main(DATASETS[ap.parse_args().data])
