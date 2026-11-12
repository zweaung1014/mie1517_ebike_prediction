"""Checks after the six reviews of the probability work (Nov 12). Train and val only; the test rows are not read.

1. In-sample skill per train day: the chosen model (seed 0, as saved by 46_prob_search.py) on its own training days,
   BSS against calibrated persistence per local day, with the day type (weekday, Saturday, Sunday; Thanksgiving
   Monday Oct 12 marked). In sample, so the level is optimistic; only the weekday / weekend pattern is used.
2. The data section of the notebook: 5 example val rows (inputs and labels) and one station's classic and e-bike
   counts over one val day at the 10-min anchors, with the 2-bike line.

Writes (reports/tables/, reports/figures/)
  prob_train_by_day.csv   per train day: rows, day type, flip rate, BSS of the chosen model (seed 0) per target
  prob_train_daytype.csv  the same pooled by day type (rows-weighted Brier, then BSS)
  prob_patience_check.csv  the base MLP with patience 6 (prob_extra_runs.csv): mean BSS, and the selection rule's
                          verdict with these runs (reported only; the choice stands, test already read)
  prob_flags.csv          renting_now / returning_now (station-status flags) min and max per split (train, val)
  prob_climatology_check.csv  train days and rows per climatology cell; its val calibration with 20 and 200
                          pseudo-rows of shrinkage
  prob_sample_rows.csv    5 val rows of one station: time, h, the counts now and before, labels
  figures/prob_station_day.png   that station on that day
Usage (from src/):  python 50_prob_checks.py
"""
import importlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import FIG_DIR
from score import TARGETS
from score_prob import Climatology, brier, hour_of_week, labels, reliability, reliability_term

ps = importlib.import_module("46_prob_search")
TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
SAVED = ps.SAVED
THANKSGIVING = "2026-10-12"
DAY = "2026-10-21"  # the val day of the station figure (a Wednesday)


def day_type(day):
    dow = pd.to_datetime(day).dayofweek
    return "Saturday" if dow == 5 else "Sunday" if dow == 6 else "Thanksgiving" if day == THANKSGIVING else "weekday"


def patience_check(chosen):
    """The base MLP with patience 6 (46_prob_search.py --patience 6) against the logged search: its mean val Brier
    and BSS, and what the selection rule of 46_prob_search.py would give with these base runs (report only: the
    choice is not changed, the test rows were already read)."""
    runs = pd.read_csv(TABLES / "prob_runs.csv")
    extra = pd.read_csv(TABLES / "prob_extra_runs.csv")
    extra = extra[extra.run_id.str.startswith("p_mlp_2x64_e8_pat6_")]
    def brier_mean(r):
        return ((r.brier_classic_all + r.brier_ebike_all) / 2).to_numpy()
    a, b = brier_mean(extra), brier_mean(runs[runs.name == chosen])
    thr = 2 * np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    out = {"base_patience": 6, "seeds": len(a), "best_epochs": "/".join(map(str, extra.best_epoch)),
           "bss_calpers_classic_mean": extra.bss_calpers_classic_all.mean(),
           "bss_calpers_classic_sd": extra.bss_calpers_classic_all.std(),
           "bss_calpers_ebike_mean": extra.bss_calpers_ebike_all.mean(),
           "bss_calpers_ebike_sd": extra.bss_calpers_ebike_all.std(),
           "mean_brier": a.mean(), "chosen_mean_brier": b.mean(), "gap": a.mean() - b.mean(), "threshold": thr,
           "rule_would_pick_base": bool(a.mean() - b.mean() < thr)}
    pd.DataFrame([out]).to_csv(TABLES / "prob_patience_check.csv", index=False)
    print(pd.DataFrame([out]).T)


def flags_and_climatology(d):
    """(a) the two station-status inputs per split (train, val): min and max; (b) climatology: train days and rows
    per station x hour-of-week cell, and its val calibration with the logged shrinkage (20 pseudo-rows) and with
    200, to see whether its miscalibration comes from noisy cells (the count now is ignored in both)."""
    rows = []
    for name, df in [("train", d["tr"]), ("val", d["va"])]:
        for c in ["renting_now", "returning_now"]:
            rows.append({"split": name, "input": c, "min": df[c].min(), "max": df[c].max(), "rows": len(df)})
    pd.DataFrame(rows).to_csv(TABLES / "prob_flags.csv", index=False)
    tr, va = d["tr"], d["va"]
    cell = pd.DataFrame({"s": tr.station_id.to_numpy(), "w": hour_of_week(tr.t_local),
                         "day": tr.t_local.str[:10].to_numpy()})
    g = cell.groupby(["s", "w"]).agg(days=("day", "nunique"), rows=("day", "size"))
    y = labels(va)
    out = {"cells": len(g), "days_per_cell_mean": g.days.mean(), "days_per_cell_max": int(g.days.max()),
           "rows_per_cell_mean": g.rows.mean()}
    for k in [20, 200]:
        p = Climatology(k=k).fit(tr).predict(va)
        for i, x in enumerate(TARGETS):
            t = reliability(p[:, i], y[:, i])
            t = t[t.rows >= 1000]
            out[f"k{k}_{x}_rel"] = reliability_term(p[:, i], y[:, i])
            out[f"k{k}_{x}_max_bin_gap"] = (t.y_rate - t.p_mean).abs().max()
            out[f"k{k}_{x}_brier"] = brier(p[:, i], y[:, i])
    pd.DataFrame([out]).to_csv(TABLES / "prob_climatology_check.csv", index=False)
    print(pd.read_csv(TABLES / "prob_flags.csv").to_string(index=False))
    print(pd.DataFrame([out]).T)


def main():
    torch.set_num_threads(6)
    d = ps.data()
    flags_and_climatology(d)
    chosen = json.loads((TABLES / "prob_choice.json").read_text())["chosen"]
    patience_check(chosen)
    ck = torch.load(SAVED / f"p_{chosen}_seed0.pt", weights_only=False)
    model = ps.ms.Net(ps.ms.Config(**ck["cfg"]), d["xt"].shape[1], len(ck["station_index"]))
    model.load_state_dict(ck["state"])
    p_val = ps.predict(model, d["xv"], d["sv"]).numpy()
    saved = np.load(SAVED / f"p_{chosen}_seed0.npy")
    assert np.abs(p_val - saved).max() < 1e-5, "the saved weights do not reproduce the saved val forecast"

    tr = d["tr"]
    p_tr = ps.predict(model, d["xt"], d["st"]).numpy().astype(np.float64)
    ref = d["refs"]["calibrated persistence"].predict(tr)
    y = labels(tr)
    day = tr.t_local.str[:10].to_numpy()
    rows = []
    for dd in sorted(set(day)):
        m = day == dd
        row = {"day": dd, "day_type": day_type(dd), "rows": int(m.sum())}
        for i, x in enumerate(TARGETS):
            row[f"flip_{x}"] = (y[m, i] != (tr[f"{x}_now"].to_numpy()[m] >= 1)).mean()
            row[f"brier_model_{x}"], row[f"brier_calpers_{x}"] = brier(p_tr[m, i], y[m, i]), brier(ref[m, i], y[m, i])
            row[f"bss_calpers_{x}"] = 1 - row[f"brier_model_{x}"] / row[f"brier_calpers_{x}"]
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "prob_train_by_day.csv", index=False)
    g = []
    for dt, s in t.groupby("day_type"):
        r = {"day_type": dt, "days": len(s), "rows": int(s.rows.sum())}
        for x in TARGETS:
            w = s.rows / s.rows.sum()
            r[f"bss_calpers_{x}"] = 1 - (w * s[f"brier_model_{x}"]).sum() / (w * s[f"brier_calpers_{x}"]).sum()
        g.append(r)
    pd.DataFrame(g).to_csv(TABLES / "prob_train_daytype.csv", index=False)
    print(pd.DataFrame(g).round(3).to_string(index=False))

    va = d["va"]
    one = va[va.t_local.str[:10] == DAY]
    flips = one.groupby("station_id").apply(lambda s: (s.classic_now.diff().abs() > 0).sum(), include_groups=False)
    sid = flips.idxmax()  # the station whose classic count changes most often that day
    s = one[one.station_id == sid].drop_duplicates("t_local").sort_values("t_local")
    cols = ["t_local", "h", "classic_now", "ebike_now", "docks_now", "classic_tm10", "classic_tm30", "capacity",
            "temp_c", "y_classic", "y_ebike"]
    rows5 = one[(one.station_id == sid) & (one.t_local.str[11:13] >= "08")].sort_values(["t_local", "h"]).head(5)
    rows5[cols].to_csv(TABLES / "prob_sample_rows.csv", index=False)
    tt = pd.to_datetime(s.t_local.str[11:16], format="%H:%M")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.step(tt, s.classic_now, where="post", color="#d62728", label="classic bikes")
    ax.step(tt, s.ebike_now, where="post", color="#1f77b4", label="e-bikes")
    ax.axhline(2, color="0.5", lw=1, ls="--")
    ax.text(tt.iloc[0], 2.2, "2: the table keeps anchors where some count (classic, e-bike or free docks) is <= 2", fontsize=8, color="0.4")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.set_xlabel(f"local time, {DAY} (one value per 10-min anchor)")
    ax.set_ylabel("bikes at the station")
    ax.set_title(f"One station ({s.capacity.iloc[0]:.0f} docks) over one validation day")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prob_station_day.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
