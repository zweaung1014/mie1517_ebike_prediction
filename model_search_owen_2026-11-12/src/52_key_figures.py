"""Figures for the key findings (progress notebook and notebooks/key_findings.ipynb), from existing tables only.

key_station.png  K3: LightGBM skill vs calibrated persistence by input set (prob_ablation.csv, val, seed 0)
key_horizon.png  K5: Brier of the chosen model and of calibrated persistence by horizon (prob_by_horizon.csv, val)
key_days.png     K6: skill vs calibrated persistence per day, val and test, weekend marked
                 (prob_by_day.csv, prob_test_by_day.csv; Oct 28 is left out: a partial day, 76,559 rows)
Run from src/: python 52_key_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T, F = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
C = "mlp_2x256_do0.3"
COL = {"classic": "#d62728", "ebike": "#1f77b4"}
NAME = {"classic": "classic", "ebike": "e-bike"}
plt.rcParams.update({"font.size": 11})

# K3: what the gain needs
ab = pd.read_csv(T / "prob_ablation.csv").set_index("name")
sets = [("lgbm_counts_now_h", "counts now + h"),
        ("lgbm_counts_lags_h", "+ counts 10 and 30 min ago"),
        ("lgbm_no_station_info", "all inputs, no station info"),
        ("lgbm_no_station_idx", "+ location and capacity"),
        ("lgbm_all", "+ station index (all inputs)")]
fig, ax = plt.subplots(figsize=(8, 3.6))
y = range(len(sets))
for k, (x, off) in enumerate([("classic", -0.18), ("ebike", 0.18)]):
    vals = [ab.loc[n, f"bss_calpers_{x}_all"] for n, _ in sets]
    ax.barh([i + off for i in y], vals, height=0.34, color=COL[x], label=NAME[x])
    for i, v in zip(y, vals):
        ax.text(v + 0.002, i + off, f"{v:.3f}", va="center", fontsize=9)
ax.set_yticks(list(y), [lab for _, lab in sets])
ax.invert_yaxis()
ax.set_xlabel("Brier skill vs calibrated persistence (val, LightGBM, seed 0)")
ax.set_title("Knowing the station gives most of the gain")
ax.legend(loc="upper right")
ax.set_xlim(0, 0.18)
fig.tight_layout()
fig.savefig(F / "key_station.png", dpi=150)

# K5: skill grows with h because the reference degrades faster
bh = pd.read_csv(T / "prob_by_horizon.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, x in zip(axes, ["classic", "ebike"]):
    for model, style, lab in [("calibrated persistence", "--", "calibrated persistence (reference)"),
                              (C, "-", "MLP 2x256, dropout 0.3")]:
        d = bh[(bh.target == x) & (bh.model == model)].sort_values("group")
        ax.plot(d.group, d.brier, style, marker="o", ms=4, color=COL[x], label=lab)
    ax.set_title(NAME[x])
    ax.set_xlabel("horizon h (min)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Brier score (val, lower is better)")
axes[0].legend(fontsize=9)
fig.suptitle("Both get less accurate further ahead; the reference degrades faster, so skill grows")
fig.tight_layout()
fig.savefig(F / "key_horizon.png", dpi=150)

# K6: per day, weekend marked
days = []
for f, split in [("prob_by_day.csv", "val"), ("prob_test_by_day.csv", "test")]:
    d = pd.read_csv(T / f)
    d = d[(d.model == C) & (d.group != "2026-10-28")].assign(split=split)
    days.append(d)
dd = pd.concat(days)
dd["date"] = pd.to_datetime(dd.group)
order = sorted(dd.date.unique())
fig, ax = plt.subplots(figsize=(9, 3.6))
for k, (x, off) in enumerate([("classic", -0.18), ("ebike", 0.18)]):
    s = dd[dd.target == x].set_index("date").loc[order]
    ax.bar([i + off for i in range(len(order))], s.bss_calpers, width=0.34, color=COL[x], label=NAME[x])
for i, t in enumerate(order):
    if pd.Timestamp(t).dayofweek >= 5:
        ax.axvspan(i - 0.5, i + 0.5, color="0.85", zorder=0)
split_at = sum(1 for t in order if pd.Timestamp(t) < pd.Timestamp("2026-10-24"))
ax.axvline(split_at - 0.5, color="k", lw=1)
ax.text(split_at / 2 - 0.5, 0.172, "validation", ha="center")
ax.text(split_at + (len(order) - split_at) / 2 - 0.5, 0.172, "test", ha="center")
ax.set_xticks(range(len(order)), [pd.Timestamp(t).strftime("%a\n%d %b") for t in order])
ax.set_ylabel("skill vs calibrated persistence")
ax.set_ylim(0, 0.185)
ax.set_title("Weekend days (grey) score lowest; test weekdays are also below validation")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(F / "key_days.png", dpi=150)
print("wrote key_station.png, key_horizon.png, key_days.png")
