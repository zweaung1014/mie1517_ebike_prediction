"""Does more training data help, and does it change dropout 0.1 against 0.3? (Step 6; val only.)

Reads the runs of 37_model_search.py --train-days K (K = 5, 10, 15: the last K train days, Oct 14/9/4 - Oct 18,
with as many updates per epoch as a pass over all train rows) and the full-train runs (20 days, Sep 29 - Oct 18)
of MLP 2x256 with dropout 0.1 (Step 3) and 0.3 (Step 5), seeds 0-2, full budget. The val split (Oct 19-23) is the
same for every size, so the gap between train and val stays the same. The test split is not used.

Writes
  reports/tables/data_size_summary.csv   per dropout and train days: train rows, seeds, best epochs, mean and sd of
                                         the val skill per target and subset
  reports/tables/data_size_gap.csv       per train days: dropout 0.3 minus 0.1 (mean gap, sd of the gap,
                                         sqrt(sd_a^2/3 + sd_b^2/3)), per target and subset
  reports/tables/data_size_slope.csv     per dropout and target: least-squares slope of the mean skill against
                                         train days over 10-20 days, and the last step 15 -> 20 days, per 7 days
  reports/figures/data_size.png          val skill against train days, both dropouts, mean +- sd over seeds

Usage (from src/):  python 44_data_size.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG_DIR
from score import TARGETS, TEAM, load

TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
CONFIGS = {"dropout 0.1": "mlp_2x256_do0.1", "dropout 0.3": "mlp_2x256_do0.3"}
DAYS = [5, 10, 15, 20]
COLS = [f"{x}_{s}" for x in TARGETS for s in ("all", "low")]


def runs_of(runs, name, days):
    if days == 20:
        return runs[(runs.name == name) & (runs.budget == "full")]
    return runs[runs.name == f"{name}_days{days}"]


def main():
    runs = pd.read_csv(TABLES / "model_search_runs.csv")
    day = load(["train"], columns=[], ds=TEAM).t_local.str[:10]
    counts = day.value_counts().sort_index()
    rows_by_days = {k: int(counts.iloc[-k:].sum()) for k in DAYS}

    summ, gaps = [], []
    for label, name in CONFIGS.items():
        for k in DAYS:
            r = runs_of(runs, name, k)
            out = {"dropout": label, "train_days": k, "first_day": counts.index[-k], "train_rows": rows_by_days[k],
                   "seeds": len(r), "best_epochs": "/".join(map(str, r.best_epoch))}
            for c in COLS:
                out[f"{c}_mean"], out[f"{c}_sd"] = r[f"skill_{c}"].mean(), r[f"skill_{c}"].std()
            summ.append(out)
    summ = pd.DataFrame(summ)
    for k in DAYS:
        a = summ[(summ.dropout == "dropout 0.3") & (summ.train_days == k)].iloc[0]
        b = summ[(summ.dropout == "dropout 0.1") & (summ.train_days == k)].iloc[0]
        for c in COLS:
            sd = np.sqrt(a[f"{c}_sd"] ** 2 / 3 + b[f"{c}_sd"] ** 2 / 3)
            gaps.append({"train_days": k, "column": c, "gap_0.3_minus_0.1": a[f"{c}_mean"] - b[f"{c}_mean"],
                         "sd_gap": sd, "z": (a[f"{c}_mean"] - b[f"{c}_mean"]) / sd})
    gaps = pd.DataFrame(gaps)
    slopes = []
    for label in CONFIGS:
        s = summ[(summ.dropout == label) & (summ.train_days >= 10)]
        for c in COLS:
            slope = np.polyfit(s.train_days, s[f"{c}_mean"], 1)[0]
            slopes.append({"dropout": label, "column": c, "slope_per_7_days_10_to_20": 7 * slope,
                           "slope_per_7_days_15_to_20": 7 / 5 * (s[s.train_days == 20][f"{c}_mean"].iloc[0]
                                                                  - s[s.train_days == 15][f"{c}_mean"].iloc[0]),
                           "skill_at_10": s[s.train_days == 10][f"{c}_mean"].iloc[0],
                           "skill_at_20": s[s.train_days == 20][f"{c}_mean"].iloc[0]})
    slopes = pd.DataFrame(slopes)
    summ.to_csv(TABLES / "data_size_summary.csv", index=False)
    gaps.to_csv(TABLES / "data_size_gap.csv", index=False)
    slopes.to_csv(TABLES / "data_size_slope.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, x, title in zip(axes, TARGETS, ["Classic bikes", "E-bikes"]):
        for (label, _), col in zip(CONFIGS.items(), ["#d62728", "#1f77b4"]):
            s = summ[summ.dropout == label]
            ax.errorbar(s.train_days, s[f"{x}_all_mean"], yerr=s[f"{x}_all_sd"], fmt="o-", color=col,
                        capsize=5, lw=2, ms=7, label=f"2×256, {label}")
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("train days used (the most recent)", fontsize=13)
        ax.set_xticks(DAYS)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val skill vs persistence (all rows)", fontsize=13)
    axes[0].legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_size.png", dpi=130)
    plt.close(fig)
    with pd.option_context("display.width", 150):
        print(summ[["dropout", "train_days", "train_rows", "best_epochs", "classic_all_mean", "classic_all_sd",
                    "ebike_all_mean", "ebike_all_sd"]].round(4).to_string(index=False))
        print(gaps.round(4).to_string(index=False))
        print(slopes.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
