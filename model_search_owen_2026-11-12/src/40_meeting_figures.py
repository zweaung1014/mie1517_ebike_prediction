"""Result figures for the team meeting of Nov 12 (val split only), with large text for a shared screen.

Reads the logs of 37_model_search.py (model_search_runs.csv, _curves.csv, _by_horizon.csv), the cheap baselines
(baselines_team_val.csv, 34_baselines.py) and the persistence MSE (mlp_team_val.csv, 33_train_mlp.py).
Writes reports/figures/:
  meeting_a_candidates.png   val skill per full-budget candidate, mean +- sd over seeds 0-2 (dots = seeds);
                             the chosen model highlighted; the Nov 10 base and the best cheap baseline as lines
  meeting_b_curves.png       val skill per epoch for the finalists (mean over seeds; thin lines = seeds)
  meeting_c_horizon.png      val skill by horizon: chosen, base, LightGBM (mean over seeds, min-max shaded)
  meeting_d_low_rows.png     all rows against low rows (count at t <= 2), mean +- sd over seeds

Usage (from src/):  python 40_meeting_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG_DIR
from score import TARGETS

TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12})
CHOSEN, BASE, LGBM = "mlp_2x256_do0.1", "mlp_2x64_e8", "lgbm_l127"
LABEL = {"mlp_2x64_e8": "MLP 2×64 (Nov 10 base)", "mlp_2x64_do0.1": "MLP 2×64, dropout 0.1",
         "mlp_2x256_e8": "MLP 2×256", "mlp_2x256_do0.1": "MLP 2×256, dropout 0.1 (chosen)",
         "mlp_2x256_do0.2": "MLP 2×256, dropout 0.2", "mlp_2x256_do0.3": "MLP 2×256, dropout 0.3",
         "mlp_2x256_do0.5": "MLP 2×256, dropout 0.5",
         "mlp_3x256_e8": "MLP 3×256", "mlp_3x256_do0.1": "MLP 3×256, dropout 0.1",
         "gru64_3x256_e8": "GRU + MLP 3×256", "gru64_3x256_do0.1": "GRU + MLP 3×256, dropout 0.1",
         "lgbm_l127": "LightGBM (non-deep)"}
TNAME = {"classic": "Classic bikes", "ebike": "E-bikes"}
GREY, BLUE, ORANGE, GREEN = "#7f7f7f", "#1f77b4", "#d62728", "#2ca02c"


def full_runs(runs):
    """Full-budget runs with 3 seeds, plus LightGBM's early-stopped runs."""
    r = runs[(runs.budget == "full") | (runs.arch == "lgbm")]
    n = r.groupby("name").seed.nunique()
    return r[r.name.isin(n[n >= 3].index)]


def fig_a(runs, base_line):
    r = full_runs(runs)
    order = r.groupby("name").skill_classic_all.mean().add(r.groupby("name").skill_ebike_all.mean())
    names = list(order.sort_values().index)
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.6 * len(names) + 2.2), sharey=True)
    for ax, x in zip(axes, TARGETS):
        for i, n in enumerate(names):
            v = r[r.name == n][f"skill_{x}_all"]
            c = ORANGE if n == CHOSEN else (GREEN if n == LGBM else BLUE)
            ax.errorbar(v.mean(), i, xerr=v.std(), fmt="o", color=c, ms=9, capsize=5, lw=2)
            ax.plot(v, [i] * len(v), "|", color=c, ms=14, alpha=0.6)
        b = r[r.name == BASE][f"skill_{x}_all"].mean()
        ax.axvline(b, color=GREY, ls="--", lw=1.5, label=f"Nov 10 base {b:.3f}")
        name, val = base_line[x]
        ax.axvline(val, color="k", ls=":", lw=1.5, label=f"best cheap baseline {val:.3f}\n({name})")
        ax.set_title(TNAME[x])
        ax.set_xlabel("val skill vs persistence (all rows)")
        ax.set_xlim(0, 0.125)
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="center left")
    axes[0].set_yticks(range(len(names)), [LABEL.get(n, n) for n in names])
    for t in axes[0].get_yticklabels():
        if CHOSEN in [k for k, v in LABEL.items() if v == t.get_text()]:
            t.set_color(ORANGE)
            t.set_fontweight("bold")
    fig.suptitle("Candidates at the full budget: mean ± sd over seeds 0–2 (ticks = seeds)", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "meeting_a_candidates.png", dpi=120)
    plt.close(fig)


def fig_b(runs, curves, pers):
    names = [BASE, "mlp_2x256_e8", "mlp_2x64_do0.1", CHOSEN, "gru64_3x256_e8"]
    colors = [GREY, BLUE, GREEN, ORANGE, "#9467bd"]
    r = runs[(runs.budget == "full") & runs.name.isin(names)]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for n, c in zip(names, colors):
        cs = curves[curves.run_id.isin(r[r.name == n].run_id)].copy()
        cs["skill"] = 1 - cs.val_mse / pers
        for _, g in cs.groupby("run_id"):
            ax.plot(g.epoch, g.skill, color=c, lw=0.8, alpha=0.4)
        m = cs.groupby("epoch").skill.agg(["mean", "size"])
        m = m[m["size"] == 3]  # epochs that all three seeds reached
        ax.plot(m.index, m["mean"], "o-", color=c, lw=3 if n == CHOSEN else 2, label=LABEL[n])
    ax.set_xlabel("epoch")
    ax.set_ylabel("val skill vs persistence\n(mean MSE of both targets)")
    ax.set_title("Learning curves on val: thick = mean over seeds 0–2, thin = each seed")
    ax.set_ylim(0.055, 0.115)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "meeting_b_curves.png", dpi=120)
    plt.close(fig)


def fig_c(runs, by_h):
    names, colors = [BASE, LGBM, CHOSEN], [GREY, GREEN, ORANGE]
    r = full_runs(runs)[["run_id", "name"]]
    b = by_h.merge(r, on="run_id")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, x in zip(axes, TARGETS):
        for n, c in zip(names, colors):
            g = b[(b.name == n) & (b.target == x)].groupby("h_target").skill
            mean, lo, hi = g.mean(), g.min(), g.max()
            ax.plot(mean.index[:-1], mean.iloc[:-1], "o-", color=c, lw=3 if n == CHOSEN else 2, label=LABEL[n])
            ax.plot(mean.index[-1:], mean.iloc[-1:], "o", color=c, mfc="white", ms=8)  # h = 15: thin cell
            ax.fill_between(lo.index, lo, hi, color=c, alpha=0.2)
        ax.set_title(TNAME[x])
        ax.set_xlabel("horizon h (min)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val skill vs persistence")
    axes[0].legend(loc="upper left")
    fig.suptitle("Skill by horizon (shaded = min–max over seeds; open dot h = 15: 15,937 rows, "
                 "others ≈ 159,000)", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "meeting_c_horizon.png", dpi=120)
    plt.close(fig)


def fig_d(runs, baselines):
    names, colors = [BASE, LGBM, CHOSEN], [GREY, GREEN, ORANGE]
    r = full_runs(runs)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, x in zip(axes, TARGETS):
        for j, subset in enumerate(["all", "low"]):
            for k, (n, c) in enumerate(zip(names, colors)):
                v = r[r.name == n][f"skill_{x}_{subset}"]
                pos = j * 4.5 + k
                ax.bar(pos, v.mean(), yerr=v.std(), color=c, capsize=6, label=LABEL[n] if j == 0 else None)
                ax.text(pos, v.mean() + 0.006, f"{v.mean():.3f}", ha="center", fontsize=12)
            bl = baselines[(baselines.target == x) & (baselines.subset == subset)]
            best = bl.loc[bl.skill.idxmax()]
            pos = j * 4.5 + 3
            ax.bar(pos, best.skill, color="white", edgecolor="k", hatch="//",
                   label="best cheap baseline" if j == 0 else None)
            ax.text(pos, best.skill + 0.006, f"{best.skill:.3f}", ha="center", fontsize=12)
        ax.set_xticks([1.5, 6], ["all rows", "low rows (count at t ≤ 2)"])
        ax.set_title(TNAME[x])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("val skill vs persistence")
    axes[0].set_ylim(0, 0.135)
    axes[1].legend(loc="upper right")
    fig.suptitle("All rows against low rows: mean ± sd over seeds 0–2", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "meeting_d_low_rows.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    runs = pd.read_csv(TABLES / "model_search_runs.csv")
    curves = pd.read_csv(TABLES / "model_search_curves.csv")
    by_h = pd.read_csv(TABLES / "model_search_by_horizon.csv")
    bl = pd.read_csv(TABLES / "baselines_team_val.csv")
    bl = bl[bl.model != "persistence"]
    best = {x: bl[(bl.target == x) & (bl.subset == "all")].sort_values("skill").iloc[-1] for x in TARGETS}
    base_line = {x: (b.model, b.skill) for x, b in best.items()}
    p = pd.read_csv(TABLES / "mlp_team_val.csv")
    pers = p[(p.model == "persistence") & (p.subset == "all")].mse.mean()  # mean of the two targets' MSE
    print(f"persistence val MSE, mean of both targets: {pers:.5f}")
    print("best cheap baseline:", {x: (n, round(v, 4)) for x, (n, v) in base_line.items()})
    fig_a(runs, base_line)
    fig_b(runs, curves, pers)
    fig_c(runs, by_h)
    fig_d(runs, bl)
    print("wrote", *sorted(p.name for p in FIG_DIR.glob("meeting_*.png")))
