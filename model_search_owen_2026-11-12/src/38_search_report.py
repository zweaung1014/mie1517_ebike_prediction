"""Tables and figures for reports/MODEL_SEARCH.md, from the logs of 37_model_search.py (val split only).

Reads reports/tables/model_search_runs.csv, model_search_curves.csv, model_search_by_horizon.csv.
Writes:
  reports/tables/model_search_summary.csv   per (step, name, budget): seeds, params, s/epoch, wall, best epoch,
                                            mean and sample sd over seeds of the val skill per target and subset
  reports/figures/search_step1.png          Step 1: val skill (short budget) per config, one dot per seed
  reports/figures/search_curves.png         finalists, full budget: train and val MSE per epoch, every seed
  reports/figures/search_skill_by_horizon.png  finalists, full budget: val skill by horizon, mean over seeds
                                               (min-max over seeds shaded)
  reports/tables/model_search_by_horizon_summary.csv  the numbers of that figure (mean, sd, min, max over seeds)

Usage (from src/):  python 38_search_report.py --finalists NAME [NAME ...]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import FIG_DIR
from score import TARGETS

TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
SKILLS = [f"skill_{x}_{s}" for x in TARGETS for s in ("all", "low")]


def summary(runs):
    g = runs.groupby(["step", "name", "budget"], sort=False)
    out = g.agg(seeds=("seed", "nunique"), n_params=("n_params", "first"), s_per_epoch=("s_per_epoch", "mean"),
                wall_s=("wall_s", "mean"), best_epoch=("best_epoch", lambda b: "/".join(map(str, b))))
    for c in SKILLS:
        out[f"{c}_mean"] = g[c].mean()
        out[f"{c}_sd"] = g[c].std()
    return out.reset_index()


REF = {2: "mlp_2x256_e8"}  # the Step-1 pick each later step changes, drawn as the first row


def fig_step(runs, step, path):
    r = runs[(runs.budget == "short") & ((runs.step == step) | ((runs.step == 1) & (runs.name == REF.get(step))))]
    r = pd.concat([r[r.name == REF.get(step)], r[r.name != REF.get(step)]])
    names = list(dict.fromkeys(r.name))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.35 * len(names) + 1.5), sharey=True)
    for ax, x in zip(axes, TARGETS):
        for i, n in enumerate(names):
            v = r[r.name == n][f"skill_{x}_all"]
            ax.plot(v, [i] * len(v), "o", color="C0", alpha=0.7)
            ax.plot(v.mean(), i, "|", color="k", ms=14)
        ax.set_yticks(range(len(names)), names)
        ax.set_title(f"{x}: val skill vs persistence (all rows)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()
    fig.suptitle(f"Step {step}, short budget (4 epochs); dot = seed, bar = mean")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def final(runs, finalists):
    """Full-budget runs of the finalists (LightGBM: its early-stopped runs)."""
    return runs[((runs.budget == "full") | (runs.arch == "lgbm")) & runs.name.isin(finalists)]


def fig_curves(runs, curves, finalists, path):
    finalists = [n for n in finalists if not n.startswith("lgbm")]  # no epochs
    r = final(runs, finalists)
    ncol = 3
    nrow = -(-len(finalists) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), sharey=True, squeeze=False)
    for ax in axes.flat[len(finalists):]:
        ax.axis("off")
    for ax, n in zip(axes.flat, finalists):
        for i, rid in enumerate(r[r.name == n].run_id):
            c = curves[curves.run_id == rid]
            ax.plot(c.epoch, c.train_mse, "--", color=f"C{i}", alpha=0.7)
            ax.plot(c.epoch, c.val_mse, "-", color=f"C{i}", label=f"seed {rid.split('seed')[-1]}")
        ax.set_title(n)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    for row in axes:
        row[0].set_ylabel("MSE (count$^2$), mean of 2 targets")
    axes[0][0].plot([], [], "k--", label="train")
    axes[0][0].plot([], [], "k-", label="val")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_horizon(runs, by_h, finalists, path):
    r = final(runs, finalists)[["run_id", "name"]]
    b = by_h.merge(r, on="run_id")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, x in zip(axes, TARGETS):
        for i, n in enumerate(finalists):
            g = b[(b.name == n) & (b.target == x)].groupby("h_target").skill
            ax.plot(g.mean().index, g.mean(), "o-", color=f"C{i}", ms=3, label=n)
            ax.fill_between(g.min().index, g.min(), g.max(), color=f"C{i}", alpha=0.2)
        ax.set_title(f"{x}")
        ax.set_xlabel("horizon h (min)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val skill vs persistence")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalists", nargs="*", default=[])
    args = ap.parse_args()
    runs = pd.read_csv(TABLES / "model_search_runs.csv")
    s = summary(runs)
    s.to_csv(TABLES / "model_search_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(s[["step", "name", "budget", "seeds", "n_params", "s_per_epoch", "best_epoch",
                 *[f"{c}_mean" for c in SKILLS[::2]], *[f"{c}_sd" for c in SKILLS[::2]]]]
              .to_string(index=False, float_format="{:.4f}".format))
    for step in sorted(runs.step.dropna().unique()):
        if ((runs.step == step) & (runs.budget == "short")).any():
            fig_step(runs, step, FIG_DIR / f"search_step{int(step)}.png")
    if args.finalists:
        curves = pd.read_csv(TABLES / "model_search_curves.csv")
        by_h = pd.read_csv(TABLES / "model_search_by_horizon.csv")
        fig_curves(runs, curves, args.finalists, FIG_DIR / "search_curves.png")
        fig_horizon(runs, by_h, args.finalists, FIG_DIR / "search_skill_by_horizon.png")
        b = by_h.merge(final(runs, args.finalists)[["run_id", "name"]], on="run_id")
        g = b.groupby(["name", "target", "h_target"]).skill
        out = pd.DataFrame({"seeds": g.size(), "skill_mean": g.mean(), "skill_sd": g.std(),
                            "skill_min": g.min(), "skill_max": g.max()}).reset_index()
        out.to_csv(TABLES / "model_search_by_horizon_summary.csv", index=False)
