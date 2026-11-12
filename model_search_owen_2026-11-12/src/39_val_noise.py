"""Val-sampling noise of the model-search gaps (after the Nov 11 reviews of MODEL_SEARCH.md). Val split only.

The seed sd in MODEL_SEARCH.md measures the weights and batch order on one fixed val set. This script adds the
noise from which val rows we happened to get: per val day, and a block bootstrap over stations and over days.

Reads the val predictions saved by `37_model_search.py --save` (retrains of logged runs; each was checked to give
the logged val skill) from DATA_DIR/models/search/<run_id>.npy, for seeds 0-2 of
  base    mlp_2x64_e8 (the Nov 10 MLP)    chosen  mlp_2x256_do0.1    lgbm  lgbm_l127 (LightGBM reference)
  do0.3   mlp_2x256_do0.3 (Step 5, the best val skill; added Nov 12)
Skill of a model = 1 - (mean over seeds of its MSE) / MSE of persistence, the same as the mean of the seed skills.

Writes (reports/tables/):
  val_noise_by_day.csv  skill per model, target, subset (all, low) and val day, and the gap of each pair in PAIRS
  val_noise_gap.csv     per pair, target and subset: the gap; the seed sd of the gap, sqrt(sd_a^2/3 + sd_b^2/3);
                        the station-block bootstrap (2,000 draws over the 1,072 val stations) sd and 95 % interval;
                        the day-block bootstrap (2,000 draws over the 5 val days) sd and 95 % interval; days with a
                        positive gap; and z = gap / sqrt(seed sd^2 + station-bootstrap sd^2)
  val_noise_optimism.csv  early-stopping optimism per full-budget run (from model_search_curves.csv): combined
                        skill (1 - val MSE / persistence MSE, both as the mean of the two targets) at the best epoch,
                        and the mean of the epochs just before and after it; then the mean per model

Usage (from src/):  python 39_val_noise.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR
from score import TARGETS, TEAM, cols, load

TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
SAVED = DATA_DIR / "models" / "search"
SEEDS = [0, 1, 2]
MODELS = {"base": "s3_mlp_2x64_e8_full_seed{s}", "chosen": "s3_mlp_2x256_do0.1_full_seed{s}",
          "lgbm": "s1_lgbm_l127_lr0.1_seed{s}", "do0.3": "s5_mlp_2x256_do0.3_full_seed{s}"}
PAIRS = [("chosen", "base"), ("chosen", "lgbm"), ("do0.3", "chosen"), ("do0.3", "base"), ("do0.3", "lgbm")]
DRAWS = 2000


def cluster_sums(err2, key, n_keys):
    """Sum of squared errors per cluster (station or day). err2: [n]."""
    return np.bincount(key, weights=err2, minlength=n_keys)


def boot(sa, sb, sp, n_draws, rng):
    """Bootstrap of the skill gap (a - b) over clusters: sa, sb = per-cluster SSE of the two models, sp = persistence."""
    w = rng.multinomial(len(sp), np.full(len(sp), 1 / len(sp)), size=n_draws).astype(np.float64)
    return (w @ sb - w @ sa) / (w @ sp)


def optimism():
    """Best epoch against its neighbours, per full-budget run: how much picking the lowest-val epoch adds."""
    runs = pd.read_csv(TABLES / "model_search_runs.csv")
    curves = pd.read_csv(TABLES / "model_search_curves.csv")
    p = pd.read_csv(TABLES / "mlp_team_val.csv")
    pers = p[(p.model == "persistence") & (p.subset == "all")].mse.mean()
    rows = []
    for r in runs[runs.budget == "full"].itertuples():
        c = curves[curves.run_id == r.run_id].set_index("epoch").val_mse
        b = int(r.best_epoch)
        nb = [e for e in (b - 1, b + 1) if e in c.index]
        rows.append({"run_id": r.run_id, "name": r.name, "seed": r.seed, "best_epoch": b,
                     "skill_best": 1 - c[b] / pers, "skill_neighbours": 1 - c[nb].mean() / pers})
    o = pd.DataFrame(rows)
    o["optimism"] = o.skill_best - o.skill_neighbours
    m = o.groupby("name")[["skill_best", "skill_neighbours", "optimism"]].mean().reset_index()
    out = pd.concat([o, m.assign(run_id="mean over seeds")])
    out.to_csv(TABLES / "val_noise_optimism.csv", index=False)
    print(m.to_string(index=False, float_format="{:.4f}".format))


def main():
    va = load(["val"], columns=["station_id", *cols(TEAM, "now"), *cols(TEAM, "truth"), *cols(TEAM, "low")],
              ds=TEAM)
    y, now = va[cols(TEAM, "truth")].to_numpy(np.float64), va[cols(TEAM, "now")].to_numpy(np.float64)
    day = pd.to_datetime(va.t_local.str[:10])
    days = sorted(day.unique())
    day_key = day.map({d: i for i, d in enumerate(days)}).to_numpy()
    st_key = va.station_id.astype("category").cat.codes.to_numpy()
    n_st = st_key.max() + 1
    print(f"val {len(va):,} rows, {len(days)} days, {n_st} stations")

    # squared error per row, model (mean over seeds) and target
    err = {}
    for m, pattern in MODELS.items():
        e = [(np.load(SAVED / f"{pattern.format(s=s)}.npy").astype(np.float64) - y) ** 2 for s in SEEDS]
        err[m] = np.mean(e, axis=0)
        err[m + "_seeds"] = e
    err["persistence"] = (now - y) ** 2

    runs = pd.read_csv(TABLES / "model_search_runs.csv")
    rng = np.random.default_rng(0)
    by_day, gaps = [], []
    for i, x in enumerate(TARGETS):
        low = va[TEAM.low.format(x=x)].to_numpy().astype(bool)
        for subset, mask in [("all", np.ones(len(va), bool)), ("low", low)]:
            sse = {m: (cluster_sums(np.where(mask, err[m][:, i], 0), day_key, len(days)),
                       cluster_sums(np.where(mask, err[m][:, i], 0), st_key, n_st))
                   for m in ["persistence", *MODELS]}
            skill_day = {m: 1 - sse[m][0] / sse["persistence"][0] for m in MODELS}
            for d_i, d in enumerate(days):
                r = {"target": x, "subset": subset, "day": f"{d:%Y-%m-%d %a}",
                     **{f"skill_{m}": skill_day[m][d_i] for m in MODELS}}
                r |= {f"gap_{a}_{b}": skill_day[a][d_i] - skill_day[b][d_i] for a, b in PAIRS}
                by_day.append(r)
            for a, b in PAIRS:
                p_tot = sse["persistence"][0].sum()
                gap = (sse[b][0].sum() - sse[a][0].sum()) / p_tot
                # seed sd of the gap from the logged per-seed skills (the same numbers as Table 1)
                sd = {m: runs[runs.run_id.isin([MODELS[m].format(s=s) for s in SEEDS])][f"skill_{x}_{subset}"].std()
                      for m in (a, b)}
                seed_sd = np.sqrt(sd[a] ** 2 / 3 + sd[b] ** 2 / 3)
                st = boot(sse[a][1], sse[b][1], sse["persistence"][1], DRAWS, rng)
                dy = boot(sse[a][0], sse[b][0], sse["persistence"][0], DRAWS, rng)
                gaps.append({"pair": f"{a} - {b}", "target": x, "subset": subset, "gap": gap,
                             "seed_sd": seed_sd,
                             "station_boot_sd": st.std(), "station_lo": np.quantile(st, 0.025),
                             "station_hi": np.quantile(st, 0.975),
                             "day_boot_sd": dy.std(), "day_lo": np.quantile(dy, 0.025),
                             "day_hi": np.quantile(dy, 0.975),
                             "days_positive": int((skill_day[a] > skill_day[b]).sum()),
                             "z": gap / np.sqrt(seed_sd ** 2 + st.std() ** 2)})
    by_day, gaps = pd.DataFrame(by_day), pd.DataFrame(gaps)
    by_day.to_csv(TABLES / "val_noise_by_day.csv", index=False)
    gaps.to_csv(TABLES / "val_noise_gap.csv", index=False)
    with pd.option_context("display.width", 140):
        print(gaps.to_string(index=False, float_format="{:.4f}".format))
        print(by_day.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
    optimism()
