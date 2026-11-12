"""Val results of the probability search (46_prob_search.py): summary, the model choice, noise, where it wins,
calibration. Val split only; the test rows are never read (score.load is called with ["train", "val"]).

Reads reports/tables/prob_runs.csv and the saved val probabilities DATA_DIR/models/prob/<run_id>.npy. The references
(score_prob.fit_references) are refit on the train rows; they are deterministic, so they equal the ones in the runs.

Writes (reports/tables/ unless said otherwise)
  prob_data.csv         per split (train, val): rows, stations, days; per target the share of rows with >= 1 bike
                        at t + h, the low-row share, and the share whose status flips between t and t + h
  prob_summary.csv      per model, target and subset (all; low = count now <= 2): seeds, mean and sd over seeds of
                        Brier, log loss, reliability term and BSS against persistence, calibrated persistence and
                        climatology; the references as rows of their own (no sd)
  prob_choice.json      the selection rule of 46_prob_search.py applied: mean val Brier (both targets, all rows) per
                        net, the best, the noise threshold per smaller net, the chosen model
  prob_noise.csv        per pair of models, target and subset: the BSS gap against calibrated persistence (a - b),
                        seed sd of the gap, station-block bootstrap sd (2,000 draws over the val stations),
                        z = gap / sqrt(seed sd^2 + station sd^2), and the val days (of 5) with a positive gap
  prob_by_count.csv     BSS per count now (0, 1, 2, 3-5, 6+) and target, chosen model and references
  prob_by_horizon.csv   BSS per target horizon (2-15 min) and target, chosen model, base model and references
  prob_last_bike.csv    rows with 1 bike now: share whose bike is gone at t + h, and the share of those cases caught
                        by flagging the 10 % / 20 % of rows with the lowest forecast (chosen, base, LightGBM, cal. pers., clim.)
  prob_reliability.csv  reliability bins (10 equal-width) per model, target and subset, seed 0 of each net
  figures/prob_reliability.png    calibration of the chosen model and the references, all rows and low rows
  figures/prob_where.png          BSS against calibrated persistence by count now and by horizon
Usage (from src/):  python 47_prob_report.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA_DIR, FIG_DIR
from score import TARGETS, TEAM, load
from score_prob import REF_COLUMNS, REF_NAMES, brier, fit_references, labels, low_mask, prob_table, reliability

TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
SAVED = DATA_DIR / "models" / "prob"
NETS = ["mlp_2x64_e8", "mlp_2x256_do0.1", "mlp_2x256_do0.3", "gru64_3x256_do0.1"]
OTHERS = ["lgbm", "logreg"]
LABEL = {"mlp_2x64_e8": "MLP 2x64 (base)", "mlp_2x256_do0.1": "MLP 2x256, dropout 0.1",
         "mlp_2x256_do0.3": "MLP 2x256, dropout 0.3", "gru64_3x256_do0.1": "GRU + 3x256, dropout 0.1",
         "lgbm": "LightGBM", "logreg": "logistic regression"}
METRICS = ["brier", "log_loss", "rel", "bss_pers", "bss_calpers", "bss_clim"]
COUNT_BINS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10_000, "6+")]
DRAWS = 2000


def runs_of(runs):
    """name -> list of run_ids, in seed order."""
    return {n: list(g.sort_values("seed").run_id) for n, g in runs.groupby("name")}


def preds(run_ids):
    return [np.load(SAVED / f"{r}.npy").astype(np.float64) for r in run_ids]


def summary_table(va, refs, by_name):
    rows = []
    for name, ids in by_name.items():
        tabs = [prob_table(va, {"m": p}, refs).query("model == 'm'") for p in preds(ids)]
        t = pd.concat(tabs)
        for (x, sub), g in t.groupby(["target", "subset"], sort=False):
            row = {"model": name, "target": x, "subset": sub, "rows": g.rows.iloc[0], "seeds": len(g)}
            for k in METRICS:
                row[f"{k}_mean"], row[f"{k}_sd"] = g[k].mean(), g[k].std() if len(g) > 1 else np.nan
            rows.append(row)
    ref = prob_table(va, {}, refs)
    for r in ref.itertuples():
        row = {"model": r.model, "target": r.target, "subset": r.subset, "rows": r.rows, "seeds": 0}
        for k in METRICS:
            row[f"{k}_mean"] = getattr(r, k)
        rows.append(row)
    return pd.DataFrame(rows)


def choose(runs, by_name):
    """The selection rule of 46_prob_search.py."""
    nets = [n for n in NETS if n in by_name]
    r = runs.set_index("run_id")
    score = {n: np.array([(r.loc[i, "brier_classic_all"] + r.loc[i, "brier_ebike_all"]) / 2 for i in by_name[n]])
             for n in nets}
    params = {n: int(r.loc[by_name[n][0], "n_params"]) for n in nets}
    best = min(nets, key=lambda n: score[n].mean())
    out = {"rule": "lowest mean val Brier (both targets, all rows, mean over seeds) among the nets; a net with fewer "
                   "parameters within 2 x sqrt(sd_a^2/n_a + sd_b^2/n_b) of it is preferred (the smallest such)",
           "nets": {n: {"seeds": len(score[n]), "mean_brier": score[n].mean(), "sd_brier": score[n].std(ddof=1),
                        "n_params": params[n]} for n in nets}, "best": best}
    chosen = best
    for n in sorted(nets, key=lambda n: params[n]):
        if params[n] >= params[best]:
            break
        a, b = score[n], score[best]
        thr = 2 * np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        out["nets"][n]["gap_to_best"], out["nets"][n]["threshold"] = a.mean() - b.mean(), thr
        if a.mean() - b.mean() < thr:
            chosen = n
            break
    out["chosen"] = chosen
    return out


def station_sums(va, p, y):
    """Per station: sum of squared errors per target [n_st, 2] and rows [n_st]."""
    codes, uniq = pd.factorize(va.station_id)
    se = (p - y) ** 2
    sums = np.stack([np.bincount(codes, se[:, i], len(uniq)) for i in range(len(TARGETS))], 1)
    return sums, np.bincount(codes, minlength=len(uniq)).astype(float), codes


def noise(va, refs, by_name, pairs):
    """BSS gap against calibrated persistence for each pair, with seed and station-bootstrap noise."""
    y = labels(va)
    ref = refs["calibrated persistence"].predict(va)
    rng = np.random.default_rng(0)
    day = va.t_local.str[:10].to_numpy()
    rows = []
    for x_i, x in enumerate(TARGETS):
        for sub in ["all", "low"]:
            m = np.ones(len(va), bool) if sub == "all" else low_mask(va, x)
            vm = va[m]
            ref_s, n_s, codes = station_sums(vm, ref[m], y[m])
            w = rng.multinomial(len(n_s), np.full(len(n_s), 1 / len(n_s)), size=DRAWS).astype(float)
            ref_boot = (w @ ref_s[:, x_i]) / (w @ n_s)
            for a, b in pairs:
                if a not in by_name or b not in by_name:
                    continue
                pa, pb = preds(by_name[a]), preds(by_name[b])
                ba = np.array([brier(p[m, x_i], y[m, x_i]) for p in pa])
                bb = np.array([brier(p[m, x_i], y[m, x_i]) for p in pb])
                b_ref = brier(ref[m, x_i], y[m, x_i])
                gap = (bb.mean() - ba.mean()) / b_ref  # BSS_a - BSS_b against calibrated persistence
                seed_sd = np.sqrt(ba.var(ddof=1) / len(ba) + bb.var(ddof=1) / len(bb)) / b_ref \
                    if min(len(ba), len(bb)) > 1 else np.nan
                sa = np.mean([station_sums(vm, p[m], y[m])[0][:, x_i] for p in pa], 0)
                sb = np.mean([station_sums(vm, p[m], y[m])[0][:, x_i] for p in pb], 0)
                boot = ((w @ sb) - (w @ sa)) / (w @ n_s) / ref_boot
                days = sorted(set(day[m]))
                pos = 0
                for d in days:
                    k = day[m] == d
                    yd, rd = y[m][k, x_i], ref[m][k, x_i]
                    g = (np.mean([brier(p[m][k, x_i], yd) for p in pb])
                         - np.mean([brier(p[m][k, x_i], yd) for p in pa])) / brier(rd, yd)
                    pos += g > 0
                st_sd = boot.std(ddof=1)
                rows.append({"a": a, "b": b, "target": x, "subset": sub, "gap_bss_calpers": gap, "seed_sd": seed_sd,
                             "station_sd": st_sd, "station_lo": np.quantile(boot, 0.025),
                             "station_hi": np.quantile(boot, 0.975),
                             "z": gap / np.sqrt(np.nan_to_num(seed_sd) ** 2 + st_sd ** 2),
                             "days_positive": f"{pos}/{len(days)}"})
    return pd.DataFrame(rows)


def by_group(va, refs, models, key, groups):
    """BSS against each reference per group (key(va, x) -> group label) and target. models: name -> [preds]."""
    y = labels(va)
    ref_p = {n: r.predict(va) for n, r in refs.items()}
    rows = []
    for x_i, x in enumerate(TARGETS):
        g_of = key(va, x)
        for g in groups:
            m = g_of == g
            if m.sum() == 0:
                continue
            yy = y[m, x_i]
            rb = {n: brier(p[m, x_i], yy) for n, p in ref_p.items()}
            cand = {**{n: [p] for n, p in ref_p.items()}, **models}
            for name, ps in cand.items():
                b = np.mean([brier(p[m, x_i], yy) for p in ps])
                row = {"target": x, "group": g, "rows": int(m.sum()), "base_rate": yy.mean(), "model": name,
                       "brier": b}
                for rname, short in REF_NAMES.items():
                    row[f"bss_{short}"] = 1 - b / rb[rname]
                rows.append(row)
    return pd.DataFrame(rows)


def count_group(va, x):
    c = va[f"{x}_now"].to_numpy()
    out = np.empty(len(va), object)
    for lo, hi, lab in COUNT_BINS:
        out[(c >= lo) & (c <= hi)] = lab
    return out


def last_bike(va, models):
    """Rider view of the '1 bike now' rows (persistence says 'go'): how often the bike is gone at t + h, and how many
    of those cases a warning on the rows with the lowest forecast would catch. models: name -> [preds per seed];
    each seed is ranked on its own and the shares are averaged over seeds."""
    y = labels(va)
    rows = []
    for x_i, x in enumerate(TARGETS):
        m = va[f"{x}_now"].to_numpy() == 1
        gone = y[m, x_i] == 0
        for name, ps in models.items():
            for flag in [0.1, 0.2]:
                caught, prec = [], []
                for p in ps:
                    pm = p[m, x_i]
                    k = int(round(flag * m.sum()))
                    low = np.argsort(pm, kind="stable")[:k]
                    caught.append(gone[low].sum() / gone.sum())
                    prec.append(gone[low].mean())
                rows.append({"target": x, "model": name, "rows_1_now": int(m.sum()), "gone_rate": gone.mean(),
                             "flag_share": flag, "gone_rate_flagged": np.mean(prec),
                             "share_of_gone_caught": np.mean(caught)})
    return pd.DataFrame(rows)


def fig_reliability(va, refs, chosen_p, chosen_name):
    y = labels(va)
    ref_p = {n: r.predict(va) for n, r in refs.items() if n != "persistence"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    rows = []
    for col, x in enumerate(TARGETS):
        for row_i, sub in enumerate(["all", "low"]):
            ax = axes[row_i, col]
            m = np.ones(len(va), bool) if sub == "all" else low_mask(va, x)
            ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
            for (name, p), c in zip([(LABEL.get(chosen_name, chosen_name), chosen_p), *ref_p.items()],
                                    ["#1f77b4", "#ff7f0e", "#2ca02c"]):
                t = reliability(p[m, col], y[m, col])
                t = t[t.rows >= 100]
                ax.plot(t.p_mean, t.y_rate, "-", color=c, label=name)
                ax.scatter(t.p_mean, t.y_rate, s=(3 + 9 * np.sqrt(t.rows / t.rows.max())) ** 2, color=c, zorder=3)
                rows += [{"model": name, "target": x, "subset": sub, **r} for r in t.to_dict("records")]
            ax.set_title(f"{'Classic' if x == 'classic' else 'E-bike'}, "
                         f"{'all rows' if sub == 'all' else 'low rows (count now <= 2)'}", fontsize=12)
            ax.set_xlabel("forecast P(>= 1 bike)")
            ax.set_ylabel("observed share with >= 1 bike")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=10, loc="upper left")
    fig.suptitle("Val calibration (Oct 19-23), 10 equal-width bins; marker size ~ rows", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prob_reliability.png", dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_where(by_count, by_h, chosen):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for x, c in zip(TARGETS, ["#d62728", "#1f77b4"]):
        t = by_count[(by_count.model == chosen) & (by_count.target == x)]
        axes[0].plot(t.group, t.bss_calpers, "o-", color=c, label="classic" if x == "classic" else "e-bike")
        t = by_h[(by_h.model == chosen) & (by_h.target == x)]
        axes[1].plot(t.group, t.bss_calpers, "o-", color=c, label="classic" if x == "classic" else "e-bike")
    axes[0].set_xlabel("bikes of that type at the station now")
    axes[1].set_xlabel("horizon h (min)")
    for ax in axes:
        ax.axhline(0, color="0.5", lw=1)
        ax.set_ylabel("BSS vs calibrated persistence")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Where {LABEL.get(chosen, chosen)} beats calibrated persistence (val, mean over seeds)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prob_where.png", dpi=130)
    plt.close(fig)


def data_table(df):
    """Per split: rows, stations, days, and per target the base rate, the low-row share and the persistence Brier
    (= share of rows whose 'at least one bike' status flips between t and t + h)."""
    rows = []
    for sp, g in df.groupby("split", sort=False):
        day = g.t_local.str[:10]
        row = {"split": sp, "rows": len(g), "stations": g.station_id.nunique(), "first_day": day.min(),
               "last_day": day.max(), "days": day.nunique()}
        y, now = labels(g), (g[[f"{x}_now" for x in TARGETS]].to_numpy() >= 1)
        for i, x in enumerate(TARGETS):
            row[f"{x}_base_rate"], row[f"{x}_low_share"] = y[:, i].mean(), low_mask(g, x).mean()
            row[f"{x}_flip_rate"] = (y[:, i] != now[:, i]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    runs = pd.read_csv(TABLES / "prob_runs.csv")
    by_name = runs_of(runs)
    df = load(["train", "val"], columns=REF_COLUMNS, ds=TEAM)
    data_table(df).to_csv(TABLES / "prob_data.csv", index=False)
    refs = fit_references(df[df.split == "train"].reset_index(drop=True))
    va = df[df.split == "val"].reset_index(drop=True)
    del df

    summ = summary_table(va, refs, by_name)
    summ.to_csv(TABLES / "prob_summary.csv", index=False)
    choice = choose(runs, by_name)
    (TABLES / "prob_choice.json").write_text(json.dumps(choice, indent=2, default=float))
    chosen = choice["chosen"]
    others = [n for n in [*NETS, *OTHERS] if n != chosen and n in by_name]
    pairs = [(chosen, n) for n in others]
    noise(va, refs, by_name, pairs).to_csv(TABLES / "prob_noise.csv", index=False)

    models = {n: preds(by_name[n]) for n in [chosen, "mlp_2x64_e8", "lgbm"] if n in by_name}
    by_count = by_group(va, refs, models, count_group, [b[2] for b in COUNT_BINS])
    by_count.to_csv(TABLES / "prob_by_count.csv", index=False)
    by_h = by_group(va, refs, models, lambda d, x: d.h_target.to_numpy(), list(range(2, 16)))
    by_h.to_csv(TABLES / "prob_by_horizon.csv", index=False)
    last_bike(va, {**models, **{n: [refs[n].predict(va)] for n in ["calibrated persistence", "climatology"]}}) \
        .to_csv(TABLES / "prob_last_bike.csv", index=False)
    rel = fig_reliability(va, refs, models[chosen][0], chosen)
    rel.to_csv(TABLES / "prob_reliability.csv", index=False)
    fig_where(by_count, by_h, chosen)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        s = summ[summ.subset == "all"][["model", "target", "seeds", "brier_mean", "brier_sd", "bss_pers_mean",
                                         "bss_calpers_mean", "bss_calpers_sd", "bss_clim_mean", "rel_mean"]]
        print(s.round(5).to_string(index=False))
        print(json.dumps({k: choice[k] for k in ["best", "chosen"]}))
        print(pd.read_csv(TABLES / "prob_noise.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
