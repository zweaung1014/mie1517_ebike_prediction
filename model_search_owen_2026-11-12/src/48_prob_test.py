"""The one test score of the chosen probability model (Owen, Nov 12). Protocol fixed before the test rows are read:

1. The model is the one in reports/tables/prob_choice.json (47_prob_report.py, the rule of 46_prob_search.py, val
   only). Its three seeds are scored; nothing else is (no LightGBM, no other net).
2. --dry first: rebuild each seed from its saved weights (.pt) and check that it reproduces its saved val
   probabilities (max abs difference < 1e-5) and the logged val Brier (to 1e-7). The test rows are not read.
3. The references are refit on the train rows only (score_prob.fit_references), as on val.
4. The test rows (Oct 24-28, score.load(["test"])) are read once. Each seed goes through
   score_prob.score_prob_test, which appends to reports/tables/prob_test_scores.csv and refuses a run already
   there, so a second run of this script stops before writing.
5. Reported: mean and sd over the seeds of Brier and BSS against each reference, all rows and low rows; the same per
   count now, per test day, and without the stations not seen in training; the test calibration bins.
   Nothing is tuned after this.

Writes (reports/tables/)
  prob_test_scores.csv     the log (one block per seed, references beside it)
  prob_test_summary.csv    mean and sd over seeds per target and subset, references as rows
  prob_test_by_count.csv   BSS per count now (0, 1, 2, 3-5, 6+), seed mean
  prob_test_by_day.csv     BSS per test day, seed mean
  prob_test_unseen.csv     all-row scores without the rows at stations not in train
  prob_test_reliability.csv  10-bin calibration of seed 0 and the references
  prob_test_data.csv       rows, stations, days, base rates and flip rates of the test split (as prob_data.csv)
Usage (from src/):  python 48_prob_test.py --dry      then      python 48_prob_test.py
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import DATA_DIR
from score import TARGETS, TEAM, load
from score_prob import REF_COLUMNS, REF_NAMES, fit_references, labels, prob_table, reliability, score_prob_test

ps = importlib.import_module("46_prob_search")
pr = importlib.import_module("47_prob_report")
ms, mlp33 = ps.ms, ps.mlp33
TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
SAVED = DATA_DIR / "models" / "prob"
METRICS = ["brier", "log_loss", "rel", "bss_pers", "bss_calpers", "bss_clim"]


def model_of(run_id):
    ck = torch.load(SAVED / f"{run_id}.pt", weights_only=False)
    cfg = ms.Config(**ck["cfg"])
    model = ms.Net(cfg, len(mlp33.TEAM_INPUTS), len(ck["station_index"]))
    model.load_state_dict(ck["state"])
    return model, ck


def probs(model, ck, df):
    x, s, _, _ = mlp33.tensors(df, ck["stats"], ck["station_index"], TEAM)
    return ps.predict(model, x, s).numpy().astype(np.float64)


def mean_sd(tabs, name):
    t = pd.concat(tabs)
    t = t[t.model.str.startswith("p_")]
    g = t.groupby(["target", "subset"], sort=False)
    out = g[METRICS].mean().add_suffix("_mean").join(g[METRICS].std().add_suffix("_sd"))
    out.insert(0, "seeds", g.size())
    return out.reset_index().assign(model=name)


def main(dry):
    choice = json.loads((TABLES / "prob_choice.json").read_text())
    name = choice["chosen"]
    runs = pd.read_csv(TABLES / "prob_runs.csv").set_index("run_id")
    run_ids = sorted(runs.index[runs.name == name])
    print(f"chosen: {name}, runs {run_ids}")
    torch.set_num_threads(6)

    df = load(["train", "val"], columns=ps.COLUMNS, ds=TEAM)
    tr, va = df[df.split == "train"].reset_index(drop=True), df[df.split == "val"].reset_index(drop=True)
    del df
    refs = fit_references(tr)
    seen = set(tr.station_id)
    del tr
    models = {}
    for r in run_ids:
        model, ck = model_of(r)
        p = probs(model, ck, va)
        saved = np.load(SAVED / f"{r}.npy").astype(np.float64)
        diff = np.abs(p - saved).max()
        b = prob_table(va, {"m": p}, refs).query("model == 'm' and subset == 'all'").set_index("target").brier
        logged = runs.loc[r, ["brier_classic_all", "brier_ebike_all"]].to_numpy(float)
        assert diff < 1e-5, f"{r}: val probabilities differ from the saved ones by {diff:.2e}"
        assert np.allclose(b[TARGETS].to_numpy(), logged, atol=1e-7), f"{r}: val Brier {b.values} != log {logged}"
        print(f"  {r}: val reproduces (max diff {diff:.1e}, Brier {b['classic']:.5f} / {b['ebike']:.5f})")
        models[r] = (model, ck)
    if dry:
        print("dry run: the test rows were not read")
        return

    te = load(["test"], columns=ps.COLUMNS, ds=TEAM)
    print(f"test rows: {len(te):,}, days {te.t_local.str[:10].min()} - {te.t_local.str[:10].max()}")
    pr.data_table(te).to_csv(TABLES / "prob_test_data.csv", index=False)
    p_te = {r: probs(m, ck, te) for r, (m, ck) in models.items()}
    tabs = [score_prob_test(r, p, te, refs, note=f"chosen {name}; protocol in src/48_prob_test.py")
            for r, p in p_te.items()]

    refs_tab = tabs[0][~tabs[0].model.str.startswith("p_")].drop(columns=["run", "scored_at", "note"])
    summ = pd.concat([mean_sd(tabs, name), refs_tab.rename(columns={k: f"{k}_mean" for k in METRICS})])
    summ.to_csv(TABLES / "prob_test_summary.csv", index=False)

    plist = list(p_te.values())
    pr.by_group(te, refs, {name: plist}, pr.count_group, [b[2] for b in pr.COUNT_BINS]) \
        .to_csv(TABLES / "prob_test_by_count.csv", index=False)
    pr.by_group(te, refs, {name: plist}, lambda d, x: d.t_local.str[:10].to_numpy(),
                sorted(te.t_local.str[:10].unique())).to_csv(TABLES / "prob_test_by_day.csv", index=False)
    keep = te.station_id.isin(seen).to_numpy()
    un = [prob_table(te[keep], {"m": p[keep]}, refs).query("model == 'm' and subset == 'all'") for p in plist]
    un = pd.concat(un).groupby("target")[["brier", "bss_pers", "bss_calpers", "bss_clim"]].mean().reset_index()
    un.insert(1, "rows_dropped", int((~keep).sum()))
    un.to_csv(TABLES / "prob_test_unseen.csv", index=False)
    y = labels(te)
    rel = []
    for n, p in [(name, plist[0]), *((k, r.predict(te)) for k, r in refs.items() if k != "persistence")]:
        for i, x in enumerate(TARGETS):
            rel += [{"model": n, "target": x, **row} for row in reliability(p[:, i], y[:, i]).to_dict("records")]
    pd.DataFrame(rel).to_csv(TABLES / "prob_test_reliability.csv", index=False)
    with pd.option_context("display.width", 160):
        print(summ[["model", "target", "subset", "brier_mean", "bss_pers_mean", "bss_calpers_mean",
                    "bss_calpers_sd", "bss_clim_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="check the val reproduction only; the test rows are not read")
    main(ap.parse_args().dry)
