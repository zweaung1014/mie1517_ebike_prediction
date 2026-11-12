"""The one test score of the model-search finalists on the team table (data-v1, weather fix; score.TEAM).

Protocol, fixed before the test rows are read (MEETING_2026-11-12.md, decision 3; run on Owen's go, Nov 12):
  models      chosen  MLP 2x256, dropout 0.1, seeds 0-2, at their logged best val epochs (weights saved by
                      37_model_search.py --save in DATA_DIR/models/search/)
              base    MLP 2x64 (the Nov 10 model), seeds 0-2, the same way
              lgbm    LightGBM, 127 leaves, seeds 0-2, retrained exactly as in the search (early stop on val)
              the best cheap baseline per target on val: station_hour_weekend_mean (classic) and
              linear_station (e-bike), fitted on train as in 34_baselines.py; both scored on both targets
  check       before the test rows are read, every model's val skill is recomputed and must equal the
              logged search value (MLP and LightGBM to 1e-6; baselines to baselines_team_val.csv)
  score       skill = 1 - MSE / MSE of persistence on the same rows, all rows and low rows (count at t <= 2);
              each model and seed is logged once in reports/tables/test_scores.csv (score.score_test);
              the nets are summarised as mean +- sd over seeds 0-2
  unseen      test rows whose station is not among the train stations are counted, and every model is also
              scored without them (they get the untrained "unknown" embedding row)
  nothing is tuned after the test numbers are seen

Writes reports/tables/test_scores.csv (appended), test_team_summary.csv (mean +- sd over seeds, all test rows and
without unseen stations), test_team_by_day.csv (skill per test day, mean over seeds for the nets).

Usage (from src/):  python 42_test_score.py [--dry]   (--dry: the val checks only; stops before the test rows)
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import DATA_DIR
from score import TARGETS, TEAM, load, score_table, score_test

ms = importlib.import_module("37_model_search")
mlp33 = importlib.import_module("33_train_mlp")
bl = importlib.import_module("34_baselines")
TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
SAVED = DATA_DIR / "models" / "search"
SEEDS = [0, 1, 2]
NETS = {"chosen_mlp_2x256_do0.1": "s3_mlp_2x256_do0.1_full_seed{s}", "base_mlp_2x64": "s3_mlp_2x64_e8_full_seed{s}"}
COLS = [*mlp33.TEAM_COLUMNS, "h_target", "station_idx"]


def val_check(name, va, pred, logged):
    t = score_table(va, {"m": pred}, TEAM)
    t = t[t.model == "m"].set_index(["target", "subset"]).skill
    for (x, sub), v in t.items():
        assert abs(v - logged[f"skill_{x}_{sub}"]) < 1e-6, f"{name} {x} {sub}: val {v:.6f} != log"
    print(f"  {name}: val skill equals the log ({t['classic', 'all']:.4f} / {t['ebike', 'all']:.4f})", flush=True)


def net_preds(tr, va, runs):
    """Saved nets (no training): val check now; returns name -> function that predicts given rows."""
    _, stats = mlp33.features(tr, ds=TEAM)
    station_index = {sid: i for i, sid in enumerate(sorted(tr.station_id.unique()))}
    xv, sv, nv, _ = mlp33.tensors(va, stats, station_index, TEAM)

    def on(net):
        def f(d):
            x, s_, n, _ = mlp33.tensors(d, stats, station_index, TEAM)
            return ms.predict(net, x, s_, n, "cpu").numpy()
        return f
    out = {}
    for name, pattern in NETS.items():
        for s in SEEDS:
            run_id = pattern.format(s=s)
            ck = torch.load(SAVED / f"{run_id}.pt", weights_only=False)
            net = ms.Net(ms.Config(**ck["cfg"]), xv.shape[1], len(station_index))
            net.load_state_dict(ck["state"])
            val_check(run_id, va, ms.predict(net, xv, sv, nv, "cpu").numpy(), runs.loc[run_id])
            out[f"{name}_seed{s}"] = on(net)
    return out


def lgbm_preds(tr, va, runs):
    import lightgbm as lgb
    feats = [*ms.INPUTS, "station_idx"]
    out = {}
    for s in SEEDS:
        pv, boosters = np.zeros((len(va), 2)), []
        for i, x in enumerate(TARGETS):
            now, fut = f"{x}_now", f"{x}_future"
            dtr = lgb.Dataset(tr[feats], tr[fut] - tr[now], categorical_feature=["station_idx"],
                              free_raw_data=False)
            dva = lgb.Dataset(va[feats], va[fut] - va[now], reference=dtr)
            params = {"objective": "regression", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
                      "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": s,
                      "num_threads": 4 if s == 0 else 6, "verbose": -1, "max_cat_to_onehot": 4,
                      "cat_smooth": 10, "max_bin": 255}  # threads as logged in the search
            b = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            pv[:, i] = np.clip(va[now] + b.predict(va[feats], num_iteration=b.best_iteration), 0, None)
            boosters.append(b)
        run_id = f"s1_lgbm_l127_lr0.1_seed{s}"
        val_check(run_id, va, pv, runs.loc[run_id])
        out[f"lgbm_l127_seed{s}"] = (lambda bs: lambda d: np.stack(
            [np.clip(d[f"{x}_now"] + b.predict(d[feats], num_iteration=b.best_iteration), 0, None)
             for x, b in zip(TARGETS, bs)], axis=1))(boosters)
    return out


def add_hour(d):
    d["hour"] = d.t_local.str[11:13].astype(int)
    d["weekend"] = d.is_weekend.astype(bool)


def baseline_preds(tr, va):
    for d in (tr, va):
        add_hour(d)
    bl.add_change(tr, TEAM)
    logged = pd.read_csv(TABLES / "baselines_team_val.csv").set_index(["model", "target", "subset"]).skill
    _, stats = mlp33.features(tr, ds=TEAM)
    out = {}
    for name, fit in [("station_hour_weekend_mean",
                       lambda d: bl.group_mean(tr, d, ["station_id", "hour", "weekend"])),
                      ("linear_station", lambda d: bl.ridge(tr, d, stats, with_station=True, ds=TEAM))]:
        t = score_table(va, {"m": bl.forecast(va, fit(va), TEAM)}, TEAM)
        t = t[t.model == "m"].set_index(["target", "subset"]).skill
        for (x, sub), v in t.items():
            assert abs(v - logged[name, x, sub]) < 1e-6, f"{name} {x} {sub}: val {v:.6f} != log"
        print(f"  {name}: val skill equals baselines_team_val.csv", flush=True)
        out[name] = (lambda fit_: lambda d: bl.forecast(d, fit_(d), TEAM))(fit)
    return out


def summarise(te, preds, mask_seen):
    rows, days = [], []
    day = te.t_local.str[:10].to_numpy()
    for rows_name, m in [("all test rows", np.ones(len(te), bool)), ("seen stations only", mask_seen)]:
        t = score_table(te[m], {k: v[m] for k, v in preds.items()}, TEAM)
        t = t[t.model != "persistence"].copy()
        t["model_group"] = t.model.str.replace(r"_seed\d$", "", regex=True)
        g = t.groupby(["model_group", "target", "subset"], sort=False)
        s = g.agg(rows=("rows", "first"), seeds=("skill", "size"), skill_mean=("skill", "mean"),
                  skill_sd=("skill", "std"), mse_mean=("mse", "mean")).reset_index()
        rows.append(s.assign(rows_used=rows_name))
    for d in sorted(set(day)):
        m = day == d
        t = score_table(te[m], {k: v[m] for k, v in preds.items()}, TEAM).query("subset == 'all'")
        t = t[t.model != "persistence"].copy()
        t["model_group"] = t.model.str.replace(r"_seed\d$", "", regex=True)
        s = t.groupby(["model_group", "target"], sort=False).skill.mean().reset_index()
        days.append(s.assign(day=pd.Timestamp(d).strftime("%Y-%m-%d %a"), rows=int(m.sum())))
    return pd.concat(rows), pd.concat(days)


def main():
    runs = pd.read_csv(TABLES / "model_search_runs.csv").set_index("run_id")
    df = load(["train", "val"], columns=COLS, ds=TEAM)
    tr, va = df[df.split == "train"].reset_index(drop=True), df[df.split == "val"].reset_index(drop=True)
    del df
    models = net_preds(tr, va, runs)  # every val check passes before the test rows are read
    models |= lgbm_preds(tr, va, runs)
    models |= baseline_preds(tr, va)
    if DRY:
        print("dry run: every val check passed; the test rows were not read")
        return
    # --- the test rows are read here, once, after the protocol above is fixed in the committed script ---
    te = load(["test"], columns=COLS, ds=TEAM)
    add_hour(te)
    print(f"train {len(tr):,}, val {len(va):,}, test {len(te):,} rows; test days "
          f"{te.t_local.str[:10].min()} to {te.t_local.str[:10].max()}", flush=True)
    seen = set(tr.station_id)
    mask_seen = te.station_id.isin(seen).to_numpy()
    print(f"test rows at stations not in train: {(~mask_seen).sum():,} "
          f"({(~mask_seen).mean():.2%}) at {te.station_id[~mask_seen].nunique()} stations", flush=True)

    preds = {name: f(te) for name, f in models.items()}

    note = "protocol fixed in 42_test_score.py before reading; val check passed"
    for name, p in preds.items():
        score_test(name, p, te, note=note, ds=TEAM)
    summary, by_day = summarise(te, preds, mask_seen)
    unseen = pd.DataFrame([{"test_rows": len(te), "rows_unseen_station": int((~mask_seen).sum()),
                            "stations_unseen": int(te.station_id[~mask_seen].nunique()),
                            "test_stations": int(te.station_id.nunique())}])
    summary.to_csv(TABLES / "test_team_summary.csv", index=False)
    by_day.to_csv(TABLES / "test_team_by_day.csv", index=False)
    unseen.to_csv(TABLES / "test_team_unseen.csv", index=False)
    with pd.option_context("display.width", 140):
        print(summary.to_string(index=False, float_format="{:.4f}".format))
        print(by_day.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    import sys
    DRY = "--dry" in sys.argv
    torch.set_num_threads(6)
    main()
