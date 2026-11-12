"""Shared split and scoring for the count forecast (NEXT item 3), so every member's model is scored the same way.

Data: export/prototype_h15_v3.parquet (31_prototype_dataset.py), the team's common benchmark: our polls up to the
end of live_snapshots_core.csv.gz (Oct 28 08:31 UTC). Task: predict each bike type's count at t + 15 min.
Split by whole local (Toronto) days of ts_local: train up to Oct 21, val Oct 22-28, test Oct 29 - Nov 4. Rows
with t in the last H min before a boundary are not used, so no target falls in the next split. Rows after Nov 4
belong to no split.
Score: MSE per bike type, and the skill against persistence (the count at t carried forward), on the same rows:
    skill = 1 - MSE_model / MSE_persistence      (0 = no better than doing nothing, 1 = perfect)
Each table also gives the rows whose input count is low (count at t <= 2, the `low_*` flags).
The test split is scored once per model and data file: score_test() records the result in
reports/tables/test_scores.csv and refuses a (model, data file) pair that is already there. v3 ends on Oct 28, so it
has train and val rows only (val to Oct 28 04:16 local); the test week needs a later export.

Second data file, TEAM (from Nov 10): the training table built by the team repo's notebooks/build_dataset.ipynb at
main 969473f (weather as of t, commit 9fe06d3). One row per (station, anchor t on a 10-min grid, horizon h); 3 random
horizons of 2-15 min per anchor; only anchors where some count at t is <= 2 are kept. Its own split column (whole local
days: train to Oct 18, val Oct 19-23, test from Oct 24; rows whose label crosses a boundary already purged). Columns:
now = {x}_now, truth = {x}_future, low = m_{x} (count at t <= 2, the same rule as low_*). Pass ds=TEAM to every function.

Usage in a model script:
    from score import load, score_table, score_test
    df = load(["train", "val"])                     # the test rows stay unread until the final score
    print(score_table(df[df.split == "val"], {"my_model": pred_val}))   # pred: [n, 2], columns TARGETS
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR, LOCAL_TZ, TEAM_TABLE

DATA = DATA_DIR / "export" / "prototype_h15_v3.parquet"
H = 15  # min
TARGETS = ["classic", "ebike"]  # bike types; the target of `x` is column f"{x}_h{H}"
SPLITS = {"train": (None, "2026-10-21"), "val": ("2026-10-22", "2026-10-28"), "test": ("2026-10-29", "2026-11-04")}
TEST_LOG = Path(__file__).resolve().parent.parent / "reports" / "tables" / "test_scores.csv"
TEAM_SPLIT_DATES = ("2026-10-19", "2026-10-24")  # val start, test start (local midnight), as printed by the notebook


@dataclass(frozen=True)
class Dataset:
    """Where a data file is and how its columns are named. {x} is a target in TARGETS."""
    key: str     # short name, used in file names and the test log
    path: Path
    now: str     # count at t (persistence)
    truth: str   # count at the horizon
    low: str     # input count <= 2
    time: str    # local time column used for the split check


V3 = Dataset("v3", DATA, "{x}", f"{{x}}_h{H}", "low_{x}", "ts_local")
TEAM = Dataset("team", TEAM_TABLE,  # common.TEAM_TABLE: env var TEAM_TABLE, else the team repo's data/processed/
               "{x}_now", "{x}_future", "m_{x}", "t_local")
DATASETS = {d.key: d for d in (V3, TEAM)}


def cols(ds, kind):
    return [getattr(ds, kind).format(x=x) for x in TARGETS]


def team_split_of(t_local):
    """Split name per row of the team table from its local time string: whole local days split at TEAM_SPLIT_DATES."""
    day = pd.to_datetime(pd.Series(t_local).str[:10])
    val, test = (pd.Timestamp(d) for d in TEAM_SPLIT_DATES)
    return np.select([day < val, day < test], ["train", "val"], "test")


def split_of(ts_local):
    """Split name per row (None outside every split) from the aware local time: whole local days, minus the last
    H min before each split's end, so the target at t + H stays in the same split."""
    ts_local = pd.DatetimeIndex(ts_local)
    day = ts_local.tz_localize(None).normalize()
    out = np.full(len(ts_local), None, object)
    for name, (a, b) in SPLITS.items():
        end = pd.Timestamp(b, tz=LOCAL_TZ) + pd.Timedelta(days=1)  # local midnight after the last day
        inside = day <= pd.Timestamp(b)
        if a is not None:
            inside &= day >= pd.Timestamp(a)
        out[inside & (ts_local < end - pd.Timedelta(minutes=H))] = name
    return out


def load(splits, columns=None, path=None, ds=V3):
    """Rows of the given splits, checked against the split rule of the data file. columns=None reads every column."""
    path = ds.path if path is None else path
    if columns is not None:
        columns = list(dict.fromkeys([ds.time, "split", *columns]))
    df = pd.read_parquet(path, columns=columns, filters=[("split", "in", list(splits))])
    df["split"] = df.split.astype(str)
    rule = split_of(df.ts_local) if ds is V3 else team_split_of(df[ds.time])
    assert (rule == df.split.values).all(), f"split column differs from the {ds.key} split rule"
    return df.reset_index(drop=True)


def persistence(df, ds=V3):
    """Doing nothing: the count at t, for each target. [n, len(TARGETS)]"""
    return df[cols(ds, "now")].to_numpy(np.float64)


def truth(df, ds=V3):
    return df[cols(ds, "truth")].to_numpy(np.float64)


def mse(pred, y):
    """MSE per target column."""
    return ((np.asarray(pred, np.float64) - y) ** 2).mean(axis=0)


def score_table(df, preds, ds=V3):
    """MSE and skill per model, target and subset (all rows; input count <= 2). preds: name -> [n, len(TARGETS)].
    Persistence is always the first row."""
    y, base = truth(df, ds), persistence(df, ds)
    preds = {"persistence": base, **preds}
    rows = []
    for x_i, x in enumerate(TARGETS):
        for subset, m in [("all", np.ones(len(df), bool)), ("low", df[ds.low.format(x=x)].to_numpy().astype(bool))]:
            ref = mse(base[m, x_i], y[m, x_i])
            for name, p in preds.items():
                e = mse(np.asarray(p)[m, x_i], y[m, x_i])
                rows.append({"model": name, "target": x, "subset": subset, "rows": int(m.sum()),
                             "mse": e, "skill": 1 - e / ref})
    return pd.DataFrame(rows)


def data_name(ds):
    """Name of the data file in the test log: the file name for v3 (as before), key/file name otherwise."""
    return ds.path.name if ds is V3 else f"{ds.key}/{ds.path.name}"


def score_test(name, pred, df, note="", log=TEST_LOG, ds=V3):
    """The one test score of a model on a data file: scores, appends to the log, and refuses a (model, data) pair
    already in the log."""
    data = data_name(ds)
    if len(df) == 0:
        raise ValueError(f"no test rows in {data}: the test week (Oct 29 - Nov 4) needs a later export")
    assert set(df.split) == {"test"}, "score_test takes the test rows only"
    if log.exists() and ((done := pd.read_csv(log)).model.eq(name) & done.data.eq(data)).any():
        raise RuntimeError(f"{name} already has a test score on {data} in {log}; scored once per model")
    t = score_table(df, {name: pred}, ds)
    t = t[t.model == name].assign(scored_at=pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
                                  data=data, note=note)
    base = score_table(df, {}, ds).set_index(["target", "subset"]).mse.rename("mse_persistence")
    t = t.join(base, on=["target", "subset"])
    log.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(log, mode="a", header=not log.exists(), index=False)
    return t
