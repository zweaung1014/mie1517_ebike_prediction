"""Scoring of the probability forecast: P(at least one bike of each type at t + h), on the team table (score.TEAM).

Output (Owen, Nov 12): for each bike type x in TARGETS, the probability that the station has at least one bike of
type x at t + h. The label is the table's y_{x} column, which equals 1[{x}_future >= 1] on every row.
The split, rows and inputs are those of score.py (TEAM): train to Oct 18, val Oct 19-23, test from Oct 24.

Why not log-loss skill against persistence: persistence says "available at t + h iff available now", a hard 0 or 1.
Each miss then costs -log(0) = infinity, so a log-loss skill against it is undefined. The Brier score is finite
for 0/1 forecasts and is a proper score, so it is the yardstick here:
    Brier = mean (p - y)^2              (0 = perfect; persistence's Brier = the share of rows whose status flips)
    BSS_ref = 1 - Brier_model / Brier_ref   (0 = no better than the reference, 1 = perfect, < 0 worse)

Three references, all fit on the train split only (fit() never sees val or test):
  persistence      p = 1[count now >= 1]. Doing nothing; the yardstick of the count model.
  calibrated pers. p = P(y = 1 | count now, h) from train, per target: the share of train rows with that count now
                   (capped at CAP) and that integer target horizon h_target whose label is 1. A cell with few rows
                   is shrunk toward its count's rate over all h (K pseudo-rows). It is persistence made into an
                   honest probability, so it is harder to beat than raw persistence: it already knows that
                   "1 bike now" means "about 85 % chance of a bike in 10 min", not 100 %.
  climatology      p = the station's train rate in that hour of the week (0-167, local time), shrunk toward the
                   station's overall train rate, which is shrunk toward the global train rate (K pseudo-rows each).
                   It ignores the count now; it is what a rider knows from habit.
  station-hour pers. (added after the reviews, val only; fit_references(..., station_hour=True)) calibrated
                   persistence plus a shrunk train correction per (count now, station, hour of day). A look-up
                   table that also knows the station and the hour: the strongest reference without learning.
Calibration: reliability() bins the forecast and gives the mean forecast and the observed rate per bin;
reliability_term() is the binned reliability (calibration) part of the Murphy decomposition of the Brier score,
sum_k n_k (pbar_k - ybar_k)^2 / N (0 = calibrated).

Usage:
    from score_prob import labels, fit_references, prob_table
    refs = fit_references(train_df)                     # dict name -> fitted reference
    t = prob_table(val_df, {"my_model": p_val}, refs)   # p_val: [n, 2] probabilities, columns TARGETS
The test split is scored once per model: score_prob_test() appends to reports/tables/prob_test_scores.csv and
refuses a model already there.
Checks on toy cases: python test_score_prob.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import LOCAL_TZ
from score import TARGETS, TEAM

CAP = 10  # count now >= CAP pooled in one cell of the calibrated persistence
K = 20  # pseudo-rows of shrinkage toward the coarser rate
EPS = 1e-6  # probability clip for the log loss of models (persistence is never given a log loss)
TEST_LOG = Path(__file__).resolve().parent.parent / "reports" / "tables" / "prob_test_scores.csv"
REF_NAMES = {"persistence": "pers", "calibrated persistence": "calpers", "climatology": "clim",
             "station-hour persistence": "sthour"}
REF_COLUMNS = ["t_local", "station_id", "h_target", *(f"{x}_now" for x in TARGETS), *(f"y_{x}" for x in TARGETS),
               *(f"m_{x}" for x in TARGETS)]


def labels(df):
    """1 if at least one bike of the type is at the station at t + h. [n, len(TARGETS)] float64"""
    return df[[f"y_{x}" for x in TARGETS]].to_numpy(np.float64)


def low_mask(df, x):
    """Rows whose count of type x now is <= 2 (the table's m_{x}, the same 'low rows' as the count model)."""
    return df[f"m_{x}"].to_numpy().astype(bool)


def hour_of_week(t_local):
    """0-167 from the local time string 'YYYY-MM-DDTHH:MM:SS-04:00' (Monday 00h = 0)."""
    t = pd.Series(t_local)
    day = pd.to_datetime(t.str[:10]).dt.dayofweek.to_numpy()
    return day * 24 + t.str[11:13].astype(int).to_numpy()


def brier(p, y):
    """Mean squared error of the probability, per target column."""
    return ((np.asarray(p, np.float64) - y) ** 2).mean(axis=0)


def log_loss(p, y, eps=EPS):
    """Mean negative log-likelihood (nats) per target column, p clipped to [eps, 1 - eps]."""
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(axis=0)


def bss(p, p_ref, y):
    """Brier skill score of p against the reference p_ref, per target column."""
    return 1 - brier(p, y) / brier(p_ref, y)


def reliability(p, y, bins=10):
    """Per equal-width bin of the forecast: rows, mean forecast, observed rate."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    t = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(rows=("y", "size"), p_mean=("p", "mean"),
                                                                    y_rate=("y", "mean"))
    return t.reset_index()


def reliability_term(p, y, bins=10):
    """Binned reliability part of the Brier score: sum_k n_k (pbar_k - ybar_k)^2 / N."""
    t = reliability(p, y, bins)
    return float((t.rows * (t.p_mean - t.y_rate) ** 2).sum() / t.rows.sum())


# ---------------------------------------------------------------- references
class Persistence:
    """p = 1[count now >= 1]. Nothing to fit."""

    def fit(self, df):
        return self

    def predict(self, df):
        return (df[[f"{x}_now" for x in TARGETS]].to_numpy() >= 1).astype(np.float64)


class CalibratedPersistence:
    """p = P(y = 1 | min(count now, CAP), h_target) from the train rows, per target, shrunk toward the count's rate.
    A (count, h) cell missing from train falls back to the count's rate over all h."""

    def fit(self, df):
        y = labels(df)
        self.table = {}
        for i, x in enumerate(TARGETS):
            c = np.minimum(df[f"{x}_now"].to_numpy(), CAP).astype(int)
            g = pd.DataFrame({"c": c, "h": df.h_target.to_numpy().astype(int), "y": y[:, i]})
            by_c = g.groupby("c").y.mean()
            cell = g.groupby(["c", "h"]).y.agg(["sum", "size"])
            prior = by_c.reindex(cell.index.get_level_values("c")).to_numpy()
            self.table[x] = ((cell["sum"] + K * prior) / (cell["size"] + K)).rename("p")
            self.table[x + "_by_c"] = by_c
            self.table[x + "_cmax"] = int(c.max())
        return self

    def predict(self, df):
        out = np.empty((len(df), len(TARGETS)))
        h = df.h_target.to_numpy().astype(int)
        for i, x in enumerate(TARGETS):  # a count above any count seen in train takes the top cell
            c = np.minimum(df[f"{x}_now"].to_numpy(), self.table[x + "_cmax"]).astype(int)
            idx = pd.MultiIndex.from_arrays([c, h], names=["c", "h"])
            p = self.table[x].reindex(idx).to_numpy().copy()
            miss = np.isnan(p)  # a (count, h) cell not in train: the count's rate over all h
            p[miss] = self.table[x + "_by_c"].reindex(c[miss]).to_numpy()
            out[:, i] = p
        assert not np.isnan(out).any(), "calibrated persistence: a count now never seen in train"
        return out


class Climatology:
    """p = station x hour-of-week rate on train, shrunk toward the station rate, shrunk toward the global rate.
    k = pseudo-rows of shrinkage (K by default; another value only for the check in 50_prob_checks.py)."""

    def __init__(self, k=K):
        self.k = k

    def fit(self, df):
        y = labels(df)
        how = hour_of_week(df.t_local)
        self.glob, self.station, self.cell = {}, {}, {}
        for i, x in enumerate(TARGETS):
            g = pd.DataFrame({"s": df.station_id.to_numpy(), "w": how, "y": y[:, i]})
            self.glob[x] = g.y.mean()
            st = g.groupby("s").y.agg(["sum", "size"])
            self.station[x] = (st["sum"] + self.k * self.glob[x]) / (st["size"] + self.k)
            cell = g.groupby(["s", "w"]).y.agg(["sum", "size"])
            prior = self.station[x].reindex(cell.index.get_level_values("s")).to_numpy()
            self.cell[x] = (cell["sum"] + self.k * prior) / (cell["size"] + self.k)
        return self

    def predict(self, df):
        out = np.empty((len(df), len(TARGETS)))
        s, how = df.station_id.to_numpy(), hour_of_week(df.t_local)
        for i, x in enumerate(TARGETS):
            p = self.cell[x].reindex(pd.MultiIndex.from_arrays([s, how])).to_numpy().copy()
            miss = np.isnan(p)  # station-hour not in train: the station rate, else (new station) the global rate
            p[miss] = self.station[x].reindex(s[miss]).fillna(self.glob[x]).to_numpy()
            out[:, i] = p
        return out


class StationHourPersistence:
    """Added after the reviews (Nov 12; val only): calibrated persistence corrected per station and hour of day.
    p = clip(p0 + r, 0, 1), with p0 = calibrated persistence (count now, h) and r = the train residual mean of
    y - p0 in the cell (min(count now, CAP), station, local hour of day), shrunk toward 0 with K pseudo-rows.
    A look-up table with no learning that knows what the screen shows plus which station and hour it is."""

    def fit(self, df):
        self.cal = CalibratedPersistence().fit(df)
        y, p0 = labels(df), self.cal.predict(df)
        hour = pd.Series(df.t_local).str[11:13].astype(int).to_numpy()
        self.cell = {}
        for i, x in enumerate(TARGETS):
            c = np.minimum(df[f"{x}_now"].to_numpy(), CAP).astype(int)
            g = pd.DataFrame({"c": c, "s": df.station_id.to_numpy(), "w": hour, "r": y[:, i] - p0[:, i]})
            cell = g.groupby(["c", "s", "w"]).r.agg(["sum", "size"])
            self.cell[x] = cell["sum"] / (cell["size"] + K)
        return self

    def predict(self, df):
        out = self.cal.predict(df)
        hour = pd.Series(df.t_local).str[11:13].astype(int).to_numpy()
        for i, x in enumerate(TARGETS):
            c = np.minimum(df[f"{x}_now"].to_numpy(), CAP).astype(int)
            idx = pd.MultiIndex.from_arrays([c, df.station_id.to_numpy(), hour])
            r = np.nan_to_num(self.cell[x].reindex(idx).to_numpy())  # cell not in train: no correction
            out[:, i] = np.clip(out[:, i] + r, 0, 1)
        return out


def fit_references(train, station_hour=False):
    """The three references of the search, fit on the train rows only; station_hour=True adds the fourth
    (StationHourPersistence, added after the reviews and reported on val only)."""
    assert set(train.split) == {"train"}, "references are fit on the train split only"
    refs = {"persistence": Persistence().fit(train), "calibrated persistence": CalibratedPersistence().fit(train),
            "climatology": Climatology().fit(train)}
    if station_hour:
        refs["station-hour persistence"] = StationHourPersistence().fit(train)
    return refs


# ---------------------------------------------------------------- tables
def prob_table(df, preds, refs, masks=None):
    """Brier, log loss, reliability term and BSS against each reference, per model, target and subset.
    preds: name -> [n, len(TARGETS)] probabilities. The references are scored as models too (log loss NaN for
    persistence). masks: extra subsets, name -> function(df, x) -> bool [n]; 'all' and 'low' are always there."""
    y = labels(df)
    ref_p = {name: r.predict(df) for name, r in refs.items()}
    subsets = {"all": lambda d, x: np.ones(len(d), bool), "low": low_mask, **(masks or {})}
    rows = []
    for i, x in enumerate(TARGETS):
        for sub, fn in subsets.items():
            m = fn(df, x)
            if m.sum() == 0:
                continue
            yy = y[m, i]
            ref_b = {name: brier(p[m, i], yy) for name, p in ref_p.items()}
            for name, p in {**ref_p, **preds}.items():
                pp = np.asarray(p)[m, i]
                b = brier(pp, yy)
                row = {"model": name, "target": x, "subset": sub, "rows": int(m.sum()), "base_rate": yy.mean(),
                       "brier": b, "log_loss": np.nan if name == "persistence" else log_loss(pp, yy),
                       "rel": reliability_term(pp, yy)}
                for rname, short in REF_NAMES.items():
                    if rname in ref_b:
                        row[f"bss_{short}"] = 1 - b / ref_b[rname]
                rows.append(row)
    return pd.DataFrame(rows)


def score_prob_test(name, pred, df, refs, note="", log=TEST_LOG):
    """The one test score of a model: scores the test rows (the references beside it, as rows with run = name),
    appends to the log, and refuses a run name already there."""
    assert set(df.split) == {"test"}, "score_prob_test takes the test rows only"
    if log.exists() and pd.read_csv(log).run.eq(name).any():
        raise RuntimeError(f"{name} already has a test score in {log}; scored once per model")
    t = prob_table(df, {name: pred}, refs)
    t = t.assign(run=name, scored_at=pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M"), note=note)
    log.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(log, mode="a", header=not log.exists(), index=False)
    return t
