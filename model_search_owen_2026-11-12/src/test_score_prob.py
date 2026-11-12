"""Checks of score_prob.py on toy cases (no data file needed). Run from src/: python test_score_prob.py

Each check prints ok / FAIL; the script exits 1 if any fails.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from score_prob import (CAP, K, CalibratedPersistence, Climatology, Persistence, StationHourPersistence, brier, bss,
                        fit_references,
                        hour_of_week, log_loss, prob_table, reliability, reliability_term, score_prob_test)

rng = np.random.default_rng(0)


def check(name, ok, info=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {info}")
    return bool(ok)


def toy(n=40_000, split="train", seed=0):
    """Rows whose label follows a known rule: P(y=1) = 0.3 if count now is 0, 0.9 if 1, 1.0 if >= 2."""
    r = np.random.default_rng(seed)
    c = r.integers(0, 4, (n, 2))
    p = np.select([c == 0, c == 1], [0.3, 0.9], 1.0)
    y = (r.random((n, 2)) < p).astype(np.int8)
    t = pd.Timestamp("2026-10-05 08:10", tz="America/Toronto") + pd.to_timedelta(r.integers(0, 7 * 24, n), "h")
    return pd.DataFrame({"split": split, "t_local": t.strftime("%Y-%m-%dT%H:%M:%S-04:00"),
                         "station_id": r.choice(["a", "b"], n), "h_target": r.integers(2, 16, n),
                         "classic_now": c[:, 0], "ebike_now": c[:, 1], "y_classic": y[:, 0], "y_ebike": y[:, 1],
                         "m_classic": c[:, 0] <= 2, "m_ebike": c[:, 1] <= 2}), p


def labels_of(df):
    return df[["y_classic", "y_ebike"]].to_numpy(float)


def main():
    ok = True
    y = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    ok &= check("Brier of a perfect forecast is 0", np.allclose(brier(y, y), 0))
    ok &= check("Brier of 0.5 is 0.25", np.allclose(brier(np.full_like(y, 0.5), y), 0.25))
    p_hard = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])  # one miss per column
    ok &= check("Brier of a hard 0/1 forecast = share of misses", np.allclose(brier(p_hard, y), [0.25, 0.25]))
    ok &= check("BSS of the reference against itself is 0", np.allclose(bss(p_hard, p_hard, y), 0))
    ok &= check("BSS of a perfect forecast is 1", np.allclose(bss(y, p_hard, y), 1))
    ok &= check("BSS < 0 for a worse forecast", (bss(1 - y, p_hard, y) < 0).all())
    ok &= check("log loss of a hard miss is finite only through the clip",
                np.isfinite(log_loss(p_hard, y)).all() and log_loss(p_hard, y)[0] > 3)
    ok &= check("log loss of 0.5 is ln 2", np.allclose(log_loss(np.full_like(y, 0.5), y), np.log(2)))

    # reliability: a calibrated forecast has a near-zero term, a shifted one has the squared shift
    p = rng.random(200_000)
    yy = (rng.random(200_000) < p).astype(float)
    ok &= check("reliability term of a calibrated forecast ~ 0", reliability_term(p, yy) < 1e-4,
                f"({reliability_term(p, yy):.2e})")
    shifted = np.clip(p + 0.1, 0, 1)
    ok &= check("reliability term of a +0.1 shift ~ 0.01 (less near 1, where the clip bites)",
                0.006 < reliability_term(shifted, yy) < 0.011, f"({reliability_term(shifted, yy):.4f})")
    t = reliability(p, yy)
    ok &= check("reliability bins: 10 bins, rows add up", len(t) == 10 and t.rows.sum() == len(p))

    # references on toy data with a known rule
    tr, truth_p = toy()
    va, va_p = toy(20_000, "val", seed=1)
    ok &= check("hour of week: Mon 00h = 0, Sun 23h = 167",
                list(hour_of_week(["2026-10-05T00:10:00-04:00", "2026-10-11T23:50:00-04:00"])) == [0, 167])
    pers = Persistence().fit(tr).predict(va)
    ok &= check("persistence = 1[count now >= 1]", (pers == (va[["classic_now", "ebike_now"]].to_numpy() >= 1)).all())
    cal = CalibratedPersistence().fit(tr).predict(va)
    err = np.abs(cal - va_p).max()
    ok &= check("calibrated persistence recovers P(y | count now) within 0.04", err < 0.04, f"(max err {err:.3f})")
    big = va.copy()
    big["classic_now"] = 50
    big["h_target"] = 99  # a horizon never seen in train
    pb = CalibratedPersistence().fit(tr).predict(big)
    ok &= check(f"calibrated persistence: count >= CAP ({CAP}) and an unseen h fall back to the count's rate",
                np.allclose(pb[:, 0], 1.0))
    clim = Climatology().fit(tr)
    pc = clim.predict(va)
    glob = tr.y_classic.mean()
    dev = np.abs(pc[:, 0] - glob).mean()  # ~120 rows per station-hour cell, so sampling sd ~0.037 per cell
    ok &= check("climatology stays near the global rate when the label ignores station and hour",
                dev < 0.04, f"(global {glob:.3f}, mean abs dev {dev:.3f})")
    new = va.head(5).assign(station_id="new")
    ok &= check("climatology: an unseen station gets the global rate", np.allclose(clim.predict(new)[:, 0], glob))
    one = tr.head(1)
    c1 = Climatology().fit(one)
    st = (one.y_classic.iloc[0] + K * one.y_classic.iloc[0]) / (1 + K)
    ok &= check("climatology shrinkage: one row -> (y + K*station)/(1 + K)",
                np.isclose(c1.predict(one)[0, 0], st))

    # station-hour persistence: a station whose bikes always stay gets a correction toward 1
    tr2 = tr.copy()
    tr2.loc[tr2.station_id == "a", ["y_classic", "y_ebike"]] = 1
    sh = StationHourPersistence().fit(tr2)
    va2 = va.copy()
    va2.loc[va2.station_id == "a", ["y_classic", "y_ebike"]] = 1
    p_sh, p_cp = sh.predict(va2), CalibratedPersistence().fit(tr2).predict(va2)
    a_rows = (va2.station_id == "a").to_numpy()
    ok &= check("station-hour persistence: moves station a toward its rate of 1",
                (p_sh[a_rows, 0] >= p_cp[a_rows, 0] - 1e-12).all() and p_sh[a_rows, 0].mean() > p_cp[a_rows, 0].mean())
    ok &= check("station-hour persistence beats calibrated persistence when stations differ",
                brier(p_sh, labels_of(va2))[0] < brier(p_cp, labels_of(va2))[0])
    ok &= check("station-hour persistence stays in [0, 1]", (p_sh >= 0).all() and (p_sh <= 1).all())
    new2 = va.head(5).assign(station_id="new")
    ok &= check("station-hour persistence: unseen station = calibrated persistence",
                np.allclose(sh.predict(new2), CalibratedPersistence().fit(tr2).predict(new2)))
    refs = fit_references(tr)
    tab = prob_table(va, {"truth": va_p}, refs)
    a = tab[(tab.subset == "all") & (tab.target == "classic")].set_index("model")
    ok &= check("table: persistence BSS against itself is 0", np.isclose(a.loc["persistence", "bss_pers"], 0))
    ok &= check("table: the true probability beats calibrated persistence by ~0 (both are the truth)",
                abs(a.loc["truth", "bss_calpers"]) < 0.01, f"({a.loc['truth', 'bss_calpers']:.4f})")
    ok &= check("table: the true probability beats raw persistence", a.loc["truth", "bss_pers"] > 0.2,
                f"({a.loc['truth', 'bss_pers']:.3f})")
    ok &= check("table: persistence has no log loss", np.isnan(a.loc["persistence", "log_loss"]))
    try:
        fit_references(va)
        ok &= check("references refuse non-train rows", False)
    except AssertionError:
        ok &= check("references refuse non-train rows", True)

    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        te, te_p = toy(1000, "test", seed=2)
        score_prob_test("m", te_p, te, refs, log=log)
        try:
            score_prob_test("m", te_p, te, refs, log=log)
            ok &= check("test log refuses a second score of the same run", False)
        except RuntimeError:
            ok &= check("test log refuses a second score of the same run", True)
        score_prob_test("m2", te_p, te, refs, log=log)
        ok &= check("test log keeps one block per run", set(pd.read_csv(log).run) == {"m", "m2"})
    print("all ok" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
