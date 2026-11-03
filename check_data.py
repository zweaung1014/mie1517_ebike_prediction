# import pandas as pd
# df = pd.read_parquet("/home/zweminhtetaung/mie1517_ebike_project/data/processed/training_table.parquet")

# print(df.head(30))
# print(df.describe().T)                                   # range and mean of every column
# print(df[(df.station_id == 7000) & (df.split == "val")].head(50))   # one station
# print(df.sample(20, random_state=0))                     # random rows

import duckdb   # pip install duckdb

PROCESSED = "/home/zweminhtetaung/mie1517_ebike_project/data/processed"
STATION_ID = 7000            # Fort York Blvd / Capreol Ct
DAY = "2026-10-21"           # a Wednesday in the val split (Toronto time)
FROM_HOUR, TO_HOUR = 7, 10   # 07:00 to 09:59

# One station through a weekday morning. For every row, look up the raw reading at label_ts
# in readings_clean.parquet and rebuild the labels from it, so they can be checked by hand.
# raw_* are the raw values at label_ts; ok_* = 1 when the rebuilt label matches y_*.
morning = duckdb.sql(f"""
  SELECT
    substr(t.t_local, 12, 5) AS t_local, t.h_target, t.h,
    strftime(r.ts AT TIME ZONE 'America/Toronto', '%H:%M:%S') AS label_local,
    t.ebike_now,   t.ebike_future,   r.ebike   AS raw_ebike,   t.y_ebike,   t.m_ebike,
    t.classic_now, t.classic_future, r.classic AS raw_classic, t.y_classic, t.m_classic,
    t.docks_now,   t.docks_future,   r.docks   AS raw_docks,   t.y_dock,    t.m_dock,
    r.renting AS raw_renting, r.returning AS raw_returning,
    (t.y_ebike   = (r.ebike   >= 1 AND r.renting)::INT)::INT   AS ok_ebike,
    (t.y_classic = (r.classic >= 1 AND r.renting)::INT)::INT   AS ok_classic,
    (t.y_dock    = (r.docks   >= 1 AND r.returning)::INT)::INT AS ok_dock
  FROM '{PROCESSED}/training_table.parquet' AS t
  LEFT JOIN '{PROCESSED}/readings_clean.parquet' AS r
    ON r.station_id = t.station_id AND r.ts = CAST(t.label_ts AS TIMESTAMPTZ)
  WHERE t.station_id = {STATION_ID}
    AND substr(t.t_local, 1, 10) = '{DAY}'
    AND CAST(substr(t.t_local, 12, 2) AS INT) BETWEEN {FROM_HOUR} AND {TO_HOUR - 1}
  ORDER BY t.t, t.h_target
""").df()

print(f"Station {STATION_ID}, {DAY}, {FROM_HOUR:02d}:00-{TO_HOUR:02d}:00, {len(morning)} rows")
print(morning.to_string(index=False))
print("\nRows where a rebuilt label disagrees:",
      int((morning[["ok_ebike", "ok_classic", "ok_dock"]] != 1).any(axis=1).sum()))
