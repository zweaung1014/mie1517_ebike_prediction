# Build plan: Bike Share Toronto training table

You are writing a Python script (`build_dataset.py`) that turns raw station snapshot CSVs into **one CSV training table** for a deep learning model. Follow this plan exactly. Where something is ambiguous, choose the option that **never lets information from after time `t` into an input column**. That is the most important rule in this whole document.

---

## 1. What the model is for (context)

- Project: MIE1517 (Intro to Deep Learning, University of Toronto). We predict Bike Share Toronto availability.
- A rider asks: *"If I arrive at station S in `h` minutes, will there be at least one classic bike / at least one e-bike / at least one free dock?"* `h` ranges from 2 to 15 minutes.
- **One model is trained on all stations.** Each row of the output table is one "pretend rider query" from the past:
  - one station,
  - one anchor time `t` (anchors every 10 minutes),
  - one horizon `h` (a few random horizons per anchor).
- Inputs describe only what was knowable at time `t`. Labels come from the real snapshot at `t + h`.
- The model outputs three probabilities (classic, e-bike, dock). Labels are **0/1** ("at least one"), not counts.
- **Only low counts are interesting.** If a station shows 20 classic bikes now, `y_classic` is 1 in 15 minutes almost every time, so that row teaches the model nothing. The table therefore keeps only anchors where at least one of `classic_now`, `ebike_now`, `docks_now` is `<= LOW_COUNT_K` (= 2). Counts of 0 are included: the model also answers "will one appear in the next 15 minutes?"
- Each row has a **mask per label** (`m_classic`, `m_ebike`, `m_dock` = 1 if that label's count now is `<= LOW_COUNT_K`). The training code uses a masked loss, so a label only counts on rows where it is uncertain (section 10).
- This script only builds the table. It does **not** normalize or scale anything. Keep values human-readable. The training code handles scaling later.

---

## 2. Input data

Several CSV files with the same columns, for example `snapshots_with_weather*.csv`. The script must accept a list of file paths or a glob pattern as a command-line argument.

Example rows:

```
ts,station_id,source,capacity,bikes,classic,ebike,docks,renting,returning,latitude,longitude,name,nearest_bike_id,datetime,temp,visibility,relative_humidity,wind_direction,wind_speed
2026-10-24 12:15:32.000Z,7302,live,11,1,1,0,10,True,True,43.674923,-79.430465,Winona Dr / Davenport Rd,31688,2026-10-24 08:15:32-04:00,8.8,,84,,
2026-10-24 12:15:32.000Z,7303,live,43,42,40,2,0,True,True,43.665269,-79.319796,Queen St E / Woodward Ave,31688,2026-10-24 08:15:32-04:00,8.8,,84,,
```

Each row is **one station at one poll time**. Polls are about every 1 minute for `source == "live"`. Stations themselves only report about every 2 minutes, so consecutive polls often repeat the same values. That is normal.

| Column | Meaning | Use |
|---|---|---|
| `ts` | Poll time, UTC | **Main time column.** Parse as tz-aware UTC. |
| `station_id` | Station ID (integer) | ID/lookup key. Never a numeric model input. |
| `source` | `live`, `history`, or `live_citybikes` | Keep for diagnostics only. |
| `capacity` | Number of docks | Input (value at `t`). |
| `bikes` | Total rentable bikes | Not used. Equals classic + ebike. |
| `classic` | Rentable classic bikes | Input and label source. |
| `ebike` | Rentable e-bikes | Input and label source. |
| `docks` | Free docks | Input and label source. |
| `renting` | Station allows rentals (bool) | Input and label logic. |
| `returning` | Station allows returns (bool) | Input and label logic. |
| `latitude`, `longitude` | Station location | Input. |
| `name` | Station name | Lookup file only. |
| `nearest_bike_id` | Unclear meaning; same value on every sample row | **Ignore.** |
| `datetime` | Same moment as `ts`, in Toronto local time | Ignore. Derive local time from `ts` instead. |
| `temp` | Air temperature, °C | Input. |
| `relative_humidity` | % | Input. |
| `visibility`, `wind_direction`, `wind_speed` | Often empty | **Ignore.** |

Not available, so do **not** create these columns: system-wide e-bike count, precipitation, station type, the station's `last_reported` time.

---

## 3. Output

1. `training_table.csv`: the main table (columns in section 5, in that exact order).
2. `station_lookup.csv`: one row per station: `station_idx, station_id, name, lat, lon, capacity_latest, first_seen_utc, last_seen_utc, n_snapshots`.
3. A printed summary report (section 8).

Also offer `--parquet` to write `training_table.parquet`, because the CSV may be several GB.

Configuration constants go at the top of the script:

```python
ANCHOR_EVERY_MIN   = 10        # anchor grid spacing
ANCHOR_TOL_MIN     = 3         # anchor must use a reading at most this old
HORIZONS           = list(range(2, 16))      # 2, 3, ..., 15
N_HORIZONS_PER_ANCHOR = 3
MAX_H_MIN          = 15        # drop rows whose actual gap h (to the label reading) exceeds this
LOW_COUNT_K        = 2         # a label is "uncertain" (masked in) when its count now is <= this
LAG_MINUTES        = [10, 30]
LAG_TOL_MIN        = 6         # lag value must come from a reading at most this old
LABEL_TOL_MIN      = 3         # label reading must be within this many minutes AFTER t+h
WEATHER_FFILL_MAX_MIN = 120
SPLIT_FRACTIONS    = (0.70, 0.15, 0.15)       # train / val / test, by time
SPLIT_DATES        = None      # optional override: ("YYYY-MM-DD", "YYYY-MM-DD") = (val_start, test_start), Toronto local
RANDOM_SEED        = 1517
LOCAL_TZ           = "America/Toronto"
```

---

## 4. Steps

### Step 1. Load and combine

- Read every file with only the needed columns: `ts, station_id, source, capacity, classic, ebike, docks, renting, returning, latitude, longitude, name, temp, relative_humidity`.
- Parse `ts` as UTC (`pd.to_datetime(..., utc=True)`).
- Convert `renting` and `returning` to bool. Values may be the strings `"True"`/`"False"`.
- Use small dtypes (`int32` for station_id, `int16` for counts, `float32` for weather and coordinates) to save memory.
- Concatenate all files.

### Step 2. Clean

Apply these steps in order, and print the number of rows removed by each:

1. Drop rows where `ts`, `station_id`, `classic`, `ebike`, or `docks` is missing.
2. Drop rows with impossible values: negative counts, `capacity <= 0`.
3. Drop stations outside Toronto: keep `43.4 <= lat <= 44.0` and `-79.8 <= lon <= -79.0`. This removes known junk entries at (0, 0) and in Montreal.
4. **Deduplicate.** If two rows have the same `station_id` and the same `ts` rounded down to the minute, keep one: prefer `source == "live"`, then the first one.
5. Sort by `station_id`, then `ts`.

Do **not** drop rows where `renting`/`returning` is False here. That is handled when building anchors and labels.

### Step 3. Weather series

Weather is the same for every station at a given time. Build one city-wide weather series:

- Take `ts, temp, relative_humidity`, group by `ts` (take the first non-null value), sort by time.
- Forward-fill missing values, but only from a reading at most `WEATHER_FFILL_MAX_MIN` old. **Never back-fill**, because that would copy future weather into the past.
- Later, attach weather to each anchor with `pd.merge_asof(direction="backward")` on time, so an anchor only gets weather from at or before `t`.

### Step 4. Anchors (the times `t`)

For each station, create anchor rows on a 10-minute grid:

- Build the grid of times `g` from the first to the last `ts` in the data, every `ANCHOR_EVERY_MIN` minutes (`floor` to the minute, in UTC).
- For each station and each grid time `g`, take the **latest reading at or before `g`**, only if it is at most `ANCHOR_TOL_MIN` minutes old. Use `pd.merge_asof(left=grid × stations, right=readings, on time, by="station_id", direction="backward", tolerance=pd.Timedelta(minutes=ANCHOR_TOL_MIN))`.
- If no reading is found, there is no anchor for that station at that grid time (the poller had a gap). Drop it.
- **Set `t` to the grid time `g`**, and take the "now" values from that reading: `classic_now, ebike_now, docks_now, renting_now, returning_now, capacity, lat, lon`.
- Drop anchors where `renting_now == False and returning_now == False` (the station is out of service).
- **Low-count filter.** Keep an anchor only if `classic_now <= LOW_COUNT_K or ebike_now <= LOW_COUNT_K or docks_now <= LOW_COUNT_K`. Drop the rest and report how many. Do this here, before the lag and label lookups, to save work. This filters on values at `t` only, so it cannot leak the future. **Never filter on `*_future`.**

Tip: building `grid × stations` creates every pair, which may be large. It is fine to do this one station at a time, or in batches of stations, and concatenate the results.

### Step 5. Lag features (recent history)

For each lag `k` in `LAG_MINUTES` (10 and 30):

- Target time = `t - k minutes`.
- Take the latest reading for the same station **at or before** that target time, at most `LAG_TOL_MIN` minutes old (`merge_asof`, `direction="backward"`, `by="station_id"`).
- Create `classic_tm{k}`, `ebike_tm{k}`, `docks_tm{k}`.
- If no reading is found, the values are missing. **Drop that anchor** for this first version, and report how many were dropped. (Do not interpolate. Do not use readings from after the target time.)

### Step 6. Horizons and labels

For each anchor:

1. Randomly choose `N_HORIZONS_PER_ANCHOR` distinct values from `HORIZONS`, using `np.random.default_rng(RANDOM_SEED)`. This makes one output row per chosen horizon, called `h_target`.
2. Target time = `t + h_target minutes`.
3. Find the **first reading for the same station at or after the target time**, at most `LABEL_TOL_MIN` minutes later (`merge_asof`, `direction="forward"`, `by="station_id"`, `tolerance=LABEL_TOL_MIN`).
   - Use `direction="forward"`, **not** `"nearest"`. With "nearest", a small `h` can pick a reading from before or at `t`, which would make the label equal to the input.
4. If no reading is found, **drop the row**. Never fill or interpolate a label.
5. Record:
   - `label_ts` = that reading's `ts`
   - `h` = `(label_ts - t)` in minutes, as a float rounded to 0.1. This is the true gap and is what the model uses as its horizon input.
   - `classic_future, ebike_future, docks_future, renting_future, returning_future`
6. **Drop rows with `h > MAX_H_MIN`** and report how many. The label reading may come up to `LABEL_TOL_MIN` after `t + h_target`, so without this a 15-minute target could give a gap of up to 18 minutes.
7. Labels (integers 0/1):
   - `y_classic = int(classic_future >= 1 and renting_future)`
   - `y_ebike   = int(ebike_future   >= 1 and renting_future)`
   - `y_dock    = int(docks_future   >= 1 and returning_future)`
8. Masks (integers 0/1), from the values at `t`:
   - `m_classic = int(classic_now <= LOW_COUNT_K)`
   - `m_ebike   = int(ebike_now   <= LOW_COUNT_K)`
   - `m_dock    = int(docks_now   <= LOW_COUNT_K)`

   Because of the low-count filter in Step 4, every row has at least one mask equal to 1.

### Step 7. Time features (from `t`, in Toronto local time)

Convert `t` to `LOCAL_TZ`. Then:

- `hour_frac = hour + minute / 60` (do not output this column; it is only used to compute the next two)
- `hour_sin = sin(2π · hour_frac / 24)`
- `hour_cos = cos(2π · hour_frac / 24)`
- `is_weekend = 1` if local day is Saturday or Sunday, else 0
- `t_local` = local time as a string, for humans

### Step 8. Weather features

Attach to each anchor, from the weather series in Step 3, using `merge_asof(direction="backward")` on `t`:

- `temp_c` (from `temp`)
- `rel_humidity` (from `relative_humidity`)

Drop rows where weather is still missing, and report how many.

### Step 9. Station index

- Assign `station_idx` = 0, 1, 2, … by sorting unique `station_id`s ascending.
- Add `station_idx` to the table and write `station_lookup.csv`.
- `station_idx` exists so the training code can use a station embedding later. It must **not** be used as a plain numeric input.

### Step 10. Train / validation / test split, by time (never random)

- If `SPLIT_DATES` is given, use those boundaries. Otherwise split the anchor times by `SPLIT_FRACTIONS` and round each boundary down to midnight local time, so that each day belongs to exactly one split.
- `split = "train"` if `t < val_start`, `"val"` if `val_start <= t < test_start`, otherwise `"test"`.
- **Purge rows near the boundary.** Drop any `train` row whose `label_ts >= val_start`, and any `val` row whose `label_ts >= test_start`. Otherwise a training label would come from inside the validation period.
- Why not random: rows 10 minutes apart at the same station are nearly identical, and different rows can share the same label snapshot. A random split would leak test answers into training and make the score look better than it is.

---

## 5. Output columns (exact order)

| # | Column | Type | Role |
|---|---|---|---|
| 1 | `station_id` | int | ID (not a model input) |
| 2 | `station_idx` | int | ID / embedding index (not a plain numeric input) |
| 3 | `t` | ISO UTC string | ID |
| 4 | `t_local` | string | ID (for humans) |
| 5 | `split` | `train`/`val`/`test` | ID |
| 6 | `h_target` | int | sampled horizon (diagnostic) |
| 7 | `h` | float, minutes | **input**: actual gap to the label reading |
| 8 | `classic_now` | int | input |
| 9 | `ebike_now` | int | input |
| 10 | `docks_now` | int | input |
| 11 | `renting_now` | 0/1 | input |
| 12 | `returning_now` | 0/1 | input |
| 13 | `classic_tm10` | int | input |
| 14 | `ebike_tm10` | int | input |
| 15 | `docks_tm10` | int | input |
| 16 | `classic_tm30` | int | input |
| 17 | `ebike_tm30` | int | input |
| 18 | `docks_tm30` | int | input |
| 19 | `capacity` | int | input (value at `t`) |
| 20 | `lat` | float | input |
| 21 | `lon` | float | input |
| 22 | `hour_sin` | float | input |
| 23 | `hour_cos` | float | input |
| 24 | `is_weekend` | 0/1 | input |
| 25 | `temp_c` | float | input |
| 26 | `rel_humidity` | float | input |
| 27 | `y_classic` | 0/1 | **label** |
| 28 | `y_ebike` | 0/1 | **label** |
| 29 | `y_dock` | 0/1 | **label** |
| 30 | `m_classic` | 0/1 | **loss mask** for `y_classic` |
| 31 | `m_ebike` | 0/1 | **loss mask** for `y_ebike` |
| 32 | `m_dock` | 0/1 | **loss mask** for `y_dock` |
| 33 | `label_ts` | ISO UTC string | reference only, never an input |
| 34 | `classic_future` | int | reference only, never an input |
| 35 | `ebike_future` | int | reference only, never an input |
| 36 | `docks_future` | int | reference only, never an input |

Columns 33–36 contain the answer. Add a comment at the top of the CSV-writing code, and a line in the summary, saying they must never be fed to the model.

---

## 6. Worked example (expected behaviour)

Readings for station 7000 (UTC):

| ts | classic | ebike | docks |
|---|---|---|---|
| 07:30:20 | 35 | 3 | 9 |
| 07:50:15 | 37 | 2 | 8 |
| 07:59:40 | 38 | 1 | 8 |
| 08:04:10 | 38 | 1 | 8 |
| 08:12:30 | 38 | 1 | 8 |
| 08:14:20 | 39 | 0 | 8 |

Anchor grid time `g = 08:00:00`:

- Latest reading at or before 08:00 → 07:59:40 (20 s old ≤ 3 min) ✓. `t = 08:00`, now = (38, 1, 8).
- Low-count filter: `ebike_now = 1 <= 2` ✓, so the anchor is kept. Masks: `m_classic = 0` (38 bikes), `m_ebike = 1`, `m_dock = 0` (8 docks).
- Lag 10: target 07:50:00 → latest at or before is 07:30:20, which is 19.7 min old > 6 min ✗. **The whole anchor is dropped.** (If a 07:49:55 reading existed, it would be used instead.)

Suppose the lags had been found and the sampled horizons are 4, 12, 14:

| h_target | target time | first reading ≥ target, within 3 min | h | ebike_future | y_ebike |
|---|---|---|---|---|---|
| 4 | 08:04:00 | 08:04:10 | 4.2 | 1 | 1 |
| 12 | 08:12:00 | 08:12:30 | 12.5 | 1 | 1 |
| 14 | 08:14:00 | 08:14:20 | 14.3 | 0 | 0 |

All three rows keep `m_ebike = 1`, `m_classic = 0`, `m_dock = 0`. If the 14-minute reading had instead arrived at 08:15:40, `h = 15.7 > 15` and that row would be dropped.

---

## 7. Assertions (the script must check these and fail loudly)

- Every lag reading time ≤ `t - k`; every anchor reading time ≤ `t`.
- Every `label_ts > t`, and `2 - 0.01 <= h <= MAX_H_MIN`.
- `y_*` and `m_*` values are only 0 or 1.
- `m_* == (*_now <= LOW_COUNT_K)` for each label, and every row has at least one `m_* == 1`.
- No duplicate `(station_id, t, h_target)`.
- All `train` times are before all `val` times, which are before all `test` times; no `train` `label_ts` ≥ `val_start`.
- No missing values in columns 1–32.

---

## 8. Summary report to print

- Rows read, rows after each cleaning step, anchors created, anchors dropped (out of service / no low count / no lag / no weather), rows dropped for missing labels, rows dropped for `h > MAX_H_MIN`.
- Number of stations; date range in Toronto local time.
- Rows per split, with each split's date range.
- Per split and per label: number of masked-in rows (`m_* == 1`), and the mean of `y_*` **over those rows only**.
- Per label, on masked-in rows: the share of rows by `*_now` value (0, 1, 2), and the mean of `y_*` for each value. This shows how often a count of 0 fills and a count of 1–2 drains.
- **Sanity baseline:** the "app" prediction is `p = 1 if *_now >= 1 else 0`. Print its Brier score `mean((p - y)^2)` for each label on the `val` split, **over masked-in rows only**. The old reference of about 0.09 was for all rows at about 30 minutes and does not apply here. A value near 0 suggests a bug (for example, future data leaking into inputs).

---

## 9. Implementation notes

- Use `pd.merge_asof` for every time lookup. It needs both frames sorted by the time column; use `by="station_id"`.
- Keep all times tz-aware UTC internally. Only convert to local time for the time features and `t_local`.
- Process station by station (or in batches) if memory is a problem.
- Use type hints, a `main()` with `argparse` (`--inputs`, `--out-dir`, `--parquet`), and log progress.
- Do not add features that are not listed here. Do not scale or normalize.

---

## 10. How the training code uses the masks (for reference, not part of this script)

- **Loss.** Masked binary cross-entropy over the three outputs:
  `loss = sum_k sum_rows m_k · BCE(p_k, y_k) / sum_k sum_rows m_k`, for `k` in {classic, ebike, dock}.
  A label with `m_k = 0` adds nothing to the loss or its gradient on that row.
- **Evaluation.** Score each label only on its masked-in rows, against the app baseline from section 8, on the same rows.
- **Inference.** If a label's count now is `> LOW_COUNT_K`, the system answers by rule (the app's answer: available). Otherwise it uses the model. When reporting whole-system numbers, combine the rule on high-count queries with the model on low-count queries.
