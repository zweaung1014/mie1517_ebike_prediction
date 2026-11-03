# Training table: column reference

What each column in `data/processed/training_table.parquet` (and `.csv`) means, and how to use it.

## What one row is

One pretend rider query from the past: **one station, one anchor time `t`, one horizon `h`**.

- The rider looks at the app at time `t` and asks: *"If I arrive in `h` minutes, will there be at least one classic bike / e-bike / free dock?"*
- Each anchor time gets 3 random horizons, so you'll see 3 rows with the same `station_id` and `t`, differing only in the horizon and the labels.
- Anchors are every 10 minutes. Horizons are 2–15 minutes.
- Only anchors where at least one of the three counts now is `<= 2` are kept (`LOW_COUNT_K`). Busy stations with plenty of everything are left out, because their labels are almost always 1.

## Roles at a glance

| Role | Columns | Feed to the model? |
|---|---|---|
| ID / bookkeeping | `station_id`, `t`, `t_local`, `split`, `h_target` | No |
| Embedding index | `station_idx` | Only through an embedding layer |
| Inputs | `h`, `*_now`, `*_tm10`, `*_tm30`, `capacity`, `lat`, `lon`, `hour_sin`, `hour_cos`, `is_weekend`, `temp_c`, `rel_humidity` | **Yes** |
| Labels (targets) | `y_classic`, `y_ebike`, `y_dock` | As targets only |
| Loss masks | `m_classic`, `m_ebike`, `m_dock` | No (multiply the loss by them) |
| Reference (the answer) | `label_ts`, `classic_future`, `ebike_future`, `docks_future` | **Never** |

When you build the input matrix, list the input columns explicitly. Don't use "everything except the labels".

## Columns

### ID and bookkeeping

| # | Column | Type | Meaning |
|---|---|---|---|
| 1 | `station_id` | int | Bike Share Toronto station ID. A label, not a quantity. |
| 2 | `station_idx` | int | 0, 1, 2, … in `station_id` order. Use it for a station embedding. Never feed it as a plain number. Maps to stations in `station_lookup.csv`. |
| 3 | `t` | UTC timestamp | Anchor time: the moment the rider looks at the app. On a 10-minute grid. |
| 4 | `t_local` | string | `t` in Toronto time, for reading only. |
| 5 | `split` | `train` / `val` / `test` | Split by time, never at random. Whole days belong to one split. |
| 6 | `h_target` | int, minutes | Horizon that was sampled (2–15). For checking only. |

### Inputs: horizon

| # | Column | Type | Meaning |
|---|---|---|---|
| 7 | `h` | float, minutes | Actual gap between `t` and the label reading (`label_ts - t`). Usually slightly above `h_target`, never above 15. **This is the horizon input**, not `h_target`. |

### Inputs: counts at `t` (what the app shows now)

| # | Column | Type | Meaning |
|---|---|---|---|
| 8 | `classic_now` | int | Rentable classic bikes at `t`. |
| 9 | `ebike_now` | int | Rentable e-bikes at `t`. |
| 10 | `docks_now` | int | Free docks at `t`. |
| 11 | `renting_now` | 0/1 | Station allows rentals at `t`. Almost always 1. |
| 12 | `returning_now` | 0/1 | Station allows returns at `t`. Almost always 1. |

### Inputs: recent history (lags)

The counts 10 and 30 minutes before `t`. Each comes from the latest reading at or before `t - k`, at most 6 minutes old. Together with the `*_now` columns, they show whether the station is filling or draining.

| # | Column | Type | Meaning |
|---|---|---|---|
| 13 | `classic_tm10` | int | Classic bikes at `t - 10 min`. |
| 14 | `ebike_tm10` | int | E-bikes at `t - 10 min`. |
| 15 | `docks_tm10` | int | Free docks at `t - 10 min`. |
| 16 | `classic_tm30` | int | Classic bikes at `t - 30 min`. |
| 17 | `ebike_tm30` | int | E-bikes at `t - 30 min`. |
| 18 | `docks_tm30` | int | Free docks at `t - 30 min`. |

### Inputs: station

| # | Column | Type | Meaning |
|---|---|---|---|
| 19 | `capacity` | int | Total docks at the station, at `t`. |
| 20 | `lat` | float | Latitude. |
| 21 | `lon` | float | Longitude. |

### Inputs: time of day and week (from `t`, Toronto time)

| # | Column | Type | Meaning |
|---|---|---|---|
| 22 | `hour_sin` | float | `sin(2π · hour / 24)`. Together with `hour_cos`, it places the time on a circle, so 23:50 and 00:10 are close. |
| 23 | `hour_cos` | float | `cos(2π · hour / 24)`. |
| 24 | `is_weekend` | 0/1 | 1 on Saturday or Sunday. |

### Inputs: weather (latest at or before `t`, from the station's nearest weather station)

| # | Column | Type | Meaning |
|---|---|---|---|
| 25 | `temp_c` | float, °C | Air temperature. |
| 26 | `rel_humidity` | float, % | Relative humidity. |

### Labels (what the model predicts)

Three **separate** yes/no labels, not one-hot. Any combination is possible (for example 1, 0, 1). Train them with three sigmoid outputs and binary cross-entropy.

| # | Column | Type | Meaning |
|---|---|---|---|
| 27 | `y_classic` | 0/1 | 1 if at `label_ts` there is at least one classic bike **and** the station allows rentals. |
| 28 | `y_ebike` | 0/1 | 1 if at `label_ts` there is at least one e-bike **and** the station allows rentals. |
| 29 | `y_dock` | 0/1 | 1 if at `label_ts` there is at least one free dock **and** the station allows returns. |

### Loss masks

A mask of 1 means that label is uncertain on this row (its count now is `<= 2`), so it counts in the loss. A mask of 0 means the count is high, the label is trivially 1, and it should be ignored. Every row has at least one mask equal to 1.

| # | Column | Type | Meaning |
|---|---|---|---|
| 30 | `m_classic` | 0/1 | `classic_now <= 2` |
| 31 | `m_ebike` | 0/1 | `ebike_now <= 2` |
| 32 | `m_dock` | 0/1 | `docks_now <= 2` |

In training:

```python
loss = (m * bce(p, y)).sum() / m.sum()   # m, p, y have shape (batch, 3)
```

Score each label only on rows where its mask is 1. At prediction time, a count above 2 is answered "available" by rule, without the model.

The share of rows masked in is uneven: about 92% for e-bikes, 51% for classic bikes and 14% for docks. So the dock output learns from far fewer rows.

### Reference only: the answer (never an input)

| # | Column | Type | Meaning |
|---|---|---|---|
| 33 | `label_ts` | UTC timestamp | Time of the real reading used as "the future": the first reading at or after `t + h_target`, at most 3 minutes later. |
| 34 | `classic_future` | int | Actual classic bikes at `label_ts`. |
| 35 | `ebike_future` | int | Actual e-bikes at `label_ts`. |
| 36 | `docks_future` | int | Actual free docks at `label_ts`. |

These are the raw counts the `y_*` labels were built from. They're useful for checking labels and for analysis. If one reaches the model as an input, the model reads off the answer and its scores mean nothing.

## Example

Station 7000 (Fort York Blvd / Capreol Ct) at `t = 2026-09-29 06:27 UTC` (02:27 Toronto time):

| Column | Value | Reading |
|---|---|---|
| `classic_now`, `ebike_now`, `docks_now` | 45, 0, 1 | App shows 45 classic bikes, no e-bikes, 1 free dock |
| `m_classic`, `m_ebike`, `m_dock` | 0, 1, 1 | Classic is trivial (45), so it's masked out |
| `h_target`, `h` | 4, ≈4.05 | Rider arrives in about 4 minutes |
| `label_ts` | 06:31:03 | Label reading |
| `classic_future`, `ebike_future`, `docks_future` | 45, 0, 1 | Nothing changed |
| `y_classic`, `y_ebike`, `y_dock` | 1, 0, 1 | Classic yes, e-bike no, dock yes |
