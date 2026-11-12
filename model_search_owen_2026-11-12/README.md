# Model search: architectures and hyperparameters for the bike-count forecast

**Model search by Owen, 12 Nov 2026. If something doesn't run or a number doesn't match, contact Owen.**

## What this is

This folder is a search over models that predict the count of classic bikes and e-bikes at a station h = 2–15 min ahead. It uses one model for all stations, with h as an input. Models were chosen on the val split only. The test split was then read once, under a protocol fixed before the run.

- **Score:** skill = 1 − MSE_model / MSE_persistence on the same rows. Persistence means doing nothing: the count now, carried forward. 0 means no better than doing nothing, and 1 means perfect.
- **Searched:** MLP widths and depths, the station embedding size, a GRU over the lag counts, and LightGBM as a non-deep reference. Then learning rate, batch size, weight decay and dropout (0, 0.1, 0.2, 0.3, 0.5).
- **Result:** the MLP 2 × 256 with dropout 0.1. Mean ± sd over seeds 0–2, classic / e-bike:
  - val 0.109 ± 0.001 / 0.109 ± 0.001, against the 2 × 64 base at 0.103 ± 0.003 / 0.102 ± 0.003;
  - test 0.080 ± 0.004 / 0.095 ± 0.002, against the base at 0.079 ± 0.001 / 0.082 ± 0.005.

Read first:

- `reports/MODEL_SEARCH.md`: every step, the noise analysis, the design choice, limits and open items.
- `reports/MEETING_2026-11-12.pdf`: the short deck for the team meeting (3 slides with build pages); `MEETING_2026-11-12_long.pdf` (7 pages) and `MEETING_2026-11-12.md` give more detail, including the test result and the training-set size check.
- `notebooks/model_search_trace.ipynb` (+ `.html`): prints every number of `MODEL_SEARCH.md` from its CSV, with the script and commit. It also checks the data file's SHA-1, and shows that the test split is never read during the search.

## Data it expects

The data is **data-v1 with the weather fix**: `data/processed/training_table.parquet`, built by this repo's `notebooks/build_dataset.ipynb` at `main` 969473f.

- SHA-1: `263bb791129bccf2073b1c4e827acf82eafeb96e`, 11,357,122 rows.
- Splits from its `split` column: train to Oct 18 (7,525,845 rows), val Oct 19–23 (2,083,617), test from Oct 24 (1,747,660).
- No raw data or model weights are in this folder.

Two environment variables control the paths (see `src/common.py`):

```bash
export TEAM_TABLE=/path/to/training_table.parquet   # default: <repo root>/data/processed/training_table.parquet
export MIE1517_DATA=/path/to/work                    # saved weights and predictions; default: ./work (git-ignored)
```

To check the file before running:

```bash
shasum "$TEAM_TABLE"
```

## Setup

```bash
pip install -r requirements.txt
cd src
```

All runs used CPU with 6 threads (`--threads`). Runs are deterministic on CPU: a retrain gives the logged val skill to 1e-6.

## Reproduce each step (from `src/`)

| Step | Command | Writes |
|---|---|---|
| Cheap baselines (val) | `python 34_baselines.py --data team` | `reports/tables/baselines_team_val.csv` |
| Nov 10 base MLP (val) | `python 33_train_mlp.py --data team --seed 0` (also 1, 2) | `mlp_team[_seedN]_val.csv`, `_curve.csv` |
| Step 1, architectures (4 epochs, seeds 0–1) | `python 37_model_search.py --step 1 --seed 0 1` and `python 37_model_search.py --lgbm --seed 0 1 2` | rows in `model_search_runs.csv`, `_curves.csv`, `_by_horizon.csv` |
| Step 2, hyperparameters (4 epochs) | `python 37_model_search.py --step 2 --seed 0 1` | the same logs |
| Steps 3–5, full budget, 3 seeds | `python 37_model_search.py --run mlp_2x256_do0.1 --tag 3 --budget full --seed 0 1 2` (one name per config; the names are in `GRIDS`) | the same logs |
| Summaries and search figures | `python 38_search_report.py --finalists mlp_2x64_e8 mlp_2x64_do0.1 mlp_2x256_e8 mlp_2x256_do0.1 mlp_2x256_do0.3 gru64_3x256_do0.1 lgbm_l127` | `model_search_summary.csv`, `model_search_by_horizon_summary.csv`, `figures/search_*.png` |
| Saved val predictions (for 39 and 42) | `python 37_model_search.py --save --run mlp_2x64_e8 mlp_2x256_do0.1 --tag 3 --budget full --seed 0 1 2`, the same for `mlp_2x256_do0.3 --tag 5`, and `--save --lgbm --seed 0 1 2` | `$MIE1517_DATA/models/search/*.npy, *.pt`; stops if the val skill differs from the log |
| Val-sampling noise | `python 39_val_noise.py` | `val_noise_gap.csv`, `val_noise_by_day.csv`, `val_noise_optimism.csv` |
| Meeting figures | `python 40_meeting_figures.py` | `figures/meeting_a–d_*.png` |
| Trace notebook | `python 41_trace_notebook.py` | `notebooks/model_search_trace.ipynb`, `.html` |
| Test score (once) | `python 42_test_score.py --dry` checks the val skill of every model without reading test. Without `--dry` it scores test, but `score.score_test` refuses any model already in `test_scores.csv`. | `test_scores.csv`, `test_team_*.csv` |
| Training-set size (Step 6) | `python 37_model_search.py --run mlp_2x256_do0.1 --tag 6 --budget full --seed 0 1 2 --train-days 5` (also 10, 15; and `mlp_2x256_do0.3`), then `python 44_data_size.py` | rows `*_daysK` in the logs; `data_size_*.csv`, `figures/data_size.png` |
| Meeting PDFs | `python 45_meeting_deck.py` (short deck) and `python 43_meeting_pdf.py` (long) | `reports/MEETING_2026-11-12.pdf`, `MEETING_2026-11-12_long.pdf`, `figures/deck_*.png`, `meeting_e_test.png` |

`37_model_search.py` appends to the logs in `reports/tables/`. To try a run without changing them, work on a copy of this folder. A quick check is `python 37_model_search.py --run mlp_2x64_e8 --tag 1 --seed 0`: 4 epochs, about 1–2 min on CPU. It should give the logged row `s1_mlp_2x64_e8_short_seed0`, val skill 0.0912 / 0.0933.

## File map

```
src/common.py          paths (TEAM_TABLE, MIE1517_DATA), time zone
src/score.py           shared split rule, skill vs persistence, the one-time test log
src/33_train_mlp.py    the table's 20 inputs (team_features), tensors, the Nov 10 MLP
  └─ src/34_baselines.py      cheap baselines (station x hour [x weekend] mean, ridge)
  └─ src/37_model_search.py   every search run (MLP, GRU, LightGBM); --save retrains with saved predictions; --train-days
       └─ reports/tables/model_search_*.csv   run log, curves, skill per horizon
            ├─ src/38_search_report.py   summaries + search figures
            ├─ src/39_val_noise.py       per-day skill, station/day bootstrap, early-stopping optimism
            ├─ src/40_meeting_figures.py figures (a)-(d)
            ├─ src/41_trace_notebook.py  trace notebook
            ├─ src/43_meeting_pdf.py     long meeting PDF
            ├─ src/44_data_size.py       training-set size (Step 6)
            └─ src/45_meeting_deck.py    short meeting deck
  └─ src/42_test_score.py     the test score (protocol in its docstring)
reports/   MODEL_SEARCH.md, MEETING_2026-11-12.md / .pdf, tables/*.csv, figures/*.png
notebooks/ model_search_trace.ipynb, model_search_trace.html
```

Not included: raw data, the training table and model weights. The weights can be rebuilt with `37_model_search.py --save`.
