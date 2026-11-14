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
- `reports/MEETING_2026-11-12.pdf`: the team-meeting deck (3 slides plus Backup); `MEETING_2026-11-12_builds.pdf` reveals the same story step by step; `MEETING_2026-11-12_long.pdf` (7 pages) and `MEETING_2026-11-12.md` give more detail, including the test result and the training-set size check.
- `notebooks/model_search_trace.ipynb` (+ `.html`): prints every number of `MODEL_SEARCH.md` from its CSV, with the script and commit. It also checks the data file's SHA-1, and shows that the test split is never read during the search.

## Probability output (added 12 Nov)

The same data, split, inputs and horizons, with a new output: the probability that at least one bike of each
type (classic, e-bike) is at the station at t + h, trained with binary cross-entropy. Details:
`reports/MODEL_SEARCH_PROB.md`; the progress-report notebook is `notebooks/progress_report_prob.ipynb` (+ `.html`). Key findings (13 Nov), grouped and ready to paste into the team notebook: `notebooks/key_findings.ipynb`.

- **Score:** Brier skill score BSS = 1 − Brier_model / Brier_ref against three references fit on train:
  persistence (1 if a bike is there now), calibrated persistence (P(bike at t + h | count now, h)) and a
  station × hour-of-week climatology (`src/score_prob.py`, toy checks in `src/test_score_prob.py`).
  Log loss against persistence is infinite (0/1 forecasts), so it is not used.
- **Searched:** the base MLP 2 × 64, MLP 2 × 256 with dropout 0.1 and 0.3, the GRU with a 3 × 256 head
  (seeds 0–2 each), LightGBM classifier (seeds 0–2) and logistic regression as references.
- **Result:** MLP 2 × 256, dropout 0.3, chosen on val by a rule fixed before any run. BSS against calibrated
  persistence, classic / e-bike: val 0.154 / 0.117, test 0.125 / 0.092 (read once, chosen model only).
  Against raw persistence: val 0.205 / 0.174, test 0.172 / 0.144. A look-up table that also knows the station and
  the hour (station-hour persistence, added after review, val only) gets 0.115 / 0.085 of it; the network beats
  that table by 0.044 / 0.035. All on the table's anchors where some count is 2 or less.

| Step | Command (from `src/`) | Writes |
|---|---|---|
| Score checks | `python test_score_prob.py` | prints 29 checks |
| Search | `python 46_prob_search.py --logreg --lgbm --seed 0 1 2` and `python 46_prob_search.py --run mlp_2x64_e8 mlp_2x256_do0.1 mlp_2x256_do0.3 gru64_3x256_do0.1 --seed 0 1 2` (`--smoke` for a 1-epoch trial that writes to a temp folder) | `prob_runs.csv`, `prob_curves.csv`; `$MIE1517_DATA/models/prob/` |
| Val report and choice | `python 47_prob_report.py` | `prob_summary.csv`, `prob_choice.json`, `prob_noise.csv`, `prob_by_*.csv`, `prob_last_bike.csv`, `prob_reliability.csv`, `figures/prob_*.png` |
| Test (once) | `python 48_prob_test.py --dry` checks the val reproduction without reading test; without `--dry` it refuses runs already in `prob_test_scores.csv` | `prob_test_*.csv` |
| Checks after the reviews | `python 46_prob_search.py --run mlp_2x64_e8 --patience 6 --seed 0 1 2 --threads 2`, `python 46_prob_search.py --lgbm-ablation all no_station_idx no_station_info no_lags counts_now_h counts_lags_h --seed 0`, `python 50_prob_checks.py` | `prob_extra_runs.csv`, `prob_ablation.csv`, `prob_patience_check.csv`, `prob_train_*.csv`, `prob_flags.csv`, `prob_climatology_check.csv`, `prob_sample_rows.csv`, `figures/prob_station_day.png` |
| Notebook | `python 49_progress_notebook.py` | `notebooks/progress_report_prob.ipynb`, `.html` |

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
| Meeting PDFs | `python 45_meeting_deck.py` (deck and builds) and `python 43_meeting_pdf.py` (long) | `reports/MEETING_2026-11-12.pdf`, `_builds.pdf`, `_long.pdf`, `figures/deck_*.png`, `meeting_e_test.png` |

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
src/score_prob.py      probability score: Brier, BSS, references, calibration, one-time test log
  └─ src/46_prob_search.py    probability models (BCE), LightGBM classifier, logistic regression
       └─ src/47_prob_report.py    val summary, choice, noise, where it wins, calibration
       └─ src/48_prob_test.py      the one test score of the chosen probability model
       └─ src/49_progress_notebook.py  progress-report notebook
       └─ src/50_prob_checks.py    checks after the reviews (patience, day types, flags, climatology, sample)
reports/   MODEL_SEARCH.md, MODEL_SEARCH_PROB.md, MEETING_2026-11-12.md / .pdf, tables/*.csv, figures/*.png
notebooks/ model_search_trace.ipynb / .html, progress_report_prob.ipynb / .html, key_findings.ipynb
```

Not included: raw data, the training table and model weights. The weights can be rebuilt with `37_model_search.py --save`.
