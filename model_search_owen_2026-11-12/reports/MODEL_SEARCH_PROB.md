# Probability of at least one bike: models and score

**Result.** The chosen model is the MLP 2 × 256 with dropout 0.3, picked on val by a rule fixed before any run.
All numbers are on the team table's station-times, where some count (classic bikes, e-bikes or free docks) is 2 or less.
Against calibrated persistence (an honest "what you see now" probability) its Brier skill score is 0.154 (classic) and 0.117 (e-bike) on val, and 0.125 / 0.092 on test.
Against raw persistence it is 0.205 / 0.174 (val) and 0.172 / 0.144 (test).
Mean over seeds 0–2; the seed sd is 0.0002–0.0006, but the val skill varies more from day to day (0.146–0.163 / 0.112–0.121; `prob_by_day.csv`).
Sources: `prob_summary.csv` (val, `src/47_prob_report.py`) and `prob_test_summary.csv` (test, `src/48_prob_test.py`).

**Most of the gain is knowing the station and the hour.** A look-up table with no learning, calibrated persistence corrected per station, hour of day and count now (station-hour persistence, added after the reviews), scores 0.115 / 0.085 against calibrated persistence on val.
The network beats that table by 0.044 / 0.035.
The table was added after the test read, so it is reported on val only.
Calibrated persistence stays the headline reference: it was fixed before the runs, uses only what the screen shows, and has a test score.

What the search shows (val):

1. **The station drives the gain; the architecture hardly matters.** LightGBM, the GRU and the three MLPs all reach 0.145–0.154 (classic) and 0.114–0.117 (e-bike).
   A LightGBM ablation (seed 0, `prob_ablation.csv`, `46_prob_search.py --lgbm-ablation`) gives: counts now and h only 0.011 / 0.006; plus the lag counts 0.080 / 0.061; all inputs without any station information (no station index, location or capacity) 0.090 / 0.071; location and capacity but no station index 0.148 / 0.112; the station but no lag counts 0.150 / 0.113; everything 0.151 / 0.114.
   So the station (its location is enough) is the largest part, and once the station is known the lag counts add little.
   Logistic regression, with neither the station nor a non-linear use of the inputs, only matches calibrated persistence (−0.000 / +0.002).
2. **Skill is highest when one bike is left** (0.182 / 0.159 at 1 bike now, 0.038 / 0.039 at 6 or more; `prob_by_count.csv`; on test 0.145 / 0.123 at 1 bike, `prob_test_by_count.csv`) **and grows with the horizon** (0.079 / 0.058 at h = 2 min, 0.177 / 0.143 at h = 14 min; `prob_by_horizon.csv`).
   In total Brier reduction, rows with 0 bikes now matter as much for classic bikes: 46 % (0 bikes) and 47 % (1 bike) of the total for classic, 37 % and 59 % for e-bikes (computed from `prob_by_count.csv`).
3. **In rider terms (val only):** with 1 bike now, the bike is gone at t + h in 7.6 % (classic) / 9.2 % (e-bike) of rows.
   Flagging the 10 % of these rows with the lowest forecast catches 48 % / 41 % of the gone cases, and 36 % / 37 % of the flags are right (about 2 in 3 are false alarms).
   Ranking within each horizon gives 46 % / 38 %, so it is not just the long horizons.
   Calibrated persistence catches 13 % / 14 %, the station-hour look-up 43 % / 36 %; the base MLP and LightGBM catch about as much as the chosen model (`prob_last_bike.csv`, random tie-breaks).
4. **Calibration:** every val bin with at least 1,000 rows is within 0.010 / 0.015 of the diagonal (seed 0; the epoch was picked on val, so this is a best case).
   On test the error is larger and systematic: classic forecasts of 0.1–0.4 are 0.012–0.029 too high, e-bike forecasts of 0.4–0.7 are 0.018–0.022 too low; calibrated persistence stays within 0.006 / 0.005 (`prob_reliability.csv`, `prob_test_reliability.csv`).
5. **Test is lower than val, and about a third of the drop is the weekend.** Test has a weekend, val has none.
   On test, Sat–Sun pooled give 0.114 / 0.081 and Mon–Tue 0.134 / 0.102 (`prob_test_by_day.csv`); leaving the weekend out closes 29 % / 39 % of the gap to val.
   Weekends are harder in sample too (train days, seed 0: weekdays 0.166, Saturdays 0.134, Sundays 0.128, classic; `prob_train_daytype.csv`, `src/50_prob_checks.py`).
   The test weekdays are still below every val day. The test week has fewer flips (2.2 % against 2.6 %, classic), and the model's own Brier is lower on test (0.0186 against 0.0210): the reference improved more than the model.
   Calmer days tend to score lower, but not all of the gap is that: the test Monday has the same classic flip rate as the val Monday (2.51 %) and still scores 0.140 against 0.149 (`prob_by_day.csv`, `prob_test_by_day.csv`).
   Untested: how much is the calmer week and how much the distance from training (test days are 6–10 days after it, val days 1–5).

## Setup

The output is the probability that the station has at least one bike of each type (classic, e-bike) at t + h, with h = 2–15 min.
It replaces the count forecast trained with MSE (`MODEL_SEARCH.md`).
Everything else is unchanged: data-v1 with the weather fix (SHA-1 263bb791…, 11,357,122 rows), the split (train to Oct 18, val Oct 19–23, test from Oct 24), the station embedding and one model for all horizons.
The 20 inputs of the table include two station-status flags, `renting_now` and `returning_now` ("the station accepts rentals / returns"), which are 1 on every train and val row (`prob_flags.csv`); after standardisation they are constant, so 18 inputs vary.
The table draws 3 random horizons per anchor; `h_target` is the whole minute of the real horizon, and h = 15 holds only 0.8 % of the rows.
The label is the table's `y_classic` / `y_ebike`, which equals 1[count at t + h ≥ 1] on every row.
The loss is binary cross-entropy on two logits; the forecast is p = sigmoid(logit).
Training: Adam (lr 10⁻³, batch 4,096), up to 30 epochs, keep the epoch with the lowest val cross-entropy, stop after 3 epochs without improvement.

## Score

Persistence ("available at t + h if available now") gives a hard 0 or 1, so a log-loss skill against it is infinite on every miss.
The Brier score is finite for 0/1 forecasts and is a proper score, so it is the yardstick (`src/score_prob.py`, 29 toy checks in `src/test_score_prob.py`):

- Brier = mean (p − y)²; persistence's Brier is the share of rows whose status flips between t and t + h.
- BSS_ref = 1 − Brier_model / Brier_ref: 0 = no better than the reference, 1 = perfect.

References, all fit on the train split only:

| Reference | Forecast | What it stands for |
|---|---|---|
| Persistence | 1[count now ≥ 1] | doing nothing; the yardstick of the count model |
| Calibrated persistence (headline) | P(y = 1 \| count now, h) from train, count capped at 10, shrunk toward the count's rate (20 pseudo-rows) | persistence made into an honest probability |
| Station-hour persistence (after the reviews; val only) | calibrated persistence + the train mean of (y − it) per (count now, station, hour of day), shrunk toward 0 (20 pseudo-rows) | the strongest look-up without learning |
| Climatology | the station's train rate in that hour of the week, shrunk toward the station rate, then the global rate (20 pseudo-rows each) | habit without the count now |

Climatology is a weak reference (BSS against persistence −4.51 / −3.93 on val) and is shown for completeness.
It is also miscalibrated (off by up to 0.24 for e-bikes, all rows) because its cells hold only 2.7 train days on average: with 200 pseudo-rows instead of 20 the e-bike gap falls to 0.08, while its Brier falls only from 0.210 to 0.198 (`prob_climatology_check.csv`). Its high Brier comes from ignoring the count now.

Calibration: 10 equal-width bins of the forecast against the observed rate (reliability plot), and the binned reliability term of the Murphy decomposition, Σ n_k (p̄_k − ȳ_k)² / N.

## Models

A small search, not the full grid (`src/46_prob_search.py`):

| Name | Model | Seeds |
|---|---|---|
| `mlp_2x64_e8` | the Nov 10 base MLP, 2 × 64 | 0–2 |
| `mlp_2x256_do0.1` | the count-search winner, 2 × 256, dropout 0.1 | 0–2 |
| `mlp_2x256_do0.3` | 2 × 256, dropout 0.3 | 0–2 |
| `gru64_3x256_do0.1` | GRU (64 units) over the counts at t − 30, t − 10 and t, then a 3 × 256 head with dropout 0.1 | 0–2 |
| `lgbm` | LightGBM binary classifier per target, the count search's settings, early stopping on val log loss | 0–2 |
| `logreg` | logistic regression per target on the 20 standardised inputs (no station) | 1 (deterministic) |

The candidates come from the count search, which was also chosen on these val days.

**Selection rule (fixed before any run, in the docstring of `46_prob_search.py`, commit 62b0d5d of the working repo):** among the four nets, the lowest val Brier averaged over the two targets (all rows, mean over seeds).
A net with fewer parameters within 2 × √(sd_a²/3 + sd_b²/3) of it is preferred.
LightGBM and logistic regression are references and are not chosen.
Only the chosen model is scored on test (`src/48_prob_test.py`, protocol committed before the read in bfc0d93 of the working repo).

## Results (val, Oct 19–23)

BSS against calibrated persistence and against station-hour persistence, all rows, mean ± sd over seeds (`prob_summary.csv`):

| Model | Params | Best epoch, seeds 0 / 1 / 2 | Classic | E-bike | vs station-hour | vs persistence |
|---|---|---|---|---|---|---|
| Persistence | – | – | −0.065 | −0.070 | −0.203 / −0.169 | 0 / 0 |
| Climatology | – | – | −4.87 | −4.27 | −5.63 / −4.76 | −4.51 / −3.93 |
| Calibrated persistence | – | – | 0 | 0 | −0.130 / −0.093 | 0.061 / 0.065 |
| Station-hour persistence | – | – | 0.115 | 0.085 | 0 / 0 | 0.169 / 0.145 |
| Logistic regression | 42 | – | −0.000 | 0.002 | −0.131 / −0.091 | 0.060 / 0.067 |
| LightGBM | – | 80–119 trees | 0.151 ± 0.0004 | 0.114 ± 0.0005 | 0.040 / 0.031 | 0.202 / 0.171 |
| MLP 2 × 64 (base) | 14,722 | 25 / 12 / 10 | 0.145 ± 0.006 | 0.114 ± 0.003 | 0.034 / 0.031 | 0.197 / 0.171 |
| MLP 2 × 256, dropout 0.1 | 82,306 | 12 / 17 / 13 | 0.153 ± 0.0006 | 0.116 ± 0.0007 | 0.043 / 0.034 | 0.204 / 0.173 |
| **MLP 2 × 256, dropout 0.3** | 82,306 | 18 / 22 / 20 | **0.154 ± 0.0002** | **0.117 ± 0.0006** | 0.044 / 0.035 | 0.205 / 0.174 |
| GRU + 3 × 256, dropout 0.1 | 176,002 | 8 / 9 / 12 | 0.152 ± 0.0014 | 0.115 ± 0.0006 | 0.042 / 0.033 | 0.204 / 0.173 |

**Choice (`prob_choice.json`).** Mean val Brier over both targets: 0.02809 (dropout 0.3), 0.02813 (dropout 0.1), 0.02815 (GRU), 0.02826 (base).
The only net with fewer parameters than the best is the base; its gap (0.000172) is above the noise threshold (0.000150), so the rule keeps dropout 0.3.
Dropout 0.1 has the same parameter count, so the rule does not prefer it.

**Noise (`prob_noise.csv`).** Gaps of the chosen model in BSS against calibrated persistence, classic / e-bike, with z = gap / √(seed sd² + station-bootstrap sd²); with 3 seeds per model the z values are rough:

| Against | Gap | z | Val days with a positive gap |
|---|---|---|---|
| MLP 2 × 256, dropout 0.1 | +0.0010 / +0.0012 | 1.6 / 2.1 | 4 / 4 of 5 |
| GRU | +0.0014 / +0.0019 | 1.4 / 3.1 | 5 / 5 of 5 |
| LightGBM | +0.0031 / +0.0034 | 2.6 / 4.6 | 4 / 5 of 5 |
| MLP 2 × 64 (base) | +0.0085 / +0.0034 | 2.1 / 2.0 | 5 / 5 of 5 |

The lead over dropout 0.1 is within about two standard errors (a tie); over the GRU it is within that on classic only (e-bike z 3.1).

**Base with patience 6 (after the reviews; `46_prob_search.py --patience 6`, `prob_extra_runs.csv`, `prob_patience_check.csv`).** The base seeds then stop at epochs 25 / 19 / 24 and reach 0.150 ± 0.003 / 0.115 ± 0.001, against 0.145 / 0.114 with patience 3.
So about half of its classic gap was early stopping.
With these runs the rule would still keep dropout 0.3 (gap 0.000084 in mean Brier, threshold 0.000063); the choice is not changed, since the test days were already read.

**Low rows** (count now ≤ 2; 1,068,040 classic and 1,929,731 e-bike val rows): the same ranking, 0.156 / 0.117 for the chosen model.
Unlike the count search, the deep model leads LightGBM on the low rows too (+0.0027 / +0.0032, z 2.1 / 4.4).

**Calibrated persistence against raw persistence:** +0.061 / +0.065, about a third of the chosen model's gain over raw persistence (0.205 / 0.174).

## Test (Oct 24–28)

For the probability models the test days were read once, for the chosen model only.
The same days were scored once before (Nov 12, 01:44) for the earlier count models (`test_scores.csv`); nothing in this search was chosen on them.
Protocol (`src/48_prob_test.py`, committed in bfc0d93 of the working repo before the read): the chosen model's three seeds; `--dry` first checks that each seed's saved weights reproduce its logged val Brier (max difference 1.2 × 10⁻¹⁰); the references are refit on train; `score_prob_test` logs each seed and refuses a second score.

| | Classic | E-bike |
|---|---|---|
| BSS vs calibrated persistence | 0.125 ± 0.0006 | 0.092 ± 0.0005 |
| BSS vs persistence | 0.172 | 0.144 |
| Calibrated persistence vs persistence | 0.053 | 0.057 |

- Per day (`prob_test_by_day.csv`), classic / e-bike: Sat 0.117 / 0.085, Sun 0.111 / 0.078, Mon 0.140 / 0.102, Tue 0.127 / 0.102. Wed Oct 28 has only 76,559 early-morning rows.
- Without the 2,324 rows at the 2 stations not seen in training, the skill changes by at most 0.0002 (`prob_test_unseen.csv`).
- Station-hour persistence has no test score: it was added after the read, and the protocol allows no second read. The clean test of the final model is data v2's unread test week (Nov 4–10).
- Nothing was tuned after the test score.

## Assumptions and limits

- The table keeps only anchors where some count (classic, e-bike or free docks) is ≤ 2 (the team's filter), so all numbers are on these rows, not on every station-minute.
- Calibrated persistence, station-hour persistence and climatology use fixed shrinkage (20 pseudo-rows) and a count cap of 10, chosen without tuning.
- Seeds measure the weights and batch order; the station bootstrap adds which val stations we have. Neither covers other weeks or seasons: the val → test drop shows that this matters.
- The candidates and the epoch were chosen on the same val days, so val is slightly optimistic.
- The selection rule's "fewer parameters" clause cannot separate dropout 0.1 from 0.3 (same size); their gap is within about two standard errors.
- Reused from `research_plan.md` (Card 1): BSS against calibrated persistence as the headline, the station × hour-of-week reference and the reliability check. Not reused: the per-(kind, h) composite, the day-block significance tests and temperature scaling. Temperature scaling may be worth it for data v2, given the test calibration error.

## Reproduce (from `src/`)

```bash
python test_score_prob.py                       # 29 toy checks of the score
python 46_prob_search.py --logreg --lgbm --seed 0 1 2
python 46_prob_search.py --run mlp_2x64_e8 mlp_2x256_do0.1 mlp_2x256_do0.3 gru64_3x256_do0.1 --seed 0 1 2
python 47_prob_report.py                        # val tables, choice, figures, station-hour reference
python 48_prob_test.py --dry                    # val reproduction check, test not read
python 46_prob_search.py --run mlp_2x64_e8 --patience 6 --seed 0 1 2 --threads 2   # after the reviews
python 46_prob_search.py --lgbm-ablation all no_station_idx no_station_info no_lags counts_now_h counts_lags_h --seed 0
python 50_prob_checks.py                        # patience check, in-sample day types, flags, climatology, sample
python 49_progress_notebook.py                  # notebooks/progress_report_prob.ipynb + .html
```
`48_prob_test.py` without `--dry` refuses to score again: the test score is in `prob_test_scores.csv`.
