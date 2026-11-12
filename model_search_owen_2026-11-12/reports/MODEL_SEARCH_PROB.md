# Probability of at least one bike: models and score

**Result.** The chosen model is the MLP 2 × 256 with dropout 0.3, picked on val by a rule fixed before any run.
Against calibrated persistence (an honest "what you see now" probability) its Brier skill score is 0.154 ± 0.0002 (classic) and 0.117 ± 0.0006 (e-bike) on val, and 0.125 ± 0.0006 / 0.092 ± 0.0005 on test, read once.
Against raw persistence it is 0.205 / 0.174 (val) and 0.172 / 0.144 (test).
Mean ± sd over seeds 0–2; `prob_summary.csv` (val, `src/47_prob_report.py`) and `prob_test_summary.csv` (test, `src/48_prob_test.py`).

What the search shows (val):

1. **Non-linearity matters, the architecture hardly does.** LightGBM, the GRU and the three MLPs all reach 0.145–0.154 (classic) and 0.114–0.117 (e-bike). Logistic regression on the same inputs only matches calibrated persistence (−0.000 / +0.002).
2. **The gain is largest when one bike is left** (0.182 / 0.159 at 1 bike now, 0.038 / 0.039 at 6 or more; `prob_by_count.csv`) **and grows with the horizon** (0.079 / 0.058 at h = 2 min, 0.177 / 0.143 at h = 14 min; `prob_by_horizon.csv`).
3. **In rider terms:** with 1 bike now, the bike is gone at t + h in 7.6 % (classic) / 9.2 % (e-bike) of rows. Flagging the 10 % of these rows with the lowest forecast catches 48 % / 41 % of the gone cases; calibrated persistence catches 16 % / 16 % (`prob_last_bike.csv`).
4. **Calibrated:** every val bin with at least 1,000 rows is within 0.010 / 0.015 of the diagonal (test: 0.029 / 0.022; `prob_reliability.csv`, `prob_test_reliability.csv`, Fig. `figures/prob_reliability.png`).
5. **Test is lower than val:** test has a weekend, val has none. Classic skill is 0.117 / 0.111 on Sat / Sun and 0.140 / 0.127 on Mon / Tue (`prob_test_by_day.csv`). The test weekdays are also below val, so the weekend is not the whole story (untested).

## Setup

The output is the probability that the station has at least one bike of each type (classic, e-bike) at t + h, with h = 2–15 min.
It replaces the count forecast trained with MSE (`MODEL_SEARCH.md`).
Everything else is unchanged: data-v1 with the weather fix (SHA-1 263bb791…, 11,357,122 rows), the split (train to Oct 18, val Oct 19–23, test from Oct 24), the 20 inputs, the station embedding and one model for all horizons.
The label is the table's `y_classic` / `y_ebike`, which equals 1[count at t + h ≥ 1] on every row.
The loss is binary cross-entropy on two logits; the forecast is p = sigmoid(logit).
Training: Adam (lr 10⁻³, batch 4,096), up to 30 epochs, keep the epoch with the lowest val cross-entropy, stop after 3 epochs without improvement.

## Score

Persistence ("available at t + h if available now") gives a hard 0 or 1, so a log-loss skill against it is infinite on every miss.
The Brier score is finite for 0/1 forecasts and is a proper score, so it is the yardstick (`src/score_prob.py`, toy checks in `src/test_score_prob.py`):

- Brier = mean (p − y)²; persistence's Brier is the share of rows whose status flips between t and t + h.
- BSS_ref = 1 − Brier_model / Brier_ref: 0 = no better than the reference, 1 = perfect.

Three references, all fit on the train split only:

| Reference | Forecast | What it stands for |
|---|---|---|
| Persistence | 1[count now ≥ 1] | doing nothing; the yardstick of the count model |
| Calibrated persistence | P(y = 1 \| count now, h) from train, count capped at 10, shrunk toward the count's rate (20 pseudo-rows) | persistence made into an honest probability |
| Climatology | the station's train rate in that hour of the week, shrunk toward the station rate, then the global rate (20 pseudo-rows each) | what a regular rider knows from habit |

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

**Selection rule (fixed before any run, in the docstring of `46_prob_search.py`, commit 62b0d5d of the working repo):** among the four nets, the lowest val Brier averaged over the two targets (all rows, mean over seeds).
A net with fewer parameters within 2 × √(sd_a²/3 + sd_b²/3) of it is preferred.
LightGBM and logistic regression are references and are not chosen.
Only the chosen model is scored on test (`src/48_prob_test.py`, protocol committed before the read in bfc0d93 of the working repo).

## Results (val, Oct 19–23)

BSS against calibrated persistence, all rows, mean ± sd over seeds (`prob_summary.csv`):

| Model | Params | Best epoch, seeds 0 / 1 / 2 | Classic | E-bike | vs persistence (classic / e-bike) |
|---|---|---|---|---|---|
| Persistence | – | – | −0.065 | −0.070 | 0 / 0 |
| Climatology | – | – | −4.87 | −4.27 | −4.51 / −3.93 |
| Calibrated persistence | – | – | 0 | 0 | 0.061 / 0.065 |
| Logistic regression | 42 | – | −0.000 | 0.002 | 0.060 / 0.067 |
| LightGBM | – | 80–119 trees | 0.151 ± 0.0004 | 0.114 ± 0.0005 | 0.202 / 0.171 |
| MLP 2 × 64 (base) | 14,722 | 25 / 12 / 10 | 0.145 ± 0.006 | 0.114 ± 0.003 | 0.197 / 0.171 |
| MLP 2 × 256, dropout 0.1 | 82,306 | 12 / 17 / 13 | 0.153 ± 0.0006 | 0.116 ± 0.0007 | 0.204 / 0.173 |
| **MLP 2 × 256, dropout 0.3** | 82,306 | 18 / 22 / 20 | **0.154 ± 0.0002** | **0.117 ± 0.0006** | 0.205 / 0.174 |
| GRU + 3 × 256, dropout 0.1 | 176,002 | 8 / 9 / 12 | 0.152 ± 0.0014 | 0.115 ± 0.0006 | 0.204 / 0.173 |

**Choice (`prob_choice.json`).** Mean val Brier over both targets: 0.02809 (dropout 0.3), 0.02813 (dropout 0.1), 0.02815 (GRU), 0.02826 (base).
The only net with fewer parameters than the best is the base; its gap (0.000172) is above the noise threshold (0.000150), so the rule keeps dropout 0.3.
Dropout 0.1 has the same parameter count, so the rule does not prefer it.

**Noise (`prob_noise.csv`).** Gaps of the chosen model in BSS against calibrated persistence, classic / e-bike, with z = gap / √(seed sd² + station-bootstrap sd²):

| Against | Gap | z | Val days with a positive gap |
|---|---|---|---|
| MLP 2 × 256, dropout 0.1 | +0.0010 / +0.0012 | 1.6 / 2.1 | 4 / 4 of 5 |
| GRU | +0.0014 / +0.0019 | 1.4 / 3.1 | 5 / 5 of 5 |
| LightGBM | +0.0031 / +0.0034 | 2.6 / 4.6 | 4 / 5 of 5 |
| MLP 2 × 64 (base) | +0.0085 / +0.0034 | 2.1 / 2.0 | 5 / 5 of 5 |

The leads over dropout 0.1 and the GRU are within about two standard errors.
The base's gap may come from early stopping rather than its size: its seed that trained to epoch 25 reached 0.153 / 0.116, while the seeds that stopped at epochs 10 and 12 reached 0.142 / 0.111 and 0.142 / 0.114 (`prob_runs.csv`). A longer patience is untested.

**Low rows** (count now ≤ 2; 1,068,040 classic and 1,929,731 e-bike val rows): the same ranking, 0.156 / 0.117 for the chosen model.
Unlike the count search, the deep model leads LightGBM on the low rows too (+0.0027 / +0.0032, z 2.1 / 4.4).

**Calibrated persistence against raw persistence:** +0.061 / +0.065. Turning "a bike now" into an honest probability is worth about a third of the chosen model's gain over raw persistence (0.205 / 0.174).

## Test (Oct 24–28, read once)

Protocol (`src/48_prob_test.py`, committed in bfc0d93 of the working repo before the read): chosen model only, its three seeds; `--dry` first checks that each seed's saved weights reproduce its logged val Brier (max difference 1.2 × 10⁻¹⁰); the references are refit on train; `score_prob_test` logs each seed and refuses a second score.

| | Classic | E-bike |
|---|---|---|
| BSS vs calibrated persistence | 0.125 ± 0.0006 | 0.092 ± 0.0005 |
| BSS vs persistence | 0.172 | 0.144 |
| BSS vs climatology | 0.879 | 0.853 |
| Calibrated persistence vs persistence | 0.053 | 0.057 |

- Per day (`prob_test_by_day.csv`), classic / e-bike: Sat 0.117 / 0.085, Sun 0.111 / 0.078, Mon 0.140 / 0.102, Tue 0.127 / 0.102. Wed Oct 28 has only 76,559 early-morning rows.
- Without the 2,324 rows at the 2 stations not seen in training, the skill changes by at most 0.0002 (`prob_test_unseen.csv`).
- Nothing was tuned after the test score.

## Assumptions and limits

- The table keeps only anchors where some count is ≤ 2 (the team's filter), so all numbers are on these rows, not on every station-minute.
- Calibrated persistence and climatology use fixed shrinkage (20 pseudo-rows) and a count cap of 10, chosen without tuning.
- Seeds measure the weights and batch order; the station bootstrap adds which val stations we have. Neither covers other weeks or seasons: the val → test drop shows that this matters.
- The selection rule's "fewer parameters" clause cannot separate dropout 0.1 from 0.3 (same size); their gap is within about two standard errors.
- Reused from `research_plan.md` (Card 1): BSS against calibrated persistence as the headline, the station × hour-of-week reference and the reliability check. Not reused: the per-(kind, h) composite, the day-block significance tests and temperature scaling (the forecasts are already calibrated on val).

## Reproduce (from `src/`)

```bash
python test_score_prob.py                       # 25 toy checks of the score
python 46_prob_search.py --logreg --lgbm --seed 0 1 2
python 46_prob_search.py --run mlp_2x64_e8 mlp_2x256_do0.1 mlp_2x256_do0.3 gru64_3x256_do0.1 --seed 0 1 2
python 47_prob_report.py                        # val tables, choice, figures
python 48_prob_test.py --dry                    # val reproduction check, test not read
python 49_progress_notebook.py                  # notebooks/progress_report_prob.ipynb + .html
```
`48_prob_test.py` without `--dry` refuses to score again: the test score is in `prob_test_scores.csv`.
