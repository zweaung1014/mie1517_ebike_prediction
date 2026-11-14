"""Build, run and export notebooks/progress_report_prob.ipynb (+ .html): the progress report notebook (model part).

Retrains nothing and reads no data file: it reads the cached tables of 46-50 (reports/tables/prob_*.csv) and the
figures. The method cells show the code of the score, the reference, the network and (condensed) the training loop,
taken from src/ when this script runs (inspect.getsource), so they match the code that produced the results.
The commands that rebuild the tables are listed at the end of the notebook; the notebook itself never retrains.

All prose in the notebook is Claude's draft for Owen: blue [Dn] = draft text to accept or rewrite; red [Cn] = a
claim to check or a choice for Owen and the team. Numbers in the prose are read from the tables when this script
runs, except the few that cite another source in the text (the data v2 sizes come from data_v2_splits.csv).
Usage (from src/):  python 49_progress_notebook.py
"""
import importlib
import inspect
import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import score_prob  # noqa: E402

ms = importlib.import_module("37_model_search")
T = ROOT / "reports" / "tables"
OUT = ROOT / "notebooks" / "progress_report_prob.ipynb"
CELLS, TAGS = [], []
N = {"D": 0, "C": 0, "K": 0}
LABEL = {"mlp_2x64_e8": "MLP 2×64 (base)", "mlp_2x256_do0.1": "MLP 2×256, dropout 0.1",
         "mlp_2x256_do0.3": "MLP 2×256, dropout 0.3", "gru64_3x256_do0.1": "GRU + 3×256, dropout 0.1",
         "lgbm": "LightGBM", "logreg": "logistic regression"}
X2 = ["classic", "ebike"]


def md(text):
    CELLS.append(nbformat.v4.new_markdown_cell(text.strip()))


def code(text, hide=False):
    c = nbformat.v4.new_code_cell(text.strip())
    if hide:  # formatting code: the HTML shows only its output
        c.metadata["tags"] = ["hide-input"]
        c.metadata["jupyter"] = {"source_hidden": True}
    CELLS.append(c)


def src_block(obj, title, start=None, stop=None):
    """A markdown cell with the source of obj (lines start:stop), so the method is visible in the HTML."""
    lines = inspect.getsource(obj).splitlines()[start:stop]
    md(f"{title}\n\n```python\n" + "\n".join(lines) + "\n```")


def d(text, kind="D"):
    """Blue draft text [Dn], or a red claim to check [Cn]."""
    N[kind] += 1
    tag = f"{kind}{N[kind]}"
    TAGS.append((tag, " ".join(text.split()[:14])))
    color = "#c00000" if kind == "C" else "#1f5fbf"
    return f'<span style="color:{color}">[{tag}] {" ".join(text.split())}</span>'


def v(x, n=3):
    return f"{x:.{n}f}"


def pct(x, n=0):
    return f"{100 * x:.{n}f} %"


# ---------------------------------------------------------------- numbers for the prose (from the tables)
S = pd.read_csv(T / "prob_summary.csv")
CH = json.loads((T / "prob_choice.json").read_text())
C = CH["chosen"]
DATA = pd.read_csv(T / "prob_data.csv").set_index("split")
TDATA = pd.read_csv(T / "prob_test_data.csv").set_index("split")
TS = pd.read_csv(T / "prob_test_summary.csv")
NO = pd.read_csv(T / "prob_noise.csv")
RUNS = pd.read_csv(T / "prob_runs.csv")
BC = pd.read_csv(T / "prob_by_count.csv")
BH = pd.read_csv(T / "prob_by_horizon.csv")
BD = pd.read_csv(T / "prob_by_day.csv")
LB = pd.read_csv(T / "prob_last_bike.csv")
REL = pd.read_csv(T / "prob_reliability.csv")
AB = pd.read_csv(T / "prob_ablation.csv").set_index("name")
PC = pd.read_csv(T / "prob_patience_check.csv").iloc[0]
TDT = pd.read_csv(T / "prob_train_daytype.csv").set_index("day_type")
CL = pd.read_csv(T / "prob_climatology_check.csv").iloc[0]
FL = pd.read_csv(T / "prob_flags.csv")
TD = pd.read_csv(T / "prob_test_by_day.csv")
TBC = pd.read_csv(T / "prob_test_by_count.csv")
UN = pd.read_csv(T / "prob_test_unseen.csv")
TR = pd.read_csv(T / "prob_test_reliability.csv")
V2 = pd.read_csv(T / "data_v2_splits.csv").query("variant == 'forward'").set_index("split")
SH = "station-hour persistence"


def s(model, x, sub, k):
    return S[(S.model == model) & (S.target == x) & (S.subset == sub)].iloc[0][f"{k}_mean"]


def sd(model, x, sub, k):
    return S[(S.model == model) & (S.target == x) & (S.subset == sub)].iloc[0][f"{k}_sd"]


def ts(model, x, sub, k):
    return TS[(TS.model == model) & (TS.target == x) & (TS.subset == sub)].iloc[0][f"{k}_mean"]


def gap(b, x, k="gap_bss_calpers"):
    return NO[(NO.a == C) & (NO.b == b) & (NO.target == x) & (NO.subset == "all")].iloc[0][k]


def grp(tab, x, g, model=C, k="bss_calpers"):
    return tab[(tab.model == model) & (tab.target == x) & (tab.group == g)].iloc[0][k]


def lb(x, model, k="share_of_gone_caught", flag=0.1):
    return LB[(LB.target == x) & (LB.model == model) & (LB.flag_share == flag)].iloc[0][k]


def relgap(x, which, tab=REL, sub="all"):
    """Largest |observed - forecast| over bins with >= 1,000 rows; which = 'chosen' or a reference name."""
    refs = ["calibrated persistence", "climatology", SH]
    m = ~tab.model.isin(refs) if which == "chosen" else tab.model == which
    if "subset" in tab:
        m &= tab.subset == sub
    r = tab[m & (tab.target == x) & (tab.rows >= 1000)]
    return (r.y_rate - r.p_mean).abs().max()


def pooled(tab, days, x, model=C, ref="calibrated persistence"):
    """BSS over several days, rows-weighted (pooled Brier)."""
    a = tab[(tab.model == model) & (tab.target == x) & tab.group.isin(days)]
    b = tab[(tab.model == ref) & (tab.target == x) & tab.group.isin(days)]
    return 1 - (a.brier * a.rows).sum() / (b.brier * b.rows).sum()


def share(x, g, model=C, ref="calibrated persistence"):
    """Share of the model's total Brier reduction against ref (rows x Brier gap) that comes from count group g."""
    t = BC[BC.target == x]
    a = t[t.model == model].set_index("group")
    b = t[t.model == ref].set_index("group")
    red = (b.brier - a.brier) * a.rows
    return red[g] / red.sum()


def flipd(tab, day, x="classic"):
    return tab[(tab.model == "persistence") & (tab.target == x) & (tab.group == day)].iloc[0].brier


def dayb(tab, day, x="classic"):
    return tab[(tab.model == C) & (tab.target == x) & (tab.group == day)].iloc[0].bss_calpers


def ab(name, x):
    return AB.loc[f"lgbm_{name}", f"bss_calpers_{x}_all"]


nets = [n for n in ["mlp_2x64_e8", "mlp_2x256_do0.1", "mlp_2x256_do0.3", "gru64_3x256_do0.1", "lgbm"] if n != C]
spread = max(abs(s(n, x, "all", "bss_calpers") - s(C, x, "all", "bss_calpers")) for n in nets for x in X2)
_end = pd.to_datetime(RUNS.finished)
span_h = (_end.max() - (_end - pd.to_timedelta(RUNS.wall_s, "s")).min()).total_seconds() / 3600
vday = BD[(BD.model == C)]
vrange = {x: (vday[vday.target == x].bss_calpers.min(), vday[vday.target == x].bss_calpers.max()) for x in X2}
WKEND, WKDAY = ["2026-10-24", "2026-10-25"], ["2026-10-26", "2026-10-27"]
t_all = {x: ts(C, x, "all", "bss_calpers") for x in X2}
t_wkday = {x: pooled(TD, WKDAY, x) for x in X2}
t_wkend = {x: pooled(TD, WKEND, x) for x in X2}
v_all = {x: s(C, x, "all", "bss_calpers") for x in X2}
wk_share = {x: (t_wkday[x] - t_all[x]) / (v_all[x] - t_all[x]) for x in X2}
tbin = TR[(TR.model == C) & (TR.rows >= 1000)].assign(g=lambda t: t.p_mean - t.y_rate)
cls_hi = tbin[(tbin.target == "classic") & tbin.bin.between(1, 3)].g
eb_lo = tbin[(tbin.target == "ebike") & tbin.bin.between(4, 6)].g
n_inputs = len(ms.INPUTS)
n_const = len(set(FL.input))
const_ok = bool((FL["min"] == FL["max"]).all())
train_rows = int(DATA.loc["train", "rows"])

# ---------------------------------------------------------------- title and summary
md(f"""
# Will there be a bike? Progress report, model part

**Team 3, MIE1517.** For every Toronto Bike Share station we forecast the probability that at least one
classic bike, and at least one e-bike, will be docked there h = 2–15 minutes from now.

**Summary.**
{d(f"Our best model so far is the {LABEL[C]} network. On the validation days, on the station-times our table keeps "
   f"(some count of classic bikes, e-bikes or free docks is 2 or less), it beats an honest 'what you see now' "
   f"forecast (calibrated persistence) by a Brier skill score of {v(v_all['classic'])} for classic bikes and "
   f"{v(v_all['ebike'])} for e-bikes. Most of that is knowing the station and the hour: a look-up table that adds "
   f"them (no learning) gets {v(s(SH, 'classic', 'all', 'bss_calpers'))} / {v(s(SH, 'ebike', 'all', 'bss_calpers'))}, "
   f"and the network beats that table by {v(s(C, 'classic', 'all', 'bss_sthour'))} / "
   f"{v(s(C, 'ebike', 'all', 'bss_sthour'))}.")}
{d(f"On the test days the skill against calibrated persistence is {v(t_all['classic'])} / {v(t_all['ebike'])}. "
   "For this model the test days were read once, for the chosen model only; the same days were scored once "
   "before (Nov 12) for our earlier count models, and nothing in this search was chosen on them.")}
{d("What is left: retrain on a larger dataset (data v2) whose test week nobody has read, longer history for the "
   "sequence models, and a rider-level evaluation; plan and dates in Section 8.")}

All prose below is a draft: blue text marked [Dn] is draft text, red text marked [Cn] is a claim still to be
checked. Every number is printed from a table written by a script of the project.
""")

# ---------------------------------------------------------------- key findings (Owen, Nov 13), grouped
DS = pd.read_csv(T / "data_size_summary.csv")
DSL = pd.read_csv(T / "data_size_slope.csv").set_index(["dropout", "column"])
ds = DS[DS.dropout == "dropout 0.1"].set_index("train_days")
KEY = f"""
## Key findings

**What we predict and how we score it**

1. {d("We predict P(at least one bike at t + h) for each bike type, because whether a bike is there is what a "
      "rider needs to know.", "K")}
2. {d("The score is the skill against a reference forecast, skill = 1 − Brier_model / Brier_ref, with calibrated "
      "persistence (an honest 'what you see now' probability) as the reference: 0 = no better than the "
      "reference, 1 = perfect.", "K")}

**What the model learns, and where it helps (validation days)**

3. {d(f"Knowing the station matters most: the chosen model, which knows it, reaches {v(s(C, 'classic', 'all', 'bss_calpers'))} / "
      f"{v(s(C, 'ebike', 'all', 'bss_calpers'))} (classic / e-bike); LightGBM without any station information "
      f"reaches {v(ab('no_station_info', 'classic'))} / {v(ab('no_station_info', 'ebike'))}.", "K")}
4. {d(f"The model helps most on the last bike: skill {v(grp(BC, 'classic', '1'))} / {v(grp(BC, 'ebike', '1'))} "
      f"with one bike now, against {v(grp(BC, 'classic', '6+'))} / {v(grp(BC, 'ebike', '6+'))} with 6 or more.", "K")}
5. {d(f"The gain over the reference grows with the horizon, from {v(grp(BH, 'classic', 2))} / {v(grp(BH, 'ebike', 2))} "
      f"at h = 2 min to {v(grp(BH, 'classic', 14))} / {v(grp(BH, 'ebike', 14))} at h = 14 min. The forecast "
      f"itself gets less accurate further ahead (Brier {v(grp(BH, 'classic', 2, k='brier'), 4)} to "
      f"{v(grp(BH, 'classic', 14, k='brier'), 4)}, classic), but the reference degrades faster.", "K")}

**How well it carries over, and what comes next**

6. {d(f"Test is lower than validation: {v(t_all['classic'])} / {v(t_all['ebike'])} against "
      f"{v(v_all['classic'])} / {v(v_all['ebike'])}. The test days include a weekend and the validation days do "
      f"not; leaving the weekend out closes {pct(wk_share['classic'])} / {pct(wk_share['ebike'])} of the gap, so "
      "the weekend is about a third of the drop.", "K")}
7. {d(f"More training data helps, with a slowing gain (earlier count model, MSE skill, data-v1): "
      f"{v(ds.loc[5, 'classic_all_mean'])} with 5 training days, {v(ds.loc[10, 'classic_all_mean'])} with 10, "
      f"{v(ds.loc[20, 'classic_all_mean'])} with 20 (classic); about "
      f"+{v(DSL.loc[('dropout 0.1', 'classic_all'), 'slope_per_7_days_10_to_20'])} per extra week over 10–20 days "
      f"and +{v(DSL.loc[('dropout 0.1', 'classic_all'), 'slope_per_7_days_15_to_20'])} over 15–20. "
      "Data v2 (more days) tests this on the probability model.", "K")}
"""
md(KEY)

code("""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
T, F = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
pd.set_option("display.width", 100)
LABEL = {"persistence": "persistence", "calibrated persistence": "calibrated persistence",
         "station-hour persistence": "station-hour look-up", "climatology": "climatology",
         "logreg": "logistic regression", "lgbm": "LightGBM",
         "mlp_2x64_e8": "MLP 2x64 (base)", "mlp_2x256_do0.1": "MLP 2x256, dropout 0.1",
         "mlp_2x256_do0.3": "MLP 2x256, dropout 0.3",
         "gru64_3x256_do0.1": "GRU + 3x256, dropout 0.1"}
""", hide=True)

# ---------------------------------------------------------------- 1. data
va = DATA.loc["val"]
md(f"""
## 1. Data collected

{d("We poll the public Toronto Bike Share feed (GBFS) every minute since Sep 29, with backup pollers, and add "
   "Environment Canada weather. The team table (data-v1) turns the polls into rows: for each station and anchor "
   "time t (every 10 min) it draws 3 random horizons h between 2 and 15 min. Each row holds the counts now and "
   "10 and 30 min before, the station, the time of day and the weather, and the counts at t + h. Only anchors "
   "where some count (classic bikes, e-bikes or free docks) is 2 or less are kept, so every number below is on "
   "these station-times, where a rider may find no bike or no dock.")}
{d("Pollers, weather source and the anchor rule as described in the team's data documentation; a coverage "
   "table (polls per day, gaps) from the data part of the team notebook should go here.", "C")}

The table gives the size of each split and how often the answer changes: the *flip rate* is the share of rows
whose 'at least one bike' status differs between t and t + h.
{d(f"Status flips are rare: {v(100 * va.classic_flip_rate, 1)} % of val rows for classic bikes and "
   f"{v(100 * va.ebike_flip_rate, 1)} % for e-bikes, so 'what you see now' is already a strong forecast.")}
""")
code("""
data = pd.concat([pd.read_csv(T / "prob_data.csv"), pd.read_csv(T / "prob_test_data.csv")])
show = data[["split", "rows", "stations", "first_day", "last_day",
             "classic_base_rate", "ebike_base_rate",
             "classic_flip_rate", "ebike_flip_rate"]].round(3)
show.columns = ["split", "rows", "stations", "from", "to",
                "P(>=1 classic)", "P(>=1 e-bike)", "flip classic", "flip e-bike"]
show.set_index("split")
""", hide=True)
md("Five rows of one station on a validation morning (inputs, horizon h in minutes, and the two labels):")
code("""
pd.read_csv(T / "prob_sample_rows.csv")
""", hide=True)
code("""
display(Image(filename=str(F / "prob_station_day.png"), width=850))
""", hide=True)

# ---------------------------------------------------------------- 2. task and score
md(f"""
## 2. Task and score

**Output.** For each bike type, p = P(at least one bike at t + h), h = 2–15 min. One model covers all stations
and horizons (h is an input). Loss: binary cross-entropy.
{d("We moved from forecasting the count (with MSE) to this probability because a rider's question is "
   "'will there be a bike?', and a probability says how sure the answer is.")}
{d("This replaces the team's count / MSE direction of Oct 28; the team agreed to the change at its Nov 12 meeting.", "C")}

**Score.** The Brier score, mean (p − y)², and the Brier skill score against a reference,
BSS = 1 − Brier_model / Brier_ref (0 = no better than the reference, 1 = perfect).
{d("We do not use log loss against persistence: persistence says 0 or 1, so every miss costs log(0) = infinity. "
   "The Brier score is finite for 0/1 forecasts and is still a proper score.")}

References, all fitted on the training days only:

| Reference | Forecast |
|---|---|
| Persistence | 1 if a bike is there now, else 0 |
| Calibrated persistence (headline) | P(bike at t + h given the count now and h) |
| Station-hour look-up (added after review; val only) | calibrated persistence corrected per station, hour of day and count now |
| Climatology | the station's rate at that hour of the week, ignoring the count now |

{d(f"Calibrated persistence is our headline 'doing nothing': it uses only what the screen shows now, and it already "
   f"knows that '1 bike now' is not a sure bike later; it beats raw persistence by "
   f"{v(s('calibrated persistence', 'classic', 'all', 'bss_pers'))} / "
   f"{v(s('calibrated persistence', 'ebike', 'all', 'bss_pers'))} (classic / e-bike, val). The station-hour "
   f"look-up is a stronger bar that needs a table per station and hour: it scores "
   f"{v(s(SH, 'classic', 'all', 'bss_calpers'))} / {v(s(SH, 'ebike', 'all', 'bss_calpers'))} against calibrated "
   f"persistence. It was added after the test read, so it has no test score. Climatology ignores the count now and "
   f"is far weaker (BSS against persistence {v(s('climatology', 'classic', 'all', 'bss_pers'), 2)} / "
   f"{v(s('climatology', 'ebike', 'all', 'bss_pers'), 2)}).")}
""")
src_block(score_prob.brier, "**Method: the score** (`score_prob.py`)")
md("```python\n" + "\n".join(inspect.getsource(score_prob.bss).splitlines()) + "\n```")
src_block(score_prob.CalibratedPersistence, "**Method: the headline reference**, fitted on the training rows only")

# ---------------------------------------------------------------- 3. model
const_txt = (f"two of them, `renting_now` and `returning_now`, are station-status flags ('accepts rentals / "
             f"returns'), equal to 1 on every training and validation row, so they are constant and carry no "
             f"information" if const_ok else "two station-status flags")
md(f"""
## 3. Model

**Label and inputs.** The label of bike type x is y = 1 if the count of x at t + h is at least 1.
{d(f"The network takes the table's {n_inputs} numeric inputs: horizon h; classic, e-bike and free-dock counts now, "
   f"10 and 30 min ago; capacity; latitude and longitude; hour of day (sine and cosine); weekend; temperature; "
   f"humidity; and {const_txt} ({n_inputs - n_const} inputs vary). They are standardised with the training "
   "mean and standard deviation. An 8-number embedding of the station is learned with the model. Two outputs "
   "give one logit per bike type.")}
""")
src_block(score_prob.labels, "**Method: the label**")
src_block(ms.Net, "**Method: the network** (`37_model_search.py`; config of the chosen model: "
          f"`{ms.config_by_name(C)}`)")
md(f"""
**Method: training** (condensed from `train()` in `46_prob_search.py`). Adam, learning rate 10⁻³, batch 4,096,
one pass over the {train_rows:,} training rows per epoch, up to 30 epochs; keep the epoch with the lowest
validation cross-entropy; stop after 3 epochs without improvement.

```python
model = Net(cfg, n_inputs=20, n_stations=1071)          # one extra embedding row for unseen stations
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()                         # binary cross-entropy on the 2 logits
best, best_epoch = inf, 0
for epoch in range(1, 31):
    model.train()
    for idx in torch.randperm(len(x_train)).split(4096):
        loss = loss_fn(model(x_train[idx], station_train[idx]), y_train[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    val = cross_entropy(sigmoid(model(x_val, station_val)), y_val)
    if val < best:
        best, best_epoch, best_state = val, epoch, copy(model.state_dict())
    elif epoch - best_epoch >= 3:                        # patience 3
        break
model.load_state_dict(best_state)
```
""")

# ---------------------------------------------------------------- 4. search
md(f"""
## 4. Step by step: from references to the chosen model (validation days)

We trained four networks with three seeds each (0, 1, 2), so a gap between models can be judged against
the spread over seeds, plus LightGBM (gradient-boosted trees, three seeds) and logistic regression.
Validation: Oct 19–23. The rule to pick the model was fixed before any run: the lowest validation Brier
(mean of the two bike types), preferring a network with fewer parameters if it is within noise.
""")
code("""
s = pd.read_csv(T / "prob_summary.csv")
runs = pd.read_csv(T / "prob_runs.csv")
order = ["persistence", "climatology", "calibrated persistence", "station-hour persistence",
         "logreg", "lgbm", "mlp_2x64_e8", "mlp_2x256_do0.1", "mlp_2x256_do0.3",
         "gru64_3x256_do0.1"]
a = s[s.subset == "all"].set_index(["model", "target"])
def cell(m, x, k="bss_calpers"):
    r = a.loc[(m, x)]
    sd = r[f"{k}_sd"]
    return f"{r[f'{k}_mean']:.3f}" + ("" if np.isnan(sd) else f" ± {sd:.4f}")
params = runs.groupby("name").n_params.first()
tab = pd.DataFrame({
    "classic": [cell(m, "classic") for m in order],
    "e-bike": [cell(m, "ebike") for m in order],
    "vs look-up": [f"{a.loc[(m, 'classic')].bss_sthour_mean:.3f} / "
                   f"{a.loc[(m, 'ebike')].bss_sthour_mean:.3f}" for m in order],
    "params": ["" if pd.isna(params.get(m)) else f"{int(params[m]):,}" for m in order]},
    index=[LABEL[m] for m in order])
print("Validation BSS against calibrated persistence (mean ± sd over 3 seeds),")
print("and against the station-hour look-up (classic / e-bike)")
tab
""", hide=True)
code("""
ch = json.loads((T / "prob_choice.json").read_text())
for n, r in ch["nets"].items():
    print(f"{LABEL[n]:26s} mean Brier {r['mean_brier']:.5f} ± {r['sd_brier']:.5f}"
          f"  ({r['n_params']:,} params)")
print("chosen by the rule:", LABEL[ch["chosen"]])
""", hide=True)
code("""
cur = pd.read_csv(T / "prob_curves.csv")
fig, ax = plt.subplots(figsize=(8, 4))
for n, c in zip(["mlp_2x64_e8", "mlp_2x256_do0.1", "mlp_2x256_do0.3",
                 "gru64_3x256_do0.1"], ["C0", "C1", "C2", "C3"]):
    g = cur[cur.run_id == f"p_{n}_seed0"]
    ax.plot(g.epoch, g.val_bce, "o-", color=c, label=LABEL[n])
    ax.plot(g.epoch, g.train_bce, ":", color=c)
ax.set_xlabel("epoch")
ax.set_ylabel("cross-entropy (solid: val, dotted: train)")
ax.set_title("Learning curves, seed 0")
ax.legend(fontsize=8)
plt.show()
""", hide=True)
md(f"""
**Result.** {d(f"The models that know the station all clear the bar by about the same amount. The chosen "
   f"{LABEL[C]} scores {v(v_all['classic'])} ± {v(sd(C, 'classic', 'all', 'bss_calpers'), 4)} (classic) and "
   f"{v(v_all['ebike'])} ± {v(sd(C, 'ebike', 'all', 'bss_calpers'), 4)} (e-bike; ± = sd over seeds); the other "
   f"networks and LightGBM are within {v(spread, 4)} of it. The day-to-day spread is larger than the seed spread: "
   f"over the five validation days the chosen model ranges {v(vrange['classic'][0])}–{v(vrange['classic'][1])} "
   f"(classic) and {v(vrange['ebike'][0])}–{v(vrange['ebike'][1])} (e-bike).")}

{d(f"What the gain needs (LightGBM, seed 0, one input set at a time; table below): the counts now and h alone give "
   f"{v(ab('counts_now_h', 'classic'))} / {v(ab('counts_now_h', 'ebike'))}; adding the lag counts gives "
   f"{v(ab('counts_lags_h', 'classic'))} / {v(ab('counts_lags_h', 'ebike'))}; all inputs without any station "
   f"information (no station index, location or capacity) give {v(ab('no_station_info', 'classic'))} / "
   f"{v(ab('no_station_info', 'ebike'))}; with the location and capacity but no station index "
   f"{v(ab('no_station_idx', 'classic'))} / {v(ab('no_station_idx', 'ebike'))}; with the station but without the "
   f"lag counts {v(ab('no_lags', 'classic'))} / {v(ab('no_lags', 'ebike'))}. So the station (its location is "
   f"enough) is the largest part of the gain, and once the station is known the lag counts add little. Logistic "
   f"regression, which has neither the station nor a non-linear use of the inputs, only matches calibrated "
   f"persistence ({v(s('logreg', 'classic', 'all', 'bss_calpers'))} / {v(s('logreg', 'ebike', 'all', 'bss_calpers'))}).")}
""")
code("""
ab = pd.read_csv(T / "prob_ablation.csv").set_index("name")
names = {"lgbm_all": "all inputs + station index",
         "lgbm_no_station_idx": "no station index (location kept)",
         "lgbm_no_station_info": "no station index, location, capacity",
         "lgbm_no_lags": "no lag counts (station kept)",
         "lgbm_counts_lags_h": "counts now + lags + h only",
         "lgbm_counts_now_h": "counts now + h only"}
print("LightGBM (seed 0) on input subsets: val BSS against calibrated persistence")
pd.DataFrame({"classic": ab.bss_calpers_classic_all, "e-bike": ab.bss_calpers_ebike_all}
             ).rename(index=names).round(3)
""", hide=True)
md(f"""
{d(f"The rule picked {LABEL[C]}. Its lead is small. Gaps against calibrated persistence, classic / e-bike, with "
   f"z = gap / √(seed variance + station-sampling variance): over dropout 0.1 +{v(gap('mlp_2x256_do0.1', 'classic'), 4)} / "
   f"+{v(gap('mlp_2x256_do0.1', 'ebike'), 4)} (z {v(gap('mlp_2x256_do0.1', 'classic', 'z'), 1)} / "
   f"{v(gap('mlp_2x256_do0.1', 'ebike', 'z'), 1)}, a tie: same size, and the rule cannot separate them); over "
   f"the GRU +{v(gap('gru64_3x256_do0.1', 'classic'), 4)} / +{v(gap('gru64_3x256_do0.1', 'ebike'), 4)} "
   f"(z {v(gap('gru64_3x256_do0.1', 'classic', 'z'), 1)} / {v(gap('gru64_3x256_do0.1', 'ebike', 'z'), 1)}); over "
   f"LightGBM +{v(gap('lgbm', 'classic'), 4)} / +{v(gap('lgbm', 'ebike'), 4)} (z {v(gap('lgbm', 'classic', 'z'), 1)} / "
   f"{v(gap('lgbm', 'ebike', 'z'), 1)}); over the base 2×64 +{v(gap('mlp_2x64_e8', 'classic'), 4)} / "
   f"+{v(gap('mlp_2x64_e8', 'ebike'), 4)} (z {v(gap('mlp_2x64_e8', 'classic', 'z'), 1)} / "
   f"{v(gap('mlp_2x64_e8', 'ebike', 'z'), 1)}). With 3 seeds per model these z values are rough. The candidates "
   "were picked from the count search on the same validation days.")}
{d(f"Part of the base's gap was early stopping: rerun with patience 6 (after the reviews), its seeds stop at "
   f"epochs {PC.best_epochs} and reach {v(PC.bss_calpers_classic_mean)} / {v(PC.bss_calpers_ebike_mean)}, against "
   f"{v(s('mlp_2x64_e8', 'classic', 'all', 'bss_calpers'))} / {v(s('mlp_2x64_e8', 'ebike', 'all', 'bss_calpers'))} "
   f"with patience 3. The rule would still keep {LABEL[C]} (gap {PC.gap:.6f} in mean Brier against a threshold of "
   f"{PC.threshold:.6f}); the choice is not changed, since the test days were already read.")}
""")

# ---------------------------------------------------------------- 5. where
md("""
## 5. Where the model helps, and is it calibrated? (validation days)
""")
code("""
display(Image(filename=str(F / "prob_where.png"), width=900))
""", hide=True)
md(f"""
{d(f"Measured as skill, the model helps most when one bike is left: BSS {v(grp(BC, 'classic', '1'))} (classic) and "
   f"{v(grp(BC, 'ebike', '1'))} (e-bike) against calibrated persistence, against {v(grp(BC, 'classic', '6+'))} / "
   f"{v(grp(BC, 'ebike', '6+'))} with 6 or more bikes (on test: {v(grp(TBC, 'classic', '1'))} / "
   f"{v(grp(TBC, 'ebike', '1'))} at one bike). Stations with no bike now matter as much for classic bikes: there "
   f"the question is whether a bike will arrive. The gain also grows with the horizon, from "
   f"{v(grp(BH, 'classic', 2))} / {v(grp(BH, 'ebike', 2))} at h = 2 min to {v(grp(BH, 'classic', 14))} / "
   f"{v(grp(BH, 'ebike', 14))} at h = 14 min. The forecast itself gets less accurate with h (Brier {v(grp(BH, 'classic', 2, k='brier'), 4)} to "
   f"{v(grp(BH, 'classic', 14, k='brier'), 4)}, classic), but calibrated persistence degrades faster "
   f"({v(grp(BH, 'classic', 2, model='calibrated persistence', k='brier'), 4)} to "
   f"{v(grp(BH, 'classic', 14, model='calibrated persistence', k='brier'), 4)}), so the gain over it grows.")}
{d(f"Share of the total Brier reduction against calibrated persistence: rows with 0 bikes now give "
   f"{pct(share('classic', '0'))} (classic) / {pct(share('ebike', '0'))} (e-bike), rows with 1 bike "
   f"{pct(share('classic', '1'))} / {pct(share('ebike', '1'))}.")}

{d(f"In rider terms (validation only): when one bike is there now, it is gone by t + h in "
   f"{pct(lb('classic', C, 'gone_rate'), 1)} of cases (classic; {pct(lb('ebike', C, 'gone_rate'), 1)} for e-bikes). "
   f"A warning on the 10 % of these cases with the lowest forecast would catch {pct(lb('classic', C))} of the times "
   f"the bike is gone ({pct(lb('ebike', C))} for e-bikes), and be right {pct(lb('classic', C, 'gone_rate_flagged'))} "
   f"/ {pct(lb('ebike', C, 'gone_rate_flagged'))} of the time, so about 2 in 3 warnings would be false alarms. "
   f"Ranking within each horizon gives nearly the same catch ({pct(lb('classic', C, 'share_caught_within_h'))} / "
   f"{pct(lb('ebike', C, 'share_caught_within_h'))}). The same warning from calibrated persistence catches "
   f"{pct(lb('classic', 'calibrated persistence'))} / {pct(lb('ebike', 'calibrated persistence'))}, and from the "
   f"station-hour look-up {pct(lb('classic', SH))} / {pct(lb('ebike', SH))}. The base MLP and LightGBM catch about "
   "as much as the chosen model, so this is a property of the models that know the station, not of this one.")}
""")
code("""
display(Image(filename=str(F / "prob_reliability.png"), width=760))
""", hide=True)
md(f"""
{d(f"On the validation days the forecasts are well calibrated: in every bin with at least 1,000 rows the observed "
   f"share is within {v(relgap('classic', 'chosen'))} (classic) and {v(relgap('ebike', 'chosen'))} (e-bike) of "
   f"the forecast (seed 0; the epoch was picked on these days, so this is a best case). Climatology is not "
   f"calibrated (off by up to {v(CL.k20_ebike_max_bin_gap, 2)} for e-bikes, all rows): each of its station × "
   f"hour-of-week cells holds only {v(CL.days_per_cell_mean, 1)} training days on average, and with 10 times more "
   f"shrinkage its e-bike gap falls to {v(CL.k200_ebike_max_bin_gap, 2)}. Its high Brier score comes from "
   f"ignoring the count now: more shrinkage lowers it only from {v(CL.k20_ebike_brier)} to "
   f"{v(CL.k200_ebike_brier)} (e-bike).")}
""")

# ---------------------------------------------------------------- 6. test
un_diff = max(abs(UN[UN.target == x].bss_calpers.iloc[0] - t_all[x]) for x in X2)
tday = TD[TD.model == C]
md(f"""
## 6. Test result

{d("The test days (Oct 24–28) were read once for this model, for the chosen model only, after the choice; the "
   "protocol was committed before the read. The same days had been scored once before (Nov 12, early morning) "
   "for our earlier count models; nothing in this search was chosen on them. Nothing was tuned after the read.")}
""")
code("""
t = pd.read_csv(T / "prob_test_summary.csv")
t = t[t.subset == "all"]
t = t[t.model != "climatology"].copy()
t["model"] = t.model.map(lambda m: LABEL.get(m, m))
t = t.rename(columns={"brier_mean": "Brier", "bss_pers_mean": "BSS vs persistence",
                      "bss_calpers_mean": "BSS vs cal. pers.", "bss_calpers_sd": "sd (seeds)"})
print("Test, Oct 24-28, all table rows")
print(t[["model", "target", "Brier", "BSS vs persistence", "BSS vs cal. pers.",
         "sd (seeds)"]].to_string(index=False, float_format=lambda x: f"{x:.4f}",
                                  na_rep=""))
""", hide=True)
code("""
tr = pd.read_csv(T / "prob_test_reliability.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, x in zip(axes, ["classic", "ebike"]):
    for m, g in tr[(tr.target == x) & (tr.rows >= 1000)].groupby("model"):
        ax.plot(g.p_mean, g.y_rate, "o-", ms=4, label=LABEL.get(m, m))
    ax.plot([0, 1], [0, 1], "--", color="0.6")
    ax.set_title(f"Test calibration, {x} (seed 0)")
    ax.set_xlabel("forecast")
    ax.set_ylabel("observed share")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
""", hide=True)
code("""
rows = []
for split, f in [("val", "prob_by_day.csv"), ("test", "prob_test_by_day.csv")]:
    t = pd.read_csv(T / f)
    m = t[t.model == "mlp_2x256_do0.3"].set_index(["group", "target"])
    p = t[t.model == "persistence"].set_index(["group", "target"])
    for day in sorted(m.index.get_level_values(0).unique()):
        rows.append({"split": split, "day": day,
                     "weekday": pd.Timestamp(day).day_name()[:3],
                     "rows": int(m.loc[(day, "classic"), "rows"]),
                     "flip classic": p.loc[(day, "classic"), "brier"],
                     "BSS classic": m.loc[(day, "classic"), "bss_calpers"],
                     "flip e-bike": p.loc[(day, "ebike"), "brier"],
                     "BSS e-bike": m.loc[(day, "ebike"), "bss_calpers"]})
print("Per day: flip rate (= persistence Brier) and BSS against calibrated persistence")
pd.DataFrame(rows).round(4).set_index(["split", "day"])
""", hide=True)
md(f"""
{d(f"The gain holds on test but is smaller: {v(t_all['classic'])} / {v(t_all['ebike'])} against calibrated "
   f"persistence (validation {v(v_all['classic'])} / {v(v_all['ebike'])}) and {v(ts(C, 'classic', 'all', 'bss_pers'))} / "
   f"{v(ts(C, 'ebike', 'all', 'bss_pers'))} against persistence. Calibration is looser than on validation, in a "
   f"systematic way: classic forecasts of 0.1–0.4 are {v(cls_hi.min())}–{v(cls_hi.max())} too high, e-bike "
   f"forecasts of 0.4–0.7 {v(-eb_lo.max())}–{v(-eb_lo.min())} too low. Calibrated persistence stays within "
   f"{v(relgap('classic', 'calibrated persistence', TR))} / {v(relgap('ebike', 'calibrated persistence', TR))}.")}
{d(f"About a third of the drop is the weekend: the test week starts on a Saturday and the validation days have no "
   f"weekend. Weekends are harder even on the training days (in sample: weekdays "
   f"{v(TDT.loc['weekday', 'bss_calpers_classic'])}, Saturdays {v(TDT.loc['Saturday', 'bss_calpers_classic'])}, "
   f"Sundays {v(TDT.loc['Sunday', 'bss_calpers_classic'])}, classic). On test, Sat–Sun pooled give "
   f"{v(t_wkend['classic'])} / {v(t_wkend['ebike'])} and Mon–Tue {v(t_wkday['classic'])} / {v(t_wkday['ebike'])}; "
   f"leaving the weekend out closes {pct(wk_share['classic'])} (classic) and {pct(wk_share['ebike'])} (e-bike) of "
   f"the gap to validation. The test weekdays are still below every validation day. The test week also has fewer flips "
   f"({pct(TDATA.loc['test', 'classic_flip_rate'], 1)} against {pct(va.classic_flip_rate, 1)}, classic), and "
   f"the model's Brier score itself is lower on test ({v(ts(C, 'classic', 'all', 'brier'), 4)} against "
   f"{v(s(C, 'classic', 'all', 'brier'), 4)}): the reference improved more than the model. Calmer days tend to score "
   f"lower (table above), but that is not all of it: the test Monday, with the same classic flip rate as the "
   f"validation Monday "
   f"({pct(flipd(TD, '2026-10-26'), 2)} against {pct(flipd(BD, '2026-10-19'), 2)}), still scores "
   f"{v(dayb(TD, '2026-10-26'))} against {v(dayb(BD, '2026-10-19'))}. Untested: how much is the calmer week and how "
   f"much the distance from training (test days are 6–10 days after it, validation days 1–5).", "C")}
{d(f"Oct 28 holds only the early-morning hours ({int(tday[(tday.target == 'classic') & (tday.group == '2026-10-28')].rows.iloc[0]):,} "
   f"rows). Dropping the {int(UN.rows_dropped.iloc[0]):,} test rows at stations never seen in training changes "
   f"the skill by at most {v(un_diff, 4)}. The station-hour look-up was added after the test read and has no "
   "test score.")}
""")

# ---------------------------------------------------------------- 7. data v2
md(f"""
## 7. Preliminary: data v2

{d(f"A second, larger table (data v2) adds the days to Nov 10: {int(V2.loc['train', 'rows']):,} training rows "
   f"(to {V2.loc['train', 'last_day']}), and a validation week ({V2.loc['val', 'first_day']} to "
   f"{V2.loc['val', 'last_day']}) and test week ({V2.loc['test', 'first_day']} to {V2.loc['test', 'last_day']}) "
   "of 5 weekdays and 2 weekend days each. Its validation week overlaps the data-v1 test days, so its numbers are "
   "not a second data-v1 test; its own test week has not been read. So far only a 1-epoch smoke test has run, "
   "with the earlier count target (MSE): its skill is 1 − MSE_model / MSE_persistence, not the Brier skill above, "
   "and only shows that the pipeline runs end to end.")}
""")
code("""
v2 = pd.read_csv(T / "data_v2_smoke_fwd.csv")
v2 = v2[(v2.subset == "all") & (v2.day_type == "all")]
print("PRELIMINARY: count target (MSE), 1 epoch, data-v2 validation week")
print(v2.pivot(index="model", columns="target", values="skill").round(3))
""", hide=True)

# ---------------------------------------------------------------- 8. what is left
v2_ratio = V2.loc[["train", "val", "test"], "rows"].sum() / sum(DATA.rows) if "rows" in DATA else np.nan
v2_ratio = V2.loc[["train", "val", "test"], "rows"].sum() / (DATA.rows.sum() + TDATA.rows.sum())
md(f"""
## 8. What is left, and is it feasible?

{d("Done: data collection (running since Sep 29), the dataset, the score and its references, a small model "
   "search, and a test score. Left, in order:")}

1. {d("Data v2 (Nov 16–19): retrain the chosen model and the station-hour look-up on the larger table and check the "
      "weekend days on its validation week.")}
2. {d("Longer history (Nov 19–22): give the GRU 60 min of past counts, which data v2 provides; freeze the final "
      "model on Nov 22 and score it once on the unread data-v2 test week (Nov 4–10).")}
3. {d("Rider-level evaluation (Nov 23–29): turn the probabilities into advice (e.g. 'go elsewhere' below a "
      "threshold) and count strandings avoided and false alarms per 100 trips.")}
4. {d("Video (Dec 6) and final report (Dec 11).")}

{d(f"Compute is not the limit: the 16 runs of this search took {v(RUNS.wall_s.sum() / 3600, 1)} h of run time, "
   f"{v(span_h, 1)} h of wall time with several runs in parallel on one laptop; data v2 has "
   f"{v(v2_ratio, 1)} times the rows, so the same search should take about {v(v2_ratio * span_h, 1)} h. "
   "The person time is the limit:")}
{d("about 5 h per person per week (the team's budget from the proposal) leaves room for steps 1–3 if each "
   "person takes one.", "C")}

**Reproduce** (from `src/` of the project repository; this notebook only reads the tables these write):

```bash
python test_score_prob.py                       # checks of the score on toy cases
python 46_prob_search.py --logreg --lgbm --seed 0 1 2
python 46_prob_search.py --run mlp_2x64_e8 mlp_2x256_do0.1 mlp_2x256_do0.3 gru64_3x256_do0.1 --seed 0 1 2
python 47_prob_report.py                        # validation tables, the choice, figures
python 48_prob_test.py --dry                    # checks the saved models; the test score is not repeated
python 50_prob_checks.py                        # checks added after the reviews
python 49_progress_notebook.py                  # this notebook
```
""")


if __name__ == "__main__":
    nb = nbformat.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["title"] = "Will there be a bike? Progress report"
    OUT.parent.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": str(OUT.parent)}}).execute()
    nbformat.write(nb, OUT)
    exporter = HTMLExporter()
    exporter.exclude_input = False
    from traitlets.config import Config
    cfg = Config()
    cfg.TagRemovePreprocessor.remove_input_tags = ("hide-input",)
    cfg.TagRemovePreprocessor.enabled = True
    exporter = HTMLExporter(config=cfg)
    body, _ = exporter.from_notebook_node(nb, resources={"metadata": {"name": "Will there be a bike? Progress report"}})
    OUT.with_suffix(".html").write_text(body)
    print("wrote", OUT, "and", OUT.with_suffix(".html"))
    kn = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(KEY.strip())])
    kn.metadata["kernelspec"] = nb.metadata["kernelspec"]
    nbformat.write(kn, OUT.with_name("key_findings.ipynb"))
    print("wrote", OUT.with_name("key_findings.ipynb"))
    print("draft tags:")
    for tag, words in TAGS:
        print(f"  [{tag}] {words} ...")
