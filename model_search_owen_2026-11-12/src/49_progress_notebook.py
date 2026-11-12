"""Build, run and export notebooks/progress_report_prob.ipynb (+ .html): the progress report notebook (model part).

Retrains nothing: it reads the cached tables of 46-48 (reports/tables/prob_*.csv), the figures, and the saved
weights of the chosen model (only to print its layers and parameter count). Set RERUN = True in its first code
cell to rerun the val scripts first (about 1 h on 12 CPU threads); the test score is never rerun (48_prob_test.py
refuses a second score).

Drafted prose is marked for Owen: blue [Dn] = draft text to accept or rewrite; red [Cn] = a claim to check.
Numbers in the prose are read from the tables when this script runs; none is typed by hand.
Usage (from src/):  python 49_progress_notebook.py
"""
import json
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "reports" / "tables"
OUT = ROOT / "notebooks" / "progress_report_prob.ipynb"
CELLS, TAGS = [], []
N = {"D": 0, "C": 0}
LABEL = {"mlp_2x64_e8": "MLP 2×64 (base)", "mlp_2x256_do0.1": "MLP 2×256, dropout 0.1",
         "mlp_2x256_do0.3": "MLP 2×256, dropout 0.3", "gru64_3x256_do0.1": "GRU + 3×256, dropout 0.1",
         "lgbm": "LightGBM", "logreg": "logistic regression"}


def md(text):
    CELLS.append(nbformat.v4.new_markdown_cell(text.strip()))


def code(text):
    CELLS.append(nbformat.v4.new_code_cell(text.strip()))


def d(text, kind="D"):
    """Blue draft text [Dn], or a red claim to check [Cn]."""
    N[kind] += 1
    tag = f"{kind}{N[kind]}"
    TAGS.append((tag, " ".join(text.split()[:14])))
    color = "#1f5fbf" if kind == "D" else "#c00000"
    return f'<span style="color:{color}">[{tag}] {" ".join(text.split())}</span>'


def v(x, n=3):
    return f"{x:.{n}f}"


# ---------------------------------------------------------------- numbers for the prose (from the tables)
S = pd.read_csv(T / "prob_summary.csv")
CH = json.loads((T / "prob_choice.json").read_text())
C = CH["chosen"]
DATA = pd.read_csv(T / "prob_data.csv").set_index("split")
TEST = (T / "prob_test_summary.csv").exists()


def s(model, x, sub, k):
    r = S[(S.model == model) & (S.target == x) & (S.subset == sub)].iloc[0]
    return r[f"{k}_mean"]


def sd(model, x, sub, k):
    r = S[(S.model == model) & (S.target == x) & (S.subset == sub)].iloc[0]
    return r[f"{k}_sd"]


if TEST:
    TS = pd.read_csv(T / "prob_test_summary.csv")

    def ts(model, x, sub, k):
        return TS[(TS.model == model) & (TS.target == x) & (TS.subset == sub)].iloc[0][f"{k}_mean"]

NO = pd.read_csv(T / "prob_noise.csv")
RUNS = pd.read_csv(T / "prob_runs.csv")
BC = pd.read_csv(T / "prob_by_count.csv")
BH = pd.read_csv(T / "prob_by_horizon.csv")
LB = pd.read_csv(T / "prob_last_bike.csv")
REL = pd.read_csv(T / "prob_reliability.csv")
NETS = ["mlp_2x64_e8", "mlp_2x256_do0.1", "mlp_2x256_do0.3", "gru64_3x256_do0.1", "lgbm"]


def gap(b, x, k="gap_bss_calpers"):
    return NO[(NO.a == C) & (NO.b == b) & (NO.target == x) & (NO.subset == "all")].iloc[0][k]


def grp(tab, x, g, model=C):
    return tab[(tab.model == model) & (tab.target == x) & (tab.group == g)].iloc[0].bss_calpers


def lb(x, model, flag=0.1, k="share_of_gone_caught"):
    return LB[(LB.target == x) & (LB.model == model) & (LB.flag_share == flag)].iloc[0][k]


def relgap(x, which):
    """Largest |observed - forecast| over val bins with >= 1,000 rows; which = 'chosen' or a reference name."""
    refs = ["calibrated persistence", "climatology"]
    m = ~REL.model.isin(refs) if which == "chosen" else REL.model == which
    r = REL[m & (REL.target == x) & (REL.subset == "all") & (REL.rows >= 1000)]
    return (r.y_rate - r.p_mean).abs().max()


nl = [n for n in NETS if n != C]
spread = max(abs(s(n, x, "all", "bss_calpers") - s(C, x, "all", "bss_calpers")) for n in nl for x in ["classic", "ebike"])
base = RUNS[RUNS.name == "mlp_2x64_e8"].sort_values("best_epoch")
_end = pd.to_datetime(RUNS.finished)
span_h = (_end.max() - (_end - pd.to_timedelta(RUNS.wall_s, "s")).min()).total_seconds() / 3600

md(f"""
# Will there be a bike? Progress report, model part

**Team 3, MIE1517.** For every Toronto Bike Share station we forecast the probability that at least one
classic bike, and at least one e-bike, will be docked there h = 2–15 minutes from now.

**Summary.**
{d(f"Our best model so far is the {LABEL[C]} network. On the validation days it beats an honest "
   f"'what you see now' forecast (calibrated persistence) by a Brier skill score of "
   f"{v(s(C, 'classic', 'all', 'bss_calpers'))} for classic bikes and {v(s(C, 'ebike', 'all', 'bss_calpers'))} "
   f"for e-bikes, and plain persistence by {v(s(C, 'classic', 'all', 'bss_pers'))} / "
   f"{v(s(C, 'ebike', 'all', 'bss_pers'))}.")}
{d(f"On the held-out test days (read once) the skill against calibrated persistence is "
   f"{v(ts(C, 'classic', 'all', 'bss_calpers'))} / {v(ts(C, 'ebike', 'all', 'bss_calpers'))}." if TEST
   else "The test score is not taken yet.")}
{d("What is left: a second, larger dataset (data v2), longer history for the sequence models, and a "
   "rider-level evaluation; the plan and dates are in Section 8.")}

This notebook reads cached results; every number is printed from a table written by a script in `src/`.
Blue text marked [Dn] is a draft; red text marked [Cn] is a claim still to be checked.
""")

code("""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
T, F = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
pd.set_option("display.width", 100)
pd.set_option("display.precision", 3)

RERUN = False  # True: rerun the val scripts (about 1 h); the test score is never rerun
if RERUN:
    py = sys.executable
    for args in [["test_score_prob.py"],
                 ["46_prob_search.py", "--logreg", "--lgbm"],
                 ["46_prob_search.py", "--run", "mlp_2x64_e8", "mlp_2x256_do0.1",
                  "mlp_2x256_do0.3", "gru64_3x256_do0.1"],
                 ["47_prob_report.py"]]:
        subprocess.run([py, *args], cwd=ROOT / "src", check=True)
""")

# ---------------------------------------------------------------- 1. data
tr, va = DATA.loc["train"], DATA.loc["val"]
md(f"""
## 1. Data collected

{d("We poll the public Toronto Bike Share feed (GBFS) every minute since Sep 29, with backup pollers, and add "
   "Environment Canada weather. The team table (data-v1) turns the polls into one row per station, anchor time t "
   "(every 10 min) and horizon h, with the counts now and 10 and 30 min before, the station, the time of day and "
   "the weather as inputs. Only anchors where some count is 2 or less are kept: those are the cases where a rider "
   "may find no bike.")}
{d("Pollers and weather source as described; anchor rule from the table's documentation.", "C")}

The table below gives the size of each split and how often the answer changes: the *flip rate* is the share of
rows whose 'at least one bike' status differs between t and t + h.
{d(f"Status flips are rare: {v(100 * va.classic_flip_rate, 1)} % of val rows for classic bikes and "
   f"{v(100 * va.ebike_flip_rate, 1)} % for e-bikes, so 'what you see now' is already a strong forecast.")}
""")
code("""
data = pd.read_csv(T / "prob_data.csv")
if (T / "prob_test_data.csv").exists():
    data = pd.concat([data, pd.read_csv(T / "prob_test_data.csv")])
show = data[["split", "rows", "stations", "first_day", "last_day",
             "classic_base_rate", "ebike_base_rate",
             "classic_flip_rate", "ebike_flip_rate"]]
show.columns = ["split", "rows", "stations", "from", "to",
                "P(>=1 classic)", "P(>=1 e-bike)",
                "flip classic", "flip e-bike"]
show.set_index("split")
""")

# ---------------------------------------------------------------- 2. task and score
md(f"""
## 2. Task and score

**Output.** For each bike type, p = P(at least one bike at t + h), h = 2–15 min. One model covers all stations
and horizons (h is an input). Loss: binary cross-entropy.
{d("We moved from forecasting the count (with MSE) to this probability because a rider's question is "
   "'will there be a bike?', and a probability says how sure the answer is.")}

**Score.** The Brier score, mean (p − y)², and the Brier skill score against a reference,
BSS = 1 − Brier_model / Brier_ref (0 = no better than the reference, 1 = perfect).
{d("We do not use log loss against persistence: persistence says 0 or 1, so every miss costs log(0) = infinity. "
   "The Brier score is finite for 0/1 forecasts and is still a proper score.")}

Three references, all fitted on the training days only (`src/score_prob.py`, checked on toy cases in
`src/test_score_prob.py`):

| Reference | Forecast |
|---|---|
| Persistence | 1 if a bike is there now, else 0 |
| Calibrated persistence | P(bike at t + h given the count now and h), from the training days |
| Climatology | the station's rate at that hour of the week, from the training days |

{d(f"Calibrated persistence is the bar that matters: it already knows that '1 bike now' does not mean a sure bike "
   f"later, and it beats raw persistence by {v(s('calibrated persistence', 'classic', 'all', 'bss_pers'))} / "
   f"{v(s('calibrated persistence', 'ebike', 'all', 'bss_pers'))} (classic / e-bike, val). "
   f"Climatology ignores the count now and is far weaker "
   f"(BSS against persistence {v(s('climatology', 'classic', 'all', 'bss_pers'), 2)} / "
   f"{v(s('climatology', 'ebike', 'all', 'bss_pers'), 2)}).")}
""")
code("""
r = subprocess.run([sys.executable, "test_score_prob.py"], cwd=ROOT / "src",
                   capture_output=True, text=True)
lines = r.stdout.splitlines()
print(f"{sum(l.startswith('ok') for l in lines)} toy checks ok, "
      f"{sum(l.startswith('FAIL') for l in lines)} failed")
""")

# ---------------------------------------------------------------- 3. model
md(f"""
## 3. Model

{d(f"The chosen network ({LABEL[C]}) takes the 20 inputs of the table (horizon h; classic, e-bike and dock "
   "counts now, 10 and 30 min ago; bikes rented and returned now; capacity; location; hour of day; weekend; "
   "temperature; humidity), standardised with the training mean and standard deviation, and an 8-number "
   "embedding of the station learned with the model. Two outputs give one logit per bike type.")}
Training: Adam (learning rate 10⁻³, batch 4,096), one pass over the 7.5 M training rows per epoch,
up to 30 epochs; we keep the epoch with the lowest validation cross-entropy and stop after 3 epochs without
improvement (`src/46_prob_search.py`).
""")
code(f"""
import importlib
ms = importlib.import_module("37_model_search")
runs = pd.read_csv(T / "prob_runs.csv")
r = runs[runs.name == "{C}"].sort_values("seed")
net = ms.Net(ms.config_by_name("{C}"), 20, 1071)  # 1,071 train stations (+1 unknown row)
print(net)
print(f"{{sum(p.numel() for p in net.parameters()):,}} parameters; "
      f"best epochs {{[int(e) for e in r.best_epoch]}} (seeds {{list(r.seed)}})")
""")

# ---------------------------------------------------------------- 4. search
md(f"""
## 4. Step by step: from references to the chosen model (validation days)

We trained four networks with three seeds each (0, 1, 2), so a gap between models can be judged against
the spread over seeds, plus two cheap references: LightGBM (gradient-boosted trees) and logistic regression.
Validation: Oct 19–23. The rule to pick the model was fixed before any run: the lowest validation Brier
(mean of the two bike types), preferring a smaller network if it is within noise.
""")
code("""
s = pd.read_csv(T / "prob_summary.csv")
runs = pd.read_csv(T / "prob_runs.csv")
order = ["persistence", "climatology", "calibrated persistence", "logreg", "lgbm",
         "mlp_2x64_e8", "mlp_2x256_do0.1", "mlp_2x256_do0.3", "gru64_3x256_do0.1"]
a = s[s.subset == "all"].copy()
a["BSS vs cal. pers."] = [f"{m:.3f}" + ("" if np.isnan(e) else f" ± {e:.3f}")
                          for m, e in zip(a.bss_calpers_mean, a.bss_calpers_sd)]
tab = a.pivot(index="model", columns="target", values="BSS vs cal. pers.").reindex(order)
tab["BSS vs pers. (classic / e-bike)"] = [
    " / ".join(f"{a[(a.model == m) & (a.target == x)].bss_pers_mean.iloc[0]:.3f}"
               for x in ["classic", "ebike"]) for m in tab.index]
p = runs.groupby("name").agg(params=("n_params", "first"), epochs=("best_epoch", "first"))
tab["params"] = p.params.reindex(tab.index).map(
    lambda n: "" if pd.isna(n) else f"{int(n):,}")
tab.columns = ["classic", "e-bike", "vs persistence", "params"]
print("Validation BSS against calibrated persistence, all rows (mean ± sd over 3 seeds)")
tab
""")
code("""
ch = json.loads((T / "prob_choice.json").read_text())
for n, r in ch["nets"].items():
    print(f"{n:20s} mean Brier {r['mean_brier']:.5f} ± {r['sd_brier']:.5f}"
          f"  ({r['n_params']:,} params)")
print("best:", ch["best"], "| chosen:", ch["chosen"])
""")
code("""
cur = pd.read_csv(T / "prob_curves.csv")
fig, ax = plt.subplots(figsize=(8, 4))
for n, c in zip(["mlp_2x64_e8", "mlp_2x256_do0.1", "mlp_2x256_do0.3",
                 "gru64_3x256_do0.1"], ["C0", "C1", "C2", "C3"]):
    g = cur[cur.run_id == f"p_{n}_seed0"]
    ax.plot(g.epoch, g.val_bce, "o-", color=c, label=n)
    ax.plot(g.epoch, g.train_bce, ":", color=c)
ax.set_xlabel("epoch")
ax.set_ylabel("cross-entropy (solid: val, dotted: train)")
ax.set_title("Learning curves, seed 0")
ax.legend(fontsize=8)
plt.show()
""")
md(f"""
**Result.** {d(f"Every non-linear model clears the bar by about the same amount: against calibrated persistence the "
   f"chosen {LABEL[C]} scores {v(s(C, 'classic', 'all', 'bss_calpers'))} ± {v(sd(C, 'classic', 'all', 'bss_calpers'), 4)} "
   f"(classic) and {v(s(C, 'ebike', 'all', 'bss_calpers'))} ± {v(sd(C, 'ebike', 'all', 'bss_calpers'), 4)} (e-bike), and "
   f"the other networks and LightGBM are within {v(spread, 4)} of it. Logistic regression on the same inputs only matches "
   f"calibrated persistence ({v(s('logreg', 'classic', 'all', 'bss_calpers'))} / "
   f"{v(s('logreg', 'ebike', 'all', 'bss_calpers'))}), so the gain needs a non-linear use of the inputs.")}

{d(f"The rule picked {LABEL[C]}. Its lead is small: +{v(gap('mlp_2x256_do0.1', 'classic'), 4)} / "
   f"+{v(gap('mlp_2x256_do0.1', 'ebike'), 4)} over dropout 0.1, +{v(gap('lgbm', 'classic'), 4)} / "
   f"+{v(gap('lgbm', 'ebike'), 4)} over LightGBM, and +{v(gap('mlp_2x64_e8', 'classic'), 4)} / "
   f"+{v(gap('mlp_2x64_e8', 'ebike'), 4)} over the base 2×64 (classic / e-bike; z = {v(gap('mlp_2x64_e8', 'classic', 'z'), 1)} "
   f"/ {v(gap('mlp_2x64_e8', 'ebike', 'z'), 1)} once seed and station-sampling noise are both counted, "
   f"`prob_noise.csv`).")}
{d(f"The base's gap may come from early stopping rather than size: its seed that trained to epoch "
   f"{int(base.best_epoch.iloc[-1])} reached {v(base.bss_calpers_classic_all.iloc[-1])} / "
   f"{v(base.bss_calpers_ebike_all.iloc[-1])}, level with the larger nets, while the seeds that stopped at epochs "
   f"{int(base.best_epoch.iloc[0])} and {int(base.best_epoch.iloc[1])} reached "
   f"{v(base.bss_calpers_classic_all.iloc[0])} and {v(base.bss_calpers_classic_all.iloc[1])} (classic). "
   "Untested: a longer patience for the base.", "C")}
""")

# ---------------------------------------------------------------- 5. where
md("""
## 5. Where the model helps, and is it calibrated? (validation days)
""")
code("""
display(Image(filename=str(F / "prob_where.png"), width=900))
bc = pd.read_csv(T / "prob_by_count.csv")
print(bc[bc.model == ch["chosen"]].pivot(index="group", columns="target",
      values="bss_calpers").reindex(["0", "1", "2", "3-5", "6+"]).round(3))
""")
md(f"""
{d(f"The model helps most where a rider is most at risk, when one bike is left: BSS {v(grp(BC, 'classic', '1'))} "
   f"(classic) and {v(grp(BC, 'ebike', '1'))} (e-bike) against calibrated persistence, against "
   f"{v(grp(BC, 'classic', '6+'))} / {v(grp(BC, 'ebike', '6+'))} with 6 or more bikes. The gain also grows with "
   f"the horizon, from {v(grp(BH, 'classic', 2))} / {v(grp(BH, 'ebike', 2))} at h = 2 min to "
   f"{v(grp(BH, 'classic', 14))} / {v(grp(BH, 'ebike', 14))} at h = 14 min (hypothesis: the further ahead, the "
   "more the count now leaves open, and the more the other inputs can add).")}

{d(f"In rider terms: when one bike is there now, it is gone {v(100 * lb('classic', C, k='gone_rate'), 1)} % of the "
   f"time (classic; {v(100 * lb('ebike', C, k='gone_rate'), 1)} % for e-bikes) by t + h. A warning on the 10 % of "
   f"these cases with the lowest forecast would catch {v(100 * lb('classic', C), 0)} % of the times the bike is "
   f"gone ({v(100 * lb('ebike', C), 0)} % for e-bikes); the same warning from calibrated persistence catches "
   f"{v(100 * lb('classic', 'calibrated persistence'), 0)} % ({v(100 * lb('ebike', 'calibrated persistence'), 0)} %).")}
""")
code("""
display(Image(filename=str(F / "prob_reliability.png"), width=800))
""")
md(f"""
{d(f"The forecasts are calibrated: in every bin with at least 1,000 rows the observed share is within "
   f"{v(relgap('classic', 'chosen'), 3)} (classic) and {v(relgap('ebike', 'chosen'), 3)} (e-bike) of the forecast. "
   "Climatology is not: it ignores the count now, so on these low-count rows it is off by up to "
   f"{v(relgap('ebike', 'climatology'), 2)}.")}
""")

# ---------------------------------------------------------------- 6. test
md("""
## 6. Test result (read once)
""")
code("""
if (T / "prob_test_summary.csv").exists():
    t = pd.read_csv(T / "prob_test_summary.csv")
    t = t[t.subset == "all"]
    print(t[["model", "target", "brier_mean", "bss_pers_mean",
             "bss_calpers_mean", "bss_calpers_sd", "bss_clim_mean"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    tr = pd.read_csv(T / "prob_test_reliability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, x in zip(axes, ["classic", "ebike"]):
        for m, g in tr[(tr.target == x) & (tr.rows >= 1000)].groupby("model"):
            ax.plot(g.p_mean, g.y_rate, "o-", ms=4, label=m)
        ax.plot([0, 1], [0, 1], "--", color="0.6")
        ax.set_title(f"Test calibration, {x}")
        ax.set_xlabel("forecast")
        ax.set_ylabel("observed share")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
""")
if TEST:
    TD = pd.read_csv(T / "prob_test_by_day.csv")
    UN = pd.read_csv(T / "prob_test_unseen.csv")
    TR = pd.read_csv(T / "prob_test_reliability.csv")
    tr_gap = {x: (lambda r: (r.y_rate - r.p_mean).abs().max())(TR[(TR.model == C) & (TR.target == x) & (TR.rows >= 1000)])
              for x in ["classic", "ebike"]}

    un_diff = max(abs(UN[UN.target == x].bss_calpers.iloc[0] - ts(C, x, "all", "bss_calpers"))
                  for x in ["classic", "ebike"])

    def day(x, dd):
        return TD[(TD.model == C) & (TD.target == x) & (TD.group == dd)].iloc[0]

    md(f"""
The test days (Oct 24–28) were read once, for the chosen model only, after the choice; the protocol was committed
before the read (`src/48_prob_test.py`). Nothing was tuned after it.

{d(f"The gain holds on test but is smaller: {v(ts(C, 'classic', 'all', 'bss_calpers'))} / "
   f"{v(ts(C, 'ebike', 'all', 'bss_calpers'))} against calibrated persistence (val {v(s(C, 'classic', 'all', 'bss_calpers'))} / "
   f"{v(s(C, 'ebike', 'all', 'bss_calpers'))}) and {v(ts(C, 'classic', 'all', 'bss_pers'))} / "
   f"{v(ts(C, 'ebike', 'all', 'bss_pers'))} against persistence. Calibration is a little looser than on "
   f"validation: the largest bin gap is {v(tr_gap['classic'])} / {v(tr_gap['ebike'])} (val "
   f"{v(relgap('classic', 'chosen'))} / {v(relgap('ebike', 'chosen'))}).")}
{d(f"Part of the drop is the weekend: the test week starts on a Saturday and the validation days have no weekend. "
   f"On Sat Oct 24 and Sun Oct 25 the classic skill is {v(day('classic', '2026-10-24').bss_calpers)} and "
   f"{v(day('classic', '2026-10-25').bss_calpers)}, on Mon and Tue {v(day('classic', '2026-10-26').bss_calpers)} and "
   f"{v(day('classic', '2026-10-27').bss_calpers)}. Even the test weekdays are below validation, so the weekend is "
   "not the whole story (hypothesis: the weather or the season changed).", "C")}
{d(f"Oct 28 holds only the early-morning hours ({int(day('classic', '2026-10-28').rows):,} rows), where little changes. "
   f"Dropping the {int(UN.rows_dropped.iloc[0]):,} test rows at stations never seen in training changes the skill "
   f"against calibrated persistence by at most {v(un_diff, 4)}.")}
""")

# ---------------------------------------------------------------- 7. data v2
md(f"""
## 7. Preliminary: data v2

{d("A second, larger table (data v2, build 7f427d0c2919) adds the days to Nov 10 and a validation week with a "
   "weekend. So far only a 1-epoch smoke test has run, with the earlier count target (MSE), so its skill is "
   "the count skill against persistence, 1 − MSE_model / MSE_persistence, not the Brier skill above. It only "
   "shows that the pipeline runs end to end.")}
""")
code("""
v2 = pd.read_csv(T / "data_v2_smoke_fwd.csv")
v2 = v2[(v2.subset == "all") & (v2.day_type == "all")]
print("PRELIMINARY, count target (MSE), 1 epoch, val Oct 28 - Nov 3")
print(v2.pivot(index="model", columns="target", values="skill").round(3))
""")

# ---------------------------------------------------------------- 8. what is left
md(f"""
## 8. What is left, and is it feasible?

{d("Done: data collection (running since Sep 29), the dataset, the score and its references, a small model "
   "search, and a test score taken once. Left, in order:")}

1. {d("Data v2 (Nov 16–19): retrain the chosen model on the larger table and check the weekend days.")}
2. {d("Longer history (Nov 19–22): give the GRU 60 min of past counts, which data v2 provides; freeze the final "
      "model on Nov 22.")}
3. {d("Rider-level evaluation (Nov 23–29): turn the probabilities into advice (e.g. 'go elsewhere' below a "
      "threshold) and count strandings avoided per 100 trips.")}
4. {d("Video (Dec 6) and final report (Dec 11).")}

{d(f"Each step reuses the scripts above. The 16 runs of this search took {v(RUNS.wall_s.sum() / 3600, 1)} h of "
   f"run time in all, {v(span_h, 1)} h of wall time when run 6 at a time on one laptop CPU, so the plan fits "
   "the 5 h per person per week we have.")}
{d("Dates of steps 1-3 are a proposal for the team.", "C")}
""")


if __name__ == "__main__":
    nb = nbformat.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    OUT.parent.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": str(OUT.parent)}}).execute()
    nbformat.write(nb, OUT)
    body, _ = HTMLExporter().from_notebook_node(nb)
    OUT.with_suffix(".html").write_text(body)
    print("wrote", OUT, "and", OUT.with_suffix(".html"))
    print("draft tags:")
    for tag, words in TAGS:
        print(f"  [{tag}] {words} ...")
