"""Build, run and export notebooks/model_search_trace.ipynb: every number of reports/MODEL_SEARCH.md from its file.

The notebook retrains nothing. It reads the CSV logs in reports/tables/, the saved val predictions of
37_model_search.py --save (to recompute their skill), the data file's bytes (SHA-1) and the train and val rows
through score.load (to show that the search scripts never read the test split).

Usage (from src/):  python 41_trace_notebook.py
Writes notebooks/model_search_trace.ipynb (run top to bottom, outputs saved) and model_search_trace.html.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "model_search_trace.ipynb"
CELLS = []


def md(text):
    CELLS.append(nbformat.v4.new_markdown_cell(text.strip()))


def code(text):
    CELLS.append(nbformat.v4.new_code_cell(text.strip()))


md("""
# Model search: where each number comes from

Each cell prints numbers of `reports/MODEL_SEARCH.md` from the file they come from,
with the source path, the script that wrote it and the last commit of that file.
Nothing is retrained. Val split only: the search never reads the test rows (section 2). The one test score, taken afterwards, is in `test_team_summary.csv` (`src/42_test_score.py`).
Built and run by `src/41_trace_notebook.py`.
""")

code("""
import hashlib, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from common import DATA_DIR
from score import TEAM, TARGETS, cols, load, score_table

T = ROOT / "reports" / "tables"
pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 12)


def commit(path):
    out = subprocess.run(["git", "log", "-1", "--format=%h %ad", "--date=short",
                          "--", str(path)], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() or "uncommitted"


def src(path, script):
    p = Path(path)
    rel = p.relative_to(ROOT) if p.is_absolute() and ROOT in p.parents else p
    print(f"  source: {rel}")
    print(f"  script: {script}")
    print(f"  commit: {commit(path)}")


runs = pd.read_csv(T / "model_search_runs.csv")
summ = pd.read_csv(T / "model_search_summary.csv")
print("repo HEAD:", commit(ROOT))
print("runs:", len(runs), " configurations:", runs.name.nunique())
src(T / "model_search_runs.csv", "src/37_model_search.py")
""")

md("## 1. Data file: SHA-1")
code("""
expected = "263bb791129bccf2073b1c4e827acf82eafeb96e"
h = hashlib.sha1()
with open(TEAM.path, "rb") as f:  # bytes only; no rows are decoded
    for block in iter(lambda: f.read(1 << 20), b""):
        h.update(block)
print("file:     $TEAM_TABLE =", TEAM.path.name)
print("sha1:    ", h.hexdigest())
print("expected:", expected)
assert h.hexdigest() == expected
""")

md("""
## 2. The search never reads the test split

(a) every `load(` call in the search scripts asks for train and val only;
(b) `score.load` filters the split inside the parquet read and checks the split rule;
(c) loading what the scripts load gives train and val rows only.
""")
code("""
for f in ["37_model_search.py", "38_search_report.py", "39_val_noise.py",
          "40_meeting_figures.py", "41_trace_notebook.py"]:
    for i, line in enumerate((ROOT / "src" / f).read_text().splitlines(), 1):
        if re.search(r"\\bload\\(\\[", line):
            print(f"{f}:{i}: {line.strip()[:80]}")
""")
code("""
import inspect, score
for line in inspect.getsource(score.load).splitlines():
    if "filters" in line or "assert" in line:
        print(line.strip()[:95])
""")
code("""
df = load(["train", "val"], columns=["station_id"], ds=TEAM)
day = df.t_local.str[:10]
g = df.assign(day=day).groupby("split")
print(pd.DataFrame({"rows": g.size(), "first day": g.day.min(),
                    "last day": g.day.max(),
                    "stations": g.station_id.nunique()}))
assert set(df.split) == {"train", "val"}
print("split values loaded:", sorted(set(df.split)))
print("test columns in the run log:",
      [c for c in runs.columns if "test" in c])
seen = set(df[df.split == "train"].station_id)
new = df[(df.split == "val") & ~df.station_id.isin(seen)]
print("val rows at stations not in train:", len(new),
      "at", new.station_id.nunique(), "station(s)")
del df
""")

md("## 3. Setup and the reproduction check")
code("""
p = pd.read_csv(T / "mlp_team_val.csv")
pers = p[(p.model == "persistence") & (p.subset == "all")]
print(pers[["target", "rows", "mse"]].round(4).to_string(index=False))
src(T / "mlp_team_val.csv", "src/33_train_mlp.py --data team")
full = runs[runs.budget == "full"]
print("longest full run (epochs):", int(full.epochs_run.max()))
lg = runs[runs.arch == "lgbm"][["run_id", "threads", "wall_s"]]
print(lg.to_string(index=False))
src(T / "model_search_runs.csv", "src/37_model_search.py")
""")
code("""
r = runs.set_index("run_id").loc["s3_mlp_2x64_e8_full_seed0"]
m = p[p.model == "mlp_team"].set_index(["target", "subset"]).skill
print("search base seed 0:", round(r.skill_classic_all, 6),
      round(r.skill_ebike_all, 6))
print("Nov 10 run seed 0: ", round(m["classic", "all"], 6),
      round(m["ebike", "all"], 6))
src(T / "mlp_team_val.csv", "src/33_train_mlp.py --data team")
""")

md("""
Saved val predictions (`37_model_search.py --save`, retrains of logged runs):
their skill recomputed here from the .npy files, against the log.
""")
code("""
va = load(["val"], columns=[*cols(TEAM, "now"), *cols(TEAM, "truth"),
                            *cols(TEAM, "low")], ds=TEAM)
saved = DATA_DIR / "models" / "search"
rows = []
for f in sorted(saved.glob("*.npy")):
    t = score_table(va, {"m": np.load(f)}, TEAM)
    t = t[(t.model == "m") & (t.subset == "all")].set_index("target").skill
    lg = runs.set_index("run_id").loc[f.stem]
    rows.append({"run": f.stem[:30], "classic": t["classic"],
                 "log": lg.skill_classic_all, "ebike": t["ebike"],
                 "log ": lg.skill_ebike_all})
chk = pd.DataFrame(rows)
print(chk.round(6).to_string(index=False))
diff = max((chk.classic - chk.log).abs().max(), (chk.ebike - chk["log "]).abs().max())
print("largest difference:", diff)
print("  source: $MIE1517_DATA/models/search (not in git)")
print("  script: src/37_model_search.py --save")
""")

md("## 4. Steps 1 and 2 (short budget, seeds 0-1)")
code("""
s = summ[summ.budget == "short"].copy()
r = runs[runs.budget == "short"].groupby("name")
s["range_c"] = s.name.map(r.skill_classic_all.agg(lambda x: x.max() - x.min()))
s["range_e"] = s.name.map(r.skill_ebike_all.agg(lambda x: x.max() - x.min()))
show = s[["step", "name", "skill_classic_all_mean",
          "skill_ebike_all_mean", "range_c", "range_e"]]
show.columns = ["step", "name", "classic", "ebike", "range c", "range e"]
print(show.round(4).to_string(index=False))
print("largest seed range, step 1:",
      round(s[s.step == 1][["range_c", "range_e"]].max().max(), 4))
src(T / "model_search_summary.csv", "src/38_search_report.py")
""")
code("""
lg = summ[summ.name == "lgbm_l127"]
r0 = runs.set_index("run_id").loc["s1_lgbm_l127_lr0.1_seed0"]
print("LightGBM seed 0:", round(r0.skill_classic_all, 4),
      round(r0.skill_ebike_all, 4))
src(T / "model_search_runs.csv", "src/37_model_search.py --lgbm")
""")

md("## 5. Table 1 (full budget, seeds 0-2)")
code("""
f = summ[(summ.budget == "full") | (summ.name == "lgbm_l127")]
t1 = pd.DataFrame({"name": f.name, "seeds": f.seeds,
                   "params": f.n_params, "best epoch": f.best_epoch})
for c in ["classic_all", "ebike_all", "classic_low", "ebike_low"]:
    m, sd = f[f"skill_{c}_mean"], f[f"skill_{c}_sd"]
    t1[c] = m.map("{:.3f}".format) + " ± " + sd.map("{:.4f}".format)
print(t1.to_string(index=False))
src(T / "model_search_summary.csv", "src/38_search_report.py")
""")
code("""
# all three chosen seeds beat all three base seeds, on all four columns
a = runs[(runs.name == "mlp_2x256_do0.1") & (runs.budget == "full")]
b = runs[(runs.name == "mlp_2x64_e8") & (runs.budget == "full")]
for c in ["classic_all", "ebike_all", "classic_low", "ebike_low"]:
    lo, hi = a[f"skill_{c}"].min(), b[f"skill_{c}"].max()
    print(f"{c:12s} chosen min {lo:.4f} > base max {hi:.4f}: {lo > hi}")
print("params ratio 3x256 / 2x256:", round(148098 / 82306, 2),
      " GRU / 2x256:", round(176002 / 82306, 2),
      " 2x256 / 2x64:", round(82306 / 14722, 1))
src(T / "model_search_runs.csv", "src/37_model_search.py")
""")

md("## 6. Step 3: curves with and without dropout (mean val MSE over seeds)")
code("""
cv = pd.read_csv(T / "model_search_curves.csv")
out = {}
for n in ["s3_mlp_2x256_e8_full_seed", "s3_mlp_2x256_do0.1_full_seed"]:
    g = cv[cv.run_id.str.startswith(n)].groupby("epoch").val_mse
    out[n[3:-10]] = g.mean().where(g.size() == 3)
print(pd.DataFrame(out).head(10).round(5).to_string())
src(T / "model_search_curves.csv", "src/37_model_search.py")
""")
code("""
sec = summ.set_index(["name", "budget"]).s_per_epoch
print("2x256 s/epoch, short vs full:", sec["mlp_2x256_e8", "short"],
      sec["mlp_2x256_e8", "full"],
      " ratio", round(sec["mlp_2x256_e8", "short"] / sec["mlp_2x256_e8", "full"], 2))
src(T / "model_search_summary.csv", "src/38_search_report.py")
""")

md("## 7. Step 5: the dropout trend on 2 x 256")
code("""
names = ["mlp_2x256_e8", "mlp_2x256_do0.1", "mlp_2x256_do0.2",
         "mlp_2x256_do0.3", "mlp_2x256_do0.5"]
d = summ[(summ.budget == "full") & summ.name.isin(names)].set_index("name").loc[names]
print(d[["best_epoch", "skill_classic_all_mean",
         "skill_ebike_all_mean"]].round(4).to_string())
src(T / "model_search_summary.csv", "src/38_search_report.py")
""")

md("## 8. Table 2: val-sampling noise")
code("""
g = pd.read_csv(T / "val_noise_gap.csv")
print(g[["pair", "target", "subset", "gap", "seed_sd", "station_boot_sd",
         "station_lo", "station_hi", "days_positive", "z"]]
      .round(4).to_string(index=False))
src(T / "val_noise_gap.csv", "src/39_val_noise.py")
""")
code("""
bd = pd.read_csv(T / "val_noise_by_day.csv")
print(bd[bd.subset == "all"][["target", "day", "skill_base", "skill_chosen",
      "skill_lgbm", "gap_chosen_base"]].round(3).to_string(index=False))
src(T / "val_noise_by_day.csv", "src/39_val_noise.py")
""")
code("""
o = pd.read_csv(T / "val_noise_optimism.csv")
o = o[o.run_id == "mean over seeds"]
print(o[["name", "skill_best", "skill_neighbours", "optimism"]]
      .round(4).to_string(index=False))
src(T / "val_noise_optimism.csv", "src/39_val_noise.py")
""")
code("""
low = p[(p.model == "persistence") & (p.target == "ebike")].set_index("subset").rows
print("e-bike low rows:", low["low"], "of", low["all"],
      f"-> rows with 3+ e-bikes: {1 - low['low'] / low['all']:.1%}")
src(T / "mlp_team_val.csv", "src/33_train_mlp.py --data team")
""")

md("## 9. By horizon")
code("""
bh = pd.read_csv(T / "model_search_by_horizon_summary.csv")
sel = bh[bh.h_target.isin([2, 14, 15]) & bh.name.isin(
    ["mlp_2x64_e8", "mlp_2x256_do0.1", "lgbm_l127"])]
print(sel.pivot_table(index=["target", "h_target"], columns="name",
                      values="skill_mean").round(4).to_string())
src(T / "model_search_by_horizon_summary.csv", "src/38_search_report.py")
rh = pd.read_csv(T / "model_search_by_horizon.csv")
n = rh[rh.run_id == "s3_mlp_2x256_do0.1_full_seed0"]
print("val rows at h = 14 and 15:",
      n.groupby("h_target").rows.first()[[14, 15]].tolist())
src(T / "model_search_by_horizon.csv", "src/37_model_search.py")
""")

md("## 10. Step 6: training-set size (val only)")
code("""
d = pd.read_csv(T / "data_size_summary.csv")
print(d[["dropout", "train_days", "first_day", "train_rows", "best_epochs",
         "classic_all_mean", "ebike_all_mean"]].round(4).to_string(index=False))
src(T / "data_size_summary.csv", "src/44_data_size.py")
g = pd.read_csv(T / "data_size_gap.csv")
print(g[g.column.isin(["classic_all", "ebike_all"])].round(4).to_string(index=False))
src(T / "data_size_gap.csv", "src/44_data_size.py")
sl = pd.read_csv(T / "data_size_slope.csv")
print(sl.round(4).to_string(index=False))
src(T / "data_size_slope.csv", "src/44_data_size.py")
""")

md("## 11. Figures")
for fig, script, caption in [
        ("search_step1.png", "38_search_report.py", "Fig. 1, Step 1"),
        ("search_step2.png", "38_search_report.py", "Fig. 2, Step 2"),
        ("search_curves.png", "38_search_report.py", "Fig. 3, curves"),
        ("search_skill_by_horizon.png", "38_search_report.py", "Fig. 4, by horizon"),
        ("meeting_a_candidates.png", "40_meeting_figures.py", "Meeting (a)"),
        ("meeting_b_curves.png", "40_meeting_figures.py", "Meeting (b)"),
        ("meeting_c_horizon.png", "40_meeting_figures.py", "Meeting (c)"),
        ("meeting_d_low_rows.png", "40_meeting_figures.py", "Meeting (d)"),
        ("data_size.png", "44_data_size.py", "Fig. 5, data size")]:
    code(f'''
print("{caption}")
src(ROOT / "reports" / "figures" / "{fig}", "src/{script}")
display(Image(filename=str(ROOT / "reports" / "figures" / "{fig}"), width=900))
''')


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
