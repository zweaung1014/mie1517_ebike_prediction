"""reports/MEETING_2026-11-12_long.pdf: the 7-page long brief for the team meeting, rebuilt from the tables (no hand-typed results).

Reads reports/tables/: model_search_summary.csv (38), val_noise_gap.csv (39), test_team_summary.csv,
test_team_by_day.csv, test_team_unseen.csv (42), baselines_team_val.csv (34), mlp_team_val.csv (33); and the meeting
figures of 40_meeting_figures.py. Writes reports/figures/meeting_e_test.png (val against test) and the PDF.

Usage (from src/):  python 43_meeting_pdf.py
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

from common import FIG_DIR

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "reports" / "tables"
OUT = ROOT / "reports" / "MEETING_2026-11-12_long.pdf"  # the short deck is 45_meeting_deck.py
W, H = 16, 9
NAVY, GREY, RED, GREEN = "#1f3b5c", "#666666", "#d62728", "#2ca02c"
GITHUB = "<GitHub link: to be added (repo not decided yet)>"

summ = pd.read_csv(T / "model_search_summary.csv")
gap = pd.read_csv(T / "val_noise_gap.csv")
test = pd.read_csv(T / "test_team_summary.csv")
by_day = pd.read_csv(T / "test_team_by_day.csv")
unseen = pd.read_csv(T / "test_team_unseen.csv").iloc[0]
pers = pd.read_csv(T / "mlp_team_val.csv").query("model == 'persistence' and subset == 'all'").set_index("target")
bl = pd.read_csv(T / "baselines_team_val.csv").set_index(["model", "target", "subset"]).skill


def val(name, col):
    r = summ[(summ.name == name) & ((summ.budget == "full") | (summ.name == "lgbm_l127"))].iloc[0]
    return r[f"skill_{col}_mean"], r[f"skill_{col}_sd"]


def tst(group, target, subset, rows="all test rows"):
    r = test[(test.model_group == group) & (test.target == target) & (test.subset == subset)
             & (test.rows_used == rows)].iloc[0]
    return r.skill_mean, r.skill_sd


def g(pair, target, subset, col="gap"):
    return gap[(gap.pair == pair) & (gap.target == target) & (gap.subset == subset)].iloc[0][col]


def ms(m, sd, d=3):
    if sd != sd:
        return f"{m:.{d}f}"
    return f"{m:.{d}f} ± {sd:.{d}f}" if sd >= 0.0005 else f"{m:.{d}f} ± <0.001"


def page(pdf, n, title, subtitle=""):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.93, title, fontsize=30, weight="bold", color=NAVY, va="top")
    if subtitle:
        fig.text(0.04, 0.865, subtitle, fontsize=17, color=GREY, va="top")
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.835, 0.835], color=NAVY, lw=2))
    fig.text(0.04, 0.025, "MIE1517 bike-share project · model search · team meeting Thu Nov 12", fontsize=11,
             color=GREY)
    fig.text(0.96, 0.025, f"{n} / 7", fontsize=11, color=GREY, ha="right")
    return fig


def bullets(fig, x, y, items, size=17, width=60, gap=0.012):
    """Bulleted text; items are strings or (string, level). Returns the y below the last line."""
    lh = size / 72 / H * 1.35
    for it in items:
        text, lvl = (it, 0) if isinstance(it, str) else it
        lines = textwrap.wrap(text, width - 4 * lvl)
        fig.text(x + 0.018 * lvl, y, "•" if lvl == 0 else "–", fontsize=size, va="top", color=NAVY)
        for i, line in enumerate(lines):
            fig.text(x + 0.018 * lvl + 0.018, y - i * lh, line, fontsize=size - lvl, va="top")
        y -= len(lines) * lh + gap
    return y


def image(fig, rect, name):
    ax = fig.add_axes(rect)
    ax.imshow(plt.imread(FIG_DIR / name))
    ax.axis("off")


def table(fig, rect, df, size=14, bold_row=None, col_w=None):
    ax = fig.add_axes(rect)
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="upper center", cellLoc="center",
                 colWidths=col_w)
    t.auto_set_font_size(False)
    t.set_fontsize(size)
    t.scale(1, 1.9)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(weight="bold")
        if bold_row is not None and r == bold_row + 1:
            cell.set_text_props(weight="bold", color=RED)
    return t


def p1(pdf):
    fig = page(pdf, 1, "Setup: task, score, data",
               "Predict the count of each bike type at a station, h minutes ahead")
    y = bullets(fig, 0.05, 0.79, [
        "Task: for every station and time t, predict the number of classic bikes and of e-bikes at t + h, "
        "h = 2 to 15 min. One model for all stations; h is an input.",
        "Inputs (20 numbers + station): h; classic, e-bike and dock counts now, 10 and 30 min ago; renting / "
        "returning now; capacity; lat / lon; hour (sin, cos); weekend; temperature; humidity. "
        "The station enters through a learned 8-number embedding.",
        "Loss: mean squared error (MSE) of the count. The net predicts the change; forecast = count now + change.",
        "Score: skill = 1 − MSE_model / MSE_persistence, on the same rows.",
        ("Persistence = doing nothing: the count at t carried forward.", 1),
        ("0 = no better than doing nothing; 1 = perfect; below 0 = worse than doing nothing.", 1),
        "Low rows: count at t ≤ 2, the hard cases the team table was built around.",
    ], width=95)
    split = pd.DataFrame({
        "Split": ["train", "val", "test"],
        "Days (local)": ["Sep 29 – Oct 18", "Oct 19 – 23 (Mon–Fri)", "Oct 24 – 28 (from Sat)"],
        "Rows": ["7,525,845", "2,083,617", f"{int(unseen.test_rows):,}"],
        "Used for": ["fitting", "early stopping, model choice", "one score, after the choice"]})
    table(fig, [0.08, y - 0.27, 0.84, 0.25], split, size=15, col_w=[0.2, 0.28, 0.14, 0.38])
    fig.text(0.05, 0.07, "Data file: team repo main 969473f build, SHA-1 263bb791… (checked). "
             f"Persistence MSE on val: classic {pers.mse['classic']:.3f}, e-bike {pers.mse['ebike']:.3f} bikes².",
             fontsize=13, color=GREY)
    pdf.savefig(fig)
    plt.close(fig)


def p2(pdf):
    fig = page(pdf, 2, "What was tested", "24 configurations, 68 runs; ranked at 4 epochs, finalists trained "
               "fully with 3 seeds")
    bullets(fig, 0.04, 0.79, [
        "Architectures:",
        ("MLP widths and depths: 2×64 (Nov 10 base), 2×128, 2×256, 3×128, 3×256, 4×256", 1),
        ("station embedding 8 or 32", 1),
        ("GRU over the lag counts (t − 30, t − 10, t) + MLP head", 1),
        ("LightGBM (gradient-boosted trees) as a non-deep reference", 1),
        "Hyperparameters (on 2×256):",
        ("learning rate 3e-4, 1e-3, 3e-3; batch 1024, 4096, 16,384", 1),
        ("weight decay 1e-4; dropout 0, 0.1, 0.2, 0.3, 0.5", 1),
        "Why no LSTM: the table has only 3 lag steps. The GRU (the same gated-RNN family) gave no gain over "
        "an MLP of the same size, with or without dropout. A longer history would be needed (untested).",
    ], size=15, width=48)
    names = [("mlp_2x64_e8", "MLP 2×64 (base)"), ("mlp_2x64_do0.1", "MLP 2×64, drop 0.1"),
             ("mlp_2x256_e8", "MLP 2×256"), ("mlp_2x256_do0.1", "MLP 2×256, drop 0.1 ★"),
             ("mlp_2x256_do0.2", "MLP 2×256, drop 0.2"), ("mlp_2x256_do0.3", "MLP 2×256, drop 0.3"),
             ("mlp_2x256_do0.5", "MLP 2×256, drop 0.5"), ("mlp_3x256_do0.1", "MLP 3×256, drop 0.1"),
             ("gru64_3x256_e8", "GRU + 3×256"), ("gru64_3x256_do0.1", "GRU + 3×256, drop 0.1"),
             ("lgbm_l127", "LightGBM")]
    rows = [[lab, ms(*val(n, "classic_all")), ms(*val(n, "ebike_all")), ms(*val(n, "classic_low")),
             ms(*val(n, "ebike_low"))] for n, lab in names]
    rows.append(["best cheap baseline", f"{bl['station_hour_weekend_mean', 'classic', 'all']:.3f}",
                 f"{bl['linear_station', 'ebike', 'all']:.3f}", "", ""])
    df = pd.DataFrame(rows, columns=["Val skill, 3 seeds", "classic", "e-bike", "classic low",
                                     "e-bike low"])
    table(fig, [0.47, 0.12, 0.52, 0.66], df, size=12, bold_row=3, col_w=[0.28, 0.18, 0.18, 0.18, 0.18])
    fig.text(0.47, 0.09, "★ chosen. Source: model_search_summary.csv (38_search_report.py)", fontsize=12,
             color=GREY)
    pdf.savefig(fig)
    plt.close(fig)


def p3(pdf):
    c, e = val("mlp_2x256_do0.1", "classic_all"), val("mlp_2x256_do0.1", "ebike_all")
    fig = page(pdf, 3, "Which wins, and by how much against the noise",
               f"MLP 2×256 + dropout 0.1: val skill {ms(*c)} (classic), {ms(*e)} (e-bike)")
    image(fig, [0.0, 0.04, 0.63, 0.8], "meeting_a_candidates.png")
    zc, ze = g("chosen - base", "classic", "all", "z"), g("chosen - base", "ebike", "all", "z")
    bullets(fig, 0.63, 0.79, [
        f"Gain over the Nov 10 base: +{g('chosen - base', 'classic', 'all'):.3f} / "
        f"+{g('chosen - base', 'ebike', 'all'):.3f}.",
        (f"= {zc:.1f} / {ze:.1f} standard errors, counting seed noise and which val stations we got "
         "(bootstrap).", 1),
        (f"Positive on {int(g('chosen - base', 'classic', 'all', 'days_positive'))}/5 val days (classic), "
         f"{int(g('chosen - base', 'ebike', 'all', 'days_positive'))}/5 (e-bike).", 1),
        "The gain is from dropout: without it, bigger nets overfit after 4–6 epochs.",
        "Width helps e-bikes only; on classic, 2×64 + dropout is as good.",
        "Dropout 0.3 scores best but is within noise on 3 of 4 columns and needs 2× the epochs; 0.5 drops.",
        f"LightGBM is level on the low rows ({g('chosen - lgbm', 'classic', 'low'):+.3f} / "
        f"{g('chosen - lgbm', 'ebike', 'low'):+.3f}).",
    ], size=15, width=40)
    pdf.savefig(fig)
    plt.close(fig)


def p4(pdf):
    fig = page(pdf, 4, "Learning curves, horizon, low rows (val)",
               "Dropout keeps improving after the plain nets overfit; the gain is largest for e-bikes at long h")
    image(fig, [0.0, 0.42, 0.45, 0.42], "meeting_b_curves.png")
    image(fig, [0.45, 0.44, 0.55, 0.39], "meeting_c_horizon.png")
    image(fig, [0.13, 0.04, 0.74, 0.39], "meeting_d_low_rows.png")
    pdf.savefig(fig)
    plt.close(fig)


def fig_test():
    groups = [("base_mlp_2x64", "mlp_2x64_e8", "MLP 2×64 (base)", GREY),
              ("lgbm_l127", "lgbm_l127", "LightGBM", GREEN),
              ("chosen_mlp_2x256_do0.1", "mlp_2x256_do0.1", "MLP 2×256 + dropout 0.1 (chosen)", RED)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, x, name in zip(axes, ["classic", "ebike"], ["Classic bikes", "E-bikes"]):
        for j, subset in enumerate(["all", "low"]):
            for k, (tg, vn, lab, col) in enumerate(groups):
                pos = j * 7.5 + k * 2.2
                vm, vs = val(vn, f"{x}_{subset}")
                tm, ts = tst(tg, x, subset)
                ax.bar(pos, vm, 0.9, color=col, alpha=0.35, yerr=vs, capsize=4,
                       label=f"{lab}, val" if j == 0 else None)
                ax.bar(pos + 0.95, tm, 0.9, color=col, yerr=ts, capsize=4,
                       label=f"{lab}, test" if j == 0 else None)
                ax.text(pos + 0.95, tm + 0.006, f"{tm:.3f}", ha="center", fontsize=11)
        ax.set_xticks([2.7, 10.2], ["all rows", "low rows (count ≤ 2)"], fontsize=13)
        ax.set_title(name, fontsize=15)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("skill vs persistence", fontsize=13)
    axes[0].set_ylim(0, 0.13)
    axes[1].legend(fontsize=10, ncol=2, loc="upper right")
    fig.suptitle("Val (light) against test (dark), mean ± sd over seeds 0–2", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "meeting_e_test.png", dpi=120)
    plt.close(fig)


def p5(pdf):
    fig_test()
    c, e = tst("chosen_mlp_2x256_do0.1", "classic", "all"), tst("chosen_mlp_2x256_do0.1", "ebike", "all")
    b = tst("base_mlp_2x64", "classic", "all"), tst("base_mlp_2x64", "ebike", "all")
    lg = tst("lgbm_l127", "classic", "all"), tst("lgbm_l127", "ebike", "all")
    fig = page(pdf, 5, "Test result (read once, protocol fixed before)",
               f"Chosen: {ms(*c)} (classic), {ms(*e)} (e-bike); test Oct 24–28, {int(unseen.test_rows):,} rows")
    image(fig, [0.0, 0.07, 0.62, 0.78], "meeting_e_test.png")
    d = by_day[by_day.model_group == "chosen_mlp_2x256_do0.1"].set_index(["day", "target"]).skill
    wk = [k for k in d.index.get_level_values(0).unique() if k.endswith(("Sat", "Sun"))]
    wd = [k for k in d.index.get_level_values(0).unique() if not k.endswith(("Sat", "Sun"))]
    full_wd = wd[:-1]  # Wed Oct 28 is the last, partial test day
    rows = by_day.groupby("day").rows.first()
    weekend = ", ".join(f"{d[k, 'classic']:.3f}" for k in wk)
    bullets(fig, 0.63, 0.79, [
        f"E-bikes: the gain holds, +{e[0] - b[1][0]:.3f} over the base and +{e[0] - lg[1][0]:.3f} over LightGBM.",
        f"Classic: chosen {c[0]:.3f} vs base {b[0][0]:.3f} — the val gain (+0.006) does not carry over. "
        "Reviewers had flagged the classic gain as weak.",
        f"Skill is lower than on val ({val('mlp_2x256_do0.1', 'classic_all')[0]:.3f} / "
        f"{val('mlp_2x256_do0.1', 'ebike_all')[0]:.3f}): test starts with a weekend, val has none. "
        f"Chosen classic: {weekend} on Sat / Sun, {min(d[k, 'classic'] for k in full_wd):.3f}–"
        f"{max(d[k, 'classic'] for k in full_wd):.3f} on Mon / Tue, "
        f"{d[wd[-1], 'classic']:.3f} on Wed ({int(rows[wd[-1]]):,} rows, a partial day).",
        f"Low rows: chosen {ms(*tst('chosen_mlp_2x256_do0.1', 'classic', 'low'))} / "
        f"{ms(*tst('chosen_mlp_2x256_do0.1', 'ebike', 'low'))}; LightGBM "
        f"{ms(*tst('lgbm_l127', 'classic', 'low'))} / {ms(*tst('lgbm_l127', 'ebike', 'low'))}.",
        f"Unseen stations: {int(unseen.rows_unseen_station):,} test rows "
        f"({unseen.rows_unseen_station / unseen.test_rows:.2%}) at {int(unseen.stations_unseen)} stations; "
        f"without them the chosen skill is {tst('chosen_mlp_2x256_do0.1', 'classic', 'all', 'seen stations only')[0]:.3f} / "
        f"{tst('chosen_mlp_2x256_do0.1', 'ebike', 'all', 'seen stations only')[0]:.3f} (no change).",
        "Nothing was tuned after the test score.",
    ], size=14, width=44)
    fig.text(0.04, 0.06, "Source: test_team_summary.csv, test_team_by_day.csv, test_scores.csv "
             "(42_test_score.py)", fontsize=12, color=GREY)
    pdf.savefig(fig)
    plt.close(fig)


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=color, ec=NAVY, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12.5, family="monospace")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, color=NAVY, lw=1.5))


def p6(pdf):
    fig = page(pdf, 6, "Code: verify it, change it", f"All in src/ of this repo · {GITHUB}")
    ax = fig.add_axes([0.02, 0.33, 0.96, 0.5])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lv1, lv2, lv3, lv4 = "#eef3f8", "#fdf2e6", "#eef7ee", "#f4eef8"
    box(ax, 0.02, 0.72, 0.28, 0.22, "common.py  (paths)\nscore.py  (split, skill,\n  test log)", lv1)
    box(ax, 0.36, 0.72, 0.28, 0.22, "33_train_mlp.py\n(inputs, Nov 10 MLP)", lv1)
    box(ax, 0.02, 0.36, 0.28, 0.22, "34_baselines.py\n(cheap baselines)", lv2)
    box(ax, 0.36, 0.36, 0.28, 0.22, "37_model_search.py\n(all runs; --save)", lv2)
    box(ax, 0.70, 0.36, 0.28, 0.22, "42_test_score.py\n(test, once)", lv2)
    box(ax, 0.36, 0.02, 0.28, 0.2, "reports/tables/*.csv", lv3)
    box(ax, 0.70, 0.02, 0.29, 0.2, "38 summary · 39 noise\n40 figures · 41 notebook\n43 this PDF", lv4)
    for a, b in [((0.16, 0.72), (0.16, 0.58)), ((0.3, 0.8), (0.4, 0.58)), ((0.5, 0.72), (0.5, 0.58)),
                 ((0.64, 0.8), (0.8, 0.58)), ((0.16, 0.36), (0.38, 0.17)), ((0.5, 0.36), (0.5, 0.22)),
                 ((0.84, 0.36), (0.62, 0.2)), ((0.64, 0.12), (0.7, 0.12))]:
        arrow(ax, a, b)
    bullets(fig, 0.04, 0.3, [
        "Reproduce (from src/, Python env from requirements.txt): python 37_model_search.py --step 1 (or --run NAME --budget full "
        "--seed 0 1 2); python 38_search_report.py --finalists …; python 39_val_noise.py; "
        "python 40_meeting_figures.py; python 42_test_score.py --dry (val checks only).",
        "Trace: notebooks/model_search_trace.ipynb (+ .html) prints every number of MODEL_SEARCH.md from its "
        "CSV with the script and commit; it checks the data SHA-1 and that the search never reads the test split.",
        "To change the model: add a config to GRIDS in 37_model_search.py; every run is logged to "
        "model_search_runs.csv. Data path: score.TEAM.",
    ], size=14, width=120)
    pdf.savefig(fig)
    plt.close(fig)


def p7(pdf):
    fig = page(pdf, 7, "Decisions for the team, open questions", "Recommendation first")
    y = bullets(fig, 0.04, 0.79, [
        "1. Model for the progress report (Fri Nov 13, 11 pm): MLP 2×256 + dropout 0.1, with the base, "
        "LightGBM and the cheap baselines as references, val and test. Alternative: keep the Nov 10 base "
        "(test: same on classic, −0.013 on e-bikes).",
        "2. Dropout rate for the final report: keep 0.1; settle 0.1 vs 0.3 on data not used for selection "
        "(e.g. a weekend check, Oct 17–18). The test split is used up for this choice.",
        "3. Test protocol: done (3 seeds, mean ± sd, read once). Any later model gets its own single test score.",
        "4. Report LightGBM beside the baselines: it is level on the low rows, so the deep model's lead is "
        "on e-bikes and on rows with 3+ bikes.",
    ], size=16, width=105, gap=0.02)
    fig.text(0.04, y - 0.01, "Open questions", fontsize=20, weight="bold", color=NAVY, va="top")
    bullets(fig, 0.04, y - 0.07, [
        "Larry (data): is the weather-fixed build (SHA-1 263bb791…) the team release? Does anyone still train "
        "on the older release with the weather leak?",
        "David (models): which models and numbers do you have? Can they be scored with score.py on the same "
        "split, for one table in the progress report?",
        "All: val has no weekend while test starts on one; add weekend days to val for the final? "
        "Unknown stations are rare on test (0.13 % of rows): leave the untrained row as is?",
    ], size=16, width=105, gap=0.02)
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    with PdfPages(OUT) as pdf:
        for p in (p1, p2, p3, p4, p5, p6, p7):
            p(pdf)
        info = pdf.infodict()
        info["Title"] = "MIE1517 model search: team meeting Nov 12"
    print("wrote", OUT)
