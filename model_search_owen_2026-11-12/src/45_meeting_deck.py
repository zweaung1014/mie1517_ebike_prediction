"""reports/MEETING_2026-11-12.pdf: the meeting deck, 3 slides (one page each) plus a Backup section; and
reports/MEETING_2026-11-12_builds.pdf: the same story revealed step by step (13 pages).

Each list slide opens with its title only; every next page adds one item, highlighted, while earlier items stay
plain and later ones are not drawn yet (all items keep their final positions). On slide 2 the figure on the right
is the evidence for the highlighted claim, and its title states that claim.

Reads reports/tables/: model_search_summary.csv, model_search_curves.csv, model_search_runs.csv (37/38),
val_noise_gap.csv (39), test_team_summary.csv, test_team_by_day.csv (42), data_size_summary.csv,
data_size_gap.csv, data_size_slope.csv (44), mlp_team_val.csv (33). Writes reports/figures/deck_*.png and the PDF.
The long 7-page version is 43_meeting_pdf.py (MEETING_2026-11-12_long.pdf).

Usage (from src/):  python 45_meeting_deck.py
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from common import FIG_DIR

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "reports" / "tables"
OUT = ROOT / "reports" / "MEETING_2026-11-12.pdf"            # 3 slides + backup, no build pages
OUT_BUILDS = ROOT / "reports" / "MEETING_2026-11-12_builds.pdf"  # the same story, revealed item by item
W, H = 16, 9
NAVY, GREY, RED, BLUE, GREEN, HL = "#1f3b5c", "#7f7f7f", "#d62728", "#1f77b4", "#2ca02c", "#fff1d6"
GITHUB = "<GitHub link: to be added (repo not decided yet)>"
plt.rcParams.update({"font.size": 15})

summ = pd.read_csv(T / "model_search_summary.csv")
curves = pd.read_csv(T / "model_search_curves.csv")
runs = pd.read_csv(T / "model_search_runs.csv")
gap = pd.read_csv(T / "val_noise_gap.csv")
test = pd.read_csv(T / "test_team_summary.csv")
by_day = pd.read_csv(T / "test_team_by_day.csv")
ds = pd.read_csv(T / "data_size_summary.csv")
ds_gap = pd.read_csv(T / "data_size_gap.csv")
ds_slope = pd.read_csv(T / "data_size_slope.csv")
pers = pd.read_csv(T / "mlp_team_val.csv").query("model == 'persistence' and subset == 'all'").mse.mean()


def val(name, col="classic_all"):
    r = summ[(summ.name == name) & ((summ.budget == "full") | (summ.name == "lgbm_l127"))].iloc[0]
    return r[f"skill_{col}_mean"], r[f"skill_{col}_sd"]


def tst(group, col):
    x, s = col.split("_")
    r = test[(test.model_group == group) & (test.target == x) & (test.subset == s)
             & (test.rows_used == "all test rows")].iloc[0]
    return r.skill_mean, r.skill_sd


def g(pair, x, s, col="gap"):
    return gap[(gap.pair == pair) & (gap.target == x) & (gap.subset == s)].iloc[0][col]


# ---------------------------------------------------------------- claim figures (title = the claim)

def bars(path, title, groups, cols=("classic_all", "ebike_all"), getter=val, ylim=(0.06, 0.125), note=""):
    """Grouped bars: one group per target, one bar per model (mean +- sd over seeds)."""
    fig, ax = plt.subplots(figsize=(9, 6.2))
    n = len(groups)
    for j, c in enumerate(cols):
        for k, (name, label, color) in enumerate(groups):
            m, sd = getter(name, c)
            pos = j * (n + 1.2) + k
            ax.bar(pos, m, 0.85, yerr=sd if sd == sd else None, color=color, capsize=6,
                   label=label if j == 0 else None)
            ax.text(pos, m + (sd if sd == sd else 0) + 0.0015, f"{m:.3f}", ha="center", fontsize=13)
    ax.set_xticks([j * (n + 1.2) + (n - 1) / 2 for j in range(len(cols))],
                  [c.replace("ebike", "e-bike").replace("_all", ", all rows").replace("_low", ", low rows")
                   for c in cols])
    ax.set_ylim(*ylim)
    ax.set_ylabel("skill vs persistence")
    ax.set_title("\n".join(textwrap.wrap(title, 48)), fontsize=17, weight="bold", color=NAVY)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=12, loc="upper left")
    if note:
        fig.text(0.02, 0.01, note, fontsize=11, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG_DIR / path, dpi=130)
    plt.close(fig)


def fig_curves(path, title):
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for name, label, color in [("mlp_2x256_e8", "2×256, no dropout", GREY),
                               ("mlp_2x256_do0.1", "2×256, dropout 0.1", RED),
                               ("mlp_2x256_do0.3", "2×256, dropout 0.3", BLUE)]:
        ids = runs[(runs.name == name) & (runs.budget == "full")].run_id
        c = curves[curves.run_id.isin(ids)].groupby("epoch").val_mse.agg(["mean", "size"])
        c = c[c["size"] == 3]
        ax.plot(c.index, 1 - c["mean"] / pers, "o-", color=color, lw=2.5, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val skill (mean of both targets)")
    ax.set_ylim(0.07, 0.115)
    ax.set_title("\n".join(textwrap.wrap(title, 48)), fontsize=17, weight="bold", color=NAVY)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13, loc="lower right")
    fig.text(0.02, 0.01, "Mean over seeds 0–2, epochs all three seeds reached (model_search_curves.csv)",
             fontsize=11, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG_DIR / path, dpi=130)
    plt.close(fig)


def fig_data(path, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 6.2), sharey=False)
    for ax, x, lab in zip(axes, ["classic", "ebike"], ["Classic", "E-bike"]):
        for d, color in [("dropout 0.1", RED), ("dropout 0.3", BLUE)]:
            s = ds[ds.dropout == d]
            ax.errorbar(s.train_days, s[f"{x}_all_mean"], yerr=s[f"{x}_all_sd"], fmt="o-", color=color,
                        capsize=5, lw=2.5, ms=7, label=d)
        ax.set_title(lab, fontsize=15)
        ax.set_xticks([5, 10, 15, 20])
        ax.set_xlabel("train days (most recent)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val skill (all rows)")
    axes[0].legend(fontsize=12)
    fig.suptitle("\n".join(textwrap.wrap(title, 52)), fontsize=17, weight="bold", color=NAVY)
    fig.text(0.02, 0.01, "Mean ± sd over seeds 0–2; same val days (data_size_summary.csv, 44_data_size.py)",
             fontsize=11, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(FIG_DIR / path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- page drawing

def frame(title, n_page, total):
    fig = plt.figure(figsize=(W, H))
    fig.text(0.04, 0.92, title, fontsize=32, weight="bold", color=NAVY, va="top")
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.83, 0.83], color=NAVY, lw=2))
    fig.text(0.04, 0.025, "MIE1517 bike-share · model search · team meeting Thu Nov 12", fontsize=11, color=GREY)
    fig.text(0.96, 0.025, f"{n_page} / {total}" if total else str(n_page), fontsize=11, color=GREY, ha="right")
    return fig


def layout(items, x, y0, width, size, gap=0.03):
    """Final positions of every item (so nothing moves between builds): list of (y_top, lines)."""
    lh = size / 72 / H * 1.32
    pos, y = [], y0
    for text in items:
        lines = textwrap.wrap(text, width)
        pos.append((y, lines))
        y -= len(lines) * lh + gap
    return pos, lh


def draw_items(fig, items, k, x, y0, width, size, box_w):
    pos, lh = layout(items, x, y0, width, size, gap=0.022)
    for i, (y, lines) in enumerate(pos[:k]):
        newest = i == k - 1
        if newest:
            fig.add_artist(FancyBboxPatch((x - 0.01, y - len(lines) * lh - 0.006), box_w, len(lines) * lh + 0.012,
                                          boxstyle="round,pad=0.006", fc=HL, ec="#e0a040", lw=1.5,
                                          transform=fig.transFigure))
        for j, line in enumerate(lines):
            fig.text(x, y - j * lh, line, fontsize=size, va="top", weight="bold" if newest else "normal",
                     color="black" if newest else "#444444")


def image(fig, rect, name):
    ax = fig.add_axes(rect)
    ax.imshow(plt.imread(FIG_DIR / name))
    ax.axis("off")


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=color, ec=NAVY, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=13, family="monospace")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, color=NAVY, lw=1.5))


def code_slide(n, total):
    """Slide 3: the script hierarchy, the commands and the trace notebook."""
    fig = frame("3. Code: verify it, change it", n, total)
    fig.text(0.04, 0.8, f"All in src/ of the project repo · {GITHUB}", fontsize=15, color=GREY, va="top")
    ax = fig.add_axes([0.02, 0.33, 0.96, 0.45])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lv1, lv2, lv3, lv4 = "#eef3f8", "#fdf2e6", "#eef7ee", "#f4eef8"
    box(ax, 0.02, 0.72, 0.28, 0.22, "common.py  (paths)\nscore.py  (split, skill,\n  test log)", lv1)
    box(ax, 0.36, 0.72, 0.28, 0.22, "33_train_mlp.py\n(inputs, Nov 10 MLP)", lv1)
    box(ax, 0.02, 0.36, 0.28, 0.22, "34_baselines.py\n(cheap baselines)", lv2)
    box(ax, 0.36, 0.36, 0.28, 0.22, "37_model_search.py\n(all runs; --save,\n--train-days)", lv2)
    box(ax, 0.70, 0.36, 0.28, 0.22, "42_test_score.py\n(test, once)", lv2)
    box(ax, 0.36, 0.02, 0.28, 0.2, "reports/tables/*.csv", lv3)
    box(ax, 0.70, 0.0, 0.29, 0.24, "38 summary · 39 noise\n40 figures · 41 notebook\n43/45 PDFs · 44 data size",
        lv4)
    for a, b in [((0.16, 0.72), (0.16, 0.58)), ((0.3, 0.8), (0.4, 0.58)), ((0.5, 0.72), (0.5, 0.58)),
                 ((0.64, 0.8), (0.8, 0.58)), ((0.16, 0.36), (0.38, 0.17)), ((0.5, 0.36), (0.5, 0.22)),
                 ((0.84, 0.36), (0.62, 0.2)), ((0.64, 0.12), (0.7, 0.12))]:
        arrow(ax, a, b)
    lines = ["Reproduce (from src/, Python env from requirements.txt): python 37_model_search.py --step 1 "
             "(or --run NAME --budget full --seed 0 1 2); python 38_search_report.py --finalists …; "
             "python 39_val_noise.py; python 44_data_size.py; python 42_test_score.py --dry (val checks only).",
             "Trace: notebooks/model_search_trace.ipynb (+ .html) prints every number of MODEL_SEARCH.md from "
             "its CSV with the script and commit; it checks the data SHA-1 and that the search never reads "
             "the test split.",
             "To change the model: add a config to GRIDS in 37_model_search.py; every run is logged to "
             "model_search_runs.csv. Data path: score.TEAM."]
    y = 0.29
    for t in lines:
        wrapped = textwrap.wrap(t, 125)
        for j, line in enumerate(wrapped):
            fig.text(0.05, y - j * 0.03, ("• " if j == 0 else "  ") + line, fontsize=14, va="top")
        y -= len(wrapped) * 0.03 + 0.012
    return fig


def main():
    c_gain, e_gain = g("chosen - base", "classic", "all"), g("chosen - base", "ebike", "all")
    zc, ze = g("chosen - base", "classic", "all", "z"), g("chosen - base", "ebike", "all", "z")
    tc = tst("chosen_mlp_2x256_do0.1", "classic_all")[0] - tst("base_mlp_2x64", "classic_all")[0]
    te = tst("chosen_mlp_2x256_do0.1", "ebike_all")[0] - tst("base_mlp_2x64", "ebike_all")[0]
    d01, d03 = ds[ds.dropout == "dropout 0.1"].set_index("train_days"), ds[ds.dropout == "dropout 0.3"].set_index(
        "train_days")
    sl15 = ds_slope.set_index(["dropout", "column"]).slope_per_7_days_15_to_20
    gp = ds_gap[ds_gap.column == "classic_all"].set_index("train_days")["gap_0.3_minus_0.1"]
    gpe = ds_gap[ds_gap.column == "ebike_all"].set_index("train_days")["gap_0.3_minus_0.1"]
    up_c, up_e = (d01.loc[20, f"{x}_all_mean"] - d01.loc[15, f"{x}_all_mean"] for x in ("classic", "ebike"))
    wk_c, wk_e = (sl15["dropout 0.1", f"{x}_all"] for x in ("classic", "ebike"))
    data_claim = (f"More train days still help: 15 → 20 days adds +{up_c:.3f} / +{up_e:.3f} "
                  f"(≈ +{wk_c:.3f} / +{wk_e:.3f} per week), but the gain is slowing")
    lead_txt = " / ".join(f"{gp[k]:+.3f}" for k in [5, 10, 15, 20])

    # claim figures
    bars("deck_1_conclusion.png", f"The pick beats the Nov 10 base on val: +{c_gain:.3f} / +{e_gain:.3f} "
         f"(≈ {zc:.1f} / {ze:.1f} standard errors)",
         [("mlp_2x64_e8", "MLP 2×64 (Nov 10 base)", GREY), ("mlp_2x256_do0.1", "MLP 2×256 + dropout 0.1", RED)],
         note="Mean ± sd over seeds 0–2 (model_search_summary.csv); z adds the val-station bootstrap "
              "(val_noise_gap.csv)")
    bars("deck_2_architecture.png", "Deeper or recurrent nets add nothing over a 2-layer MLP",
         [("mlp_2x256_do0.1", "MLP 2×256", RED), ("mlp_3x256_do0.1", "MLP 3×256", BLUE),
          ("gru64_3x256_do0.1", "GRU + MLP 3×256", "#9467bd")],
         note="All with dropout 0.1; mean ± sd over seeds 0–2 (model_search_summary.csv)")
    fig_curves("deck_3_dropout.png", "Without dropout, 2×256 overfits after epoch 4–6; with it, it keeps improving")
    bars("deck_4_width.png", "Width (64 → 256) helps e-bikes, not classic bikes",
         [("mlp_2x64_do0.1", "MLP 2×64 + dropout 0.1", GREY), ("mlp_2x256_do0.1", "MLP 2×256 + dropout 0.1", RED)],
         note="Mean ± sd over seeds 0–2 (model_search_summary.csv)")
    fig_data("deck_5_dropout03.png", f"Dropout 0.3 leads 0.1 by a little, and the lead does not grow with data "
             f"(classic {lead_txt})")
    fig_data("deck_6_data.png", data_claim)
    bars("deck_7_test.png", f"Test (read once): the e-bike gain holds (+{te:.3f}), the classic gain does not "
         f"(+{tc:.3f})",
         [("base_mlp_2x64", "MLP 2×64 (base)", GREY), ("lgbm_l127", "LightGBM", GREEN),
          ("chosen_mlp_2x256_do0.1", "MLP 2×256 + dropout 0.1", RED)], getter=tst, ylim=(0.0, 0.115),
         note="Test Oct 24–28, mean ± sd over seeds 0–2 (test_team_summary.csv, 42_test_score.py)")

    wk = by_day[(by_day.model_group == "chosen_mlp_2x256_do0.1") & (by_day.target == "classic")]
    sat_sun = wk[wk.day.str.endswith(("Sat", "Sun"))].skill
    mon_tue = wk[wk.day.str.endswith(("Mon", "Tue"))].skill
    lead_c, lead_e = g("do0.3 - chosen", "classic", "all"), g("do0.3 - chosen", "ebike", "all")

    s1 = ["Loss: mean squared error (MSE) of the future count of each bike type.",
          "Score: skill = 1 − MSE_model / MSE_persistence. Persistence = doing nothing (count now carried "
          "forward). 0 = no better than doing nothing, 1 = perfect.",
          "We pick models by their skill on the val days (Oct 19–23). The test days were scored once, at the end."]
    s2 = [(f"Pick: MLP 2×256 + dropout 0.1: +{c_gain:.3f} / +{e_gain:.3f} over the Nov 10 base "
           f"(≈ {min(zc, ze):.1f}–{max(zc, ze):.1f} SE).", "deck_1_conclusion.png"),
          ("Architecture: a plain MLP. 3 layers or a GRU add nothing; no LSTM, as there are only 3 lag steps.",
           "deck_2_architecture.png"),
          ("Dropout is the main gain: without it, the bigger nets overfit after epoch 4–6.", "deck_3_dropout.png"),
          ("Width (64 → 256) helps e-bikes only.", "deck_4_width.png"),
          (f"Why not dropout 0.3? Small lead (+{lead_c:.3f} / +{lead_e:.3f}), nil on low rows, 2× the epochs, "
           "and it does not grow with data.", "deck_5_dropout03.png"),
          (f"More data helps most: about +{wk_c:.3f} / +{wk_e:.3f} per extra week of training days.",
           "deck_6_data.png"),
          (f"Test (read once): the e-bike gain holds (+{te:.3f}), the classic gain does not (+{tc:.3f}); "
           "weekends score lower.", "deck_7_test.png")]
    total = (len(s1) + 1) + (len(s2) + 1) + 1
    n = 0
    with PdfPages(OUT_BUILDS) as pdf:
        for k in range(len(s1) + 1):
            n += 1
            fig = frame("1. Setup", n, total)
            draw_items(fig, s1, k, 0.06, 0.75, 70, 24, 0.9)
            pdf.savefig(fig)
            plt.close(fig)
        for k in range(len(s2) + 1):
            n += 1
            fig = frame("2. What we tested, and what we conclude", n, total)
            draw_items(fig, [t for t, _ in s2], k, 0.05, 0.79, 42, 16, 0.43)
            if k:
                image(fig, [0.48, 0.06, 0.5, 0.76], s2[k - 1][1])
            pdf.savefig(fig)
            plt.close(fig)
        n += 1
        fig = code_slide(n, total)
        pdf.savefig(fig)
        plt.close(fig)
        pdf.infodict()["Title"] = "MIE1517 model search: team meeting Nov 12 (short)"
    print("wrote", OUT_BUILDS, f"({total} pages)")
    short_deck(dict(c_gain=c_gain, e_gain=e_gain, zc=zc, ze=ze, te=te, tc=tc, lead_c=lead_c, lead_e=lead_e,
                    wk_c=wk_c, wk_e=wk_e, up_c=up_c, up_e=up_e, sat_sun=sat_sun, mon_tue=mon_tue, s1=s1))
    print("data claim:", data_claim)
    print("0.3 lead, classic, 5/10/15/20 days:", lead_txt)


def panel_bars(ax, groups, getter=val, ylim=(0.09, 0.122)):
    n = len(groups)
    for j, c in enumerate(("classic_all", "ebike_all")):
        for k, (name, label, color) in enumerate(groups):
            m, sd = getter(name, c)
            ax.bar(j * (n + 1) + k, m, 0.85, yerr=sd, color=color, capsize=4, label=label if j == 0 else None)
            ax.text(j * (n + 1) + k, m + sd + 0.0007, f"{m:.3f}", ha="center", fontsize=10)
    ax.set_xticks([(n - 1) / 2, (n + 1) + (n - 1) / 2], ["classic", "e-bike"], fontsize=12)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9.5, loc="upper left")


def short_deck(v):
    total = 3
    with PdfPages(OUT) as pdf:
        # slide 1
        fig = frame("1. Loss and score", 1, total)
        items = ["Loss: mean squared error (MSE) of the future count of each bike type.",
                 "Score: skill = 1 − MSE_model / MSE_persistence, because a raw MSE has no reference.",
                 "Persistence = doing nothing: predict that the count now stays the same.",
                 "Skill 0 = no better than doing nothing; 1 = perfect.",
                 "We pick models by their skill on the val days (Oct 19–23)."]
        y = 0.74
        for t in items:
            for j, line in enumerate(textwrap.wrap(t, 70)):
                fig.text(0.07, y - j * 0.06, ("•  " if j == 0 else "    ") + line, fontsize=26, va="top")
            y -= 0.06 * len(textwrap.wrap(t, 70)) + 0.045
        pdf.savefig(fig)
        plt.close(fig)

        # slide 2
        fig = frame("2. What works, and why", 2, total)
        fig.add_artist(FancyBboxPatch((0.04, 0.705), 0.92, 0.095, boxstyle="round,pad=0.006", fc=HL, ec="#e0a040",
                                      lw=1.5, transform=fig.transFigure))
        fig.text(0.05, 0.775, f"Pick: MLP 2×256 + dropout 0.1:  +{v['c_gain']:.3f} / +{v['e_gain']:.3f} val skill "
                 f"over the Nov 10 base (≈ {min(v['zc'], v['ze']):.1f}–{max(v['zc'], v['ze']):.1f} SE)",
                 fontsize=19, weight="bold", va="top")
        fig.text(0.05, 0.735, "Mean ± sd over seeds 0–2 on val; each panel title is the claim, the line below it the "
                 "reason", fontsize=13, color=GREY, va="top")
        titles = ["Architecture: a 2-layer MLP.\nDeeper or a GRU add nothing.",
                  "Dropout is the main gain.",
                  "Width (64 → 256) helps\ne-bikes only.",
                  "Why not dropout 0.3, which\nlooks better?"]
        reasons = ["No LSTM: the table has only 3\nlag steps (t−30, t−10, t).",
                   "Without it, 2×256 overfits after\nepoch 4–6; with it, it keeps learning.",
                   "On classic bikes, 2×64 + dropout\nis within noise of 2×256.",
                   f"Lead +{v['lead_c']:.3f} / +{v['lead_e']:.3f}: noise except\nclassic all rows; nil on low "
                   "rows;\n2× the epochs; not growing with data."]
        xs = [0.055, 0.29, 0.525, 0.76]
        axs = [fig.add_axes([x + 0.015, 0.2, 0.185, 0.36]) for x in xs]
        panel_bars(axs[0], [("mlp_2x256_do0.1", "2×256", RED), ("mlp_3x256_do0.1", "3×256", BLUE),
                            ("gru64_3x256_do0.1", "GRU+3×256", "#9467bd")])
        for name, label, color in [("mlp_2x256_e8", "no dropout", GREY), ("mlp_2x256_do0.1", "dropout 0.1", RED)]:
            ids = runs[(runs.name == name) & (runs.budget == "full")].run_id
            c = curves[curves.run_id.isin(ids)].groupby("epoch").val_mse.agg(["mean", "size"])
            c = c[c["size"] == 3]
            axs[1].plot(c.index, 1 - c["mean"] / pers, "o-", color=color, lw=2, ms=4, label=label)
        axs[1].set_ylim(0.09, 0.112)
        axs[1].set_yticks(np.arange(0.09, 0.1121, 0.005))
        axs[1].set_xlabel("epoch", fontsize=12)
        axs[1].tick_params(labelsize=10)
        axs[1].grid(alpha=0.3)
        axs[1].legend(fontsize=10, loc="lower right")
        panel_bars(axs[2], [("mlp_2x64_do0.1", "2×64", GREY), ("mlp_2x256_do0.1", "2×256", RED)])
        for col, x, mk in [(RED, "classic_all", "o-"), (BLUE, "ebike_all", "s-")]:
            gg = ds_gap[ds_gap.column == x]
            axs[3].errorbar(gg.train_days, gg["gap_0.3_minus_0.1"], yerr=gg.sd_gap, fmt=mk, color=col, capsize=4,
                            lw=2, label=x.replace("_all", "").replace("ebike", "e-bike"))
        axs[3].axhline(0, color="k", lw=1)
        axs[3].set_xticks([5, 10, 15, 20])
        axs[3].set_xlabel("train days", fontsize=12)
        axs[3].set_ylabel("skill, 0.3 − 0.1", fontsize=11)
        axs[3].tick_params(labelsize=10)
        axs[3].grid(alpha=0.3)
        axs[3].legend(fontsize=10)
        axs[0].set_ylabel("val skill", fontsize=12)
        axs[1].set_ylabel("val skill (both)", fontsize=11)
        for x, t, r in zip(xs, titles, reasons):
            fig.text(x, 0.66, t, fontsize=15, weight="bold", color=NAVY, va="top")
            fig.text(x, 0.12, r, fontsize=12.5, va="top")
        fig.text(0.04, 0.055, "Sources: model_search_summary.csv, model_search_curves.csv, val_noise_gap.csv, "
                 "data_size_gap.csv (src/38, 39, 44)", fontsize=10, color=GREY)
        pdf.savefig(fig)
        plt.close(fig)

        # slide 3
        fig = code_slide(3, total)
        pdf.savefig(fig)
        plt.close(fig)

        # backup
        fig = plt.figure(figsize=(W, H))
        fig.text(0.5, 0.55, "Backup", fontsize=60, weight="bold", color=NAVY, ha="center", va="center")
        fig.text(0.5, 0.42, "B1  More data  ·  B2  Test result", fontsize=22, color=GREY, ha="center")
        pdf.savefig(fig)
        plt.close(fig)
        for code, title, img, lines in [
                ("B1", "More data helps more than any architecture change", "deck_6_data.png",
                 [f"Refit on the most recent 5 / 10 / 15 / 20 train days, same val days, same updates per epoch.",
                  f"15 → 20 days still adds +{v['up_c']:.3f} / +{v['up_e']:.3f} (≈ +{v['wk_c']:.3f} / "
                  f"+{v['wk_e']:.3f} per week): the model is data-limited, though the gain is slowing.",
                  "Compare: the whole architecture and dropout search gained +0.006 / +0.007.",
                  "The pollers kept running after Oct 28, so a later table build adds training days."]),
                ("B2", "Test result (read once)", "deck_7_test.png",
                 [f"Chosen vs Nov 10 base on test: +{v['tc']:.3f} classic, +{v['te']:.3f} e-bike.",
                  f"Test starts on a weekend, val has none: chosen classic {v['sat_sun'].min():.3f}–"
                  f"{v['sat_sun'].max():.3f} on Sat/Sun vs {v['mon_tue'].min():.3f}–{v['mon_tue'].max():.3f} "
                  "on Mon/Tue.",
                  "Protocol fixed and committed before reading; 3 seeds; nothing tuned after."])]:
            fig = frame(f"{code}. {title}", "Backup", "")
            image(fig, [0.0, 0.06, 0.55, 0.75], img)
            y = 0.76
            for t in lines:
                for j, line in enumerate(textwrap.wrap(t, 46)):
                    fig.text(0.57, y - j * 0.045, ("•  " if j == 0 else "    ") + line, fontsize=17, va="top")
                y -= 0.045 * len(textwrap.wrap(t, 46)) + 0.03
            pdf.savefig(fig)
            plt.close(fig)
        pdf.infodict()["Title"] = "MIE1517 model search: team meeting Nov 12"
    print("wrote", OUT, "(3 slides + backup divider + 2 backup pages)")


if __name__ == "__main__":
    main()
