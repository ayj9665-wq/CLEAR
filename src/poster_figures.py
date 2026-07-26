"""Poster figures for CLEAR (week-4 deliverable).

Two figures, the pair the fairness report (§11) named as the largest remaining
output:

  fig1  Racial clearance gap, actual vs. model-predicted  -> the diagnosis.
  fig2  Accuracy-fairness trade-off, two mitigations on one axis -> the prescription.

Reads only the small tracked result CSVs under outputs/ (no retraining). Labels
are English-only on purpose: the default matplotlib font has no Hangul glyphs,
so Korean plot text renders as blank boxes (see CLAUDE.md env notes).

Run from src/:  python poster_figures.py  -> outputs/poster_fig1_race_gap.png
                                             outputs/poster_fig2_tradeoff.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

# ---- validated light-mode palette (dataviz skill reference instance) ---------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"   # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
})

OUT = C.OUTPUT_DIR


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure 1 — the racial clearance gap: reality vs. what the GNN predicts.
# graphsage_geo is the deployed (sighted) model. Groups at the n>=1000 floor.
# ---------------------------------------------------------------------------
def fig1_race_gap():
    g = pd.read_csv(OUT / "fairness_group_metrics.csv")
    g = g[(g.attribute == "Victim Race") & (g.model == "graphsage_geo")]
    order = ["White", "Black", "Asian/Pacific Islander"]
    g = g.set_index("group").loc[order]

    actual = g["base_rate"].to_numpy() * 100      # true clearance rate
    predicted = g["selection_rate"].to_numpy() * 100  # model "solved" rate
    labels = ["White", "Black", "Asian /\nPac. Isl."]

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=200)
    _clean_axes(ax)

    b1 = ax.bar(x - w / 2, actual, w, label="Actual clearance rate",
                color=ORANGE, zorder=3)
    b2 = ax.bar(x + w / 2, predicted, w, label='Model "solved" rate',
                color=BLUE, zorder=3)

    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.0f}%",
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11, color=INK2)

    # Call out the White-Black gap widening (the report's robust contrast).
    wb_actual = actual[0] - actual[1]
    wb_pred = predicted[0] - predicted[1]
    amp = wb_pred / wb_actual
    ax.text(
        0.30, 91,
        f"White-Black gap:\n{wb_actual:.1f} pt in reality  →  "
        f"{wb_pred:.1f} pt predicted\n({amp:.1f}× amplification)",
        fontsize=11.5, color=INK, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fdece4", ec=ORANGE, lw=1.2))

    ax.set_ylim(0, 92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Clearance rate (%)", fontsize=12)
    ax.set_title("The model widens the racial clearance gap",
                 fontsize=15.5, fontweight="bold", color=INK, pad=14, loc="left")
    ax.text(0, 1.015, "Homicide cases, CA + TX + MI  ·  GraphSAGE (geo edges), "
            "test set", transform=ax.transAxes, fontsize=10.5, color=MUTED)
    ax.legend(frameon=False, fontsize=11.5, loc="upper right")
    ax.tick_params(length=0)

    fig.tight_layout()
    p = OUT / "poster_fig1_race_gap.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Figure 2 — accuracy-fairness trade-off, both mitigations on one axis.
# x: demographic-parity amplification (1.0 = data-level gap, 0 = parity).
# y: MCC change vs. each method's OWN unmitigated baseline. Plotting the
#    *relative* cost is what makes the two comparable -- post-processing (08)
#    scores on the eval half, the loss penalty (10) on the whole test set, so
#    their absolute MCC levels are not on the same scale (CLAUDE.md / report 10-4).
# Both are blind geo GraphSAGE, White-vs-Black race gap.
# ---------------------------------------------------------------------------
def fig2_tradeoff():
    # loss penalty (10_fairloss), beta=0 main curve. Restrict to the useful
    # region alpha<=50: the report shows it stays a trade-off curve only there,
    # above which the penalty swamps the BCE term and MCC destabilizes (10-2).
    fl_all = pd.read_csv(OUT / "fairloss_tradeoff.csv")
    fl_all = fl_all[(fl_all.beta == 0) & (fl_all.attribute == "Victim Race")]
    fl_base = fl_all.loc[fl_all.alpha == 0, "acc_mcc"].iloc[0]
    fl = fl_all[fl_all.alpha <= 50].sort_values("alpha")
    fl_amp = fl["dp_amplification"].to_numpy()
    fl_dmcc = (fl["acc_mcc"] - fl_base).to_numpy() * 1000  # milli-MCC
    fl_alpha = fl["alpha"].to_numpy()

    # post-processing (08_mitigate), demographic-parity criterion
    mt = pd.read_csv(OUT / "mitigation_tradeoff.csv")
    mt = mt[(mt.model == "graphsage_geo_blind") & (mt.criterion == "dp")
            & (mt.attribute == "Victim Race")].sort_values("lambda")
    mt_base = mt.loc[mt["lambda"] == 0, "acc_mcc"].iloc[0]
    mt_amp = mt["dp_amplification"].to_numpy()
    mt_dmcc = (mt["acc_mcc"] - mt_base).to_numpy() * 1000

    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=200)
    _clean_axes(ax)

    ax.axhline(0, color=MUTED, lw=1.0, ls=(0, (4, 4)), zorder=1)
    ax.axvline(1.0, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
    ax.text(1.03, 1.35, "data-level gap", color=MUTED, fontsize=9.5,
            va="top", ha="left")

    # post-processing: threshold sweep, monotone toward parity
    ax.plot(mt_amp, mt_dmcc, "-o", color=ORANGE, lw=2, ms=6, zorder=3,
            label="Group thresholds (post-hoc)")
    # loss penalty: alpha sweep (folds back above alpha~50)
    ax.plot(fl_amp, fl_dmcc, "-o", color=BLUE, lw=2, ms=6, zorder=4,
            label="Fairness loss penalty (in-training)")

    # annotate the loss-penalty alpha values
    for a, xv, yv in zip(fl_alpha, fl_amp, fl_dmcc):
        ax.annotate(f"α={a:.0f}", (xv, yv), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=10, color=BLUE)

    # the penalty's useful region ends at alpha~50; note the inversion rather
    # than plotting the alpha>50 fold, which would blow out the y-scale.
    ax.text(0.03, -4.4, "beyond α≈50 the penalty\ninverts (see report §10-2)",
            fontsize=9.5, color=MUTED, ha="left", va="top")

    ax.set_ylim(-5.2, 1.6)
    ax.set_xlim(-0.05, 1.55)
    ax.invert_xaxis()  # parity (0) on the right = "better fairness ->"
    ax.set_xlabel("Demographic-parity amplification  "
                  "(1.0 = data gap  ·  0 = parity)", fontsize=12)
    ax.set_ylabel("MCC change vs. unmitigated  (×10⁻³)", fontsize=12)
    ax.set_title("Fairness is cheap to repay — and repayable two ways",
                 fontsize=15.5, fontweight="bold", color=INK, pad=14, loc="left")
    ax.text(0, 1.015, "Blind GraphSAGE (geo), White-vs-Black race gap  ·  "
            "relative accuracy cost", transform=ax.transAxes,
            fontsize=10.5, color=MUTED)
    ax.legend(frameon=False, fontsize=11.5, loc="lower left")
    ax.tick_params(length=0)

    fig.tight_layout()
    p = OUT / "poster_fig2_tradeoff.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    fig1_race_gap()
    fig2_tradeoff()
