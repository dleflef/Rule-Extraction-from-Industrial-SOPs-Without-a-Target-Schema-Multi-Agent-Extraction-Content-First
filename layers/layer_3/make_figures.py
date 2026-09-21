"""make_figures.py
==================================================
Render every results figure in the thesis from the committed artifacts.

Each figure is generated from a CSV this repository already ships, never from a
number retyped out of the text, so a figure cannot quietly disagree with the
table or the prose it accompanies. The source artifact for each is named in the
docstring of the function that draws it, and every function asserts the headline
value it depends on before drawing, so a stale artifact fails loudly instead of
producing a figure that misreports the run.

Figures written to layers/layer_3/figures/:

    fig_threshold.png      F1_content across the eleven acceptance thresholds
    fig_strictness.png     detection recall under stricter acceptance
    fig_correlated.png     the correlated anomaly, end to end

Run:  python3 layers/layer_3/make_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, ConnectionPatch  # noqa: E402

# --- paths -----------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS = os.path.join(_ROOT, "layers", "step3_results")
LAYER4 = os.path.join(_ROOT, "layers", "layer_4")
ORACLE = os.path.join(_ROOT, "layers", "oracle_baseline")
TIMESERIES = os.path.join(_ROOT, "data", "dataset", "sensors",
                          "timeseries_raw.csv")
OUT = os.path.join(_HERE, "figures")

# --- house style -----------------------------------------------------------
# Serif to match the thesis body (mathptmx). Muted, print-safe palette whose
# hues stay distinguishable when the page is photocopied in greyscale.
INK = "#1A1A1A"
MUTED = "#6E6E6E"
FAINT = "#D8D8D8"
BLUE = "#2C5F8A"      # the multi-agent pipeline
BLUE_LT = "#7FA6C4"   # external corpora
SLATE = "#40485A"     # development corpus
RED = "#B04A4A"       # blind spots, breaches, ground-truth windows
GREEN = "#4A7C59"     # what the metric does detect
SAND = "#C89B5A"      # the oracle / annotation's own rules

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Corpus order used throughout the thesis: development first, then the three
# external corpora as they appear in Table 5.1, never sorted by score.
CORPUS_ORDER = [
    ("dev_production_line", "Production line\n(development)"),
    ("external_test_biogas", "Biogas\ndigestion"),
    ("external_test_desalination", "RO\ndesalination"),
    ("external_test_sulfuric_acid", "Sulfuric acid\nproduction"),
]


def _finish(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)
    return ax


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _ROOT)}")


def _check(actual, expected, what, tol=0.0015):
    if abs(actual - expected) > tol:
        sys.exit(f"ARTIFACT MISMATCH: {what} is {actual:.3f}, "
                 f"thesis reports {expected:.3f}. Refusing to draw.")


# ---------------------------------------------------------------------------
# 1. Transfer across the four corpora
#    source: layers/step3_results/RESULTS.csv
#            layers/step3_results/validity/validity_perturbations.csv (floor)
# ---------------------------------------------------------------------------
# 1. Acceptance-threshold sweep
#    source: layers/step3_results/validity/validity_threshold_curve.csv
# ---------------------------------------------------------------------------
def fig_threshold():
    df = pd.read_csv(os.path.join(RESULTS, "validity",
                                  "validity_threshold_curve.csv"))
    df = df.set_index("corpus")
    taus = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    cols = [f"f1@{t:g}" for t in taus]

    _check(float(df.loc["dev_production_line", "f1@0.6"]), 0.719,
           "dev F1 at the operating point")

    styles = {
        "dev_production_line": (SLATE, "-", "o"),
        "external_test_biogas": (BLUE, "--", "s"),
        "external_test_desalination": (GREEN, "-.", "^"),
        "external_test_sulfuric_acid": (RED, (0, (3, 1.5)), "D"),
    }
    names = {
        "dev_production_line": "Production line (development)",
        "external_test_biogas": "Biogas digestion",
        "external_test_desalination": "RO desalination",
        "external_test_sulfuric_acid": "Sulfuric acid production",
    }

    fig, ax = plt.subplots(figsize=(6.3, 3.6))

    ax.axvline(0.60, color=INK, lw=0.9, ls=(0, (2, 2)), zorder=2)
    ax.text(0.605, 0.955, "operating point  $\\tau = 0.6$", fontsize=8,
            color=INK, va="top", ha="left", style="italic")

    for key, (c, ls, mk) in styles.items():
        y = [float(df.loc[key, col]) for col in cols]
        ax.plot(taus, y, color=c, ls=ls, marker=mk, ms=3.6, lw=1.4,
                label=names[key], zorder=3, markeredgewidth=0)
        ax.plot([0.60], [float(df.loc[key, "f1@0.6"])], marker=mk, ms=7.5,
                color=c, zorder=4, markeredgecolor="white",
                markeredgewidth=1.0)

    ax.set_xlabel(r"acceptance threshold $\tau$")
    ax.set_ylabel(r"$F1_{\mathrm{content}}$")
    ax.set_xticks(taus)
    ax.set_xlim(0.375, 0.925)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color=FAINT, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=1)
    _finish(ax)
    _save(fig, "fig_threshold.png")


# ---------------------------------------------------------------------------
# 3. What the metric detects, and what it cannot see
#    source: layers/step3_results/validity/validity_perturbations.csv
# ---------------------------------------------------------------------------
# 4. Which system recovered which labelled anomaly
#    source: layers/oracle_baseline/oracle_coverage.csv
#            layers/layer_4/detection_generic_results/gt_coverage*.csv
#            layers/layer_4/graph_generic_results/graph_detection_summary.csv
# ---------------------------------------------------------------------------
# 2. Detection recall under stricter acceptance
#    source: layers/layer_4/graph_generic_results/graph_detection_summary.csv
# ---------------------------------------------------------------------------
def fig_strictness():
    g = pd.read_csv(os.path.join(LAYER4, "graph_generic_results",
                                 "graph_detection_summary.csv")).iloc[0]

    rows = [
        ("Any overlap\n(operating\npoint)", float(g["recall_with_relation"]),
         int(g["gt_covered_with_relation"])),
        ("Coverage\n$\\geq 25\\%$", float(g["recall_cov25"]),
         int(g["tp_cov25"])),
        ("Latency\n$\\leq 30$ min", float(g["recall_lat30"]),
         int(g["tp_lat30"])),
        ("Coverage $\\geq 25\\%$\nand latency\n$\\leq 30$ min",
         float(g["recall_cov25_lat30"]), int(g["tp_cov25_lat30"])),
        ("Coverage\n$\\geq 50\\%$", float(g["recall_cov50"]),
         int(g["tp_cov50"])),
    ]

    _check(rows[0][1], 1.000, "any-overlap recall")
    _check(rows[1][1], 0.714, "recall at 25% coverage")
    _check(rows[4][1], 0.500, "recall at 50% coverage")

    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    x = np.arange(len(rows))
    vals = [r[1] for r in rows]
    colors = [BLUE] + [BLUE_LT] * (len(rows) - 1)

    ax.bar(x, vals, width=0.56, color=colors, zorder=3, edgecolor="white",
           linewidth=0.6)
    for xi, (lab, v, tp) in enumerate(rows):
        ax.text(xi, v + 0.028, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK,
                fontweight="bold" if xi == 0 else "normal")
        ax.text(xi, v / 2, f"{tp}/14", ha="center", va="center", fontsize=8.5,
                color="white", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.8)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("detection recall")
    ax.yaxis.grid(True, color=FAINT, lw=0.6)
    ax.set_axisbelow(True)
    _finish(ax)
    _save(fig, "fig_strictness.png")


# ---------------------------------------------------------------------------
# 3. The correlated anomaly, end to end
#    source: data/dataset/sensors/timeseries_raw.csv
#            parameters mirror detect_correlated() in layer_4/step4_graph_generic.py
# ---------------------------------------------------------------------------
def fig_correlated():
    SRC, TGT = "ST02_SEALING_CUR", "ST04_PACKAGING_SPD"
    # The bounds below are the ones the PIPELINE extracted from SOP-002, not
    # the plant's own columns; they coincide, which is the point of the check
    # in Section 10.1. warn_hi on the source is the trigger.
    SRC_WARN_HI, SRC_CRIT_HI = 13.5, 15.0
    TGT_WARN_LO = 1.0
    Z_THRESH = -2.5

    usecols = ["timestamp", "sensor_id", "value"]
    df = pd.read_csv(TIMESERIES, usecols=usecols)
    df = df[df["sensor_id"].isin([SRC, TGT])].copy()
    df["ts"] = pd.to_datetime(df["timestamp"])

    src = df[df.sensor_id == SRC].sort_values("ts").reset_index(drop=True)
    tgt = df[df.sensor_id == TGT].sort_values("ts").reset_index(drop=True)

    # z-score about the median over the WHOLE series, as the detector computes it
    med = tgt.value.median()
    mad = (tgt.value - med).abs().median()
    scale = 1.4826 * mad if mad > 0 else (tgt.value.std() or 1.0)
    tgt["z"] = (tgt.value - med) / scale
    z_cut = med + Z_THRESH * scale

    w0 = pd.Timestamp("2026-01-08T07:30:00")
    w1 = pd.Timestamp("2026-01-08T11:30:00")
    gt0 = pd.Timestamp("2026-01-08T09:30:00")
    gt1 = pd.Timestamp("2026-01-08T10:30:00")

    s = src[(src.ts >= w0) & (src.ts <= w1)]
    t = tgt[(tgt.ts >= w0) & (tgt.ts <= w1)]

    hot = s[s.value > SRC_WARN_HI]
    breach = hot.ts.min()

    # the figure only claims what the detector found
    assert not hot.empty, "source never breaches its extracted warning bound"
    assert (t[(t.ts >= gt0) & (t.ts <= gt1)].z < Z_THRESH).any()
    assert t.value.min() > TGT_WARN_LO, \
        "target crosses its own bound; the correlated argument would not hold"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.3, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.30})

    # ---- top: the source sensor breaches a bound the pipeline extracted ----
    ax1.axhspan(SRC_WARN_HI, 16.2, color=RED, alpha=0.07, lw=0, zorder=0)
    ax1.plot(s.ts, s.value, color=SLATE, lw=1.0, zorder=3)
    ax1.axhline(SRC_WARN_HI, color=RED, lw=1.1, ls=(0, (4, 2.5)), zorder=2)
    ax1.axhline(SRC_CRIT_HI, color=RED, lw=1.1, ls=(0, (1.5, 2)), zorder=2)
    ax1.text(w1, SRC_WARN_HI + 0.08, "extracted warn_hi  13.5", ha="right",
             va="bottom", fontsize=7.8, color=RED)
    ax1.text(w1, SRC_CRIT_HI + 0.08, "extracted crit_hi  15.0", ha="right",
             va="bottom", fontsize=7.8, color=RED)

    ax1.plot([breach], [float(s[s.ts == breach].value.iloc[0])], marker="v",
             ms=8, color=RED, zorder=5, markeredgecolor="white",
             markeredgewidth=0.8)
    ax1.annotate("breach of the extracted bound\ntriggers the traversal",
                 xy=(breach, float(s[s.ts == breach].value.iloc[0])),
                 xytext=(-6, 26), textcoords="offset points", fontsize=7.8,
                 color=RED, ha="right",
                 arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))

    peak = float(s.value.max()); peak_t = s.loc[s.value.idxmax(), "ts"]
    ax1.text(w0 + pd.Timedelta(minutes=4), 11.10,
             f"first crossing {breach:%H:%M}   |   peak {peak:.2f} A",
             fontsize=7.4, color=SLATE, ha="left", va="center", zorder=11)

    ax1.set_ylabel("current (A)")
    ax1.set_title(f"{SRC}  —  governed by an extracted threshold rule",
                  loc="left", fontsize=9, color=INK)
    ax1.set_ylim(11.0, 16.2)
    ax1.yaxis.grid(True, color=FAINT, lw=0.6)
    ax1.set_axisbelow(True)
    _finish(ax1)

    # ---- bottom: the target is abnormal only for itself --------------------
    ax2.axvspan(gt0, gt1, color=SAND, alpha=0.20, lw=0, zorder=0)
    ax2.plot(t.ts, t.value, color=BLUE, lw=1.0, zorder=3)
    ax2.axhline(TGT_WARN_LO, color=MUTED, lw=1.1, ls=(0, (4, 2.5)), zorder=2)
    ax2.axhline(z_cut, color=GREEN, lw=1.1, ls=(0, (2, 2)), zorder=2)
    ax2.text(w0, TGT_WARN_LO + 0.010,
             "extracted warn_lo  1.0  —  never crossed",
             ha="left", va="bottom", fontsize=7.8, color=MUTED, zorder=6)
    ax2.text(w0, z_cut - 0.008,
             f"$z = -2.5$ about the median  ({z_cut:.3f})", ha="left", va="top",
             fontsize=7.8, color=GREEN)
    ax2.text(gt0 + (gt1 - gt0) / 2, 1.205, "GT-0009",
             ha="center", va="center", fontsize=8, color="#8A6A2F", zorder=6)

    gt = t[(t.ts >= gt0) & (t.ts <= gt1)]
    zin = (gt.value - med) / scale
    ax2.text(w1 - pd.Timedelta(minutes=3), 1.128,
             "in the shaded window\n"
             f"min {gt.value.min():.4f} m/s\n"
             f"clears 1.0 by {gt.value.min() - TGT_WARN_LO:.4f}\n"
             f"mean $z$ {zin.mean():.1f}\n"
             f"{int((zin < Z_THRESH).sum())}/{len(gt)} below the cut",
             fontsize=7.2, color=BLUE, ha="right", va="top", zorder=11,
             linespacing=1.55)

    ax2.set_ylim(0.985, 1.285)
    ax2.set_ylabel("speed (m/s)")
    ax2.set_title(f"{TGT}  —  inside every bound, abnormal only for itself",
                  loc="left", fontsize=9, color=INK)
    ax2.yaxis.grid(True, color=FAINT, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.set_xlabel("2026-01-08")
    _finish(ax2)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # ---- the edge that links the two panels --------------------------------
    # ConnectionPatch transforms its endpoints itself and cannot consume a
    # Timestamp, so the two x positions are converted to date numbers here.
    con = ConnectionPatch(
        xyA=(mdates.date2num(breach), SRC_WARN_HI - 0.35),
        coordsA=ax1.transData,
        xyB=(mdates.date2num(gt0 + (gt1 - gt0) / 2), float(t.value.max())),
        coordsB=ax2.transData,
        arrowstyle="-|>", mutation_scale=13, lw=1.4, color=GREEN,
        connectionstyle="arc3,rad=0.16", zorder=10)
    fig.add_artist(con)

    # Label the edge inside the top panel's empty lower band, where the arrow
    # passes, rather than in the gap between panels where the lower title sits.
    ax1.text(w0 + (w1 - w0) * 0.68, 11.40,
             "CORRELATES_WITH edge, held in the graph",
             fontsize=7.7, color=GREEN, ha="center", va="center", style="italic",
             zorder=11,
             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=GREEN,
                       lw=0.8))

    _save(fig, "fig_correlated.png")


if __name__ == "__main__":
    print("Rendering thesis figures from committed artifacts:")
    fig_threshold()
    fig_strictness()
    fig_correlated()
    print("Done.")
