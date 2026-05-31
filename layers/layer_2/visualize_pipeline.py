from __future__ import annotations

import argparse
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from step2_TEST import _ABL_TAG_MAP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── colours ────────────────────────────────────────────────────────────────────
C_EXTRACTOR = "#dbeafe"   # blue  — extractors
C_CONTROL   = "#fef9c3"   # yellow — control nodes
C_IO        = "#dcfce7"   # green  — save / CSV
C_ENDPOINT  = "#f3f4f6"   # grey   — START / END
B_EXTRACTOR = "#3b82f6"
B_CONTROL   = "#ca8a04"
B_IO        = "#16a34a"
B_ENDPOINT  = "#6b7280"
T_EXTRACTOR = "#1e3a8a"
T_CONTROL   = "#713f12"
T_IO        = "#14532d"
T_ENDPOINT  = "#374151"

BOX_W  = 0.18   # node box width  (axes fraction)
BOX_H  = 0.07   # node box height (axes fraction)
R_CIRC = 0.035  # radius for START / END circles


def _box(ax, cx, cy, label, fill, edge, text_color, fontsize=9):
    """Draw a rounded rectangle centred at (cx, cy)."""
    rect = FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.01",
        facecolor=fill, edgecolor=edge, linewidth=1.4,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", transform=ax.transAxes,
            multialignment="center")


def _circle(ax, cx, cy, label, fill, edge, text_color):
    """Draw a circle node (START / END)."""
    circ = mpatches.Circle(
        (cx, cy), R_CIRC,
        facecolor=fill, edgecolor=edge, linewidth=1.2,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(circ)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=8,
            color=text_color, fontweight="bold", transform=ax.transAxes)


def _arrow(ax, x0, y0, x1, y1, label="", color="#555", rad=0.0,
           label_dx=0.01, label_dy=0.0):
    """Draw an annotated arrow between two points in axes coordinates."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.2,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )
    if label:
        mx = (x0 + x1) / 2 + label_dx
        my = (y0 + y1) / 2 + label_dy
        ax.text(mx, my, label, ha="left", va="center", fontsize=7,
                color="#555", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))


def visualize(ablation: str | None = None) -> None:
    has_judge = ablation != "no_judge"

    # ── node labels ──────────────────────────────────────────────────────────
    coord_lbl    = "coordinator\nkeyword regex" if ablation == "no_cm"        else "coordinator\nLLM classify"
    merge_lbl    = "merge\ndeterministic only"  if ablation == "no_adj"       else "merge\nadjudicator LLM"
    validate_lbl = "validate\npassthrough"       if ablation == "no_validator" else "validate\nLLM check"

    # ── y positions (top-down, values decrease) ───────────────────────────────
    # Rows are laid out explicitly so extractors are always at the same level.
    Y = {
        "START":  0.96,
        "COORD":  0.86,
        "EXTR":   0.73,   # shared y for all three extractors
        "MRG":    0.60,
        "VAL":    0.50,
        "JDG":    0.40,
        "PRE":    0.29,
        "NRM":    0.29 if not has_judge else 0.19,
        "SAV":    0.19 if not has_judge else 0.09,
        "END":    0.09 if not has_judge else 0.01,
    }

    # ── figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("", pad=0)

    # ── draw nodes ───────────────────────────────────────────────────────────
    cx = 0.50   # centre x for main column

    _circle(ax, cx, Y["START"], "START", C_ENDPOINT, B_ENDPOINT, T_ENDPOINT)
    _box(ax, cx, Y["COORD"], coord_lbl, C_CONTROL, B_CONTROL, T_CONTROL)

    # Three extractors at the same y — fixed x positions
    _box(ax, 0.18, Y["EXTR"], "extract_narrative\nExtractor-A", C_EXTRACTOR, B_EXTRACTOR, T_EXTRACTOR)
    _box(ax, 0.50, Y["EXTR"], "extract_tabular\nExtractor-B",   C_EXTRACTOR, B_EXTRACTOR, T_EXTRACTOR)
    _box(ax, 0.82, Y["EXTR"], "extract_matrix\nExtractor-C",    C_EXTRACTOR, B_EXTRACTOR, T_EXTRACTOR)

    _box(ax, cx, Y["MRG"],  merge_lbl,    C_CONTROL, B_CONTROL, T_CONTROL)
    _box(ax, cx, Y["VAL"],  validate_lbl, C_CONTROL, B_CONTROL, T_CONTROL)

    if has_judge:
        _box(ax, cx, Y["JDG"], "judge\ngap detector", C_CONTROL, B_CONTROL, T_CONTROL)
        _box(ax, cx, Y["PRE"], "pre_retry",            C_CONTROL, B_CONTROL, T_CONTROL)

    _box(ax, cx, Y["NRM"], "normalize\nruleId canon.", C_CONTROL, B_CONTROL, T_CONTROL)
    _box(ax, cx, Y["SAV"], "save\n→ CSV",              C_IO,      B_IO,      T_IO)
    _circle(ax, cx, Y["END"], "END", C_ENDPOINT, B_ENDPOINT, T_ENDPOINT)

    # ── draw edges (forward) ─────────────────────────────────────────────────
    top   = lambda y: y + BOX_H / 2
    bot   = lambda y: y - BOX_H / 2
    top_c = lambda y: y + R_CIRC

    # START → COORD
    _arrow(ax, cx, top_c(Y["START"]) - R_CIRC, cx, top(Y["COORD"]))

    # COORD → extractors
    _arrow(ax, cx, bot(Y["COORD"]), 0.18, top(Y["EXTR"]), "narrative / mixed", rad=-0.15)
    _arrow(ax, cx, bot(Y["COORD"]), 0.50, top(Y["EXTR"]), "tabular")
    _arrow(ax, cx, bot(Y["COORD"]), 0.82, top(Y["EXTR"]), "matrix", rad=0.15)

    # extractors → MRG
    _arrow(ax, 0.18, bot(Y["EXTR"]), cx, top(Y["MRG"]), rad=0.15)
    _arrow(ax, 0.50, bot(Y["EXTR"]), cx, top(Y["MRG"]))
    _arrow(ax, 0.82, bot(Y["EXTR"]), cx, top(Y["MRG"]), rad=-0.15)

    # MRG → VAL → ...
    _arrow(ax, cx, bot(Y["MRG"]), cx, top(Y["VAL"]))

    if has_judge:
        _arrow(ax, cx, bot(Y["VAL"]), cx, top(Y["JDG"]))
        # "pass": label moved left so it doesn't land on top of the PRE box
        _arrow(ax, cx - 0.02, bot(Y["JDG"]), cx - 0.02, top(Y["NRM"]), "pass",
               label_dx=-0.12, label_dy=0.04)
        _arrow(ax, cx + 0.02, bot(Y["JDG"]), cx + 0.02, top(Y["PRE"]), "needs retry")

        # PRE → extractors retry: simple diagonals from PRE corners to extractor outer edges
        _arrow(ax, cx - BOX_W / 2, bot(Y["PRE"]),
               0.18 - BOX_W / 2, Y["EXTR"], color="#999")
        _arrow(ax, cx + BOX_W / 2, bot(Y["PRE"]),
               0.82 + BOX_W / 2, Y["EXTR"], color="#999")
    else:
        _arrow(ax, cx, bot(Y["VAL"]), cx, top(Y["NRM"]))

    _arrow(ax, cx, bot(Y["NRM"]), cx, top(Y["SAV"]))
    _arrow(ax, cx, bot(Y["SAV"]), cx, Y["END"] + R_CIRC)

    # ── save ─────────────────────────────────────────────────────────────────
    tag     = f"abl_{ablation}" if ablation else "full"
    out_png = os.path.join(_DIR, f"pipeline_{tag}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize step2_TEST pipeline graph")
    parser.add_argument(
        "--ablation", choices=list(_ABL_TAG_MAP.keys()), default=None,
        help="Visualize an ablation variant (omit for full pipeline)",
    )
    parser.add_argument(
        "--all", dest="all_variants", action="store_true",
        help="Render full pipeline + all ablation variants",
    )
    args = parser.parse_args()

    if args.all_variants:
        visualize(None)
        for abl in _ABL_TAG_MAP:
            visualize(abl)
    else:
        visualize(args.ablation)
