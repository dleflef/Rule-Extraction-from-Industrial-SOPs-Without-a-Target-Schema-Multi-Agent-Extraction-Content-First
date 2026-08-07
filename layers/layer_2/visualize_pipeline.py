"""visualize_pipeline.py
==================================================
Render the LangGraph topology that build_graph() constructs in
step2_multi_agent_generic.py.

The figure is generated from that module's own constants --- the node names, the
model assigned to each agent role, and the fan-out concurrency are imported
rather than retyped --- so a diagram cannot quietly describe a pipeline the code
no longer builds. Anything the figure states that is not importable (which stages
use no LLM, how many calls each stage issues) is derived from the graph structure
and labelled on the diagram itself.

Topology, mirroring build_graph() edge for edge:

    START -> load_segment -> [scout]* -> induce -> assemble -> [audit]* -> save -> END
                                                        \\_________________________/
                                                          (bypass when nothing to audit)

Starred stages fan out: dispatch_scouts emits one Send per chunk and
dispatch_auditors one per chunk that produced records, so the two agent roles are
instantiated per chunk and run concurrently. The scout -> induce edge is a
fan-in: induce runs once, after every scout branch returns, because the arbiter's
purpose is to see the corpus's complete observed vocabulary rather than one
chunk's.

Usage:
    python3 visualize_pipeline.py                # writes pipeline_full.png
    python3 visualize_pipeline.py --out FILE     # custom path
"""

from __future__ import annotations

import argparse
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from step2_multi_agent_generic import (  # noqa: E402
    DEFAULTS, DEFAULT_MAX_CONCURRENCY, DEFAULT_CHUNK_CHARS,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# ── colours ────────────────────────────────────────────────────────────────────
# Three node kinds are distinguished, because the distinction is the point: an
# LLM node is where non-determinism and cost enter, a deterministic node is where
# neither does, and the reader should be able to see at a glance that two of the
# six stages involve no model at all.
C_AGENT = "#dbeafe"   # blue   — LLM agent role
C_DET   = "#fef9c3"   # yellow — deterministic (no LLM)
C_IO    = "#dcfce7"   # green  — output
C_END   = "#f3f4f6"   # grey   — START / END
B_AGENT, B_DET, B_IO, B_END = "#3b82f6", "#ca8a04", "#16a34a", "#6b7280"
T_AGENT, T_DET, T_IO, T_END = "#1e3a8a", "#713f12", "#14532d", "#374151"

# x-positions for the parallel instances drawn side by side. Kept inside
# [0.15, 0.85] so the conditional bypass edge has a clear lane down the left
# margin without crossing any node.
FAN_X = (0.27, 0.50, 0.73)
FAN_W = 0.215


def _box(ax, cx, cy, w, h, label, fill, edge, text_color, fontsize=8.5, z=3):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.01",
        facecolor=fill, edgecolor=edge, linewidth=1.4,
        transform=ax.transAxes, clip_on=False, zorder=z,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", transform=ax.transAxes,
            multialignment="center", zorder=z + 1)


def _circle(ax, cx, cy, r, label, fill, edge, text_color):
    circ = mpatches.Circle((cx, cy), r, facecolor=fill, edgecolor=edge,
                           linewidth=1.2, transform=ax.transAxes,
                           clip_on=False, zorder=3)
    ax.add_patch(circ)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=8,
            color=text_color, fontweight="bold", transform=ax.transAxes, zorder=4)


def _arrow(ax, x0, y0, x1, y1, label="", color="#555", rad=0.0,
           label_dx=0.012, label_dy=0.0, style="-|>", lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=2)
    if label:
        ax.text((x0 + x1) / 2 + label_dx, (y0 + y1) / 2 + label_dy, label,
                ha="left", va="center", fontsize=7, color="#555",
                transform=ax.transAxes, zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def _layout(rows, start_y):
    """Stack rows top-down; each row is (name, height, gap_after)."""
    pos, cursor = {}, start_y
    for name, h, gap in rows:
        pos[name] = (cursor - h / 2, h)
        cursor -= h + gap
    return pos


def visualize(out_path: str | None = None) -> None:
    rows = [
        ("start",    0.040, 0.045),
        ("segment",  0.085, 0.070),
        ("scout",    0.095, 0.070),
        ("induce",   0.090, 0.045),
        ("assemble", 0.085, 0.070),
        ("audit",    0.095, 0.060),
        ("save",     0.065, 0.045),
        ("end",      0.040, 0.0),
    ]
    pos = _layout(rows, start_y=0.955)
    top = lambda n: pos[n][0] + pos[n][1] / 2   # noqa: E731
    bot = lambda n: pos[n][0] - pos[n][1] / 2   # noqa: E731
    cy  = lambda n: pos[n][0]                   # noqa: E731

    fig, ax = plt.subplots(figsize=(11.5, 13))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("step2_multi_agent_generic.py — LangGraph pipeline\n"
                 "no target schema, no rule taxonomy, no worked example is supplied to any stage",
                 fontsize=12.5, fontweight="bold", pad=16)

    # ── nodes ────────────────────────────────────────────────────────────────
    _circle(ax, 0.50, cy("start"), pos["start"][1] / 2, "START", C_END, B_END, T_END)

    _box(ax, 0.50, cy("segment"), 0.46, pos["segment"][1],
         "load_segment  —  no LLM\n"
         f"blank-line blocks packed into ~{DEFAULT_CHUNK_CHARS:,}-char chunks\n"
         "every content line numbered in code; headings carried by shape",
         C_DET, B_DET, T_DET, fontsize=7.8)

    # Only the centre instance is annotated in full; the flanking two carry the
    # bare role name, so the row reads as "many of these" without three copies
    # of the same paragraph.
    scout_full = ("scout  (1 call / chunk)\n"
                  f"{DEFAULTS['scout_model']}\n"
                  "field names copied from the\ndocument's own labels")
    for i, x in enumerate(FAN_X):
        _box(ax, x, cy("scout"), FAN_W, pos["scout"][1],
             scout_full if i == 1 else "scout\n" + DEFAULTS["scout_model"],
             C_AGENT, B_AGENT, T_AGENT, fontsize=7.4 if i == 1 else 8)

    _box(ax, 0.50, cy("induce"), 0.50, pos["induce"][1],
         "induce  —  1 call / CORPUS\n"
         f"{DEFAULTS['inducer_model']}\n"
         "sees only the observed field-name inventory,\nnever a document",
         C_AGENT, B_AGENT, T_AGENT, fontsize=7.8)

    _box(ax, 0.50, cy("assemble"), 0.46, pos["assemble"][1],
         "assemble  —  no LLM\n"
         "group by the document's own record boundary;\n"
         "rename each field to its canonical name",
         C_DET, B_DET, T_DET, fontsize=7.8)

    for i, x in enumerate(FAN_X):
        _box(ax, x, cy("audit"), FAN_W, pos["audit"][1],
             ("audit  (1 call / chunk)\n"
              f"{DEFAULTS['auditor_model']}\n"
              "re-reads each record against\nthe lines it cites")
             if i == 1 else "audit\n" + DEFAULTS["auditor_model"],
             C_AGENT, B_AGENT, T_AGENT, fontsize=7.4 if i == 1 else 8)

    _box(ax, 0.50, cy("save"), 0.42, pos["save"][1],
         "save  →  one column per discovered field\n+ facts sidecar + induced schema",
         C_IO, B_IO, T_IO, fontsize=7.8)

    _circle(ax, 0.50, cy("end"), pos["end"][1] / 2, "END", C_END, B_END, T_END)

    # ── edges ────────────────────────────────────────────────────────────────
    _arrow(ax, 0.50, bot("start"), 0.50, top("segment"))

    # fan-out to scouts (Send per chunk)
    for i, x in enumerate(FAN_X):
        _arrow(ax, 0.50, bot("segment"), x, top("scout"),
               "Send() per chunk" if i == 1 else "",
               rad=(-0.12 if i == 0 else (0.12 if i == 2 else 0.0)), label_dy=0.014)
    # fan-in to induce
    for i, x in enumerate(FAN_X):
        _arrow(ax, x, bot("scout"), 0.50, top("induce"),
               "fan-in: induce runs once,\nafter every scout returns" if i == 2 else "",
               rad=(0.12 if i == 0 else (-0.12 if i == 2 else 0.0)),
               label_dx=0.02, label_dy=-0.005)

    _arrow(ax, 0.50, bot("induce"), 0.50, top("assemble"))

    for i, x in enumerate(FAN_X):
        _arrow(ax, 0.50, bot("assemble"), x, top("audit"),
               "Send() per chunk" if i == 1 else "",
               rad=(-0.12 if i == 0 else (0.12 if i == 2 else 0.0)), label_dy=0.014)
    for i, x in enumerate(FAN_X):
        _arrow(ax, x, bot("audit"), 0.50, top("save"),
               rad=(0.12 if i == 0 else (-0.12 if i == 2 else 0.0)))

    # Conditional bypass: assemble -> save directly when no chunk produced
    # records to audit. Routed down the left margin so it crosses nothing.
    _arrow(ax, 0.50 - 0.23, bot("assemble") + 0.012, 0.09, cy("audit"),
           color="#9ca3af", rad=0.28, lw=1.0)
    _arrow(ax, 0.09, cy("audit"), 0.50 - 0.21, top("save") + 0.006,
           color="#9ca3af", rad=0.28, lw=1.0)
    ax.text(0.075, cy("audit"), "bypass when no\nrecords to audit",
            ha="center", va="center", fontsize=6.6, color="#6b7280",
            transform=ax.transAxes, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    _arrow(ax, 0.50, bot("save"), 0.50, top("end"))

    # ── annotations ──────────────────────────────────────────────────────────
    ax.text(0.995, cy("scout"),
            f"parallel,\nmax {DEFAULT_MAX_CONCURRENCY}\nconcurrent",
            ha="right", va="center", fontsize=7, color="#3b82f6",
            transform=ax.transAxes, style="italic")
    ax.text(0.995, cy("audit"),
            f"parallel,\nmax {DEFAULT_MAX_CONCURRENCY}\nconcurrent",
            ha="right", va="center", fontsize=7, color="#3b82f6",
            transform=ax.transAxes, style="italic")

    handles = [
        mpatches.Patch(facecolor=C_AGENT, edgecolor=B_AGENT, label="LLM agent role"),
        mpatches.Patch(facecolor=C_DET,   edgecolor=B_DET,   label="deterministic (no LLM)"),
        mpatches.Patch(facecolor=C_IO,    edgecolor=B_IO,    label="output"),
        mpatches.Patch(facecolor=C_END,   edgecolor=B_END,   label="START / END"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.055),
              fontsize=8.5, frameon=False, ncol=4)

    out_png = out_path or os.path.join(_DIR, "pipeline_full.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Visualize the LangGraph pipeline built by step2_multi_agent_generic.py")
    ap.add_argument("--out", default=None, help="Output PNG path.")
    args = ap.parse_args()
    visualize(args.out)
