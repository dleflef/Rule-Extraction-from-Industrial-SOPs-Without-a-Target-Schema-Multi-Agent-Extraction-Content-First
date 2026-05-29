from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from step2_TEST import build_pipeline, _build_pipeline_for_ablation, ABLATION_CONFIGS

ABLATION_OPTIONS = ABLATION_CONFIGS  # convenience alias

# Output scale: 1 = native SVG size, 2 = 2× (sharper but larger file)
PNG_SCALE = 1


def _postprocess_mermaid(mermaid_code: str) -> str:
    """
    1. Switch layout direction to LR so the pipeline flows left→right and
       parallel branches (same rank) naturally stack top→bottom.
    2. Add ~~~ invisible links between nodes that share the same parent-set
       and child-set to hint dagre they belong at the same rank.
    """
    # Switch TD → LR
    code = mermaid_code.replace("graph TD;", "graph LR;")

    edge_re = re.compile(r"^\t(\S+)\s+(?:-->|-\.->)\s+(\S+);")
    parents_of:  dict[str, set[str]] = defaultdict(set)
    children_of: dict[str, set[str]] = defaultdict(set)
    for line in code.splitlines():
        m = edge_re.match(line)
        if m:
            src, dst = m.group(1), m.group(2)
            children_of[src].add(dst)
            parents_of[dst].add(src)

    all_nodes = set(parents_of) | set(children_of)
    sig_to_nodes: dict[tuple, list[str]] = defaultdict(list)
    for node in all_nodes:
        if node.startswith("__"):
            continue
        sig = (frozenset(parents_of[node]), frozenset(children_of[node]))
        if sig[0] or sig[1]:
            sig_to_nodes[sig].append(node)

    invisible = []
    for nodes in sig_to_nodes.values():
        if len(nodes) >= 2:
            ns = sorted(nodes)
            for a, b in zip(ns, ns[1:]):
                invisible.append(f"\t{a} ~~~ {b};")

    if invisible:
        lines = code.splitlines()
        insert_at = next(
            (i for i, l in enumerate(lines) if l.strip().startswith("classDef")),
            len(lines),
        )
        lines[insert_at:insert_at] = invisible
        code = "\n".join(lines)

    return code


def _parse_svg_dimensions(svg_path: str) -> tuple[int, int]:
    """Read width/height from the root <svg> element, falling back to viewBox."""
    with open(svg_path, encoding="utf-8") as f:
        header = f.read(4096)
    # Try explicit width/height attributes first
    w = re.search(r'<svg[^>]+\bwidth="([\d.]+)"', header)
    h = re.search(r'<svg[^>]+\bheight="([\d.]+)"', header)
    if w and h:
        return int(float(w.group(1))) + 40, int(float(h.group(1))) + 40
    # Fall back to viewBox="minX minY width height"
    vb = re.search(r'<svg[^>]+\bviewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"', header)
    if vb:
        return int(float(vb.group(1))) + 40, int(float(vb.group(2))) + 40
    return 3200, 2000


def _render_mermaid_png(mermaid_code: str, output_path: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mmd", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(mermaid_code)
        tmp.flush()
        tmp_path = tmp.name

    svg_path = output_path.replace(".png", "_tmp.svg")
    try:
        # Step 1: render to SVG (auto-sized to diagram content)
        subprocess.run(
            [
                "npx", "@mermaid-js/mermaid-cli",
                "-i", tmp_path,
                "-o", svg_path,
                "--backgroundColor", "white",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Step 2: measure the SVG's natural size
        w, h = _parse_svg_dimensions(svg_path)

        # Step 3: render to PNG using those exact dimensions × scale
        subprocess.run(
            [
                "npx", "@mermaid-js/mermaid-cli",
                "-i", tmp_path,
                "-o", output_path,
                "--width",  str(w),
                "--height", str(h),
                "--scale",  str(PNG_SCALE),
                "--backgroundColor", "white",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"PNG saved → {output_path}  ({w * PNG_SCALE}×{h * PNG_SCALE} px)")
    finally:
        os.unlink(tmp_path)
        if os.path.exists(svg_path):
            os.unlink(svg_path)


def visualize(ablation: str | None = None) -> None:
    if ablation:
        app = _build_pipeline_for_ablation(ablation)
        tag = f"abl_{ablation}"
    else:
        app = build_pipeline()
        tag = "full"

    out_png = os.path.join(_DIR, f"pipeline_{tag}.png")
    out_md  = os.path.join(_DIR, f"pipeline_{tag}.md")

    mermaid = _postprocess_mermaid(app.get_graph().draw_mermaid())

    try:
        _render_mermaid_png(mermaid, out_png)
    except Exception as e:
        # Fallback: save the Mermaid source and print it
        print(f"PNG rendering failed ({e}), falling back to Mermaid markdown.")
        with open(out_md, "w") as f:
            f.write(f"```mermaid\n{mermaid}\n```\n")
        print(f"Mermaid saved → {out_md}")
        print("Paste the code below at https://mermaid.live/ to view:")
        print(mermaid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize step2_TEST pipeline graph")
    parser.add_argument(
        "--ablation", choices=list(ABLATION_OPTIONS), default=None,
        help="Visualize an ablation variant instead of the full pipeline",
    )
    parser.add_argument(
        "--all", dest="all_variants", action="store_true",
        help="Render full pipeline + all four ablation variants",
    )
    args = parser.parse_args()

    if args.all_variants:
        visualize(None)
        for abl in ABLATION_OPTIONS:
            visualize(abl)
    else:
        visualize(args.ablation)