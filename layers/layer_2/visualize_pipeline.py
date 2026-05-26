from __future__ import annotations

import argparse
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from step2_TEST import build_pipeline, build_ablation_pipeline, ABLATION_OPTIONS


def visualize(ablation: str | None = None) -> None:
    if ablation:
        app = build_ablation_pipeline(ablation)
        tag = f"abl_{ablation}"
    else:
        app = build_pipeline()
        tag = "full"

    out_png = os.path.join(_DIR, f"pipeline_{tag}.png")
    out_md  = os.path.join(_DIR, f"pipeline_{tag}.md")

    mermaid = app.get_graph().draw_mermaid()

    try:
        app.get_graph().draw_mermaid_png(output_file_path=out_png)
        print(f"PNG saved → {out_png}")
    except Exception as e:
        print(f"PNG failed ({e}), falling back to Mermaid markdown.")
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
