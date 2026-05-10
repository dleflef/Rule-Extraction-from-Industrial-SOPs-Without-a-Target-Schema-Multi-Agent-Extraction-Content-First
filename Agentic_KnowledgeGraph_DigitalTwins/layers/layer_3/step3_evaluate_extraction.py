"""
step3_evaluate_extraction.py
================================
Evaluation pipeline to benchmark LLM-extracted SOP rules against Ground Truth.
Uses Hungarian-algorithm optimal matching with SBERT semantic scoring.

Saves one CSV per (model, paradigm) run, a combined file, and a short summary.

Usage
-----
    python3 layer_3/step3_evaluate_extraction.py
    python3 layer_3/step3_evaluate_extraction.py --model qwen2.5-7b-instruct
    python3 layer_3/step3_evaluate_extraction.py --paradigm naive
    python3 layer_3/step3_evaluate_extraction.py --model qwen2.5-7b-instruct --paradigm cot_basic

Output
------
    step3_results/eval_<model>_<paradigm>_run1.csv   -- per-run detail
    step3_results/comprehensive_evaluation_results.csv -- all runs merged
    step3_results/summary_metrics.csv                 -- one row per run, key metrics only
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util

# ── CONFIG ────────────────────────────────────────────────────────────────────

GROUND_TRUTH_FILE = "data/seed_rules/dataset/kg_seeds/ground_truth.csv"
RESULTS_DIR       = "step2_results"
STEP3_RESULTS_DIR = "step3_results"
COMBINED_OUTPUT_FILE = os.path.join(STEP3_RESULTS_DIR, "comprehensive_evaluation_results.csv")

MATCHING_THRESHOLD = 0.6  # minimum rule score to count as a valid TP match

# Fields compared in each rule pair
FIELDS = [
    "ruleId", "class", "station", "sensor", "sensorType",
    "condition", "action", "severity",
    "critHi", "warnHi", "critLo", "warnLo", "unit",
]

FIELD_WEIGHTS = {
    "condition":  3.0,
    "action":     3.0,
    "station":    2.0,
    "sensor":     2.0,
    "critHi":     1.5,
    "warnHi":     1.5,
    "critLo":     1.5,
    "warnLo":     1.5,
    "class":      1.0,
    "sensorType": 1.0,
    "severity":   1.0,
    "unit":       0.0,  # kept for output only
    "ruleId":     0.0,  # kept for reference, not scored
}

CATEGORICAL_FIELDS = ["class", "station", "sensor", "sensorType", "severity", "unit"]
NUMERIC_FIELDS     = ["critHi", "warnHi", "critLo", "warnLo"]
TEXT_FIELDS        = ["condition", "action"]


# ── EMBEDDING CACHE ───────────────────────────────────────────────────────────

_embedding_cache: dict = {}


def get_embedding(text: str, model: SentenceTransformer):
    text = str(text).strip()
    if not text:
        return None
    if text not in _embedding_cache:
        _embedding_cache[text] = model.encode(text, convert_to_tensor=True)
    return _embedding_cache[text]


# ── COMPARISON FUNCTIONS ──────────────────────────────────────────────────────


def compare_categorical(v1, v2) -> float:
    s1 = re.sub(r"[^a-z0-9]", "", str(v1 or "").lower())
    s2 = re.sub(r"[^a-z0-9]", "", str(v2 or "").lower())
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return 1.0 if s1 == s2 else 0.0


def compare_numeric(v1, v2) -> float:
    s1, s2 = str(v1 or "").strip(), str(v2 or "").strip()
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    try:
        f1 = float(re.sub(r"[^\d\.\-]", "", s1))
        f2 = float(re.sub(r"[^\d\.\-]", "", s2))
        return max(0.0, 1.0 - abs(f1 - f2) / max(abs(f1), abs(f2), 1e-9))
    except ValueError:
        return 1.0 if s1.lower() == s2.lower() else 0.0


def compare_text(text1, text2, model: SentenceTransformer) -> float:
    emb1 = get_embedding(text1, model)
    emb2 = get_embedding(text2, model)
    if emb1 is None and emb2 is None:
        return 1.0
    if emb1 is None or emb2 is None:
        return 0.0
    return max(0.0, min(1.0, util.cos_sim(emb1, emb2).item()))


def compute_rule_similarity(gt_rule: dict, ext_rule: dict, model: SentenceTransformer) -> float:
    score_sum, weight_sum = 0.0, 0.0
    for f in CATEGORICAL_FIELDS:
        w = FIELD_WEIGHTS[f]
        score_sum += compare_categorical(gt_rule.get(f), ext_rule.get(f)) * w
        weight_sum += w
    for f in NUMERIC_FIELDS:
        w = FIELD_WEIGHTS[f]
        score_sum += compare_numeric(gt_rule.get(f), ext_rule.get(f)) * w
        weight_sum += w
    for f in TEXT_FIELDS:
        w = FIELD_WEIGHTS[f]
        score_sum += compare_text(gt_rule.get(f), ext_rule.get(f), model) * w
        weight_sum += w
    return score_sum / weight_sum if weight_sum > 0 else 0.0


# ── SOURCE MAPPING ────────────────────────────────────────────────────────────


def _source_file_to_id(source_file: str) -> str:
    """Map 'SOP_001_OperatingProcedures.txt' -> 'SOP-001'."""
    m = re.match(r"(SOP)_(\d+)", str(source_file), re.IGNORECASE)
    return f"{m.group(1)}-{m.group(2)}" if m else str(source_file)


# ── EVALUATION PIPELINE ───────────────────────────────────────────────────────


def _match_source(
    gt_rules: list[dict],
    ext_rules: list[dict],
    sbert: SentenceTransformer,
) -> tuple[set, set, list]:
    """
    Hungarian matching for one (source, run) pair.
    Returns (matched_gt_indices, matched_ex_indices, [(gt_idx, ex_idx, score)]).
    """
    if not ext_rules:
        return set(), set(), []

    sim_matrix = np.zeros((len(gt_rules), len(ext_rules)))
    for i, gt in enumerate(gt_rules):
        for j, ext in enumerate(ext_rules):
            sim_matrix[i, j] = compute_rule_similarity(gt, ext, sbert)

    matched_gt: set[int] = set()
    matched_ex: set[int] = set()
    pairs: list[tuple]   = []

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    for r, c in zip(row_ind, col_ind):
        if sim_matrix[r, c] >= MATCHING_THRESHOLD:
            matched_gt.add(r)
            matched_ex.add(c)
            pairs.append((r, c, float(sim_matrix[r, c])))

    return matched_gt, matched_ex, pairs


def evaluate_run(
    gt_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    run_id: str,
    model_name: str,
    paradigm: str,
    sbert: SentenceTransformer,
) -> list[dict]:
    """
    Match extracted rules to ground truth per SOP source using the Hungarian algorithm.
    Grouping by source prevents cross-SOP false matches.
    Returns detail rows (TP, FN, FP) for the run.
    """
    gt_df  = gt_df.copy()
    ext_df = ext_df.copy()
    gt_df["_src"]  = gt_df["source"].fillna("unknown")       if "source"      in gt_df.columns  else "unknown"
    ext_df["_src"] = ext_df["source_file"].apply(_source_file_to_id) if "source_file" in ext_df.columns else "unknown"

    all_sources = sorted(set(gt_df["_src"].unique()) | set(ext_df["_src"].unique()))

    # Per-source matching — collect results first, then compute global metrics
    source_results = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for src in all_sources:
        gt_src  = gt_df[gt_df["_src"]  == src].to_dict("records")
        ext_src = ext_df[ext_df["_src"] == src].to_dict("records")
        if not gt_src and not ext_src:
            continue
        matched_gt, matched_ex, pairs = _match_source(gt_src, ext_src, sbert)
        tp = len(pairs)
        total_tp += tp
        total_fp += len(ext_src) - tp
        total_fn += len(gt_src)  - tp
        source_results.append((src, gt_src, ext_src, matched_gt, matched_ex, pairs))

    # Run-level aggregated metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    run_metrics = {
        "model_name":    model_name,
        "paradigm":      paradigm,
        "Run_F1":        round(f1, 4),
        "Run_Precision": round(precision, 4),
        "Run_Recall":    round(recall, 4),
        "Run_TP":        total_tp,
        "Run_FP":        total_fp,
        "Run_FN":        total_fn,
    }

    def build_row(src: str, match_type: str, score: float, gt_data: dict, ext_data: dict) -> dict:
        row = {"run_id": run_id, "source": src, **run_metrics,
               "match_type": match_type, "rule_match_score": round(score, 3)}
        for f in FIELDS:
            row[f"GT_{f}"]  = gt_data.get(f, "")
            row[f"EXT_{f}"] = ext_data.get(f, "")
        return row

    rows: list[dict] = []
    for src, gt_src, ext_src, matched_gt, matched_ex, pairs in source_results:
        for r, c, score in pairs:
            rows.append(build_row(src, "TP (Match)", score, gt_src[r], ext_src[c]))
        for i, gt in enumerate(gt_src):
            if i not in matched_gt:
                rows.append(build_row(src, "FN (Missed GT)", 0.0, gt, {}))
        for j, ext in enumerate(ext_src):
            if j not in matched_ex:
                rows.append(build_row(src, "FP (Hallucinated/Unmatched)", 0.0, {}, ext))

    return rows


# ── RESULT LOADING ────────────────────────────────────────────────────────────


def discover_result_files(
    results_dir: str,
    filter_model: str | None = None,
    filter_paradigm: str | None = None,
) -> list[str]:
    """Return sorted list of complete ext_*.csv files matching optional filters."""
    files = sorted(glob.glob(os.path.join(results_dir, "ext_*.csv")))
    files = [f for f in files if "_partial" not in os.path.basename(f)]

    if filter_model or filter_paradigm:
        filtered = []
        for f in files:
            ext_df = pd.read_csv(f, nrows=1).fillna("")
            model = ext_df["model_name"].iloc[0] if "model_name" in ext_df.columns else ""
            parad = ext_df["paradigm"].iloc[0]   if "paradigm"   in ext_df.columns else ""
            if filter_model    and filter_model    != model: continue
            if filter_paradigm and filter_paradigm != parad:  continue
            filtered.append(f)
        return filtered

    return files


# ── MAIN ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate step2 extracted rules against ground truth."
    )
    parser.add_argument("--model",    type=str, default=None, help="Filter by model name (e.g. qwen2.5-7b-instruct).")
    parser.add_argument("--paradigm", type=str, default=None, help="Filter by paradigm (e.g. naive, cot_basic).")
    args = parser.parse_args()

    os.makedirs(STEP3_RESULTS_DIR, exist_ok=True)

    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Error: ground truth not found at {GROUND_TRUTH_FILE}")
        return

    print("Loading ground truth...")
    gt_df = pd.read_csv(GROUND_TRUTH_FILE).fillna("")
    print(f"  {len(gt_df)} rules loaded.")

    print("Loading SBERT model (all-MiniLM-L6-v2)...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    result_files = discover_result_files(RESULTS_DIR, args.model, args.paradigm)
    if not result_files:
        print(f"No result files found in {RESULTS_DIR}/")
        return
    print(f"  {len(result_files)} run file(s) to evaluate.\n")

    all_rows: list[dict] = []

    for file_path in result_files:
        filename   = os.path.basename(file_path)
        run_id     = filename.replace("ext_", "").replace(".csv", "")
        ext_df     = pd.read_csv(file_path).fillna("")
        model_name = ext_df["model_name"].iloc[0] if "model_name" in ext_df.columns and len(ext_df) > 0 else ""
        paradigm   = ext_df["paradigm"].iloc[0]   if "paradigm"   in ext_df.columns and len(ext_df) > 0 else ""

        print(f"  Evaluating  {model_name} | {paradigm} ...")
        rows = evaluate_run(gt_df, ext_df, run_id, model_name, paradigm, sbert)
        all_rows.extend(rows)

        run_df  = pd.DataFrame(rows)
        run_out = os.path.join(STEP3_RESULTS_DIR, f"eval_{run_id}.csv")
        run_df.to_csv(run_out, index=False)
        f1 = rows[0]["Run_F1"] if rows else 0.0
        print(f"    F1={f1:.3f} | saved -> {run_out}")

    # Combined file
    final_df = pd.DataFrame(all_rows)
    final_df.sort_values(by=["Run_F1", "run_id", "match_type"], ascending=[False, True, False], inplace=True)
    final_df.to_csv(COMBINED_OUTPUT_FILE, index=False)

    # Summary: one row per run
    summary_df = (
        final_df.drop_duplicates(subset=["run_id"])[
            ["model_name", "paradigm", "run_id", "Run_F1", "Run_Precision", "Run_Recall", "Run_TP", "Run_FP", "Run_FN"]
        ]
        .sort_values(by=["model_name", "Run_F1"], ascending=[True, False])
    )
    summary_out = os.path.join(STEP3_RESULTS_DIR, "summary_metrics.csv")
    summary_df.to_csv(summary_out, index=False)

    print(f"\n{'='*65}")
    print("EVALUATION COMPLETE")
    print(f"  Individual:  {STEP3_RESULTS_DIR}/eval_<model>_<paradigm>_run1.csv")
    print(f"  Combined:    {COMBINED_OUTPUT_FILE}")
    print(f"  Summary:     {summary_out}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
