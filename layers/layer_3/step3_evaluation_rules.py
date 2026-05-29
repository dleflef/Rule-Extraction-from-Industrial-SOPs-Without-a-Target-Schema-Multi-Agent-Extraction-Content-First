"""
step3_evaluation.py
==================================================
Evaluation framework for industrial SOP rule extraction.

Implements the evaluation methodology strictly from the iMAKS benchmark:
  1. F1_strict: Exact match of ruleIds after normalisation.
  2. F1_content: Content-first metric using Hungarian assignment (primary).
     Score per pair is the weighted average of active field scores
     (fields present in the GT, i.e. score ≠ N/A):

         content_score(g,l) = Σ(w_f * s_f) / Σ(w_f)   for f in F*(g,l)

     Field scoring by type:
       (a) categorical fields: exact match  → s_f ∈ {0, 1}
       (b) numeric fields:     tolerance    → s_f = max(0, 1 - |gt-llm|/max(|gt|,|llm|,ε))
       (c) text fields:        SBERT cosine → s_f ∈ [0, 1]
     Weights are raw (not pre-normalised); the denominator Σw_f normalises
     dynamically over whichever fields are active in the GT row.

Optimizations:
  - Hardware acceleration: Auto-detects and utilises MPS or CUDA.
  - Strict Column Ordering: Enforces requested column sequence in final summary CSV.
"""

import argparse
import csv
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util

# ── PATH CONFIGURATION ─────────────────────────────────────────────────────────

_LAYERS_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.normpath(os.path.join(_LAYERS_DIR, ".."))

GROUND_TRUTH_FILE = os.path.join(
    _PROJECT_ROOT, "data", "dataset", "kg_seed", "ground_truth.csv"
)
RESULTS_DIR       = os.path.join(_LAYERS_DIR, "layer_2", "step2_results")
STEP3_RESULTS_DIR = os.path.join(_LAYERS_DIR, "step3_results")

os.makedirs(STEP3_RESULTS_DIR, exist_ok=True)


# ── EVALUATION CONFIG ──────────────────────────────────────────────────────────

# Raw field weights per rule class, directly from the README weight table.
# Weights are NOT pre-normalised; the content_agreement function divides by
# the sum of weights for active fields (those present in the GT row).
#
# Field types:
#   Categorical (exact match): class, station, sensor, sensorType, severity, unit
#   Numeric (tolerance score): critHi, critLo, warnHi, warnLo
#   Text (SBERT cosine):       condition, action
#
# '—' in the table means the field is absent for that class (no key → skipped).
# Note: condition/action have low weights for ThresholdRule because the GT stores
# them in synthetic form (e.g. "T > 80°C"); the numeric thresholds are primary.

FIELD_WEIGHTS: dict[str, dict[str, float]] = {
    "ThresholdRule": {
        "class":      1.5,
        "station":    2.0,
        "sensor":     2.5,
        "sensorType": 1.0,
        "severity":   0.5,
        "unit":       1.0,
        "critHi":     3.0,
        "critLo":     3.0,
        "warnHi":     1.5,
        "warnLo":     1.5,
        "condition":  0.5,
        "action":     0.5,
    },
    "OperationalRule": {
        "class":      1.5,
        "station":    2.0,
        "sensor":     1.5,
        "sensorType": 0.5,
        "severity":   1.0,
        "critHi":     1.0,
        "critLo":     1.0,
        "condition":  4.0,
        "action":     2.0,
    },
    "MaintenanceRule": {
        "class":      1.5,
        "station":    2.0,
        "sensor":     2.5,
        "sensorType": 1.0,
        "severity":   1.5,
        "condition":  3.5,
        "action":     2.0,
    },
    "AccessRule": {
        "class":      1.5,
        "station":    2.0,
        "severity":   1.5,
        "condition":  2.0,
        "action":     1.5,
    },
    "default": {
        "class":      1.0,
        "station":    1.5,
        "sensor":     1.5,
        "severity":   1.0,
        "condition":  2.0,
        "action":     1.0,
    },
}

# Field type lookup — determines which scoring function to apply.
_CATEGORICAL_FIELDS = {"class", "station", "sensor", "sensorType", "severity", "unit"}
_NUMERIC_FIELDS     = {"critHi", "critLo", "warnHi", "warnLo"}
_TEXT_FIELDS        = {"condition", "action"}


# ── HARDWARE ACCELERATION & CACHING ────────────────────────────────────────────

_sbert_model = None

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_sbert() -> SentenceTransformer:
    global _sbert_model
    if _sbert_model is None:
        device = get_device()
        print(f"    Loading SBERT model (all-MiniLM-L6-v2) on device: [{device.upper()}]...")
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _sbert_model


# ── SIMILARITY FUNCTIONS ───────────────────────────────────────────────────────

def _norm_str_id(s: Any) -> str:
    """Aggressive normalisation for ruleId: lowercased, only alphanumeric."""
    if pd.isna(s) or s is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _norm_str_field(s: Any) -> str:
    """Simple normalisation for categorical fields: strip, lower."""
    if pd.isna(s) or s is None:
        return ""
    return str(s).strip().lower()

def _numeric_tolerance(gt_val: Any, llm_val: Any, epsilon: float = 1e-9) -> float:
    """Symmetric iMAKS tolerance score: max(0, 1 - |a-b| / max(|a|,|b|,ε))."""
    try:
        f_gt = float(str(gt_val).strip())
        f_llm = float(str(llm_val).strip())
    except ValueError:
        return 1.0 if str(gt_val).strip() == str(llm_val).strip() else 0.0

    denom = max(abs(f_gt), abs(f_llm), epsilon)
    return max(0.0, 1.0 - abs(f_gt - f_llm) / denom)

def _batch_encode_texts(texts: List[str]) -> Dict[str, Any]:
    unique = list({t for t in texts if t})
    if not unique:
        return {}
    model = get_sbert()
    embs = model.encode(unique, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
    return {t: embs[i] for i, t in enumerate(unique)}


def _semantic_similarity(text_a: str, text_b: str, emb_cache: Dict[str, Any] = None) -> float:
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    model = get_sbert()
    emb_a = emb_cache.get(text_a) if emb_cache else None
    emb_b = emb_cache.get(text_b) if emb_cache else None
    if emb_a is None:
        emb_a = model.encode(text_a, convert_to_tensor=True)
    if emb_b is None:
        emb_b = model.encode(text_b, convert_to_tensor=True)

    cos_sim = util.cos_sim(emb_a, emb_b).item()
    return max(0.0, float(cos_sim))


def content_agreement(r_gt: Dict, r_ext: Dict, emb_cache: Dict[str, Any] = None) -> float:
    """
    Weighted average content score per the README formula:
        content_score = Σ(w_f * s_f) / Σ(w_f)   for f in F*(g,l)
    where F*(g,l) = fields with a non-empty GT value.
    Weights are raw (not pre-normalised); the denominator handles normalisation.
    """
    rule_class = r_gt.get("class", "")
    weights = FIELD_WEIGHTS.get(rule_class, FIELD_WEIGHTS["default"])

    total_weight = 0.0
    total_score  = 0.0

    for field, w in weights.items():
        gt_val = r_gt.get(field)
        if not (pd.notna(gt_val) and str(gt_val).strip()):
            continue

        total_weight += w
        ext_val = r_ext.get(field, "")   # 👈 default to ""

        if field in _CATEGORICAL_FIELDS:
            score = 1.0 if _norm_str_field(gt_val) == _norm_str_field(ext_val) else 0.0
        elif field in _NUMERIC_FIELDS:
            score = _numeric_tolerance(gt_val, ext_val)
        elif field in _TEXT_FIELDS:
            score = _semantic_similarity(str(gt_val), str(ext_val), emb_cache)
        else:
            score = 0.0

        total_score += w * score

    if total_weight == 0:
        return 0.0
    return total_score / total_weight


# ── EVALUATION METRICS ─────────────────────────────────────────────────────────

def evaluate_strict(gt_rules: List[Dict], ext_rules: List[Dict]) -> Tuple[float, float, float, int, int, int]:
    gt_ids = set(_norm_str_id(r.get("ruleId", "")) for r in gt_rules if r.get("ruleId"))
    ext_ids = set(_norm_str_id(r.get("ruleId", "")) for r in ext_rules if r.get("ruleId"))

    gt_ids.discard("")
    ext_ids.discard("")

    tp = len(gt_ids.intersection(ext_ids))
    fp = len(ext_ids - gt_ids)
    fn = len(gt_ids - ext_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall, tp, fp, fn


def evaluate_content(gt_rules: List[Dict], ext_rules: List[Dict], threshold: float = 0.6) -> Tuple[float, float, float, int, int, int]:
    if not gt_rules or not ext_rules:
        return 0.0, 0.0, 0.0, 0, len(ext_rules), len(gt_rules)

    all_texts = [str(r.get(f, "")) for r in gt_rules + ext_rules for f in _TEXT_FIELDS]
    emb_cache = _batch_encode_texts(all_texts)

    cost_matrix = np.zeros((len(gt_rules), len(ext_rules)))
    for i, gt in enumerate(gt_rules):
        for j, ext in enumerate(ext_rules):
            cost_matrix[i, j] = 1.0 - content_agreement(gt, ext, emb_cache)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp = sum(1 for i, j in zip(row_ind, col_ind) if (1.0 - cost_matrix[i, j]) >= threshold)
    fp = len(ext_rules) - tp
    fn = len(gt_rules) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall, tp, fp, fn


# ── MAIN EXECUTION ─────────────────────────────────────────────────────────────

def run_evaluation(gt_path: str, pred_path: str) -> dict:
    try:
        gt_df = pd.read_csv(gt_path)
        pred_df = pd.read_csv(pred_path)
        gt_rules = gt_df.fillna("").to_dict("records")
        ext_rules = pred_df.fillna("").to_dict("records")
    except Exception as e:
        print(f"    Error loading files for {os.path.basename(pred_path)}: {e}")
        return {}

    f1_s, pr_s, re_s, tp_s, fp_s, fn_s = evaluate_strict(gt_rules, ext_rules)
    f1_c, pr_c, re_c, tp_c, fp_c, fn_c = evaluate_content(gt_rules, ext_rules)

    return {
        "file_name": os.path.basename(pred_path),
        "total_extracted": len(ext_rules),
        "strict_f1": round(f1_s, 3), "strict_pr": round(pr_s, 3), "strict_re": round(re_s, 3),
        "strict_tp": tp_s, "strict_fp": fp_s, "strict_fn": fn_s,
        "content_f1": round(f1_c, 3), "content_pr": round(pr_c, 3), "content_re": round(re_c, 3),
        "content_tp": tp_c, "content_fp": fp_c, "content_fn": fn_c
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM Rule Extraction")
    parser.add_argument("--gt", default=GROUND_TRUTH_FILE, help="Path to ground truth CSV")
    parser.add_argument("--pred", default=None, help="Path to specific extraction CSV. If omitted, evaluates all in RESULTS_DIR.")
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        print(f"CRITICAL ERROR: Ground truth file not found at {args.gt}")
        exit(1)

    print(f"\n{'='*65}")
    print(f"  iMAKS Evaluation Framework (GPU Accelerated)")
    print(f"  Ground Truth: {args.gt}")
    print(f"{'='*65}")

    get_sbert()

    if args.pred:
        print(f"\nEvaluating single file: {args.pred}")
        metrics = run_evaluation(args.gt, args.pred)
        if metrics:
            print(f"\n--- F1_content (Semantic Assignment) ---")
            print(f"F1:        {metrics['content_f1']:.3f}")
            print(f"Precision: {metrics['content_pr']:.3f}")
            print(f"Recall:    {metrics['content_re']:.3f}")
            print(f"TP: {metrics['content_tp']} | FP: {metrics['content_fp']} | FN: {metrics['content_fn']}")

            print(f"\n--- F1_strict (Exact ruleId match) ---")
            print(f"F1:        {metrics['strict_f1']:.3f}")
            print(f"Precision: {metrics['strict_pr']:.3f}")
            print(f"Recall:    {metrics['strict_re']:.3f}")

    else:
        print(f"\nBatch Mode: Scanning {RESULTS_DIR} for results...")
        if not os.path.exists(RESULTS_DIR):
            print(f"Directory not found: {RESULTS_DIR}")
            exit(1)

        csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv') and not f.endswith('_partial.csv')]

        if not csv_files:
            print(f"No prediction CSVs found in {RESULTS_DIR}")
            exit(0)

        print(f"Found {len(csv_files)} files to evaluate.\n")

        all_metrics = []
        for i, file_name in enumerate(csv_files, 1):
            file_path = os.path.join(RESULTS_DIR, file_name)
            print(f"  [{i}/{len(csv_files)}] Evaluating {file_name}...")
            m = run_evaluation(args.gt, file_path)
            if m:
                all_metrics.append(m)

        if all_metrics:
            summary_df = pd.DataFrame(all_metrics)

            ordered_cols = [
                "file_name", "total_extracted",
                "strict_f1", "strict_pr", "strict_re", "strict_tp", "strict_fp", "strict_fn",
                "content_f1", "content_pr", "content_re", "content_tp", "content_fp", "content_fn"
            ]
            summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]
            summary_df = summary_df.sort_values(by="content_f1", ascending=False)

            summary_out_path = os.path.join(STEP3_RESULTS_DIR, "evaluation_summary.csv")
            summary_df.to_csv(summary_out_path, index=False)

            print(f"\n{'='*65}")
            print(f"Batch Evaluation Complete.")
            print(f"Summary report saved to: {summary_out_path}")
            print(f"{'='*65}")
            print(summary_df[["file_name", "content_f1", "content_pr", "content_re"]].head(10).to_string(index=False))