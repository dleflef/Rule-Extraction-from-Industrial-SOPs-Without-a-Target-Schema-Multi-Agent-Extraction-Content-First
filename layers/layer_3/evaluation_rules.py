# evaluation_rules.py
#
# Source‑aware field‑level evaluation.
# Extended to support multiple extraction paradigms.
# Usage:
#   python evaluation_rules.py --paradigm baseline
#   python evaluation_rules.py --all

import json
import os
import csv
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

from sentence_transformers import SentenceTransformer

# ── CONFIG ────────────────────────────────────────────────────────────────────
GT_CSV_PATH = "layers/data/seed_rules/dataset/kg_seeds/ground_truth.csv"
EXTRACTED_DIR = "outputs/layer_2"
OUTPUT_REPORT_TEMPLATE = "evaluation_report_{paradigm}.json"  # {paradigm} is replaced

# FIELD_WEIGHTS — unchanged
FIELD_WEIGHTS = {
    "ruleId":      0.0,
    "class":       0.08,
    "station":     0.12,
    "sensor":      0.12,
    "sensorType":  0.08,
    "condition":   0.20,
    "action":      0.16,
    "severity":    0.04,
    "critHi":      0.05,
    "warnHi":      0.05,
    "critLo":      0.05,
    "warnLo":      0.05,
    "unit":        0.0,
    "source":      0.0,
}

NUMERIC_FIELDS = ["critHi", "warnHi", "critLo", "warnLo"]
NUMERIC_TOLERANCE = 0.15
CATEGORICAL_FIELDS = ["class", "station", "sensor", "sensorType", "severity", "unit", "source"]
TEXT_FIELDS = ["condition", "action"]
MATCH_THRESHOLD = 0.5

# ── Helper functions (identical to original, except where noted) ───────────────

def load_ground_truth(path: str) -> List[Dict]:
    rules = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for num_field in NUMERIC_FIELDS:
                v = row.get(num_field, "").strip()
                row[num_field] = float(v) if v else None
            rules.append(row)
    return rules


def load_extracted_rules_by_source_and_paradigm(
    directory: str, paradigm: Optional[str] = None
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Returns nested dict: {paradigm: {source_name: [list of rule dicts]}}.
    If paradigm is specified, only that paradigm is loaded; otherwise all are loaded.
    """
    paradigm_sources = defaultdict(lambda: defaultdict(list))
    for fname in os.listdir(directory):
        if not fname.endswith(".json"):
            continue
        # Expect names like SOP_001_agent2a_baseline.json
        parts = fname.rsplit("_agent2a_", 1)
        if len(parts) != 2:
            continue  # skip files not following the new naming convention
        base, rest = parts
        parad = rest.replace(".json", "")
        if paradigm and parad != paradigm:
            continue

        # Derive source ID (e.g., SOP-001)
        base_parts = base.split("_")
        if len(base_parts) >= 2:
            source = "SOP-" + base_parts[1]
        else:
            source = base

        with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
            for rule in data.get("rules", []):
                if not rule.get("source"):
                    rule["source"] = source
                paradigm_sources[parad][source].append(rule)

    return dict(paradigm_sources)


def categorical_similarity(v1: Optional[str], v2: Optional[str]) -> float:
    if v1 is None or v2 is None:
        return 0.0
    return 1.0 if v1.strip().lower() == v2.strip().lower() else 0.0


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def numeric_similarity(v1, v2) -> float:
    v1 = _to_float(v1)
    v2 = _to_float(v2)
    if v1 is None and v2 is None:
        return 1.0
    if v1 is None or v2 is None:
        return 0.0
    denom = max(abs(v1), 1e-9)
    tol = max(NUMERIC_TOLERANCE * denom, 0.5)
    return 1.0 if abs(v1 - v2) <= tol else 0.0


def text_similarity(t1: str, t2: str, model: SentenceTransformer) -> float:
    if not t1 or not t2:
        return 0.0
    from sentence_transformers import util
    emb = model.encode([t1, t2], show_progress_bar=False)
    sim = float(util.cos_sim(emb[0], emb[1]))
    return max(0.0, sim)


def rule_similarity(gt_dict: Dict, ex_dict: Dict, model: SentenceTransformer) -> float:
    score = 0.0
    total_weight = 0.0
    for field, w in FIELD_WEIGHTS.items():
        if w == 0.0:
            continue
        total_weight += w
        gt_val = gt_dict.get(field)
        ex_val = ex_dict.get(field)
        if field in NUMERIC_FIELDS:
            sim = numeric_similarity(gt_val, ex_val)
        elif field in CATEGORICAL_FIELDS:
            sim = categorical_similarity(gt_val, ex_val)
        elif field in TEXT_FIELDS:
            sim = text_similarity(gt_val or "", ex_val or "", model)
        else:
            sim = categorical_similarity(gt_val, ex_val)
        score += w * sim
    return score / total_weight if total_weight > 0 else 0.0


def match_rules_greedy(gt_list: List, ex_list: List, model: SentenceTransformer) -> Tuple[List, List, List]:
    matched = []
    available_ex = list(range(len(ex_list)))
    for gt_idx, gt_rule in enumerate(gt_list):
        best_sim = -1.0
        best_ex = None
        for ex_idx in available_ex:
            sim = rule_similarity(gt_rule, ex_list[ex_idx], model)
            if sim > best_sim:
                best_sim = sim
                best_ex = ex_idx
        if best_sim >= MATCH_THRESHOLD and best_ex is not None:
            matched.append((gt_idx, best_ex, best_sim))
            available_ex.remove(best_ex)
    unmatched_gt = [i for i in range(len(gt_list)) if i not in [p[0] for p in matched]]
    unmatched_ex = [i for i in range(len(ex_list)) if i not in [p[1] for p in matched]]
    return matched, unmatched_gt, unmatched_ex


def evaluate_source(gt_source: List, ex_source: List, model: SentenceTransformer) -> Tuple[Dict, List]:
    matched, unm_gt, unm_ex = match_rules_greedy(gt_source, ex_source, model)
    tp = len(matched)
    precision = tp / len(ex_source) if ex_source else 0.0
    recall = tp / len(gt_source) if gt_source else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_sim = sum(s for _, _, s in matched) / tp if tp else 0.0
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "num_gt": len(gt_source),
        "num_ex": len(ex_source),
        "unmatched_gt": [gt_source[i]["ruleId"] for i in unm_gt],
        "unmatched_ex": [(ex_source[i].get("condition") or "")[:80] for i in unm_ex],
        "avg_similarity": avg_sim,
    }
    return metrics, matched


def evaluate_paradigm(paradigm: str, gt_by_source: Dict[str, List], sbert: SentenceTransformer):
    """
    Run evaluation for one paradigm, return the full report dict.
    """
    # Load extracted rules for this paradigm
    paradigm_data = load_extracted_rules_by_source_and_paradigm(EXTRACTED_DIR, paradigm)
    extracted_by_source = paradigm_data.get(paradigm, {})

    all_sources = set(gt_by_source.keys()) | set(extracted_by_source.keys())
    per_source_metrics = {}
    total_tp = 0
    total_gt = 0
    total_ex = 0
    all_matched_pairs = []

    for src in sorted(all_sources):
        gt_list = gt_by_source.get(src, [])
        ex_list = extracted_by_source.get(src, [])
        if not gt_list and not ex_list:
            continue
        res, matched = evaluate_source(gt_list, ex_list, sbert)
        per_source_metrics[src] = res
        total_tp += res["tp"]
        total_gt += len(gt_list)
        total_ex += len(ex_list)

        for gt_idx, ex_idx, sim in matched:
            gt_rule = gt_list[gt_idx]
            ex_rule = ex_list[ex_idx]
            all_matched_pairs.append({
                "source": src,
                "similarity": round(sim, 4),
                "gt_ruleId": gt_rule.get("ruleId"),
                "extracted_ruleId": ex_rule.get("ruleId"),
                "ground_truth_fields": gt_rule,
                "extracted_fields": ex_rule,
            })

    global_precision = total_tp / total_ex if total_ex else 0.0
    global_recall = total_tp / total_gt if total_gt else 0.0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0

    report = {
        "paradigm": paradigm,
        "global_metrics": {
            "precision": round(global_precision, 4),
            "recall": round(global_recall, 4),
            "f1": round(global_f1, 4),
            "tp": total_tp,
            "total_gt": total_gt,
            "total_extracted": total_ex,
        },
        "per_source_metrics": {
            src: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
            for src, m in per_source_metrics.items()
        },
        "all_matched_pairs": all_matched_pairs,
    }
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate extracted rules against ground truth."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--paradigm", type=str,
        help="Evaluate a single paradigm (e.g., baseline, cot_structured)."
    )
    group.add_argument(
        "--all", action="store_true",
        help="Evaluate all paradigms found in the extracted directory."
    )
    args = parser.parse_args()

    # Load ground truth
    print("Loading ground truth...")
    all_gt = load_ground_truth(GT_CSV_PATH)
    gt_by_source = defaultdict(list)
    for rule in all_gt:
        src = rule.get("source", "unknown")
        gt_by_source[src].append(rule)
    print(f"Total GT rules: {len(all_gt)}")

    # Load SBERT once
    print("Loading SBERT model...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    if args.paradigm:
        paradigms = [args.paradigm]
    else:
        # Discover all paradigms from filenames
        discovered = set()
        for fname in os.listdir(EXTRACTED_DIR):
            if "_agent2a_" in fname and fname.endswith(".json"):
                parad = fname.rsplit("_agent2a_", 1)[1].replace(".json", "")
                discovered.add(parad)
        paradigms = sorted(discovered) if discovered else []
        if not paradigms:
            print("No paradigm files found.")
            return
        print(f"Found paradigms: {paradigms}")

    for paradigm in paradigms:
        print(f"\n=== Evaluating paradigm: {paradigm} ===")
        report = evaluate_paradigm(paradigm, gt_by_source, sbert)
        # Print summary
        gm = report["global_metrics"]
        print(f"Precision: {gm['precision']:.3f}, Recall: {gm['recall']:.3f}, F1: {gm['f1']:.3f}")
        print(f"TP: {gm['tp']}, GT: {gm['total_gt']}, Extracted: {gm['total_extracted']}")

        # Save report
        out_path = OUTPUT_REPORT_TEMPLATE.format(paradigm=paradigm)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()


# python evaluation_rules.py --all