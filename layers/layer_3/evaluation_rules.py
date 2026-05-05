# evaluation_rules.py
#
# Source‑aware field‑level evaluation.
# Matches extracted rules against ground truth only when they come from the same SOP document.
# Embeds full explicit fields in the final JSON output for easy comparison.

import json
import os
import csv
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

from sentence_transformers import SentenceTransformer  # provides semantic similarity for text fields

# ── CONFIG ────────────────────────────────────────────────────────────────────
GT_CSV_PATH = "layers/data/seed_rules/dataset/kg_seeds/ground_truth.csv"  # path to ground truth CSV
EXTRACTED_DIR = "outputs/layer_2"  # directory containing agent2a JSON outputs (one per SOP)
OUTPUT_REPORT = "evaluation_report_v2.json"  # where to save the detailed report

# FIELD_WEIGHTS: Each field's contribution to the rule similarity score.
# Weights sum to 1.0 across all non‑zero fields. Fields with 0.0 are ignored entirely.
# The assignment reflects the semantic importance of each field.
FIELD_WEIGHTS = {
    "ruleId":      0.0,   # intentionally zero – rule IDs are unpredictable, not semantically meaningful
    "class":       0.08,  # rule type (Operational, Threshold, etc.) – low weight
    "station":     0.12,  # which station/zone the rule belongs to – medium weight
    "sensor":      0.12,  # which sensor is involved – medium weight
    "sensorType":  0.08,  # abbreviation of sensor type – low weight
    "condition":   0.20,  # the trigger condition – highest weight because it carries the core semantics
    "action":      0.16,  # the recommended action – high weight, closely tied to condition
    "severity":    0.04,  # severity keyword – low weight, often ambiguous
    "critHi":      0.05,  # critical high threshold – numeric but important
    "warnHi":      0.05,  # warning high threshold
    "critLo":      0.05,  # critical low threshold
    "warnLo":      0.05,  # warning low threshold
    "unit":        0.0,   # measurement unit – usually not discriminative
    "source":      0.0,   # document source – already used to partition evaluation, so zero weight here
}

# Fields containing numeric threshold values (physically measurable quantities).
NUMERIC_FIELDS = ["critHi", "warnHi", "critLo", "warnLo"]
# For numeric fields, we allow a relative tolerance of ±15% but at least 0.5 units absolute.
NUMERIC_TOLERANCE = 0.15
# Fields where we require an exact string match (case‑insensitive, trimmed).
CATEGORICAL_FIELDS = ["class", "station", "sensor", "sensorType", "severity", "unit", "source"]
# Fields where we compute semantic similarity using Sentence‑BERT.
TEXT_FIELDS = ["condition", "action"]
# Minimum rule‑pair similarity required to consider a match (0.5 = moderate similarity).
MATCH_THRESHOLD = 0.5

def load_ground_truth(path: str) -> List[Dict]:
    """Read the ground truth CSV and convert numeric fields from string to float (or None)."""
    rules = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # each row becomes a dict with column headers as keys
        for row in reader:
            # Convert each numeric field to float; treat empty strings as None.
            for num_field in NUMERIC_FIELDS:
                v = row.get(num_field, "").strip()
                row[num_field] = float(v) if v else None
            rules.append(row)
    return rules

def load_extracted_rules_by_source(directory: str) -> Dict[str, List[Dict]]:
    """
    Returns dict {source_name: [list of rule dicts]}.
    Grouping key is always the file-name-derived source (e.g. SOP_001_..._agent2a.json → SOP-001)
    so every rule from a file lands in the right bucket regardless of whether the LLM populated
    the source field. The rule's own source field is patched with the derived value when absent,
    keeping the rule self-consistent for downstream display.
    """
    source_rules = defaultdict(list)  # default empty list for new keys
    for fname in os.listdir(directory):
        if fname.endswith("_agent2a.json"):
            base = fname.split("_agent2a")[0]  # e.g., "SOP_001_OperatingProcedures"
            # Extract the second underscore-separated part to build source ID like SOP-001.
            source = "SOP-" + base.split("_")[1] if "_" in base else base
            with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                for rule in data.get("rules", []):
                    # If the LLM failed to set source (or set it to a wrong value), override it.
                    if not rule.get("source"):
                        rule["source"] = source
                    source_rules[source].append(rule)
    return dict(source_rules)  # convert to plain dict for easier usage

def categorical_similarity(v1: Optional[str], v2: Optional[str]) -> float:
    """Exact match on two categorical values. Returns 1.0 if equal (case‑insensitive), 0.0 otherwise."""
    if v1 is None or v2 is None:
        return 0.0
    return 1.0 if v1.strip().lower() == v2.strip().lower() else 0.0

def _to_float(v) -> Optional[float]:
    """Safely convert a value (string, int, float, or None) to float; return None if impossible."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def numeric_similarity(v1, v2) -> float:
    """
    Compare two numeric values using a tolerance band.
    If both are None, they are considered equal (1.0).
    If only one is None, they differ (0.0).
    Otherwise, the difference must be ≤ max(15% of the ground truth value, 0.5 units).
    """
    v1 = _to_float(v1)  # ground truth
    v2 = _to_float(v2)  # extracted
    if v1 is None and v2 is None:
        return 1.0  # both missing → agree
    if v1 is None or v2 is None:
        return 0.0  # one missing, one present → disagreement
    # Compute tolerance: at least 15% of the ground truth magnitude, but never less than 0.5 absolute.
    denom = max(abs(v1), 1e-9)  # avoid division by zero
    tol = max(NUMERIC_TOLERANCE * denom, 0.5)
    return 1.0 if abs(v1 - v2) <= tol else 0.0

def text_similarity(t1: str, t2: str, model: SentenceTransformer) -> float:
    """
    Compute semantic similarity between two text strings using Sentence‑BERT.
    Returns cosine similarity clamped to [0.0, 1.0]. If either string is empty, returns 0.0.
    """
    if not t1 or not t2:
        return 0.0
    from sentence_transformers import util
    emb = model.encode([t1, t2], show_progress_bar=False)
    # util.cos_sim handles normalization explicitly, making this model-agnostic.
    sim = float(util.cos_sim(emb[0], emb[1]))
    return max(0.0, sim)

def rule_similarity(gt_dict: Dict, ex_dict: Dict, model: SentenceTransformer) -> float:
    """
    Compute a weighted similarity between a single ground truth rule and a single extracted rule.
    Iterates over all fields, applies the appropriate similarity function, and averages by weight.
    """
    score = 0.0
    total_weight = 0.0
    for field, w in FIELD_WEIGHTS.items():
        if w == 0.0:
            continue  # skip zero‑weight fields entirely
        total_weight += w
        gt_val = gt_dict.get(field)  # ground truth value for this field
        ex_val = ex_dict.get(field)  # extracted value for this field
        if field in NUMERIC_FIELDS:
            sim = numeric_similarity(gt_val, ex_val)
        elif field in CATEGORICAL_FIELDS:
            sim = categorical_similarity(gt_val, ex_val)
        elif field in TEXT_FIELDS:
            sim = text_similarity(gt_val or "", ex_val or "", model)
        else:
            # fallback (should not happen) treat as categorical.
            sim = categorical_similarity(gt_val, ex_val)
        score += w * sim
    # Normalise by the sum of weights (should be 1.0, but just in case).
    return score / total_weight if total_weight > 0 else 0.0

def match_rules_greedy(gt_list: List, ex_list: List, model: SentenceTransformer) -> Tuple[List, List, List]:
    """
    Greedy matching between ground truth (gt_list) and extracted (ex_list) rules.
    Each ground truth rule is paired with the most similar yet‑unmatched extracted rule,
    but only if the similarity ≥ MATCH_THRESHOLD.
    Returns:
      matched: list of (gt_idx, ex_idx, similarity)
      unmatched_gt: list of ground truth indices without a match
      unmatched_ex: list of extracted indices without a match
    """
    matched = []
    available_ex = list(range(len(ex_list)))  # indices of extracted rules still available for matching
    for gt_idx, gt_rule in enumerate(gt_list):
        best_sim = -1.0
        best_ex = None
        # Scan over available extracted rules to find the best match for this ground truth rule.
        for ex_idx in available_ex:
            sim = rule_similarity(gt_rule, ex_list[ex_idx], model)
            if sim > best_sim:
                best_sim = sim
                best_ex = ex_idx
        # If the best similarity is above threshold, record the pair and remove the extracted rule.
        if best_sim >= MATCH_THRESHOLD and best_ex is not None:
            matched.append((gt_idx, best_ex, best_sim))
            available_ex.remove(best_ex)

    # Indices of ground truth rules that never found a match.
    unmatched_gt = [i for i in range(len(gt_list)) if i not in [p[0] for p in matched]]
    # Indices of extracted rules that were never chosen.
    unmatched_ex = [i for i in range(len(ex_list)) if i not in [p[1] for p in matched]]
    return matched, unmatched_gt, unmatched_ex

def evaluate_source(gt_source: List, ex_source: List, model: SentenceTransformer) -> Tuple[Dict, List]:
    """
    Compute metrics for a single source (SOP document).
    Returns a metrics dict (precision, recall, f1, …) and the list of matched pairs.
    """
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

def main():
    # ── Load ground truth ──────────────────────────────────────────────────
    print("Loading ground truth...")
    all_gt = load_ground_truth(GT_CSV_PATH)
    print(f"Total GT rules: {len(all_gt)}")

    # ── Load extracted rules, grouped by source ────────────────────────────
    print("Loading extracted rules...")
    extracted_by_source = load_extracted_rules_by_source(EXTRACTED_DIR)
    # Union of sources present in ground truth and extracted data.
    all_sources = set(rule["source"] for rule in all_gt if rule.get("source")) | set(extracted_by_source.keys())
    print(f"Sources found: {sorted(all_sources)}")

    # ── Load SBERT model (runs once, shared across all comparisons) ────────
    print("Loading SBERT model...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Group ground truth rules by source ──────────────────────────────────
    gt_by_source = defaultdict(list)
    for rule in all_gt:
        src = rule.get("source", "unknown")
        gt_by_source[src].append(rule)

    # ── Aggregate metrics across all sources ───────────────────────────────
    per_source_metrics = {}
    total_tp = 0
    total_gt = 0
    total_ex = 0
    all_matched_pairs = []

    for src in sorted(all_sources):
        gt_list = gt_by_source.get(src, [])
        ex_list = extracted_by_source.get(src, [])
        if not gt_list and not ex_list:
            continue  # skip sources with no rules on either side
        # Evaluate this source, obtaining metrics and matched pairs.
        res, matched = evaluate_source(gt_list, ex_list, sbert)
        per_source_metrics[src] = res
        total_tp += res["tp"]
        total_gt += len(gt_list)
        total_ex += len(ex_list)

        # Save detailed info for each matched pair (including full fields) for manual analysis.
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

    # ── Global metrics ─────────────────────────────────────────────────────
    global_precision = total_tp / total_ex if total_ex else 0.0
    global_recall = total_tp / total_gt if total_gt else 0.0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0

    # ── Console summary ────────────────────────────────────────────────────
    print("\n========== GLOBAL RESULTS (source-aware) ============")
    print(f"Total GT: {total_gt}, Total Extracted: {total_ex}, Matched: {total_tp}")
    print(f"Precision: {global_precision:.3f}")
    print(f"Recall:    {global_recall:.3f}")
    print(f"F1:        {global_f1:.3f}")

    print("\n========== PER-SOURCE RESULTS ==========")
    for src, m in sorted(per_source_metrics.items()):
        print(f"{src}: P={m['precision']:.2f}, R={m['recall']:.2f}, F1={m['f1']:.2f}  (GT={m['num_gt']}, Ex={m['num_ex']}, TP={m['tp']})")

    # ── Write detailed JSON report ─────────────────────────────────────────
    report = {
        "global_metrics": {
            "precision": round(global_precision, 4),
            "recall": round(global_recall, 4),
            "f1": round(global_f1, 4),
            "tp": total_tp,
            "total_gt": total_gt,
            "total_extracted": total_ex,
        },
        # For each source, round floats to 4 decimal places for clean JSON.
        "per_source_metrics": {src: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()} for src, m in per_source_metrics.items()},
        "all_matched_pairs": all_matched_pairs,
    }

    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(OUTPUT_REPORT) or ".", exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()