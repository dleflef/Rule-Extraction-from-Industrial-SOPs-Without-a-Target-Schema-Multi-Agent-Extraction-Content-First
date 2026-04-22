"""
Layer 3 — Field-by-Field Evaluation Against Ground Truth

Each ground_truth.csv row is a structured rule record.
Each extracted rule (from Agent 2C aligned_rule_chunks) is also a structured record
with the same schema.

Matching strategy (two-phase):
  Phase 1 — Anchor match: station AND sensor both match exactly.
  Phase 2 — Semantic fallback: condition word-overlap ≥ 0.5 AND same severity level.

Scoring (per matched pair):
  station    — exact match (weight 2)
  sensor     — exact match (weight 2)
  sensorType — exact match (weight 1)
  condition  — word-overlap similarity (weight 2)
  action     — word-overlap similarity (weight 1)
  severity   — exact match (weight 1)
  unit       — normalised exact match (weight 1)
  critHi     — numeric within ±15% (weight 0.5)
  warnHi     — numeric within ±15% (weight 0.5)
  critLo     — numeric within ±15% (weight 0.5)
  warnLo     — numeric within ±15% (weight 0.5)

An extracted rule counts as a True Positive (TP) if its best match score ≥ 0.5.
Unmatched GT rows → False Negatives (FN).
Unmatched extracted rules → False Positives (FP).
"""

import os
import re
import csv
import json
import glob
from typing import Any, Dict, List, Optional, Tuple

LAYER_2C_DIR = "outputs/layer_2c"
GT_CSV_PATH  = os.path.join("layers", "extracted_seed", "dataset", "kg_seeds", "ground_truth.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_unit(u: Optional[str]) -> str:
    if not u:
        return ""
    return u.strip().lower().replace("°", "").replace(" ", "")


def _word_overlap(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    clean = lambda s: re.sub(r"[^a-z0-9]", " ", s.lower())
    wa = set(clean(a).split())
    wb = set(clean(b).split())
    wa.discard("")
    wb.discard("")
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _numeric_match(ext_val: Any, gt_val: Any, tolerance: float = 0.15) -> bool:
    try:
        e = float(ext_val)
        g = float(gt_val)
        if g == 0:
            return e == 0
        return abs(e - g) / abs(g) <= tolerance
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Ground Truth Loading
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(csv_path):
        print(f"ERROR: Ground truth CSV not found at {csv_path}")
        return rows

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "ruleId":     row.get("ruleId", "").strip() or None,
                "rule_class": row.get("class", "").strip() or None,
                "station":    row.get("station", "").strip() or None,
                "sensor":     row.get("sensor", "").strip() or None,
                "sensorType": row.get("sensorType", "").strip() or None,
                "condition":  row.get("condition", "").strip() or None,
                "action":     row.get("action", "").strip() or None,
                "severity":   row.get("severity", "").strip() or None,
                "critHi":     row.get("critHi", "").strip() or None,
                "warnHi":     row.get("warnHi", "").strip() or None,
                "critLo":     row.get("critLo", "").strip() or None,
                "warnLo":     row.get("warnLo", "").strip() or None,
                "unit":       row.get("unit", "").strip() or None,
            })
    return rows


# ---------------------------------------------------------------------------
# Extracted Rules Loading
# ---------------------------------------------------------------------------

def load_extracted_rules(layer_2c_dir: str) -> List[Dict[str, Any]]:
    """Flattens all aligned_rules from all Agent 2C output files."""
    rules = []
    json_files = glob.glob(os.path.join(layer_2c_dir, "*_agent2c_aligned.json"))

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for chunk in data.get("aligned_rule_chunks", []):
            for rule in chunk.get("aligned_rules", []):
                # Drop internal alignment metadata before comparison
                clean = {k: v for k, v in rule.items() if not k.startswith("_")}
                clean["_source_file"] = data.get("source_file", "")
                clean["_chunk_id"]    = chunk.get("chunk_id", "")
                rules.append(clean)
    return rules


# ---------------------------------------------------------------------------
# Pairwise Scoring
# ---------------------------------------------------------------------------

def _score_pair(ext: Dict, gt: Dict) -> float:
    """
    Returns a composite similarity score in [0, 1].
    Weighted sum of field matches, normalised by total possible weight.
    """
    score = 0.0
    total = 0.0

    # Station (weight 2)
    if gt.get("station") and ext.get("station"):
        total += 2.0
        if ext["station"].strip().upper() == gt["station"].strip().upper():
            score += 2.0

    # Sensor (weight 2)
    if gt.get("sensor") and ext.get("sensor"):
        total += 2.0
        if ext["sensor"].strip().upper() == gt["sensor"].strip().upper():
            score += 2.0

    # SensorType (weight 1)
    if gt.get("sensorType") and ext.get("sensorType"):
        total += 1.0
        if ext["sensorType"].strip().upper() == gt["sensorType"].strip().upper():
            score += 1.0

    # Condition (weight 2) — semantic word overlap
    if gt.get("condition") and ext.get("condition"):
        total += 2.0
        score += 2.0 * _word_overlap(ext["condition"], gt["condition"])

    # Action (weight 1) — semantic word overlap
    if gt.get("action") and ext.get("action"):
        total += 1.0
        score += 1.0 * _word_overlap(ext["action"], gt["action"])

    # Severity (weight 1)
    if gt.get("severity") and ext.get("severity"):
        total += 1.0
        if ext["severity"].strip().upper() == gt["severity"].strip().upper():
            score += 1.0

    # Unit (weight 1)
    if gt.get("unit") and ext.get("unit"):
        total += 1.0
        if _norm_unit(ext["unit"]) == _norm_unit(gt["unit"]):
            score += 1.0

    # Numeric thresholds (weight 0.5 each)
    for field in ("critHi", "warnHi", "critLo", "warnLo"):
        if gt.get(field) is not None and ext.get(field) is not None:
            total += 0.5
            if _numeric_match(ext[field], gt[field]):
                score += 0.5

    if total == 0.0:
        return 0.0
    return score / total


TP_THRESHOLD = 0.5   # minimum score to count as a True Positive


def _best_match(
    ext: Dict,
    gt_pool: List[Dict]
) -> Tuple[Optional[int], float]:
    """
    Returns (index_in_pool, score) of the best GT match, or (None, 0) if below threshold.

    Priority:
      1. Anchor match — station AND sensor both match exactly → score that pair
      2. Semantic fallback — condition overlap ≥ 0.5 AND same severity → score that pair
    """
    best_idx: Optional[int] = None
    best_score = 0.0

    ext_station = (ext.get("station") or "").strip().upper()
    ext_sensor  = (ext.get("sensor")  or "").strip().upper()
    ext_sev     = (ext.get("severity") or "").strip().upper()

    for i, gt in enumerate(gt_pool):
        gt_station = (gt.get("station") or "").strip().upper()
        gt_sensor  = (gt.get("sensor")  or "").strip().upper()
        gt_sev     = (gt.get("severity") or "").strip().upper()

        # Phase 1 — anchor match
        anchor = (
            ext_station and gt_station and ext_station == gt_station and
            ext_sensor  and gt_sensor  and ext_sensor  == gt_sensor
        )

        # Phase 2 — semantic fallback (no anchor)
        cond_overlap = _word_overlap(ext.get("condition"), gt.get("condition"))
        semantic = not anchor and cond_overlap >= 0.5 and ext_sev == gt_sev and ext_sev != ""

        if anchor or semantic:
            s = _score_pair(ext, gt)
            if s > best_score:
                best_score = s
                best_idx = i

    if best_score >= TP_THRESHOLD:
        return best_idx, best_score
    return None, best_score


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline():
    print(f"{'='*60}")
    print("LAYER 3: FIELD-BY-FIELD RULE EVALUATION (BASELINE)")
    print(f"{'='*60}")

    ground_truth = load_ground_truth(GT_CSV_PATH)
    extracted    = load_extracted_rules(LAYER_2C_DIR)

    print(f"Ground truth rules : {len(ground_truth)}")
    print(f"Extracted rules    : {len(extracted)}\n")

    gt_remaining = list(range(len(ground_truth)))   # indices of unmatched GT rows
    true_positives:  List[Dict] = []
    false_positives: List[Dict] = []

    for ext in extracted:
        candidate_pool = [ground_truth[i] for i in gt_remaining]
        match_local_idx, score = _best_match(ext, candidate_pool)

        if match_local_idx is not None:
            global_idx = gt_remaining[match_local_idx]
            true_positives.append({
                "extracted":           ext,
                "matched_ground_truth": ground_truth[global_idx],
                "match_score":          round(score, 3),
            })
            gt_remaining.pop(match_local_idx)
        else:
            false_positives.append(ext)

    false_negatives = [ground_truth[i] for i in gt_remaining]

    TP = len(true_positives)
    FP = len(false_positives)
    FN = len(false_negatives)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- Per rule-class breakdown ---
    class_stats: Dict[str, Dict[str, int]] = {}
    for record in true_positives:
        rc = record["matched_ground_truth"].get("rule_class") or "Unknown"
        class_stats.setdefault(rc, {"TP": 0, "FP": 0, "FN": 0})["TP"] += 1
    for record in false_positives:
        rc = record.get("rule_class") or "Unknown"
        class_stats.setdefault(rc, {"TP": 0, "FP": 0, "FN": 0})["FP"] += 1
    for record in false_negatives:
        rc = record.get("rule_class") or "Unknown"
        class_stats.setdefault(rc, {"TP": 0, "FP": 0, "FN": 0})["FN"] += 1

    # --- Per-field hit rate among TPs ---
    field_hits: Dict[str, int] = {
        "station": 0, "sensor": 0, "sensorType": 0,
        "condition": 0, "action": 0, "severity": 0, "unit": 0,
        "critHi": 0, "warnHi": 0, "critLo": 0, "warnLo": 0
    }
    field_possible: Dict[str, int] = dict.fromkeys(field_hits, 0)

    for record in true_positives:
        ext = record["extracted"]
        gt  = record["matched_ground_truth"]
        if gt.get("station") and ext.get("station"):
            field_possible["station"] += 1
            if ext["station"].strip().upper() == gt["station"].strip().upper():
                field_hits["station"] += 1
        if gt.get("sensor") and ext.get("sensor"):
            field_possible["sensor"] += 1
            if ext["sensor"].strip().upper() == gt["sensor"].strip().upper():
                field_hits["sensor"] += 1
        if gt.get("sensorType") and ext.get("sensorType"):
            field_possible["sensorType"] += 1
            if ext["sensorType"].strip().upper() == gt["sensorType"].strip().upper():
                field_hits["sensorType"] += 1
        if gt.get("condition") and ext.get("condition"):
            field_possible["condition"] += 1
            if _word_overlap(ext["condition"], gt["condition"]) >= 0.6:
                field_hits["condition"] += 1
        if gt.get("action") and ext.get("action"):
            field_possible["action"] += 1
            if _word_overlap(ext["action"], gt["action"]) >= 0.5:
                field_hits["action"] += 1
        if gt.get("severity") and ext.get("severity"):
            field_possible["severity"] += 1
            if ext["severity"].strip().upper() == gt["severity"].strip().upper():
                field_hits["severity"] += 1
        if gt.get("unit") and ext.get("unit"):
            field_possible["unit"] += 1
            if _norm_unit(ext["unit"]) == _norm_unit(gt["unit"]):
                field_hits["unit"] += 1
        for field in ("critHi", "warnHi", "critLo", "warnLo"):
            if gt.get(field) is not None and ext.get(field) is not None:
                field_possible[field] += 1
                if _numeric_match(ext[field], gt[field]):
                    field_hits[field] += 1

    # --- Console output ---
    print("--- OVERALL METRICS ---")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1 Score  : {f1:.3f}")
    print()
    print("--- PER RULE-CLASS BREAKDOWN ---")
    for rc, counts in sorted(class_stats.items()):
        tp = counts['TP']; fp = counts['FP']; fn = counts['FN']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        print(f"  {rc:<22}  TP={tp}  FP={fp}  FN={fn}  P={p:.2f}  R={r:.2f}  F1={f:.2f}")
    print()
    print("--- FIELD ACCURACY (among TPs) ---")
    for field in ("station", "sensor", "sensorType", "condition", "action",
                  "severity", "unit", "critHi", "warnHi", "critLo", "warnLo"):
        possible = field_possible[field]
        hits = field_hits[field]
        acc = hits / possible if possible > 0 else 0.0
        print(f"  {field:<12} {hits:3}/{possible:3}  ({acc*100:.0f}%)")
    print(f"{'='*60}")

    # --- JSON report ---
    report = {
        "summary": {
            "ground_truth_rules": len(ground_truth),
            "extracted_rules":    len(extracted),
            "metrics": {
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1_score":  round(f1, 4),
            },
            "counts": {"TP": TP, "FP": FP, "FN": FN},
            "per_rule_class": {
                rc: {
                    "TP": v["TP"], "FP": v["FP"], "FN": v["FN"],
                    "precision": round(v["TP"]/(v["TP"]+v["FP"]), 3) if (v["TP"]+v["FP"]) > 0 else 0,
                    "recall":    round(v["TP"]/(v["TP"]+v["FN"]), 3) if (v["TP"]+v["FN"]) > 0 else 0,
                }
                for rc, v in class_stats.items()
            },
            "field_accuracy": {
                field: {
                    "hits": field_hits[field],
                    "possible": field_possible[field],
                    "accuracy": round(field_hits[field] / field_possible[field], 3)
                    if field_possible[field] > 0 else 0.0
                }
                for field in field_hits
            },
        },
        "true_positives":  true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    output_path = "outputs/baseline_evaluation_report.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"Detailed report saved to: {output_path}")


if __name__ == "__main__":
    evaluate_pipeline()
