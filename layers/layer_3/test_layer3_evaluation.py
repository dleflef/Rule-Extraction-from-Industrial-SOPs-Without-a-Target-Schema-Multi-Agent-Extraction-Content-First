"""
Layer 3 — Field-by-Field Evaluation Against Ground Truth

Pipeline:
  1. Load Agent 2C output (aligned reified triples).
  2. Group triples by Rule node subject (subjects starting with RULE-).
  3. Reconstruct a structured record per Rule node using the predicate→field mapping.
  4. Compare each reconstructed record against ground_truth.csv field-by-field.
  5. Report Precision / Recall / F1 plus per-rule-class and per-field breakdowns.

Predicate → GT field mapping:
  applies_to_station  → station
  applies_to_sensor   → sensor
  applies_to_zone     → station   (AccessRules: zone plays the station role)
  has_sensor_type     → sensorType
  has_condition       → condition
  triggers_action     → action
  has_severity        → severity
  has_crit_hi         → critHi  (converted to float)
  has_warn_hi         → warnHi
  has_crit_lo         → critLo
  has_warn_lo         → warnLo
  has_unit            → unit
  rdf:type            → rule_class

Matching strategy:
  Phase 1 — Anchor: station AND sensor both match exactly.
  Phase 2 — Semantic fallback: condition word-overlap ≥ 0.5 AND same severity.
  A pair is a True Positive if composite score ≥ 0.5.
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
# Predicate → structured-record field
# ---------------------------------------------------------------------------
PREDICATE_TO_FIELD: Dict[str, str] = {
    "rdf:type":           "rule_class",
    "applies_to_station": "station",
    "applies_to_sensor":  "sensor",
    "applies_to_zone":    "station",   # AccessRule: zone maps to station field
    "has_sensor_type":    "sensorType",
    "has_condition":      "condition",
    "triggers_action":    "action",
    "has_severity":       "severity",
    "has_crit_hi":        "critHi",
    "has_warn_hi":        "warnHi",
    "has_crit_lo":        "critLo",
    "has_warn_lo":        "warnLo",
    "has_unit":           "unit",
}

NUMERIC_FIELDS = {"critHi", "warnHi", "critLo", "warnLo"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_unit(u: Optional[str]) -> str:
    if not u:
        return ""
    return re.sub(r"[^a-z0-9]", "", u.lower())


def _word_overlap(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    clean = lambda s: set(re.sub(r"[^a-z0-9]", " ", s.lower()).split())
    wa, wb = clean(a), clean(b)
    wa.discard(""); wb.discard("")
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _numeric_within(ext: Any, gt: Any, tol: float = 0.15) -> bool:
    try:
        e, g = float(ext), float(gt)
        return abs(e - g) / abs(g) <= tol if g != 0 else e == 0
    except (TypeError, ValueError):
        return False


def _is_rule_subject(subject: str) -> bool:
    return subject.upper().startswith("RULE-")


# ---------------------------------------------------------------------------
# 1. Reconstruct structured records from reified triples
# ---------------------------------------------------------------------------

def reconstruct_rules_from_triples(all_triples: List[Dict]) -> List[Dict]:
    """
    Groups triples by Rule node subject and reconstructs one structured
    record per Rule node.
    """
    groups: Dict[str, Dict] = {}

    for t in all_triples:
        subj = t.get("subject", "")
        pred = t.get("predicate", "")
        obj  = t.get("object", "")

        if not _is_rule_subject(subj):
            continue  # structural (monitors, etc.) — not a rule record

        if subj not in groups:
            groups[subj] = {"ruleId": subj}

        field = PREDICATE_TO_FIELD.get(pred)
        if field is None:
            continue

        if field in NUMERIC_FIELDS:
            try:
                groups[subj][field] = float(obj)
            except (TypeError, ValueError):
                groups[subj][field] = obj
        else:
            groups[subj][field] = obj

    return list(groups.values())


# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(csv_path):
        print(f"ERROR: Ground truth CSV not found at {csv_path}")
        return rows
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "ruleId":     row.get("ruleId", "").strip() or None,
                "rule_class": row.get("class",  "").strip() or None,
                "station":    row.get("station","").strip() or None,
                "sensor":     row.get("sensor", "").strip() or None,
                "sensorType": row.get("sensorType","").strip() or None,
                "condition":  row.get("condition","").strip() or None,
                "action":     row.get("action",  "").strip() or None,
                "severity":   row.get("severity","").strip() or None,
                "critHi":     row.get("critHi",  "").strip() or None,
                "warnHi":     row.get("warnHi",  "").strip() or None,
                "critLo":     row.get("critLo",  "").strip() or None,
                "warnLo":     row.get("warnLo",  "").strip() or None,
                "unit":       row.get("unit",    "").strip() or None,
            })
    return rows


def load_extracted_rules(layer_2c_dir: str) -> List[Dict]:
    """
    Reads all Agent 2C output files, collects aligned triples, and returns
    the list of reconstructed structured rule records.
    """
    all_triples: List[Dict] = []
    json_files = glob.glob(os.path.join(layer_2c_dir, "*_agent2c_aligned.json"))

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for chunk in data.get("results", []):
            for t in chunk.get("aligned_relations", []):
                all_triples.append(t)

    return reconstruct_rules_from_triples(all_triples)


# ---------------------------------------------------------------------------
# 3. Pairwise scoring
# ---------------------------------------------------------------------------

def _score_pair(ext: Dict, gt: Dict) -> float:
    score = total = 0.0

    def add(weight: float, hit: bool):
        nonlocal score, total
        total += weight
        if hit:
            score += weight

    if gt.get("station") and ext.get("station"):
        add(2.0, ext["station"].strip().upper() == gt["station"].strip().upper())
    if gt.get("sensor") and ext.get("sensor"):
        add(2.0, ext["sensor"].strip().upper() == gt["sensor"].strip().upper())
    if gt.get("sensorType") and ext.get("sensorType"):
        add(1.0, ext["sensorType"].strip().upper() == gt["sensorType"].strip().upper())
    if gt.get("condition") and ext.get("condition"):
        add(2.0, False)
        score += 2.0 * _word_overlap(ext["condition"], gt["condition"])
    if gt.get("action") and ext.get("action"):
        add(1.0, False)
        score += 1.0 * _word_overlap(ext["action"], gt["action"])
    if gt.get("severity") and ext.get("severity"):
        add(1.0, ext["severity"].strip().upper() == gt["severity"].strip().upper())
    if gt.get("unit") and ext.get("unit"):
        add(1.0, _norm_unit(ext["unit"]) == _norm_unit(gt["unit"]))
    for field in ("critHi", "warnHi", "critLo", "warnLo"):
        if gt.get(field) is not None and ext.get(field) is not None:
            add(0.5, _numeric_within(ext[field], gt[field]))

    return (score / total) if total > 0 else 0.0


TP_THRESHOLD = 0.5


def _best_match(ext: Dict, gt_pool: List[Dict]) -> Tuple[Optional[int], float]:
    ext_station = (ext.get("station") or "").strip().upper()
    ext_sensor  = (ext.get("sensor")  or "").strip().upper()
    ext_sev     = (ext.get("severity") or "").strip().upper()

    best_idx: Optional[int] = None
    best_score = 0.0

    for i, gt in enumerate(gt_pool):
        gt_station = (gt.get("station") or "").strip().upper()
        gt_sensor  = (gt.get("sensor")  or "").strip().upper()
        gt_sev     = (gt.get("severity") or "").strip().upper()

        # Phase 1 — anchor (station + sensor both exact)
        anchor = (
            ext_station and gt_station and ext_station == gt_station and
            ext_sensor  and gt_sensor  and ext_sensor  == gt_sensor
        )
        # Phase 2 — semantic fallback
        cond_sim  = _word_overlap(ext.get("condition"), gt.get("condition"))
        semantic  = not anchor and cond_sim >= 0.5 and ext_sev == gt_sev and ext_sev != ""

        if anchor or semantic:
            s = _score_pair(ext, gt)
            if s > best_score:
                best_score = s
                best_idx   = i

    return (best_idx, best_score) if best_score >= TP_THRESHOLD else (None, best_score)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def evaluate_pipeline():
    print(f"{'='*60}")
    print("LAYER 3: FIELD-BY-FIELD RULE EVALUATION (BASELINE)")
    print(f"{'='*60}")

    ground_truth = load_ground_truth(GT_CSV_PATH)
    extracted    = load_extracted_rules(LAYER_2C_DIR)

    print(f"Ground truth rules  : {len(ground_truth)}")
    print(f"Reconstructed rules : {len(extracted)}\n")

    gt_remaining = list(range(len(ground_truth)))
    true_positives:  List[Dict] = []
    false_positives: List[Dict] = []

    for ext in extracted:
        pool = [ground_truth[i] for i in gt_remaining]
        local_idx, score = _best_match(ext, pool)

        if local_idx is not None:
            global_idx = gt_remaining[local_idx]
            true_positives.append({
                "extracted":            ext,
                "matched_ground_truth": ground_truth[global_idx],
                "match_score":          round(score, 3),
            })
            gt_remaining.pop(local_idx)
        else:
            false_positives.append(ext)

    false_negatives = [ground_truth[i] for i in gt_remaining]

    TP = len(true_positives)
    FP = len(false_positives)
    FN = len(false_negatives)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per rule-class breakdown
    class_stats: Dict[str, Dict[str, int]] = {}
    for r in true_positives:
        rc = r["matched_ground_truth"].get("rule_class") or "Unknown"
        class_stats.setdefault(rc, {"TP": 0, "FP": 0, "FN": 0})["TP"] += 1
    for r in false_positives:
        rc = r.get("rule_class") or "Unknown"
        class_stats.setdefault(rc, {"TP": 0, "FP": 0, "FN": 0})["FP"] += 1
    for r in false_negatives:
        rc = r.get("rule_class") or "Unknown"
        class_stats.setdefault(rc, {"TP": 0, "FP": 0, "FN": 0})["FN"] += 1

    # Per-field hit rate (among TPs)
    FIELDS = ("station", "sensor", "sensorType", "condition",
              "action", "severity", "unit", "critHi", "warnHi", "critLo", "warnLo")
    hits     = dict.fromkeys(FIELDS, 0)
    possible = dict.fromkeys(FIELDS, 0)

    for record in true_positives:
        ext = record["extracted"]
        gt  = record["matched_ground_truth"]
        for f in ("station", "sensor", "sensorType", "severity"):
            if gt.get(f) and ext.get(f):
                possible[f] += 1
                if ext[f].strip().upper() == gt[f].strip().upper():
                    hits[f] += 1
        for f in ("condition", "action"):
            if gt.get(f) and ext.get(f):
                possible[f] += 1
                if _word_overlap(ext[f], gt[f]) >= 0.5:
                    hits[f] += 1
        if gt.get("unit") and ext.get("unit"):
            possible["unit"] += 1
            if _norm_unit(ext["unit"]) == _norm_unit(gt["unit"]):
                hits["unit"] += 1
        for f in ("critHi", "warnHi", "critLo", "warnLo"):
            if gt.get(f) is not None and ext.get(f) is not None:
                possible[f] += 1
                if _numeric_within(ext[f], gt[f]):
                    hits[f] += 1

    # --- Console ---
    print("--- OVERALL METRICS ---")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1 Score  : {f1:.3f}")
    print()
    print("--- PER RULE-CLASS BREAKDOWN ---")
    for rc, c in sorted(class_stats.items()):
        p = c['TP'] / (c['TP'] + c['FP']) if (c['TP'] + c['FP']) > 0 else 0.0
        r = c['TP'] / (c['TP'] + c['FN']) if (c['TP'] + c['FN']) > 0 else 0.0
        fv = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        print(f"  {rc:<22}  TP={c['TP']}  FP={c['FP']}  FN={c['FN']}  "
              f"P={p:.2f}  R={r:.2f}  F1={fv:.2f}")
    print()
    print("--- FIELD ACCURACY (among TPs) ---")
    for f in FIELDS:
        p_ = possible[f]
        acc = hits[f] / p_ if p_ > 0 else 0.0
        print(f"  {f:<12}  {hits[f]:3}/{p_:3}  ({acc*100:.0f}%)")
    print(f"{'='*60}")

    # --- JSON report ---
    report = {
        "summary": {
            "ground_truth_rules":  len(ground_truth),
            "reconstructed_rules": len(extracted),
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
                f: {
                    "hits": hits[f], "possible": possible[f],
                    "accuracy": round(hits[f] / possible[f], 3) if possible[f] > 0 else 0.0
                }
                for f in FIELDS
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
