"""
Layer 3 — Field-by-Field Evaluation Against Ground Truth
"""

import os
import re
import csv
import json
import glob
from typing import Any, Dict, List, Optional, Tuple

# SBERT for semantic condition/action matching (professor recommendation).
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    _SBERT_AVAILABLE = True
    print("[Evaluation] SBERT (all-MiniLM-L6-v2) loaded for semantic matching.")
except Exception as _sbert_err:
    _sbert_model = None
    _SBERT_AVAILABLE = False
    print(f"[Evaluation] SBERT unavailable ({_sbert_err}). Falling back to word-overlap.")

LAYER_2C_DIR  = "outputs/layer_2c"
GT_CSV_PATH   = os.path.join("layers", "extracted_seed", "dataset", "kg_seeds", "ground_truth.csv")
NODES_CSV_PATH = os.path.join("layers", "extracted_seed", "dataset", "kg_seeds", "nodes_factory.csv")

# Official sensor names loaded once at module level — used to guard auto-inference
_OFFICIAL_SENSORS: set = set()
if os.path.exists(NODES_CSV_PATH):
    with open(NODES_CSV_PATH, "r", encoding="utf-8") as _nf:
        for _nr in csv.DictReader(_nf):
            if _nr.get("label") == "Sensor":
                _OFFICIAL_SENSORS.add(_nr.get("name", "").strip())

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

# Hallucinated placeholder phrases the LLM emits when a field is absent.
# These should be treated as missing (None) rather than real extracted values.
_HALLUCINATED_TEXT = {
    "no specific action mentioned", "not specified in text", "not specified",
    "no action specified", "action not specified", "no action", "none",
    "no specific action", "action not mentioned",
    "no specific condition mentioned", "condition not specified",
    "no condition specified", "no specific condition", "condition not mentioned",
    "not mentioned",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_unit(u: Optional[str]) -> str:
    if not u: return ""
    return re.sub(r"[^a-z0-9]", "", u.lower())

def _word_overlap(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b: return 0.0
    clean = lambda s: set(re.sub(r"[^a-z0-9]", " ", s.lower()).split())
    wa, wb = clean(a), clean(b)
    wa.discard(""); wb.discard("")
    if not wa or not wb: return 0.0
    return len(wa & wb) / min(len(wa), len(wb))

def _text_sim(a: Optional[str], b: Optional[str]) -> float:
    """Semantic similarity via SBERT cosine; falls back to word-overlap."""
    if not a or not b: return 0.0
    if _SBERT_AVAILABLE and _sbert_model is not None:
        vecs = _sbert_model.encode([a, b], convert_to_numpy=True)
        denom = (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])) + 1e-8
        return float(max(0.0, np.dot(vecs[0], vecs[1]) / denom))
    return _word_overlap(a, b)

def _numeric_within(ext: Any, gt: Any, tol: float = 0.15) -> bool:
    try:
        e, g = float(ext), float(gt)
        return abs(e - g) / abs(g) <= tol if g != 0 else e == 0
    except (TypeError, ValueError):
        return False

def _is_rule_subject(subject: str) -> bool:
    s = subject.upper()
    return s.startswith("RULE-") or s.startswith("MAINT-") or "_RULE-" in s or "_MAINT-" in s

# ---------------------------------------------------------------------------
# 1. Reconstruct structured records from reified triples
# ---------------------------------------------------------------------------

def reconstruct_rules_from_triples(all_triples: List[Dict]) -> List[Dict]:
    groups:      Dict[str, Dict] = {}
    group_preds: Dict[str, set]  = {} 

    for t in all_triples:
        subj = t.get("subject", "")
        pred = t.get("predicate", "")
        obj  = t.get("object", "")

        if not _is_rule_subject(subj):
            continue 

        if subj not in groups:
            groups[subj]      = {"ruleId": subj}
            group_preds[subj] = set()

        group_preds[subj].add(pred)

        field = PREDICATE_TO_FIELD.get(pred)
        if field is None:
            continue

        # Drop hallucinated placeholder values before storing
        if isinstance(obj, str) and obj.lower().strip() in _HALLUCINATED_TEXT:
            continue

        if field in NUMERIC_FIELDS:
            try:
                groups[subj][field] = float(obj)
            except (TypeError, ValueError):
                groups[subj][field] = obj
        else:
            groups[subj][field] = obj

    _MAINTENANCE_KEYWORDS = {"maintenance", "schedule", "replace", "lubrication",
                             "calibration", "sensor failure", "preventive"}
    _THRESHOLD_PREDS = {"has_crit_hi", "has_warn_hi", "has_crit_lo", "has_warn_lo"}
    for rule_id, record in groups.items():
        if record.get("rule_class"):
            continue
        preds = group_preds.get(rule_id, set())
        # Fix 2: require >= 2 threshold bounds to avoid misclassifying single-bound rules
        if len(_THRESHOLD_PREDS & preds) >= 2:
            record["rule_class"] = "ThresholdRule"
        elif "applies_to_zone" in preds:
            record["rule_class"] = "AccessRule"
        else:
            action = (record.get("action") or "").lower()
            if any(kw in action for kw in _MAINTENANCE_KEYWORDS):
                record["rule_class"] = "MaintenanceRule"
            else:
                record["rule_class"] = "OperationalRule"

    # Reconstruct missing sensor IDs from station + sensorType when LLM omits them.
    # Guard: only accept the inference when the resulting name is a known official sensor.
    # This blocks garbage like "STATION_01_CNT" (unaligned station) from polluting results.
    _station_code_re = re.compile(r'^[A-Z0-9]+_[A-Z0-9]')
    for rule_id, record in groups.items():
        if not record.get("sensor") and record.get("station") and record.get("sensorType"):
            station = record["station"]
            if _station_code_re.match(station) and " " not in station:
                inferred_sensor = f"{station}_{record['sensorType']}"
                if _OFFICIAL_SENSORS and inferred_sensor not in _OFFICIAL_SENSORS:
                    print(f"[Layer 3] Skipped auto-inference: '{inferred_sensor}' not an official sensor for {rule_id}")
                    continue
                record["sensor"] = inferred_sensor
                print(f"[Layer 3] Auto-inferred missing sensor: {inferred_sensor} for {rule_id}")

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
    all_triples: List[Dict] = []
    json_files = sorted(glob.glob(os.path.join(layer_2c_dir, "*_agent2c_aligned.json")))

    for file_path in json_files:
        basename = os.path.basename(file_path)
        m = re.match(r"(SOP_\d+)", basename)
        file_pfx = m.group(1).replace("_", "") if m else re.sub(r"[^A-Z0-9]", "", basename.upper())[:6]

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for chunk in data.get("results", []):
            chunk_id = chunk.get("chunk_id", "X")
            for t in chunk.get("aligned_relations", []):
                t_copy = dict(t)
                subj = t_copy.get("subject", "")
                if _is_rule_subject(subj):
                    t_copy["subject"] = f"{file_pfx}_C{chunk_id}_{subj}"
                all_triples.append(t_copy)

    return reconstruct_rules_from_triples(all_triples)

# ---------------------------------------------------------------------------
# 3. Pairwise scoring (CORRECTED WEIGHTS PER PROFESSOR NOTES)
# ---------------------------------------------------------------------------

def _score_pair(ext: Dict, gt: Dict) -> float:
    score = total = 0.0

    def add(weight: float, hit: bool):
        nonlocal score, total
        total += weight
        if hit:
            score += weight

    # 1. High Weight: Condition and Action
    if gt.get("condition") and ext.get("condition"):
        add(3.0, False)
        score += 3.0 * _text_sim(ext["condition"], gt["condition"])
    if gt.get("action") and ext.get("action"):
        add(3.0, False)
        score += 3.0 * _text_sim(ext["action"], gt["action"])

    # 2. Medium Weight: Station and Sensor
    if gt.get("station") and ext.get("station"):
        add(2.0, ext["station"].strip().upper() == gt["station"].strip().upper())
    if gt.get("sensor") and ext.get("sensor"):
        add(2.0, ext["sensor"].strip().upper() == gt["sensor"].strip().upper())

    # 3. Lower Weight: Class, Severity, Unit, SensorType
    if gt.get("rule_class") and ext.get("rule_class"):
        add(1.0, ext["rule_class"].strip().upper() == gt["rule_class"].strip().upper())
    if gt.get("severity") and ext.get("severity"):
        add(1.0, ext["severity"].strip().upper() == gt["severity"].strip().upper())
    if gt.get("sensorType") and ext.get("sensorType"):
        add(1.0, ext["sensorType"].strip().upper() == gt["sensorType"].strip().upper())
    if gt.get("unit") and ext.get("unit"):
        add(1.0, _norm_unit(ext["unit"]) == _norm_unit(gt["unit"]))

    # 4. Numerics
    for field in ("critHi", "warnHi", "critLo", "warnLo"):
        if gt.get(field) is not None and ext.get(field) is not None:
            add(1.0, _numeric_within(ext[field], gt[field]))

    return (score / total) if total > 0 else 0.0

TP_THRESHOLD = 0.5

def _best_match(ext: Dict, gt_pool: List[Dict]) -> Tuple[Optional[int], float]:
    ext_station = (ext.get("station") or "").strip().upper()
    ext_sensor  = (ext.get("sensor")  or "").strip().upper()

    best_idx: Optional[int] = None
    best_score = 0.0

    for i, gt in enumerate(gt_pool):
        gt_station = (gt.get("station") or "").strip().upper()
        gt_sensor  = (gt.get("sensor")  or "").strip().upper()

        full_anchor = (
            ext_station and gt_station and ext_station == gt_station and
            ext_sensor  and gt_sensor  and ext_sensor  == gt_sensor
        )

        station_anchor = (
            ext_station and gt_station and ext_station == gt_station and
            not ext_sensor and not gt_sensor
        )

        anchor = full_anchor or station_anchor

        cond_sim = _text_sim(ext.get("condition"), gt.get("condition"))
        semantic = not anchor and cond_sim >= 0.5

        action_sim = _text_sim(ext.get("action"), gt.get("action"))
        action_fallback = not anchor and not semantic and action_sim >= 0.55

        if anchor or semantic or action_fallback:
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

    FIELDS = ("rule_class", "station", "sensor", "sensorType", "condition",
              "action", "severity", "unit", "critHi", "warnHi", "critLo", "warnLo")
    hits     = dict.fromkeys(FIELDS, 0)
    possible = dict.fromkeys(FIELDS, 0)

    for record in true_positives:
        ext = record["extracted"]
        gt  = record["matched_ground_truth"]
        for f in ("rule_class", "station", "sensor", "sensorType", "severity"):
            if gt.get(f) and ext.get(f):
                possible[f] += 1
                if ext[f].strip().upper() == gt[f].strip().upper():
                    hits[f] += 1
        for f in ("condition", "action"):
            if gt.get(f) and ext.get(f):
                possible[f] += 1
                if _text_sim(ext[f], gt[f]) >= 0.5:
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
        print(f"  {rc:<22}  TP={c['TP']:<2} FP={c['FP']:<2} FN={c['FN']:<2} "
              f"P={p:.2f}  R={r:.2f}  F1={fv:.2f}")
    print()
    print("--- FIELD ACCURACY (among TPs) ---")
    for f in FIELDS:
        p_ = possible[f]
        acc = hits[f] / p_ if p_ > 0 else 0.0
        print(f"  {f:<12}  {hits[f]:3}/{p_:3}  ({acc*100:.0f}%)")
    print(f"{'='*60}")

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