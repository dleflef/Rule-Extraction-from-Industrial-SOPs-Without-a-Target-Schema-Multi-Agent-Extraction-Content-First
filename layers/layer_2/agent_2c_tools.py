"""
Agent 2C Tools — Ontology Alignment and KG Triple Conversion

Two responsibilities:
1. Align station/sensor names in each extracted rule to official node IDs from the
   KG seed (4-pass fuzzy matching: exact → normalised exact → substring → Jaccard).
2. Convert each aligned structured rule into KG triples that carry ALL rule fields
   as edge properties, so no information is lost before Neo4j ingestion.
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip, collapse whitespace, remove special chars."""
    return re.sub(r"[^a-z0-9_]", "", text.lower().replace(" ", "_"))


def _jaccard(a: str, b: str) -> float:
    tokens_a = set(re.split(r"[^a-z0-9]+", a.lower()))
    tokens_b = set(re.split(r"[^a-z0-9]+", b.lower()))
    tokens_a.discard("")
    tokens_b.discard("")
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _fuzzy_align(raw: Optional[str], official_nodes: Dict[str, str]) -> Tuple[Optional[str], str]:
    """
    Returns (aligned_name, status) where status is 'exact', 'fuzzy', or 'unaligned'.
    official_nodes: {name: label}
    """
    if not raw:
        return None, "unaligned"

    # Pass 1 — exact
    if raw in official_nodes:
        return raw, "exact"

    # Pass 2 — normalised exact
    raw_norm = _normalise(raw)
    for name in official_nodes:
        if _normalise(name) == raw_norm:
            return name, "exact"

    # Pass 3 — substring (official name contained in raw or vice-versa)
    raw_up = raw.upper()
    for name in official_nodes:
        name_up = name.upper()
        if name_up in raw_up or raw_up in name_up:
            return name, "fuzzy"

    # Pass 4 — Jaccard ≥ 0.4
    best_score = 0.0
    best_name: Optional[str] = None
    for name in official_nodes:
        score = _jaccard(raw, name)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score >= 0.4 and best_name is not None:
        return best_name, "fuzzy"

    return raw, "unaligned"


# ---------------------------------------------------------------------------
# 2. Rule Alignment
# ---------------------------------------------------------------------------

def align_rule(rule: Dict[str, Any], official_nodes: Dict[str, str]) -> Dict[str, Any]:
    """
    Aligns the station and sensor fields of a structured rule record to official
    node IDs, then annotates with alignment metadata.
    """
    aligned = dict(rule)  # shallow copy — all scalar fields are immutable

    station_aligned, station_status = _fuzzy_align(rule.get("station"), official_nodes)
    sensor_aligned, sensor_status = _fuzzy_align(rule.get("sensor"), official_nodes)

    aligned["station"] = station_aligned
    aligned["sensor"] = sensor_aligned
    aligned["_station_alignment"] = station_status
    aligned["_sensor_alignment"] = sensor_status
    aligned["_overall_alignment"] = (
        "exact" if station_status == "exact" and sensor_status in ("exact", "unaligned")
        else "fuzzy" if station_status in ("exact", "fuzzy") or sensor_status in ("exact", "fuzzy")
        else "unaligned"
    )
    return aligned


# ---------------------------------------------------------------------------
# 3. Conversion from aligned rule → KG triples
# ---------------------------------------------------------------------------

def rule_to_kg_triples(rule: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts one aligned structured rule into KG triples.

    Each triple has: subject, predicate, object, properties (dict).

    Schema:
      OperationalRule / MaintenanceRule:
        (station, monitors, sensor)                           — structural
        (sensor, triggers, action)  props: condition, severity, unit, thresholds

      ThresholdRule:
        (station, monitors, sensor)                           — structural
        (sensor, has_threshold, <sensor>_threshold)  props: critHi, warnHi, critLo, warnLo, unit

      AccessRule:
        (zone, has_access_rule, action)  props: condition, severity
    """
    triples: List[Dict[str, Any]] = []
    station  = rule.get("station")
    sensor   = rule.get("sensor")
    action   = rule.get("action")
    condition = rule.get("condition")
    severity  = rule.get("severity")
    rule_class = rule.get("rule_class", "OperationalRule")

    threshold_props = {k: rule.get(k) for k in ("critHi", "warnHi", "critLo", "warnLo", "unit")}
    # Remove None values from props
    threshold_props = {k: v for k, v in threshold_props.items() if v is not None}

    if rule_class in ("OperationalRule", "MaintenanceRule"):
        if station and sensor:
            triples.append({
                "subject": station, "predicate": "monitors", "object": sensor,
                "properties": {}
            })
        if sensor and action:
            props = {"condition": condition, "severity": severity}
            props.update(threshold_props)
            props = {k: v for k, v in props.items() if v is not None}
            triples.append({
                "subject": sensor, "predicate": "triggers", "object": action,
                "properties": props
            })

    elif rule_class == "ThresholdRule":
        if station and sensor:
            triples.append({
                "subject": station, "predicate": "monitors", "object": sensor,
                "properties": {}
            })
        if sensor:
            props = {"condition": condition, "severity": severity}
            props.update(threshold_props)
            props = {k: v for k, v in props.items() if v is not None}
            triples.append({
                "subject": sensor, "predicate": "has_threshold",
                "object": f"{sensor}_threshold",
                "properties": props
            })

    elif rule_class == "AccessRule":
        zone = station  # for access rules the 'station' field holds the zone
        if zone and action:
            props = {"condition": condition, "severity": severity}
            props = {k: v for k, v in props.items() if v is not None}
            triples.append({
                "subject": zone, "predicate": "has_access_rule", "object": action,
                "properties": props
            })

    return triples


# ---------------------------------------------------------------------------
# 4. Batch processing entry point (called from agent_2c.py)
# ---------------------------------------------------------------------------

def align_and_convert(
    chunks_with_rules: List[Dict[str, Any]],
    official_nodes: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Processes all chunks from Agent 2B.

    Returns:
        aligned_rule_chunks  — same structure as input but with aligned + annotated rules
        kg_triples           — flat deduplicated list of KG triples
    """
    aligned_rule_chunks: List[Dict[str, Any]] = []
    all_triples: List[Dict[str, Any]] = []
    seen_structural: set = set()  # deduplicate (station, monitors, sensor)

    for chunk in chunks_with_rules:
        raw_rules = chunk.get("extracted_rules", [])
        aligned_rules = []
        for rule in raw_rules:
            ar = align_rule(rule, official_nodes)
            aligned_rules.append(ar)

            for triple in rule_to_kg_triples(ar):
                # Deduplicate structural monitors triples
                if triple["predicate"] == "monitors":
                    key = (triple["subject"], triple["object"])
                    if key in seen_structural:
                        continue
                    seen_structural.add(key)
                all_triples.append(triple)

        aligned_rule_chunks.append({
            "chunk_id": chunk.get("chunk_id"),
            "metadata": chunk.get("metadata", {}),
            "aligned_rules": aligned_rules
        })

    return aligned_rule_chunks, all_triples
