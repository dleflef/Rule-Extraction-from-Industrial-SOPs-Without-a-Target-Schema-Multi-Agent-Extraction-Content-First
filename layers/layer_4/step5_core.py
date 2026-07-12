"""
step5_core.py

Shared engine for Phase 2 anomaly detection (iMAKS, guide §3.1/§3.3).
Everything that step5_detect.py, step5_sensitivity.py, step5_baselines.py
and step5_holdout.py have in common lives here, so there is
exactly one definition of the detectors, the scoring rule, and the metrics.

Detectors (all driven ONLY by extracted rules — never by GT):
  threshold  : value crosses an extracted critHi/warnHi/warnLo/critLo bound
  stuck      : value frozen for N samples          ("stuck >N samples")
  drift      : rolling delta over a time window    ("drift >X over Y min/h")
  sustained  : bound violated for N minutes        (">X for >Y min/h")

Out of scope — CORRELATED events (GT-0009): the guide (§2.3.2) defines the
CORRELATED type as requiring three-way multi-source fusion (timeseries
co-occurrence + SOP-001 causal rule + SOP-003 corroboration). No fusion
detector is implemented; CORRELATED events are scored by the same uniform
rule as everything else and are expected to show as GAP. The guide's
separate GT-0009 binary is therefore NOT ATTEMPTED at this stage.

Anti-leakage invariants (hold for every entry point in this module):
  - Detection reads ONLY timestamp / sensor_id / value from the timeseries.
  - GT (nodes.csv / edges.csv / annotated timeseries) is read only by the
    scoring helpers, which callers invoke strictly AFTER detection.
  - Rules come from LLM extraction (Neo4j or the step4_results CSV dump);
    the one deliberate exception is the oracle baseline in
    step5_baselines.py, which is documented as the leak-everything bound.
"""

from __future__ import annotations

import csv
import os
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))

TIMESERIES_CSV = os.path.join(_PROJECT_ROOT, "data", "dataset", "sensors", "timeseries_raw.csv")
NODES_CSV      = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "nodes.csv")
EDGES_CSV      = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "edges.csv")
RESULTS_DIR    = os.path.join(_SCRIPT_DIR, "detection_results")
STEP4_RESULTS  = os.path.join(_SCRIPT_DIR, "step4_results")

NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# ── Tunable constants ─────────────────────────────────────────────────────────
# Every constant below is exposed as a parameter of stream_and_detect() /
# merge_alarms() and exercised by step5_sensitivity.py. Shipped defaults were
# fixed before the sensitivity analysis and are NOT retuned against GT.

# WARNING on-delay: only alarm after 5 continuous minutes of violation
# (ISA-18.2 alarm rationalisation). CRITICAL fires immediately.
ON_DELAY_WARNING_MIN = 5

# Merge alarms on the same sensor within 15 min into one event. The
# sensitivity table shows 30 min would remove both residual FPs — kept at
# 15 to avoid post-hoc tuning against GT (see EVALUATION_LIMITATIONS.md).
GAP_MERGE_MIN = 15

# Drift detector: recent window sized PER SENSOR from that sensor's own
# extracted rule duration — a global cap smaller than a rule's "over N min"
# makes that rule structurally unable to fire (bit MAINT-06 once).
DEFAULT_DRIFT_WINDOW_MIN = 60
# Baseline duration = ratio × recent-window duration (2 h vs 1 h shipped).
BASELINE_WINDOW_RATIO = 2
# Minimum baseline samples before drift_delta() reports a nonzero value.
DRIFT_MIN_REF_SAMPLES = 10
# Dataset sampling cadence (guide §1.2: 30 s).
SAMPLE_INTERVAL_SEC = 30

# Quarantine threshold rules violating >50% of their sensor's readings —
# almost certainly a mis-extracted bound. Unsupervised (no GT labels), but
# data-dependent; its effect is quantified in sensitivity_analysis.csv.
PLAUSIBILITY_MAX_VIOLATION_RATE = 0.5

# Calibration window for the deployable (non-transductive) variant of the
# plausibility filter: violation rates computed over only the FIRST N hours
# of the stream — a commissioning period, as a real deployment would use
# before arming alarms. PRE-REGISTERED at 8 h = 10% of the 80 h stream (an
# a-priori round fraction, not tuned against results). step5_sensitivity.py
# verifies whether this window quarantines the same rules as the shipped
# full-stream filter.
CALIBRATION_HOURS = 8

STUCK_TOL = 1e-6            # values within this range count as identical
DEFAULT_STUCK_BUFFER = 64   # fallback only — sensors with a stuck rule get
                            # a buffer sized from the rule's own sample count

# Phase 2 pass bar (guide Table 2: coverage fraction ≥ 70%).
PHASE2_COVERAGE_THRESHOLD = 0.70


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Rule:
    rule_id:   str
    cls:       str
    sensor:    str
    crit_hi:   Optional[float]
    warn_hi:   Optional[float]
    warn_lo:   Optional[float]
    crit_lo:   Optional[float]
    condition: str
    action:    str
    source:    str
    severity:  str = ""
    station:   str = ""


@dataclass
class GtAnomaly:
    gt_id:   str
    sensor:  str
    atype:   str
    start:   datetime
    end:     datetime


@dataclass
class Alarm:
    sensor:   str
    ts:       datetime
    value:    float
    severity: str    # WARNING | CRITICAL
    detector: str    # threshold | stuck | drift | sustained
    rule_id:  str


@dataclass
class Event:
    sensor:    str
    start:     datetime
    end:       datetime
    severity:  str
    detectors: list[str]
    rule_ids:  list[str]
    n_alarms:  int


# ── Rule loading ──────────────────────────────────────────────────────────────

def _node_float(node, key: str):
    v = node.get(key)
    return float(v) if v is not None else None


def _to_float(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def load_rules_from_neo4j() -> list[Rule]:
    """ACTIVE rules only — those step4b linked to a real ABox sensor via
    GOVERNS_ABOX. Returns Rule nodes only (RETURN DISTINCT r); the GT
    thresholds on ABoxNode:Sensor nodes are never read here."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=NEO4J_DATABASE) as session:
        records = list(session.run(
            "MATCH (r:Rule)-[:GOVERNS_ABOX]->() "
            "RETURN DISTINCT r ORDER BY r.ruleId"))
    driver.close()

    rules = []
    for rec in records:
        n = rec["r"]
        cls = (n.get("class") or "").strip()
        if cls == "AccessRule":
            continue
        rules.append(Rule(
            rule_id=n.get("ruleId") or "",
            cls=cls,
            sensor=(n.get("sensor") or "").strip(),
            crit_hi=_node_float(n, "critHi"), warn_hi=_node_float(n, "warnHi"),
            warn_lo=_node_float(n, "warnLo"), crit_lo=_node_float(n, "critLo"),
            condition=n.get("condition") or "", action=n.get("action") or "",
            source=n.get("sourceFile") or "",
            severity=(n.get("severity") or "").strip(),
            station=(n.get("station") or "").strip(),
        ))
    return rules


def load_rules_from_csv() -> list[Rule]:
    """Same ACTIVE population, rebuilt from the step4_results CSV dumps —
    lets diagnostic scripts run without Neo4j. Equivalence to the Neo4j
    loader is asserted by step5_baselines.py's consistency check."""
    with open(os.path.join(STEP4_RESULTS, "rule_validation.csv"),
              newline="", encoding="utf-8") as f:
        active_ids = {r["ruleId"] for r in csv.DictReader(f)
                      if r["status"] == "ACTIVE"}
    rules = []
    with open(os.path.join(STEP4_RESULTS, "rules_all.csv"),
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ruleId"] not in active_ids:
                continue
            if (r.get("class") or "").strip() == "AccessRule":
                continue
            rules.append(Rule(
                rule_id=r["ruleId"], cls=(r.get("class") or "").strip(),
                sensor=(r.get("sensor") or "").strip(),
                crit_hi=_to_float(r.get("critHi")), warn_hi=_to_float(r.get("warnHi")),
                warn_lo=_to_float(r.get("warnLo")), crit_lo=_to_float(r.get("critLo")),
                condition=r.get("condition") or "", action=r.get("action") or "",
                source=r.get("source_file") or "",
                severity=(r.get("severity") or "").strip(),
                station=(r.get("station") or "").strip(),
            ))
    return rules


# ── Condition parsers ─────────────────────────────────────────────────────────

def _bound_severity(val: float, r: Rule) -> Optional[str]:
    # Crit bounds are checked before warn bounds, so a rule extracted with
    # critHi == warnHi (a real LLM defect, e.g. RULE-CAF01-01) classifies a
    # warn-band violation as CRITICAL and bypasses the WARNING on-delay.
    # Recall/precision are unaffected; per-event latency can be understated
    # by up to ON_DELAY_WARNING_MIN — see EVALUATION_LIMITATIONS.md.
    if r.crit_hi is not None and val > r.crit_hi:
        return "CRITICAL"
    if r.crit_lo is not None and val < r.crit_lo:
        return "CRITICAL"
    if r.warn_hi is not None and val > r.warn_hi:
        return "WARNING"
    if r.warn_lo is not None and val < r.warn_lo:
        return "WARNING"
    return None


def parse_maint_params(rules: list[Rule]) -> list[dict]:
    """MaintenanceRule condition strings → stuck/drift/sustained detector
    parameters. Text patterns only — no GT thresholds involved."""
    params = []
    for r in rules:
        if r.cls != "MaintenanceRule" or not r.sensor:
            continue
        cond = r.condition
        severity = r.severity or "WARNING"

        m = re.search(r"stuck\s*[>≥]\s*(\d+)\s*samples", cond, re.I)
        if m:
            params.append({"type": "stuck", "rule_id": r.rule_id,
                           "sensor": r.sensor, "n": int(m.group(1)) + 1,
                           "severity": severity})
            continue

        m = re.search(
            r"drift\s*[>≥]\s*([\d.]+)\s*\S*\s*(?:over|for)\s*[>≥]?\s*(\d+)\+?\s*(min|h)",
            cond, re.I)
        if m:
            delta = float(m.group(1))
            dur   = int(m.group(2)) * (60 if m.group(3).lower() == "h" else 1)
            params.append({"type": "drift", "rule_id": r.rule_id,
                           "sensor": r.sensor, "delta": delta, "dur_min": dur,
                           "severity": severity})
            continue

        m = re.search(
            r"([><%≥≤]+)\s*([\d.]+)\s*\S*\s+for\s+[>≥]?\s*(\d+)\s*(min|h)",
            cond, re.I)
        if m:
            op  = m.group(1)
            thr = float(m.group(2))
            dur = int(m.group(3)) * (60 if m.group(4).lower() == "h" else 1)
            params.append({"type": "sustained", "rule_id": r.rule_id,
                           "sensor": r.sensor, "op": op,
                           "threshold": thr, "dur_min": dur,
                           "severity": severity})
    return params


# ── Plausibility guard ────────────────────────────────────────────────────────

def compute_violation_rates(rules: list[Rule],
                            timeseries_csv: str = TIMESERIES_CSV,
                            calibration_hours: Optional[float] = None,
                            ) -> dict[str, float]:
    """Fraction of readings each Threshold/OperationalRule flags.
    Reads only timestamp/sensor_id/value — no GT columns, no labels.

    calibration_hours=None (shipped default): rates over the full stream
    (transductive). Set to CALIBRATION_HOURS for the deployable variant
    that uses only the first N hours as a commissioning period — see
    step5_sensitivity.py's calibration check for the equivalence result.
    """
    candidates = {r.rule_id: r for r in rules
                  if r.cls in ("ThresholdRule", "OperationalRule")}
    if not candidates:
        return {}

    by_sensor: dict[str, list[Rule]] = {}
    for r in candidates.values():
        by_sensor.setdefault(r.sensor, []).append(r)

    totals     = {rid: 0 for rid in candidates}
    violations = {rid: 0 for rid in candidates}

    cutoff = None
    with open(timeseries_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if calibration_hours is not None:
                ts = datetime.fromisoformat(row["timestamp"])
                if cutoff is None:
                    cutoff = ts + timedelta(hours=calibration_hours)
                if ts > cutoff:
                    continue
            sid = row["sensor_id"]
            if sid not in by_sensor:
                continue
            try:
                val = float(row["value"])
            except (ValueError, KeyError):
                continue
            for r in by_sensor[sid]:
                totals[r.rule_id] += 1
                if _bound_severity(val, r) is not None:
                    violations[r.rule_id] += 1

    return {rid: violations[rid] / totals[rid]
            for rid in totals if totals[rid] > 0}


# ── Per-sensor state ──────────────────────────────────────────────────────────

def _baseline_samples_for(window_min: float, ratio: float) -> int:
    return max(1, round(window_min * ratio * 60 / SAMPLE_INTERVAL_SEC))


@dataclass
class SensorState:
    drift_window_min: float = DEFAULT_DRIFT_WINDOW_MIN
    stuck_buffer_len: int   = DEFAULT_STUCK_BUFFER
    baseline_ratio:   float = BASELINE_WINDOW_RATIO
    drift_min_ref:    int   = DRIFT_MIN_REF_SAMPLES

    violation_start: Optional[datetime] = None
    recent_vals:   deque = field(default=None)
    recent_win:    deque = field(default_factory=deque)   # (ts, val)
    reference_win: deque = field(default=None)

    def __post_init__(self) -> None:
        if self.recent_vals is None:
            self.recent_vals = deque(maxlen=self.stuck_buffer_len)
        if self.reference_win is None:
            self.reference_win = deque(
                maxlen=_baseline_samples_for(self.drift_window_min,
                                             self.baseline_ratio))

    def push(self, ts: datetime, val: float) -> None:
        self.recent_vals.append(val)
        self.recent_win.append((ts, val))
        cutoff = ts - timedelta(minutes=self.drift_window_min)
        while self.recent_win and self.recent_win[0][0] < cutoff:
            _, old = self.recent_win.popleft()
            self.reference_win.append(old)

    def is_stuck(self, n: int) -> bool:
        if len(self.recent_vals) < n:
            return False
        window = list(self.recent_vals)[-n:]
        return (max(window) - min(window)) < STUCK_TOL

    def drift_delta(self) -> float:
        if len(self.reference_win) < self.drift_min_ref or len(self.recent_win) < 2:
            return 0.0
        base = median(self.reference_win)
        recent_mean = sum(v for _, v in self.recent_win) / len(self.recent_win)
        return recent_mean - base

    def drift_span_min(self) -> float:
        if not self.recent_win:
            return 0.0
        return (self.recent_win[-1][0] - self.recent_win[0][0]).total_seconds() / 60.0


# ── Detection engine ──────────────────────────────────────────────────────────

def stream_and_detect(rules: list[Rule],
                      quarantined: set[str],
                      timeseries_csv: str = TIMESERIES_CSV,
                      on_delay_warning_min: float = ON_DELAY_WARNING_MIN,
                      baseline_window_ratio: float = BASELINE_WINDOW_RATIO,
                      drift_min_ref_samples: int = DRIFT_MIN_REF_SAMPLES,
                      ) -> tuple[list[Alarm], int]:
    """Single pass over the timeseries. Reads ONLY timestamp/sensor_id/value.

    All constants are parameters so step5_sensitivity.py can vary them and
    step5_holdout.py can point timeseries_csv at an injected copy.
    """
    by_sensor: dict[str, list[Rule]] = {}
    for r in rules:
        if r.sensor:
            by_sensor.setdefault(r.sensor, []).append(r)

    maint_params_by_sensor: dict[str, list[dict]] = {}
    for p in parse_maint_params(rules):
        maint_params_by_sensor.setdefault(p["sensor"], []).append(p)

    # Per-sensor windows sized from the sensor's own rules — a global cap
    # smaller than a rule's requirement would make it structurally unable
    # to fire regardless of the data.
    drift_window_by_sensor: dict[str, float] = {}
    stuck_buffer_by_sensor: dict[str, int] = {}
    for sid, params in maint_params_by_sensor.items():
        durs = [p["dur_min"] for p in params if p["type"] == "drift"]
        if durs:
            drift_window_by_sensor[sid] = max(durs)
        ns = [p["n"] for p in params if p["type"] == "stuck"]
        if ns:
            stuck_buffer_by_sensor[sid] = max(max(ns), DEFAULT_STUCK_BUFFER)

    states:           dict[str, SensorState] = {}
    sustained_starts: dict[tuple, Optional[datetime]] = {}
    alarms:           list[Alarm] = []
    row_count = 0

    with open(timeseries_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts  = datetime.fromisoformat(row["timestamp"])
            sid = row["sensor_id"]
            try:
                val = float(row["value"])
            except (ValueError, KeyError):
                continue

            if sid not in states:
                states[sid] = SensorState(
                    drift_window_min=drift_window_by_sensor.get(
                        sid, DEFAULT_DRIFT_WINDOW_MIN),
                    stuck_buffer_len=stuck_buffer_by_sensor.get(
                        sid, DEFAULT_STUCK_BUFFER),
                    baseline_ratio=baseline_window_ratio,
                    drift_min_ref=drift_min_ref_samples)
            st = states[sid]
            st.push(ts, val)
            row_count += 1

            # ── threshold ─────────────────────────────────────────────────
            best_sev:  Optional[str] = None
            best_rule: str = ""
            for r in by_sensor.get(sid, []):
                if r.cls not in ("ThresholdRule", "OperationalRule"):
                    continue
                if r.rule_id in quarantined:
                    continue
                sev = _bound_severity(val, r)
                if sev is None:
                    continue
                if best_sev is None or (sev == "CRITICAL" and best_sev == "WARNING"):
                    best_sev, best_rule = sev, r.rule_id

            if best_sev is None:
                st.violation_start = None
            else:
                if st.violation_start is None:
                    st.violation_start = ts
                elapsed = (ts - st.violation_start).total_seconds() / 60.0
                delay   = 0 if best_sev == "CRITICAL" else on_delay_warning_min
                if elapsed >= delay:
                    alarms.append(Alarm(sid, ts, val, best_sev, "threshold", best_rule))

            # ── stuck / drift / sustained (MaintenanceRule-driven) ────────
            for p in maint_params_by_sensor.get(sid, []):
                if p["type"] == "stuck" and st.is_stuck(p["n"]):
                    alarms.append(Alarm(sid, ts, val, p["severity"], "stuck",
                                        p["rule_id"]))
                elif p["type"] == "drift":
                    if (abs(st.drift_delta()) >= p["delta"]
                            and st.drift_span_min() >= p["dur_min"]):
                        alarms.append(Alarm(sid, ts, val, p["severity"], "drift",
                                            p["rule_id"]))
                elif p["type"] == "sustained":
                    key = (sid, p["rule_id"])
                    # "≥"/"≤" are single unicode chars — test both spellings
                    # or the direction silently flips.
                    upper = (">" in p["op"]) or ("≥" in p["op"])
                    in_viol = (val > p["threshold"] if upper
                               else val < p["threshold"])
                    if not in_viol:
                        sustained_starts[key] = None
                        continue
                    if sustained_starts.get(key) is None:
                        sustained_starts[key] = ts
                    elapsed = (ts - sustained_starts[key]).total_seconds() / 60.0
                    if elapsed >= p["dur_min"]:
                        alarms.append(Alarm(sid, ts, val, p["severity"],
                                            "sustained", p["rule_id"]))

    return alarms, row_count


# ── Merge alarms → events ─────────────────────────────────────────────────────

def merge_alarms(alarms: list[Alarm],
                 gap_merge_min: float = GAP_MERGE_MIN) -> list[Event]:
    """Merge consecutive alarms on the same sensor into events."""
    if not alarms:
        return []
    by_sensor: dict[str, list[Alarm]] = {}
    for a in sorted(alarms, key=lambda x: (x.sensor, x.ts)):
        by_sensor.setdefault(a.sensor, []).append(a)

    events: list[Event] = []
    for sensor, sal in by_sensor.items():
        cs, ce = sal[0].ts, sal[0].ts
        csev  = sal[0].severity
        dets  = {sal[0].detector}
        rids  = {sal[0].rule_id}
        count = 1
        for a in sal[1:]:
            if (a.ts - ce).total_seconds() / 60.0 <= gap_merge_min:
                ce = a.ts
                if a.severity == "CRITICAL":
                    csev = "CRITICAL"
                dets.add(a.detector)
                rids.add(a.rule_id)
                count += 1
            else:
                events.append(Event(sensor, cs, ce, csev,
                                    sorted(dets), sorted(rids), count))
                cs, ce, csev = a.ts, a.ts, a.severity
                dets, rids, count = {a.detector}, {a.rule_id}, 1
        events.append(Event(sensor, cs, ce, csev,
                            sorted(dets), sorted(rids), count))
    return sorted(events, key=lambda e: e.start)


# ── GT loading (call ONLY after detection) ────────────────────────────────────

def load_gt_windows() -> list[GtAnomaly]:
    """GT anomaly windows from nodes.csv + edges.csv (Phase 2 reference,
    guide §2.3). Never call before detection has finished."""
    with open(NODES_CSV, newline="", encoding="utf-8") as f:
        nodes = {r["nodeId"]: r for r in csv.DictReader(f)}
    with open(EDGES_CSV, newline="", encoding="utf-8") as f:
        trigger_map = {e["toId"]: e["fromId"]
                       for e in csv.DictReader(f)
                       if e["type"].lower() == "triggers"}

    anomalies = []
    for nid, n in nodes.items():
        if n.get("label") != "AnomalyEvent":
            continue
        src_id = trigger_map.get(nid, "")
        sensor = nodes.get(src_id, {}).get("name", "")
        anomalies.append(GtAnomaly(
            gt_id=n["gtId"], sensor=sensor, atype=n["anomalyType"],
            start=datetime.fromisoformat(n["startTs"]),
            end=datetime.fromisoformat(n["endTs"])))
    return sorted(anomalies, key=lambda a: a.start)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_coverage(events: list[Event],
                   gt_windows: list[GtAnomaly]) -> list[dict]:
    """COVERED/GAP per GT window (guide Table 2). Shipped rule: any nonzero
    temporal overlap on the same sensor. Leniency is quantified separately
    (apply_strictness / scoring_strictness.csv). All GT types are scored by
    this same uniform rule — no type gets a dedicated detector or carve-out;
    CORRELATED events are expected to show as GAP (see module docstring)."""
    results = []
    for gt in gt_windows:
        covering = [e for e in events
                    if e.sensor == gt.sensor
                    and e.start <= gt.end and e.end >= gt.start]
        if covering:
            det_start   = min(e.start for e in covering)
            det_end     = max(e.end   for e in covering)
            latency_min = (det_start - gt.start).total_seconds() / 60.0
            gt_dur_sec  = (gt.end - gt.start).total_seconds()
            # coveragePct = UNION of per-event overlaps with the GT window,
            # not the [det_start, det_end] envelope — with multiple disjoint
            # covering events the envelope would count the uncovered gaps
            # between them as covered. Identical to the envelope for a
            # single covering event (every case in the current results).
            clipped = sorted((max(gt.start, e.start), min(gt.end, e.end))
                             for e in covering)
            ov_sec, cur_s, cur_e = 0.0, None, None
            for s, e in clipped:
                if cur_e is None or s > cur_e:
                    if cur_e is not None:
                        ov_sec += (cur_e - cur_s).total_seconds()
                    cur_s, cur_e = s, e
                else:
                    cur_e = max(cur_e, e)
            if cur_e is not None:
                ov_sec += (cur_e - cur_s).total_seconds()
            cov_pct = 100.0 * ov_sec / gt_dur_sec if gt_dur_sec > 0 else 100.0
            results.append({
                "gtId": gt.gt_id, "sensor": gt.sensor, "type": gt.atype,
                "status": "COVERED",
                "detector": "|".join(sorted({d for e in covering
                                             for d in e.detectors})),
                "matchedRules": "|".join(sorted({rid for e in covering
                                                 for rid in e.rule_ids})),
                "gtStart": gt.start.isoformat(), "gtEnd": gt.end.isoformat(),
                "detStart": det_start.isoformat(), "detEnd": det_end.isoformat(),
                "latencyMin": round(latency_min, 1),
                "coveragePct": round(cov_pct, 1),
            })
        else:
            results.append({
                "gtId": gt.gt_id, "sensor": gt.sensor, "type": gt.atype,
                "status": "GAP", "detector": "", "matchedRules": "",
                "gtStart": gt.start.isoformat(), "gtEnd": gt.end.isoformat(),
                "detStart": "", "detEnd": "", "latencyMin": "", "coveragePct": "",
            })
    return results


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI — always report next to small-n recall."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom  = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_anomaly_metrics(coverage: list[dict], events: list[Event],
                            fp_extra_windows: list[GtAnomaly] | None = None) -> dict:
    """Event-level metrics. Definitions (state these in the thesis — TP and
    FP count DIFFERENT units, standard for event-level evaluation):
      TP = GT windows overlapped by >=1 detected event (counts GT windows)
      FN = GT windows with no overlapping event
      FP = detected events overlapping NO GT window on their sensor (counts
           detected events); several events covering one GT window all count
           as matched, none as FP.
      recall = TP/(TP+FN);  precision = TP/(TP+FP)  (mixed units).
    fp_extra_windows: additional windows that suppress FP status without
    entering recall — used by the holdout run, where the original 14 GT
    events remain in the stream but only injected events are scored."""
    covered = [v for v in coverage if v["status"] == "COVERED"]
    gaps    = [v for v in coverage if v["status"] == "GAP"]

    all_gt_windows = [
        (v["sensor"], datetime.fromisoformat(v["gtStart"]),
         datetime.fromisoformat(v["gtEnd"]))
        for v in coverage
    ] + [(g.sensor, g.start, g.end) for g in (fp_extra_windows or [])]

    def _overlaps_any_gt(e: Event) -> bool:
        return any(e.sensor == s and e.start <= end and e.end >= start
                   for s, start, end in all_gt_windows)

    fp = sum(1 for e in events if not _overlaps_any_gt(e))
    tp, fn = len(covered), len(gaps)
    recall    = tp / len(coverage) if coverage else 0.0
    precision = tp / (tp + fp)     if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    lo, hi = wilson_ci(tp, len(coverage))
    return {
        "anomaly_events_total": len(coverage),
        "anomaly_tp": tp, "anomaly_fn_gap": fn, "anomaly_fp": fp,
        "anomaly_recall": round(recall, 3),
        "anomaly_recall_ci95_lo": round(lo, 3),
        "anomaly_recall_ci95_hi": round(hi, 3),
        "anomaly_precision": round(precision, 3),
        "anomaly_f1": round(f1, 3),
        "total_events_detected": len(events),
    }


def apply_strictness(coverage: list[dict],
                     min_cov_pct: float = 0.0,
                     max_latency_min: float | None = None,
                     max_latency_frac: float | None = None) -> tuple[int, list[str]]:
    """Re-score COVERED rows under a stricter acceptance criterion.
    Returns (tp, dropped_gtIds). Negative latency (early detection) always
    satisfies a latency cap."""
    tp, dropped = 0, []
    for v in coverage:
        if v["status"] != "COVERED":
            continue
        cov_pct = float(v["coveragePct"])
        latency = float(v["latencyMin"])
        gt_dur_min = (datetime.fromisoformat(v["gtEnd"])
                      - datetime.fromisoformat(v["gtStart"])).total_seconds() / 60.0
        ok = cov_pct >= min_cov_pct
        if ok and max_latency_min is not None:
            ok = latency <= max_latency_min
        if ok and max_latency_frac is not None:
            ok = latency <= max_latency_frac * gt_dur_min
        if ok:
            tp += 1
        else:
            dropped.append(v["gtId"])
    return tp, dropped


# ── CSV writer ────────────────────────────────────────────────────────────────

def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
