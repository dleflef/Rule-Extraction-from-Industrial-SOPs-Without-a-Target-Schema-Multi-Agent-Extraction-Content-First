"""
step5_core.py

The shared engine for Phase 2 anomaly detection, following the two-phase
evaluation protocol published with the iMAKS dataset
(https://doi.org/10.5281/zenodo.20075430).

The detection primitives shared by the downstream harnesses are kept here, so the detectors, the scoring rule, and the metric definitions
exist exactly once and no two experiments can disagree because of
duplicated logic.

Four detectors are provided, all driven ONLY by the extracted rules and
never by ground truth:
  threshold  : the value crosses an extracted critHi/warnHi/warnLo/critLo bound
  stuck      : the value has been frozen for N samples   ("stuck >N samples")
  drift      : a rolling delta over a time window        ("drift >X over Y min/h")
  sustained  : a bound is violated for N minutes         (">X for >Y min/h")

Out of scope for THIS module — CORRELATED events (GT-0009). A correlated
fault is a relation between two sensors, which none of the four detectors
below can express: they each see one signal at a time. CORRELATED events
are therefore scored by the same uniform rule as everything else here and
appear as GAP. The cross-sensor case is handled one level up, in
step4_graph_generic.py, where the coupling is read back out of the graph
as a CORRELATES_WITH edge; nothing in this file participates in it.

Anti-leakage invariants, held by every entry point in this module:
  - Only timestamp / sensor_id / value are read from the timeseries during
    detection.
  - Ground truth (nodes.csv / edges.csv) is read only by the scoring
    helpers, which are invoked strictly AFTER detection has finished.
  - The rules are taken from an extraction CSV; GT bounds are never used
    as detection rules.
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

# Connection settings live with the harness that opens a connection
# (step4_graph_generic.py); this module reads CSVs and never opens one.

# ── Tunable constants ─────────────────────────────────────────────────────────
# Every constant below is exposed as a parameter of stream_and_detect() /
# merge_alarms(). The shipped
# defaults were fixed before the sensitivity analysis was run and were
# never retuned against ground truth.

# WARNING alarms are raised only after 5 continuous minutes of violation,
# following ISA-18.2 alarm rationalisation practice; CRITICAL alarms are
# raised immediately.
ON_DELAY_WARNING_MIN = 5

# Alarms on the same sensor separated by less than 15 minutes are merged
# into one event. A longer window would absorb the residual false positives
# into the events beside them; 15 is kept precisely so the value is not
# tuned post hoc against ground truth.
GAP_MERGE_MIN = 15

# The drift detector's recent window is sized PER SENSOR from that
# sensor's own extracted rule duration. A global cap smaller than a rule's
# "over N min" clause would make the rule structurally unable to fire —
# a failure mode that was actually hit once with MAINT-06.
DEFAULT_DRIFT_WINDOW_MIN = 60
# The baseline (reference) window is this many times longer than the
# recent window: with the shipped ratio of 2, the last hour is compared
# against the two hours before it.
BASELINE_WINDOW_RATIO = 2
# No drift value is reported until the baseline holds at least this many
# samples, so that startup noise is never compared against an empty or
# meaningless reference.
DRIFT_MIN_REF_SAMPLES = 10
# The dataset's sampling cadence: one reading every 30 s, as stated by the
# iMAKS deposit and confirmed against timeseries_raw.csv.
SAMPLE_INTERVAL_SEC = 30

# Threshold rules that flag more than half of their sensor's readings are
# quarantined as almost certainly mis-extracted. The check is unsupervised
# (no labels are consulted); the `quarantined` column of every summary this
# module feeds reports how many rules it removed on a given run.
PLAUSIBILITY_MAX_VIOLATION_RATE = 0.5

# The deployable (non-transductive) variant of the plausibility filter
# computes violation rates over only the FIRST N hours of the stream — a
# commissioning period, as would be available before alarms are armed in
# a real deployment. The value was pre-registered at 8 h (10% of the 80 h
# stream, an a-priori round fraction) and was not tuned against results.
# Whether this window quarantines the same rules as the full-stream
# filter is exercised by the downstream harnesses.
CALIBRATION_HOURS = 8

# Two readings are treated as identical by the stuck detector when they
# differ by less than this tolerance.
STUCK_TOL = 1e-6
# Fallback buffer length only — sensors that carry a stuck rule are given
# a buffer sized from the rule's own sample count instead.
DEFAULT_STUCK_BUFFER = 64

# No pass-bar constant is defined for Phase 2 coverage: the coverage fraction a
# run achieves is reported directly by compute_anomaly_metrics / score_coverage,
# so no threshold has to be agreed on here for a figure to be readable.


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




# ── Condition parsers ─────────────────────────────────────────────────────────

def _bound_severity(val: float, r: Rule) -> Optional[str]:
    # Crit bounds are checked before warn bounds. As a consequence, a rule
    # extracted with critHi == warnHi (a real LLM defect, e.g.
    # RULE-CAF01-01) classifies a warn-band violation as CRITICAL and
    # bypasses the WARNING on-delay. Recall and precision are unaffected;
    # per-event latency can be understated by up to ON_DELAY_WARNING_MIN —
    # see EVALUATION_LIMITATIONS.md.
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
    """MaintenanceRule condition strings are parsed into stuck/drift/
    sustained detector parameters. Only the extracted text is consulted —
    no GT thresholds are involved at any point."""
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
    """The fraction of readings flagged by each Threshold/OperationalRule
    is computed here. Only timestamp/sensor_id/value are read — no GT
    columns and no labels.

    With calibration_hours=None (the shipped default) the rates are taken
    over the full stream, which is transductive. When CALIBRATION_HOURS is
    passed instead, only the first N hours are used — the deployable
    commissioning-period variant, whose equivalence to the shipped filter
    is exercised by the downstream harnesses.
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
    """The whole detection run is a single pass over the timeseries, and
    ONLY timestamp/sensor_id/value are read from it.

    All engine constants are accepted as parameters so that
    The harnesses sweep them one at a time; the defaults are the
    shipped values.
    """
    by_sensor: dict[str, list[Rule]] = {}
    for r in rules:
        if r.sensor:
            by_sensor.setdefault(r.sensor, []).append(r)

    maint_params_by_sensor: dict[str, list[dict]] = {}
    for p in parse_maint_params(rules):
        maint_params_by_sensor.setdefault(p["sensor"], []).append(p)

    # The per-sensor windows are sized from each sensor's own rules; a
    # global cap smaller than a rule's requirement would make that rule
    # structurally unable to fire regardless of the data.
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
                    # "≥" and "≤" are single unicode characters, so both
                    # spellings must be tested; otherwise the comparison
                    # direction is silently flipped.
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


# ── Merge alarms , events ─────────────────────────────────────────────────────

def merge_alarms(alarms: list[Alarm],
                 gap_merge_min: float = GAP_MERGE_MIN) -> list[Event]:
    """Consecutive alarms on the same sensor are merged into events; a new
    event is started whenever the gap to the previous alarm exceeds
    gap_merge_min. The merged event keeps the highest severity seen and
    the union of contributing detectors and rules."""
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
    """The GT anomaly windows are read from nodes.csv + edges.csv (the
    Phase 2 reference). This must never be called before
    detection has finished — GT is a scoring input only."""
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
    """Each GT window is scored COVERED or GAP. Under the
    shipped rule, any nonzero temporal overlap on the same sensor counts;
    how much recall that leniency buys is quantified separately by
    apply_strictness / scoring_strictness.csv. Every GT type is scored by
    this same uniform rule — no type is given a dedicated detector or a
    carve-out, so CORRELATED events are expected to appear as GAP (see the
    module docstring)."""
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
            # coveragePct is computed as the UNION of per-event overlaps
            # with the GT window, not as the [det_start, det_end] envelope:
            # with multiple disjoint covering events, the envelope would
            # count the uncovered gaps between them as covered. For a
            # single covering event — every case in the current results —
            # the two are identical.
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
    """The Wilson score 95% confidence interval. It is reported next to
    every recall figure because n is small and a point estimate alone
    would overstate the certainty."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom  = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_anomaly_metrics(coverage: list[dict], events: list[Event]) -> dict:
    """Event-level metrics. The definitions should be stated explicitly in
    the thesis, because TP and FP count DIFFERENT units — which is the
    standard convention for event-level evaluation:
      TP = GT windows overlapped by >=1 detected event (counts GT windows)
      FN = GT windows with no overlapping event
      FP = detected events that overlap NO GT window on their sensor
           (counts detected events); several events covering one GT window
           are all counted as matched, and none as FP.
      recall = TP/(TP+FN);  precision = TP/(TP+FP)  (mixed units)."""
    covered = [v for v in coverage if v["status"] == "COVERED"]
    gaps    = [v for v in coverage if v["status"] == "GAP"]

    all_gt_windows = [
        (v["sensor"], datetime.fromisoformat(v["gtStart"]),
         datetime.fromisoformat(v["gtEnd"]))
        for v in coverage
    ]

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
    """COVERED rows are re-scored under a stricter acceptance criterion,
    and (tp, dropped_gtIds) is returned. Negative latency — detection
    before the GT window opened — always satisfies a latency cap."""
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

