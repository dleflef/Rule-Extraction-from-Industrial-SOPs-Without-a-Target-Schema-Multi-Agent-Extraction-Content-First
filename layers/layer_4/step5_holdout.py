"""
step5_holdout.py

Held-out evaluation by seeded anomaly injection — the fix for the one
weakness disclosure alone cannot repair: the detector stack was designed
while observing outcomes on all 14 original GT events, so those events
cannot yield an unbiased estimate. A second iMAKS release is not
available; instead this script manufactures NEW, never-seen anomaly
instances that follow the dataset's own generative definitions (guide
§2.3.2) and magnitude scales (§2.3.1), injects them into anomaly-free
stretches of the original timeseries, and runs the FROZEN pipeline on the
result.

Two modes:
  (default)      the canonical ONE-SHOT run (seed 20260710): 42 events,
                 outputs holdout_manifest.csv / holdout_coverage.csv /
                 holdout_summary.csv / injected_timeseries.csv.
  --multiseed    the PRE-COMMITTED ten-seed sweep (20260711–20260720):
                 removes injection-schedule sampling variance from the
                 estimate. Every seed is reported, none discarded; a
                 POOLED row aggregates raw counts with a Wilson 95% CI.
                 Outputs holdout_multiseed.csv. This complements — does
                 not replace — the one-shot run: that answers "did we
                 evaluate exactly once without iterating?", this answers
                 "how much does the estimate move across schedules?".

Generative definitions implemented (guide §2.3.2, verbatim semantics):
  SPIKE        — Gaussian-shaped transient, peak in first 20% of window,
                 duration < 5 min.
  DRIFT        — monotonic linear ramp across the full event window.
  STUCK        — value frozen at window onset.
  OUT_OF_RANGE — fixed sustained offset above WARN_HI / below WARN_LO.
CORRELATED events are NOT injected: the pipeline implements no fusion
detector (see step5_core.py docstring), so injecting them would only
re-measure a known structural gap.

Sensor pools per type mirror the dataset's own placement pattern (e.g.
STUCK occurs on ST03_LABELLING_TEN, OUT_OF_RANGE on auxiliary-zone
sensors), so the injected set estimates performance on the same generative
process, not on a different benchmark.

Honesty protocol (state this in the thesis):
  - Seeds are fixed in this file BEFORE running; the run records the seed
    and the git commit of the detector code. All runs are reported.
  - Any future code change requires fresh seeds and reporting all runs.
  - The injector uses GT bands (nodes.csv) to scale magnitudes — that is
    benchmark construction, not detection. The frozen pipeline reads only
    timestamp/sensor_id/value (proven by step5_leakage_audit.py).
  - The original 14 GT windows remain in the stream. They are NOT scored
    here (recall is over injected events only); detections overlapping
    them are excluded from the false-positive count.

Usage:
    python layers/layer_4/step5_holdout.py               # one-shot run
    python layers/layer_4/step5_holdout.py --seed N      # custom seed
    python layers/layer_4/step5_holdout.py --multiseed   # 10-seed sweep
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from step5_core import (
    RESULTS_DIR, TIMESERIES_CSV, NODES_CSV,
    PLAUSIBILITY_MAX_VIOLATION_RATE,
    GtAnomaly,
    load_rules_from_csv, compute_violation_rates, stream_and_detect,
    merge_alarms, load_gt_windows, score_coverage, compute_anomaly_metrics,
    apply_strictness, wilson_ci, write_csv,
)

HOLDOUT_DIR  = os.path.join(RESULTS_DIR, "holdout")
INJECTED_CSV = os.path.join(HOLDOUT_DIR, "injected_timeseries.csv")

# Seed history (one-shot protocol: every pipeline revision gets a fresh
# seed): 20260709 evaluated an earlier, since-revised pipeline and was
# retired with it; 20260710 is the current pipeline's canonical seed.
CANONICAL_SEED = 20260710

# PRE-COMMITTED multiseed list — fixed here before running, all reported.
PRECOMMITTED_SEEDS = list(range(20260711, 20260721))

# Number of injected events per type (42 total per schedule).
N_SPIKE, N_DRIFT, N_STUCK, N_OOR = 10, 10, 8, 14

# Same-sensor exclusion buffer around original GT windows and around other
# injected windows, so events never blur into each other.
CLEARANCE_MIN = 120

# Sensor pools mirroring the dataset's own type→sensor placement (§2.3.1).
POOL = {
    "SPIKE": ["ST02_SEALING_TMP", "ST01_FILLING_FLW", "ST04_PACKAGING_VIB",
              "ST01_FILLING_TMP", "ST02_SEALING_PRS"],
    "DRIFT": ["ST01_FILLING_PRS", "ST04_PACKAGING_VIB", "ST02_SEALING_CUR",
              "WRH01_WAREHOUSE_TMP", "RND01_RDLAB_HUM"],
    "STUCK": ["ST03_LABELLING_TEN"],
    "OUT_OF_RANGE": ["SRV01_SERVERROOM_TMP", "CAF01_CAFETERIA_TMP",
                     "ST03_LABELLING_CNT", "CHM01_CHEMICALSTORAGE_TMP",
                     "SRV01_SERVERROOM_HUM", "CAF01_CAFETERIA_HUM",
                     "CHM01_CHEMICALSTORAGE_HUM", "RND01_RDLAB_TMP",
                     "WRH01_WAREHOUSE_HUM"],
}


def _git_sha() -> str:
    """Short commit hash of the detector CODE, with a -dirty marker if any
    layer_4 *.py file differs from HEAD. Scoped to code deliberately:
    regenerated result CSVs must not mark the frozen code as dirty."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10,
                             cwd=here).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "*.py"],
            capture_output=True, text=True, timeout=10,
            cwd=here).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def load_sensor_bands() -> dict[str, dict]:
    """nominal/warn/crit bands per sensor from nodes.csv — used ONLY to
    scale injected magnitudes (benchmark construction)."""
    bands = {}
    with open(NODES_CSV, newline="", encoding="utf-8") as f:
        for n in csv.DictReader(f):
            if n.get("label") != "Sensor":
                continue
            def g(k):
                v = (n.get(k) or "").strip()
                return float(v) if v else None
            bands[n["name"]] = {"nominal": g("nominalValue"),
                                "warnHi": g("warnHi"), "critHi": g("critHi"),
                                "warnLo": g("warnLo"), "critLo": g("critLo")}
    return bands


def sim_bounds() -> tuple[datetime, datetime]:
    """First/last timestamp of the raw stream (read from the file itself)."""
    with open(TIMESERIES_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        first = next(r)["timestamp"]
        for row in r:
            last = row["timestamp"]
    return datetime.fromisoformat(first), datetime.fromisoformat(last)


def build_schedule(rng: random.Random, bands: dict[str, dict],
                   sim_start: datetime, sim_end: datetime,
                   original_gt: list[GtAnomaly]) -> list[dict]:
    """Seeded injection schedule. Each entry: hoId, type, sensor, start,
    end, magnitude. Same-sensor windows keep CLEARANCE_MIN from original
    GT windows and from each other."""
    taken: dict[str, list[tuple[datetime, datetime]]] = {}
    for gt in original_gt:
        taken.setdefault(gt.sensor, []).append((gt.start, gt.end))

    def _clear(sensor: str, s: datetime, e: datetime) -> bool:
        buf = timedelta(minutes=CLEARANCE_MIN)
        return all(e + buf < ts or s - buf > te
                   for ts, te in taken.get(sensor, []))

    def _place(sensor: str, dur_min: float) -> tuple[datetime, datetime] | None:
        lo = sim_start + timedelta(hours=4)          # detector warm-up
        hi = sim_end - timedelta(minutes=dur_min + 60)
        span = (hi - lo).total_seconds()
        for _ in range(300):
            s = lo + timedelta(seconds=rng.uniform(0, span))
            s = s.replace(second=0 if s.second < 30 else 30, microsecond=0)
            e = s + timedelta(minutes=dur_min)
            if _clear(sensor, s, e):
                taken.setdefault(sensor, []).append((s, e))
                return s, e
        return None

    def _updir(b) -> bool:
        """Prefer the side that has both warn and crit bounds."""
        up = b["warnHi"] is not None and b["critHi"] is not None
        dn = b["warnLo"] is not None and b["critLo"] is not None
        if up and dn:
            return rng.random() < 0.5
        return up

    def _crit_band(b, up: bool) -> float | None:
        nom = b["nominal"]
        crit = b["critHi"] if up else b["critLo"]
        if nom is None or crit is None:
            return None
        return abs(crit - nom)

    schedule, k = [], 0

    def _add(atype, sensor, dur_min, **params):
        nonlocal k
        placed = _place(sensor, dur_min)
        if placed is None:
            print(f"  WARNING: could not place {atype} on {sensor} — skipped")
            return
        k += 1
        schedule.append({"hoId": f"HO-{k:04d}", "type": atype,
                         "sensor": sensor,
                         "start": placed[0], "end": placed[1], **params})

    for _ in range(N_SPIKE):                       # SPIKE: <5 min, peak 1.2–2×crit
        sensor = rng.choice(POOL["SPIKE"])
        b = bands[sensor]
        up = _updir(b)
        band = _crit_band(b, up) or 1.0
        _add("SPIKE", sensor, rng.choice([1.5, 2, 3, 4]),
             magnitude=round((1 if up else -1) * rng.uniform(1.2, 2.0) * band, 4))

    for _ in range(N_DRIFT):                       # DRIFT: 45–180 min ramp
        sensor = rng.choice(POOL["DRIFT"])
        b = bands[sensor]
        up = _updir(b)
        band = _crit_band(b, up) or 1.0
        _add("DRIFT", sensor, rng.choice([45, 60, 90, 120, 180]),
             magnitude=round((1 if up else -1) * rng.uniform(0.6, 1.2) * band, 4))

    for _ in range(N_STUCK):                       # STUCK: 10–20 min frozen
        _add("STUCK", rng.choice(POOL["STUCK"]),
             rng.choice([10, 12, 15, 20]), magnitude=0.0)

    for _ in range(N_OOR):                         # OOR: offset beyond WARN
        sensor = rng.choice(POOL["OUT_OF_RANGE"])
        b = bands[sensor]
        up = _updir(b)
        nom = b["nominal"]
        warn = b["warnHi"] if up else b["warnLo"]
        crit = b["critHi"] if up else b["critLo"]
        if nom is None or warn is None or crit is None:
            continue
        off = abs(warn - nom) + rng.uniform(0.3, 1.0) * abs(crit - warn)
        _add("OUT_OF_RANGE", sensor, rng.choice([20, 25, 30, 45]),
             magnitude=round((1 if up else -1) * off, 4))

    return sorted(schedule, key=lambda x: x["start"])


def inject(schedule: list[dict], out_csv: str = INJECTED_CSV) -> None:
    """Stream the raw CSV once, add each event's offset per §2.3.2, write a
    3-column copy (timestamp,sensor_id,value). Only `value` is modified."""
    by_sensor: dict[str, list[dict]] = {}
    for ev in schedule:
        by_sensor.setdefault(ev["sensor"], []).append(ev)

    frozen: dict[str, float] = {}       # hoId → frozen value for STUCK
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(TIMESERIES_CSV, newline="", encoding="utf-8") as fin, \
         open(out_csv, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["timestamp", "sensor_id", "value"])
        for row in csv.DictReader(fin):
            sid = row["sensor_id"]
            try:
                val = float(row["value"])
            except (ValueError, KeyError):
                continue
            ts = datetime.fromisoformat(row["timestamp"])

            for ev in by_sensor.get(sid, []):
                if not (ev["start"] <= ts <= ev["end"]):
                    continue
                dur = (ev["end"] - ev["start"]).total_seconds()
                pos = (ts - ev["start"]).total_seconds()
                mag = ev["magnitude"]
                if ev["type"] == "SPIKE":
                    # peak in first 20% of window (guide §2.3.2)
                    t_peak = 0.15 * dur
                    sigma  = max(dur / 6.0, 15.0)
                    val += mag * math.exp(-((pos - t_peak) ** 2) / (2 * sigma ** 2))
                elif ev["type"] == "DRIFT":
                    val += mag * (pos / dur if dur else 1.0)   # monotonic ramp
                elif ev["type"] == "OUT_OF_RANGE":
                    val += mag                                  # fixed offset
                elif ev["type"] == "STUCK":
                    if ev["hoId"] not in frozen:
                        frozen[ev["hoId"]] = val                # freeze at onset
                    val = frozen[ev["hoId"]]
            w.writerow([row["timestamp"], sid, f"{val:.4f}"])


def run_one(seed: int, rules, bands, original_gt,
            sim_start: datetime, sim_end: datetime,
            injected_csv: str) -> tuple[list[dict], list[dict], dict, set, int]:
    """One full holdout pass: schedule → inject → frozen detection → score.
    Returns (schedule, coverage, metrics, quarantined, tp_at_25pct)."""
    rng = random.Random(seed)
    schedule = build_schedule(rng, bands, sim_start, sim_end, original_gt)
    inject(schedule, out_csv=injected_csv)

    vrates = compute_violation_rates(rules, timeseries_csv=injected_csv)
    quarantined = {rid for rid, rate in vrates.items()
                   if rate > PLAUSIBILITY_MAX_VIOLATION_RATE}
    alarms, _ = stream_and_detect(rules, quarantined,
                                  timeseries_csv=injected_csv)
    events = merge_alarms(alarms)

    ho_windows = [GtAnomaly(ev["hoId"], ev["sensor"], ev["type"],
                            ev["start"], ev["end"]) for ev in schedule]
    coverage = score_coverage(events, ho_windows)
    # Original 14 GT windows stay in the stream: exclude detections that
    # overlap them from the FP count, but never from recall.
    m = compute_anomaly_metrics(coverage, events, fp_extra_windows=original_gt)
    tp25, _ = apply_strictness(coverage, min_cov_pct=25)
    return schedule, coverage, m, quarantined, tp25


# ── One-shot mode ─────────────────────────────────────────────────────────────

def main_single(seed: int, sha: str, rules, bands, original_gt,
                sim_start, sim_end) -> None:
    print(f"HOLDOUT INJECTION RUN — seed={seed}, detector code {sha}")
    print("One-shot protocol: do not iterate on this result; a code fix "
          "requires a fresh seed.\n")
    print(f"[1/3] Building schedule + injecting ({sim_start} → {sim_end}) …")
    os.makedirs(HOLDOUT_DIR, exist_ok=True)
    schedule, coverage, m, quarantined, tp25 = run_one(
        seed, rules, bands, original_gt, sim_start, sim_end, INJECTED_CSV)
    print(f"  {len(schedule)} events: "
          f"{dict(Counter(ev['type'] for ev in schedule))}")
    print(f"[2/3] Frozen pipeline ran "
          f"(quarantined: {sorted(quarantined) or 'none'})")
    print(f"[3/3] Writing results …")

    write_csv(os.path.join(HOLDOUT_DIR, "holdout_manifest.csv"),
              ["hoId", "type", "sensor", "start", "end", "magnitude"],
              [{**ev, "start": ev["start"].isoformat(),
                "end": ev["end"].isoformat()} for ev in schedule])
    write_csv(os.path.join(HOLDOUT_DIR, "holdout_coverage.csv"),
              ["gtId", "sensor", "type", "status", "detector", "matchedRules",
               "gtStart", "gtEnd", "detStart", "detEnd", "latencyMin",
               "coveragePct"], coverage)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for v in coverage:
        by_type[v["type"]].append(v)
    summary = [
        {"metric": "seed",                    "value": seed},
        {"metric": "detector_code_git_sha",   "value": sha},
        {"metric": "events_injected",         "value": m["anomaly_events_total"]},
        {"metric": "holdout_tp",              "value": m["anomaly_tp"]},
        {"metric": "holdout_fn_gap",          "value": m["anomaly_fn_gap"]},
        {"metric": "holdout_fp",              "value": m["anomaly_fp"]},
        {"metric": "holdout_recall",          "value": m["anomaly_recall"]},
        {"metric": "holdout_recall_ci95_lo",  "value": m["anomaly_recall_ci95_lo"]},
        {"metric": "holdout_recall_ci95_hi",  "value": m["anomaly_recall_ci95_hi"]},
        {"metric": "holdout_precision",       "value": m["anomaly_precision"]},
        {"metric": "holdout_f1",              "value": m["anomaly_f1"]},
        {"metric": "holdout_recall_mincov25", "value": round(tp25 / len(coverage), 3) if coverage else 0.0},
        {"metric": "rules_quarantined",       "value": len(quarantined)},
    ]
    for atype in sorted(by_type):
        vs = by_type[atype]
        tp = sum(1 for v in vs if v["status"] == "COVERED")
        lo, hi = wilson_ci(tp, len(vs))
        summary.append({"metric": f"recall_{atype}",
                        "value": f"{tp}/{len(vs)} = {tp/len(vs):.3f} "
                                 f"[{lo:.3f},{hi:.3f}]"})
    write_csv(os.path.join(HOLDOUT_DIR, "holdout_summary.csv"),
              ["metric", "value"], summary)

    print()
    print("=" * 64)
    print("  HELD-OUT INJECTION RESULTS (one-shot, frozen pipeline)")
    print("=" * 64)
    print(f"  Injected events : {m['anomaly_events_total']}")
    print(f"  Recall          : {m['anomaly_recall']:.3f} "
          f"(95% CI [{m['anomaly_recall_ci95_lo']:.3f}, "
          f"{m['anomaly_recall_ci95_hi']:.3f}])")
    print(f"  Recall (≥25% overlap): {tp25}/{len(coverage)} "
          f"= {tp25/len(coverage):.3f}")
    print(f"  Precision       : {m['anomaly_precision']:.3f} "
          f"(FP={m['anomaly_fp']}, original GT windows excluded from FP)")
    print(f"\n  Per type:")
    for atype in sorted(by_type):
        vs = by_type[atype]
        tp = sum(1 for v in vs if v["status"] == "COVERED")
        print(f"    {atype:<14} {tp}/{len(vs)}")
    print(f"\n  NOTE: a single 42-event schedule carries sampling variance —")
    print(f"  quote the pooled --multiseed estimate as the headline number.")
    print(f"\n  Written to detection_results/holdout/")
    print("=" * 64)


# ── Multiseed mode ────────────────────────────────────────────────────────────

def main_multiseed(sha: str, rules, bands, original_gt,
                   sim_start, sim_end) -> None:
    print(f"MULTI-SEED HOLDOUT — {len(PRECOMMITTED_SEEDS)} pre-committed "
          f"seeds, detector code {sha}")
    print(f"Seeds: {PRECOMMITTED_SEEDS}\n")

    rows = []
    pooled_tp, pooled_total, pooled_fp, pooled_tp25 = 0, 0, 0, 0
    pooled_type_tp: dict[str, int] = defaultdict(int)
    pooled_type_n:  dict[str, int] = defaultdict(int)

    print(f"  {'seed':<10} {'events':<7} {'tp':<4} {'recall':<8} "
          f"{'rec>=25%':<9} {'precision':<10} fp")
    print(f"  {'-'*10} {'-'*7} {'-'*4} {'-'*8} {'-'*9} {'-'*10} {'-'*3}")

    for seed in PRECOMMITTED_SEEDS:
        with tempfile.TemporaryDirectory() as tmp:
            injected = os.path.join(tmp, f"injected_{seed}.csv")
            schedule, coverage, m, quarantined, tp25 = run_one(
                seed, rules, bands, original_gt, sim_start, sim_end, injected)

        pooled_tp    += m["anomaly_tp"]
        pooled_total += m["anomaly_events_total"]
        pooled_fp    += m["anomaly_fp"]
        pooled_tp25  += tp25
        by_type: dict[str, list] = defaultdict(list)
        for v in coverage:
            by_type[v["type"]].append(v)
        type_detail = []
        for t in sorted(by_type):
            tp_t = sum(1 for v in by_type[t] if v["status"] == "COVERED")
            pooled_type_tp[t] += tp_t
            pooled_type_n[t]  += len(by_type[t])
            type_detail.append(f"{t}:{tp_t}/{len(by_type[t])}")

        rows.append({"seed": seed, "git_sha": sha,
                     "events_injected": m["anomaly_events_total"],
                     "tp": m["anomaly_tp"], "fp": m["anomaly_fp"],
                     "recall": m["anomaly_recall"],
                     "recall_ci95_lo": m["anomaly_recall_ci95_lo"],
                     "recall_ci95_hi": m["anomaly_recall_ci95_hi"],
                     "recall_mincov25": round(
                         tp25 / m["anomaly_events_total"], 3),
                     "precision": m["anomaly_precision"],
                     "rules_quarantined": len(quarantined),
                     "per_type": "|".join(type_detail)})
        print(f"  {seed:<10} {m['anomaly_events_total']:<7} "
              f"{m['anomaly_tp']:<4} {m['anomaly_recall']:<8.3f} "
              f"{tp25/m['anomaly_events_total']:<9.3f} "
              f"{m['anomaly_precision']:<10.3f} {m['anomaly_fp']}")

    lo, hi = wilson_ci(pooled_tp, pooled_total)
    pooled_recall = pooled_tp / pooled_total if pooled_total else 0.0
    pooled_prec   = (pooled_tp / (pooled_tp + pooled_fp)
                     if (pooled_tp + pooled_fp) else 0.0)
    recalls = [r["recall"] for r in rows]
    type_detail = [f"{t}:{pooled_type_tp[t]}/{pooled_type_n[t]}"
                   for t in sorted(pooled_type_n)]
    rows.append({"seed": "POOLED", "git_sha": sha,
                 "events_injected": pooled_total,
                 "tp": pooled_tp, "fp": pooled_fp,
                 "recall": round(pooled_recall, 3),
                 "recall_ci95_lo": round(lo, 3),
                 "recall_ci95_hi": round(hi, 3),
                 "recall_mincov25": round(pooled_tp25 / pooled_total, 3),
                 "precision": round(pooled_prec, 3),
                 "rules_quarantined": "",
                 "per_type": "|".join(type_detail)})

    os.makedirs(HOLDOUT_DIR, exist_ok=True)
    out = os.path.join(HOLDOUT_DIR, "holdout_multiseed.csv")
    write_csv(out,
              ["seed", "git_sha", "events_injected", "tp", "fp", "recall",
               "recall_ci95_lo", "recall_ci95_hi", "recall_mincov25",
               "precision", "rules_quarantined", "per_type"],
              rows)

    print(f"\n  POOLED over {len(PRECOMMITTED_SEEDS)} seeds: "
          f"{pooled_tp}/{pooled_total} recall {pooled_recall:.3f} "
          f"(95% CI [{lo:.3f}, {hi:.3f}])")
    print(f"  Per-seed recall range: {min(recalls):.3f}–{max(recalls):.3f}")
    print(f"  Pooled recall (>=25% overlap): "
          f"{pooled_tp25/pooled_total:.3f}   Pooled precision: {pooled_prec:.3f}")
    print(f"  Per type pooled: {'  '.join(type_detail)}")
    print(f"\n  Written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED,
                    help="single-run seed (default: canonical one-shot)")
    ap.add_argument("--multiseed", action="store_true",
                    help="run the pre-committed 10-seed sweep instead")
    args = ap.parse_args()

    sha = _git_sha()
    rules = load_rules_from_csv()
    bands = load_sensor_bands()
    original_gt = load_gt_windows()
    sim_start, sim_end = sim_bounds()

    if args.multiseed:
        main_multiseed(sha, rules, bands, original_gt, sim_start, sim_end)
    else:
        main_single(args.seed, sha, rules, bands, original_gt,
                    sim_start, sim_end)


if __name__ == "__main__":
    main()
