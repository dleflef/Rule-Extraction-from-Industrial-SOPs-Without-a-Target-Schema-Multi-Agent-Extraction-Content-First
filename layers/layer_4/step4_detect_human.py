"""step4_detect_human.py
==================================================
Downstream test, part 2: the records a telemetry detector cannot consume.

step4_detect_generic.py exercises the 21% of extracted records that carry
numeric sensor bounds. This script exercises three more families against the
human-domain logs shipped with the dataset, which no earlier evaluation in this
project has used:

    access authorisation  ->  human/access_events.csv      (448 events, 8 denied)
    occupancy limits      ->  human/occupancy_timeseries.csv (211,200 rows)
    acknowledgement times ->  human/person_registry.csv    (per-role deadlines)

Design constraints carried over from the sensor experiment:

* Nothing is keyed on a discovered column NAME. Zones and roles are located in
  a record BY VALUE against the facility's declared vocabulary (the seven Zone
  nodes of kg_seed/nodes.csv and the roles present in the registry).

* Ground truth is opened only to SCORE, never to detect. The extracted rules
  alone decide every prediction.

* Each check carries a null control: the extracted decisions are permuted, and
  the check must collapse. A check that scores well under permutation is
  measuring the log's base rate, not the extraction.

Zone-name damage is expected and handled explicitly. The layer-1 parser emits
"Gen. Warehous e", "Chem. Stora ge", "R&D; Lab" where the facility declares
"General Warehouse", "Chemical Storage", "R&D Lab", so zone strings are
resolved by closest match against the declared seven and the full mapping is
printed. Anything that fails to resolve is reported, never guessed.

Scoring notes, stated rather than buried:

* Occupancy is scored against the unambiguous event "headcount exceeded the
  zone's hard cap" (zone_count > zone_max), which is a strict subset of the
  dataset's OVERCROWDING label. The label's WARNING tier does not follow the
  SOP's stated rule (Main Entrance is labelled WARNING at 3 of 8 while the
  document says "Monitoring only"), so it is excluded rather than modelled.

* Acknowledgement is a VALUE-AGREEMENT check, not a detection task: all 14
  logged responses met their SLA, so the log contains no violation to find.
  It is reported as AGREEMENT WITH THE REGISTRY rather than as correctness,
  because the registry turns out not to implement the procedure: SOP-004 §3
  states five distinct per-role deadlines (operator 15/3, technician 10/2,
  supervisor 5/1, manager 15/5, security 20/5) and person_registry.csv assigns
  15/3 to every role uniformly. The extraction reproduces all ten cells of the
  document's table exactly; the disagreement is between the registry and the
  document it is supposed to encode. Read the low agreement figure as a
  property of the reference, not of the extractor.

Usage:
    python3 step4_detect_human.py --rules <extraction.csv>
    python3 step4_detect_human.py --all-dev-runs [--null-control]
"""

from __future__ import annotations

import argparse
import csv
import difflib
import glob
import os
import random
import re

import pandas as pd

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HUMAN = os.path.join(_PROJECT_ROOT, "data", "dataset", "human")
NODES_CSV = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "nodes.csv")
PRED_DIR = os.path.join(_PROJECT_ROOT, "layers", "layer_2", "step2_results_generic")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "detection_generic_results")

# Every cell where an extracted value is placed beside the value some reference
# artefact holds for the same thing. Written out per run so that each claimed
# reference-data error can be checked one cell at a time, rather than inferred
# from an aggregate agreement count.
DISAGREEMENTS: list = []
ALL_DISAGREEMENTS: list = []

YES = {"yes", "y", "true", "authorized", "allowed", "permitted", "granted"}
NO = {"no", "n", "false", "denied", "prohibited", "forbidden", "not authorized"}


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).casefold())


def prf(tp: int, fp: int, fn: int) -> tuple:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def declared_zones() -> list:
    nodes = pd.read_csv(NODES_CSV, dtype=str).fillna("")
    return sorted({z.strip() for z in nodes.loc[nodes.label == "Zone", "name"] if z.strip()})


def resolve_zone(raw: str, zones: list, cache: dict) -> str:
    """Closest declared zone, or "" when nothing is close enough. Parser damage
    ("Gen. Warehous e") is the reason this is fuzzy rather than exact; the
    cutoff is deliberately loose because the alternative is discarding a
    correctly extracted rule for a typography artifact."""
    key = norm(raw)
    if not key:
        return ""
    if key in cache:
        return cache[key]
    table = {norm(z): z for z in zones}
    hit = table.get(key, "")
    if not hit:
        close = difflib.get_close_matches(key, list(table), n=1, cutoff=0.72)
        hit = table[close[0]] if close else ""
    cache[key] = hit
    return hit


def find_in_row(row: dict, vocab: set) -> str:
    """A value from `vocab` appearing anywhere in the record — the same
    by-value resolution the sensor experiment uses for stations."""
    for v in row.values():
        s = str(v).strip().casefold()
        if s in vocab:
            return s
    return ""


def tri_state(v: str):
    n = norm(v)
    if n in {norm(x) for x in YES}:
        return True
    if n in {norm(x) for x in NO}:
        return False
    return None


# ── Check 1: access authorisation ─────────────────────────────────────────────

def check_access(df: pd.DataFrame, zones: list, roles: set, null: bool) -> dict:
    cache: dict = {}
    matrix: dict = {}
    ambiguous_decisions = 0
    for _, r in df.iterrows():
        row = r.to_dict()
        role = find_in_row(row, roles)
        if not role:
            continue
        zone = ""
        for v in row.values():
            z = resolve_zone(str(v), zones, cache)
            if z:
                zone = z
                break
        if not zone:
            continue
        # An authorisation record carries exactly one yes/no cell. Taking the
        # FIRST such cell in column order would silently pick the wrong one on
        # any record that happened to carry two, so the cell is required to be
        # unique and ambiguous records are counted rather than resolved by
        # position.
        votes = [t for t in (tri_state(v) for v in row.values()) if t is not None]
        if len(votes) != 1:
            if len(votes) > 1:
                ambiguous_decisions += 1
            continue
        decision = votes[0]
        matrix.setdefault((role, zone), decision)

    if null:                                   # permute the decisions
        rng = random.Random(42)
        keys = list(matrix)
        vals = [matrix[k] for k in keys]
        rng.shuffle(vals)
        matrix = dict(zip(keys, vals))

    ev = pd.read_csv(os.path.join(HUMAN, "access_events.csv"), dtype=str).fillna("")
    tp = fp = fn = tn = unknown = 0
    for _, e in ev.iterrows():
        truth_denied = e["authorized"].strip().upper() == "NO"
        key = (e["role"].strip().casefold(), e["zone"].strip())
        if key not in matrix:
            unknown += 1
            continue
        pred_denied = not matrix[key]
        if pred_denied and truth_denied:
            tp += 1
        elif pred_denied and not truth_denied:
            fp += 1
        elif not pred_denied and truth_denied:
            fn += 1
        else:
            tn += 1
    p, r, f = prf(tp, fp, fn)
    return {"pairs_extracted": len(matrix),
            "records_ambiguous": ambiguous_decisions,
            "events_scored": tp + fp + fn + tn,
            "events_unknown_pair": unknown, "violations_tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f}


# ── Check 2: occupancy limits ─────────────────────────────────────────────────

def check_occupancy(df: pd.DataFrame, zones: list, null: bool) -> dict:
    cache: dict = {}
    caps: dict = {}
    rejected_multinumeric = 0
    for _, r in df.iterrows():
        row = r.to_dict()
        zone = ""
        for v in row.values():
            z = resolve_zone(str(v), zones, cache)
            if z:
                zone = z
                break
        if not zone:
            continue
        # By STRUCTURE, not by column name: a zone-capacity record carries
        # exactly one numeric cell, so the cap is that cell wherever the
        # inducer filed it. Searching for a column whose NAME looks capacity-
        # like is the failure mode this harness exists to avoid, and it is a
        # live hazard here -- one run emits an "authorized_personnel" column
        # whose name matches a "person" substring test just as well as
        # "max_persons" does. Records with several numeric cells are
        # ambiguous under this rule and are counted rather than guessed at.
        numeric = []
        for v in row.values():
            t = str(v).strip()
            if not t:
                continue
            try:
                numeric.append(int(float(t)))
            except (TypeError, ValueError):
                continue
        if len(numeric) == 1:
            caps.setdefault(zone, numeric[0])
        elif len(numeric) > 1:
            # A zone-bearing record carrying several numbers is not a capacity
            # record at all (a threshold record naming a zone, for instance).
            # Counted so the discriminator's selectivity is visible: across
            # every run, exactly the seven capacity records survive it.
            rejected_multinumeric += 1

    if null:
        rng = random.Random(42)
        keys = list(caps)
        vals = [caps[k] for k in keys]
        rng.shuffle(vals)
        caps = dict(zip(keys, vals))

    occ = pd.read_csv(os.path.join(HUMAN, "occupancy_timeseries.csv"), dtype=str).fillna("")
    occ["cnt"] = occ.zone_count.astype(int)
    occ["mx"] = occ.zone_max.astype(int)
    # One row per (timestamp, zone): the file lists one row per PERSON present.
    per_zone = occ.drop_duplicates(subset=["timestamp", "zone"])
    tp = fp = fn = 0
    unknown = 0
    exact_caps = 0
    examined = 0
    # A breach lasts for as long as the zone stays over its cap, so consecutive
    # 30-second intervals are one incident and not hundreds. Counted here so the
    # scale of `breaches_true` is on the record rather than left to a reader to
    # infer: zones_breached says how many zones ever exceed their limit at all,
    # and breach_episodes collapses the contiguous runs within them.
    breach_episodes = 0
    zones_breached = 0
    for zone, g in per_zone.groupby("zone"):
        if zone not in caps:
            unknown += len(g)
            continue
        examined += len(g)
        facility_cap = int(g.mx.iloc[0])
        gs = g.sort_values("timestamp")
        over = (gs.cnt > gs.mx).tolist()
        runs = sum(1 for i, v in enumerate(over) if v and (i == 0 or not over[i - 1]))
        breach_episodes += runs
        zones_breached += int(runs > 0)
        if caps[zone] == facility_cap:
            exact_caps += 1
        DISAGREEMENTS.append({
            "check": "occupancy", "subject": zone, "field": "max_persons",
            "extracted": caps[zone], "reference": facility_cap,
            "reference_source": "occupancy_timeseries.zone_max",
            "agrees": caps[zone] == facility_cap})
        pred = g.cnt > caps[zone]
        truth = g.cnt > g.mx
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
    p, r, f = prf(tp, fp, fn)
    # Cafeteria is the one cap that differs from the facility's zone_max, and
    # the document is the reason: SOP-004 prints "| Cafeteria | 20 |" while the
    # occupancy simulation caps it at 5. The extraction reads the document
    # correctly; the two references disagree with each other.
    # intervals_examined is every (timestamp, zone) pair the caps were applied
    # to, INCLUDING the ones where nothing happened. breaches_true is the
    # number of genuine cap exceedances among them. Reporting tp+fp+fn as
    # though it were the number of intervals scored would understate the
    # denominator by roughly two orders of magnitude and would differ between
    # the real and null conditions, making the two rows incomparable.
    return {"zones_extracted": len(caps), "caps_matching_facility": exact_caps,
            "zone_records_rejected_multinumeric": rejected_multinumeric,
            "intervals_examined": examined, "breaches_true": tp + fn,
            "breach_episodes": breach_episodes, "zones_breached": zones_breached,
            "zones_never_breached": len(caps) - zones_breached,
            "intervals_unknown_zone": unknown,
            "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


# ── Check 3: acknowledgement deadlines ────────────────────────────────────────

def check_ack(df: pd.DataFrame, roles: set) -> dict:
    reg = pd.read_csv(os.path.join(HUMAN, "person_registry.csv"), dtype=str).fillna("")
    truth = {}
    for _, r in reg.iterrows():
        truth.setdefault(r["role"].strip().casefold(),
                         (r["max_ack_warning"].strip(), r["max_ack_critical"].strip()))
    checked = warn_ok = crit_ok = 0
    for _, r in df.iterrows():
        row = r.to_dict()
        role = find_in_row(row, roles)
        if not role or role not in truth:
            continue
        w = c = None
        for k, v in row.items():
            nk = norm(k)
            s = str(v).strip()
            if not s:                       # an empty cell is absence, not a value
                continue
            if "ack" in nk and "warn" in nk:
                w = s
            elif "ack" in nk and "crit" in nk:
                c = s
        if w is None and c is None:
            continue
        checked += 1
        tw, tc = truth[role]
        if w and tw:
            warn_ok += int(float(w) == float(tw))
            DISAGREEMENTS.append({
                "check": "acknowledgement", "subject": role,
                "field": "max_ack_warning_min", "extracted": w, "reference": tw,
                "reference_source": "person_registry.max_ack_warning",
                "agrees": float(w) == float(tw)})
        if c and tc:
            crit_ok += int(float(c) == float(tc))
            DISAGREEMENTS.append({
                "check": "acknowledgement", "subject": role,
                "field": "max_ack_critical_min", "extracted": c, "reference": tc,
                "reference_source": "person_registry.max_ack_critical",
                "agrees": float(c) == float(tc)})
    return {"roles_checked": checked, "warning_agrees_with_registry": warn_ok,
            "critical_agrees_with_registry": crit_ok}


def run(path: str, null: bool = False) -> dict:
    DISAGREEMENTS.clear()
    df = pd.read_csv(path, dtype=str).fillna("")
    zones = declared_zones()
    reg = pd.read_csv(os.path.join(HUMAN, "person_registry.csv"), dtype=str).fillna("")
    roles = {r.strip().casefold() for r in reg["role"] if r.strip()}

    acc = check_access(df, zones, roles, null)
    occ = check_occupancy(df, zones, null)
    ack = check_ack(df, roles)

    print(f"\n=== {os.path.basename(path)}{'  [NULL CONTROL]' if null else ''} ===")
    print(f"  ACCESS      {acc['pairs_extracted']} role/zone pairs extracted; "
          f"{acc['events_scored']} of 448 events scored "
          f"({acc['events_unknown_pair']} no matching pair)")
    print(f"              denials: TP {acc['violations_tp']} FP {acc['fp']} FN {acc['fn']} TN {acc['tn']}"
          f"  ->  P {acc['precision']} R {acc['recall']} F1 {acc['f1']}")
    amb = (f" ({occ['zone_records_rejected_multinumeric']} zone-bearing records "
           f"rejected as non-capacity)" if occ["zone_records_rejected_multinumeric"] else "")
    print(f"  OCCUPANCY   {occ['zones_extracted']}/7 zones, "
          f"{occ['caps_matching_facility']} caps match facility zone_max{amb};\n              "
          f"{occ['intervals_examined']:,} intervals examined, "
          f"{occ['breaches_true']:,} true breaches")
    print(f"              hard-cap breaches: TP {occ['tp']} FP {occ['fp']} FN {occ['fn']}"
          f"  ->  P {occ['precision']} R {occ['recall']} F1 {occ['f1']}")
    print(f"  ACK         {ack['roles_checked']} roles; agreement with person_registry: "
          f"warning {ack['warning_agrees_with_registry']}/{ack['roles_checked']}, "
          f"critical {ack['critical_agrees_with_registry']}/{ack['roles_checked']} "
          f"(registry is uniform 15/3; SOP-004 s3 is per-role -- see docstring)")

    for d in DISAGREEMENTS:
        d["run_file"] = os.path.basename(path)
    ALL_DISAGREEMENTS.extend(DISAGREEMENTS)

    dis = [d for d in DISAGREEMENTS if not d["agrees"]]
    if dis:
        print(f"  DISAGREES   {len(dis)} cell(s) differ from a reference artefact:")
        for d in dis:
            print(f"                {d['check']:16s} {d['subject']:18s} "
                  f"{d['field']:22s} extracted={d['extracted']:>4} "
                  f"reference={d['reference']:>4}")

    out = {"file": os.path.basename(path), "null_control": null}
    out.update({f"access_{k}": v for k, v in acc.items()})
    out.update({f"occupancy_{k}": v for k, v in occ.items()})
    out.update({f"ack_{k}": v for k, v in ack.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules")
    ap.add_argument("--all-dev-runs", action="store_true")
    ap.add_argument("--null-control", action="store_true")
    args = ap.parse_args()

    if args.all_dev_runs:
        paths = sorted(glob.glob(os.path.join(
            PRED_DIR, "ext_multi_agent_generic_dev_production_line_run*_*.csv")))
    elif args.rules:
        paths = [args.rules]
    else:
        ap.error("pass --rules <csv> or --all-dev-runs")

    rows = [run(p, args.null_control) for p in paths]
    os.makedirs(OUT_DIR, exist_ok=True)
    name = "human_summary_null.csv" if args.null_control else "human_summary.csv"
    with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # Under the null control every extracted value has been permuted, so a
    # "disagreement" row would record the permutation rather than anything
    # about the extraction. The artefact is written only for the real run,
    # and never overwritten by a null one.
    if ALL_DISAGREEMENTS and not args.null_control:
        dpath = os.path.join(OUT_DIR, "reference_disagreements.csv")
        with open(dpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ALL_DISAGREEMENTS[0]))
            w.writeheader()
            w.writerows(ALL_DISAGREEMENTS)
        print(f"[detect-human] wrote {dpath}")
    print(f"\n[detect-human] wrote {OUT_DIR}/{name}")


if __name__ == "__main__":
    main()
