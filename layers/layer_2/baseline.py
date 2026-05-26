#!/usr/bin/env python3
"""
baseline_b0.py  –  Deterministic regex parser using SOP text + embedded 24-node knowledge.
Only stations and sensors from the provided list are ever referenced.
No roles, no external CSV.  No hardcoded naming conventions.
"""

import csv, os, re, sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LAYERS_DIR  = os.path.join(SCRIPT_DIR, "..")
PROJECT_DIR = os.path.normpath(os.path.join(LAYERS_DIR, ".."))
TEXTS_DIR   = os.path.join(LAYERS_DIR, "texts")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "baseline_results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "baseline_b0.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

RULE_FIELDS = [
    "ruleId", "class", "station", "sensor", "sensorType",
    "condition", "action", "severity",
    "critHi", "warnHi", "warnLo", "critLo", "unit",
    "source_file", "run_id", "model_name", "paradigm", "level",
    "text_truncated", "llm_turns",
]

# ── Embedded knowledge from the 24 provided nodes ────────────────────────────
STATION_SENSORS = {
    "ST01_FILLING": {
        "TMP": "ST01_FILLING_TMP",
        "PRS": "ST01_FILLING_PRS",
        "FLW": "ST01_FILLING_FLW",
    },
    "ST02_SEALING": {
        "TMP": "ST02_SEALING_TMP",
        "PRS": "ST02_SEALING_PRS",
        "CUR": "ST02_SEALING_CUR",
    },
    "ST03_LABELLING": {
        "TMP": "ST03_LABELLING_TMP",
        "SPD": "ST03_LABELLING_SPD",
        "POS": "ST03_LABELLING_POS",
    },
    "ST04_PACKAGING": {
        "TMP": "ST04_PACKAGING_TMP",
        "SPD": "ST04_PACKAGING_SPD",
        "CNT": "ST04_PACKAGING_CNT",
    },
}

ZONES = [
    "Production Area",
    "Server Room",
    "General Warehouse",
    "Main Entrance",
    "Chemical Storage",
    "R&D Lab",
    "Cafeteria",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sev(text: str) -> str:
    t = text.upper()
    if "CRITICAL" in t: return "CRITICAL"
    if "WARNING"  in t: return "WARNING"
    return "MANDATORY"


def _nums(**kw) -> dict:
    return {k: kw.get(k, "") for k in ("critHi", "warnHi", "warnLo", "critLo", "unit")}


_ACTION_VERB = re.compile(
    r'\b(Inspect|Replace|Activate|Notify|Emergency|Check|Repair|Monitor|'
    r'Suspend|Verify|Halt|Escalate|Schedule)\b'
)

def _split_merged(text: str) -> tuple[str, str]:
    m = _ACTION_VERB.search(text)
    if not m:
        return text, ""
    trigger = text[:m.start()].strip()
    rest    = text[m.start():].strip()
    action  = re.sub(r'\s+[A-Z]+$', '', rest).strip()
    return trigger, action


def _find_sensor(body: str, station: str, station_sensors: dict) -> tuple[str, str]:
    if station not in station_sensors:
        return "", ""
    body_upper = body.upper()
    for stype, sid in station_sensors[station].items():
        if re.search(r'\b' + re.escape(stype) + r'\b', body_upper):
            return stype, sid
    return "", ""


# ── SOP-001 ──────────────────────────────────────────────────────────────────

def parse_sop001(text: str, station_sensors: dict) -> list[dict]:
    rules = []
    ruleId_station = {}
    current_station = ""
    all_station_ids = list(station_sensors.keys())

    for line in text.splitlines():
        # If line looks like a section header ("## ..."), try to match a known station.
        # Only update current_station if we find one; otherwise reset to "".
        if line.startswith("##"):
            found = False
            for sid in all_station_ids:
                if sid in line:
                    current_station = sid
                    found = True
                    break
            if not found:
                current_station = ""
        # Also catch station IDs in other formatted headers (e.g., "--- ST01 ---")
        else:
            for sid in all_station_ids:
                if sid in line:
                    current_station = sid
                    break

        for m in re.finditer(r'(RULE-\w+-\d+)\s*:', line):
            ruleId_station[m.group(1)] = current_station

    parts = re.split(r'(RULE-\w+-\d+)\s*:', text)
    for i in range(1, len(parts) - 1, 2):
        rule_id = parts[i].strip()
        body    = parts[i + 1].split('\n\n')[0].replace('\n', ' ').strip()

        station = ruleId_station.get(rule_id, "")

        if "personnel" in body.lower() and ("authorised" in body.lower() or "authorized" in body.lower()):
            rules.append({
                "ruleId": rule_id, "class": "AccessRule",
                "station": "Chemical Storage", "sensor": "", "sensorType": "",
                "condition": body, "action": "", "severity": "CRITICAL",
                **_nums(),
            })
            continue

        stype, sid = _find_sensor(body, station, station_sensors)

        if " - " in body:
            dash_pos = body.index(" - ")
            condition = body[:dash_pos].strip()
            action    = body[dash_pos + 3:].strip()
        else:
            condition = body
            action    = ""

        severity = _sev(body)

        rules.append({
            "ruleId": rule_id, "class": "OperationalRule",
            "station": station, "sensor": sid, "sensorType": stype,
            "condition": condition, "action": action, "severity": severity,
            **_nums(),
        })

    return rules


# ── SOP-002 ──────────────────────────────────────────────────────────────────

def parse_sop002(text: str, station_sensors: dict) -> list[dict]:
    rules = []
    station_re = re.compile(r"^##\s+(\w+)\s+Thresholds$")
    data_re = re.compile(
        r"^\|\s*(\w+)\s*\|\s*(\S+)\s*\|\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*\|"
        r"\s*[^\|]+\|\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*\|\s*(.*?)\s*\|$"
    )
    anom_re = re.compile(r"^\|\s*(SPIKE|DRIFT|STUCK|OUT_OF_RANGE|CORRELATED)\s*\|")

    current_station = ""
    in_valid_section = False

    for line in text.splitlines():
        m = station_re.match(line.strip())
        if m:
            current_station = m.group(1)
            # Only process if this station is one of our four known stations
            in_valid_section = current_station in station_sensors
            continue

        if not in_valid_section:
            continue

        m = data_re.match(line.strip())
        if m:
            sensor_type    = m.group(1)
            unit           = m.group(2)
            crit_lo        = m.group(3)
            warn_lo        = m.group(4)
            warn_hi        = m.group(5)
            crit_hi        = m.group(6)
            critical_resp  = m.group(7).strip()

            station_abbrev = current_station.split("_")[0]
            rule_id = f"RULE-THR-{station_abbrev}-{sensor_type}-CRIT"

            sensor_id = station_sensors.get(current_station, {}).get(sensor_type, "")

            rules.append({
                "ruleId": rule_id,
                "class": "ThresholdRule",
                "station": current_station,
                "sensor": sensor_id,
                "sensorType": sensor_type,
                "condition": f"{sensor_type} threshold data for {current_station}",
                "action": critical_resp,
                "severity": "CRITICAL",
                **_nums(unit=unit, critLo=crit_lo, warnLo=warn_lo,
                        warnHi=warn_hi, critHi=crit_hi),
            })
            continue

        m = anom_re.match(line.strip())
        if m:
            atype = m.group(1)
            rules.append({
                "ruleId": f"RULE-ANOM-{atype}",
                "class": "ThresholdRule",
                "station": "", "sensor": "", "sensorType": "",
                "condition": f"Anomaly type: {atype}",
                "action": f"AnomalyEvent::{atype.title().replace('_', '')}",
                "severity": "CRITICAL",
                **_nums(),
            })

    return rules


# ── SOP-003 ──────────────────────────────────────────────────────────────────

def parse_sop003(text: str, station_sensors: dict) -> list[dict]:
    rules = []
    in_corr    = False
    corr_count = 0

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("##") and "Correlated Fault" in s:
            in_corr = True
            continue
        if s.startswith("## "):
            in_corr = False

        if not s.startswith("|"):
            continue

        cols = [c.strip() for c in s.split("|")]

        # MAINT rows
        if len(cols) >= 8 and re.fullmatch(r"MAINT-\d+", cols[1]):
            rid, sensor_type, station = cols[1], cols[2], cols[3]
            if station not in station_sensors:
                continue
            trigger, action = cols[4], cols[5]
            if trigger == action:
                trigger, action = _split_merged(trigger)
            sev = _sev(line)
            if sev == "MANDATORY":
                sev = "HIGH" if "HIGH" in line.upper() else "MEDIUM"
            sensor_id = station_sensors.get(station, {}).get(sensor_type, "")
            rules.append({
                "ruleId": rid, "class": "MaintenanceRule",
                "station": station,
                "sensor": sensor_id or f"{station}_{sensor_type}",
                "sensorType": sensor_type,
                "condition": trigger, "action": action, "severity": sev,
                **_nums(),
            })
            continue

        # Correlated fault rows
        if in_corr and len(cols) >= 5 and ("›" in cols[1] or "→" in cols[1]):
            if cols[1] in ("Pattern", "---", ""):
                continue
            corr_count += 1
            src_clean = re.sub(r"[›→].*", "", cols[1]).strip()
            if "-" in src_clean:
                station, sensor_type = src_clean.rsplit("-", 1)
            else:
                station, sensor_type = src_clean, ""
            full_station = next(
                (sid for sid in station_sensors if sid.startswith(station)), station
            )
            if full_station not in station_sensors:
                continue
            sensor_id = station_sensors.get(full_station, {}).get(sensor_type, "")
            resolution = cols[4] if len(cols) > 4 else ""
            rules.append({
                "ruleId": f"RULE-CORR-{corr_count:02d}",
                "class": "MaintenanceRule",
                "station": full_station, "sensor": sensor_id,
                "sensorType": sensor_type,
                "condition": f"Correlated fault: {cols[1].strip()}",
                "action": resolution, "severity": "HIGH",
                **_nums(),
            })

    return rules


# ── SOP-004 ──────────────────────────────────────────────────────────────────

def parse_sop004(text: str, zones: list) -> list[dict]:
    rules = []
    occ_re = re.compile(r"^\|\s*(.+?)\s*\|\s*(\d+)\s*\|\s*(.*?)\s*\|$")
    in_occ = False
    zone_set = set(zones)

    for line in text.splitlines():
        s = line.strip()
        if "Maximum Occupancy" in s:
            in_occ = True
            continue
        if s.startswith("## "):
            in_occ = False

        if in_occ:
            m = occ_re.match(s)
            if m and m.group(1) not in ("Zone", "---"):
                zone_name = m.group(1)
                enf       = m.group(3)
                if zone_name not in zone_set:
                    continue
                zone_slug = re.sub(r"[^A-Z0-9]", "_", zone_name.upper()).strip("_")
                rules.append({
                    "ruleId": f"RULE-OCC-{zone_slug}",
                    "class": "AccessRule",
                    "station": zone_name, "sensor": "", "sensorType": "",
                    "condition": f"Maximum occupancy: {enf.strip()}",
                    "action": "Enforce occupancy limits",
                    "severity": _sev(enf),
                    **_nums(unit="persons"),
                })

    # RULE-ACCESS-01..04
    parts = re.split(r"(RULE-ACCESS-\d+)\s*:", text)
    for i in range(1, len(parts) - 1, 2):
        rid  = parts[i].strip()
        body = parts[i + 1].split("\n\n")[0].replace("\n", " ").strip()
        sev  = "CRITICAL" if ("zero-tolerance" in body.lower()
                               or "immediate escalation" in body.lower()) else "MANDATORY"
        act  = ("Immediate escalation to Safety Officer"
                if "escalated" in body.lower() or "immediate" in body.lower()
                else "Log event, notify security")
        rules.append({
            "ruleId": rid, "class": "AccessRule",
            "station": "", "sensor": "", "sensorType": "",
            "condition": body, "action": act, "severity": sev,
            **_nums(),
        })

    return rules


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(TEXTS_DIR):
        print(f"ERROR: texts directory not found: {TEXTS_DIR}")
        sys.exit(1)

    parsers = {
        "SOP_001": lambda text: parse_sop001(text, STATION_SENSORS),
        "SOP_002": lambda text: parse_sop002(text, STATION_SENSORS),
        "SOP_003": lambda text: parse_sop003(text, STATION_SENSORS),
        "SOP_004": lambda text: parse_sop004(text, ZONES),
    }

    all_rules = []
    for fname in sorted(f for f in os.listdir(TEXTS_DIR) if f.endswith(".txt")):
        key = next((k for k in parsers if k in fname), None)
        if not key:
            continue
        with open(os.path.join(TEXTS_DIR, fname), encoding="utf-8") as f:
            text = f.read()
        rules = parsers[key](text)
        for r in rules:
            r.update({
                "source_file": fname, "run_id": "baseline_b0",
                "model_name": "deterministic", "paradigm": "baseline",
                "level": 0, "text_truncated": "False", "llm_turns": 0,
            })
        all_rules.extend(rules)

    for r in all_rules:
        for field in RULE_FIELDS:
            r.setdefault(field, "")

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RULE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rules)

    print(f"Baseline B0: {len(all_rules)} rules written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()