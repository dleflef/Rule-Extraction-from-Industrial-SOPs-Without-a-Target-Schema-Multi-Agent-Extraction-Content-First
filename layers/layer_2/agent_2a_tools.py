# layer_2/agent_2a_tools.py — Agent 2A Entity Extraction Tool
#
# Purpose:
#   This module provides the core tool for Agent 2A: calling an LLM to extract
#   structured operational rules from a single text chunk.  The chunk comes from
#   Agent 1B's output (clean markdown with section context).  The prompt is
#   fully domain-agnostic; all domain knowledge is injected via the seed node
#   table.  Rule IDs are never created by the LLM — they are assigned
#   deterministically afterwards to avoid hallucination.

import json
import os
import re
from typing import Any, Dict, List, Optional

# ── Seed node loading ─────────────────────────────────────────────────────────
# The seed node CSV (nodes_factory.csv) contains the official KG node registry.
# We load it and format it as a markdown table for the prompt, so the LLM knows
# which exact station/sensor names to use.

def load_seed_nodes(csv_path: str) -> List[Dict[str, str]]:
    # Import csv locally; it's only needed here.
    import csv
    nodes = []
    # Open the CSV with utf-8 encoding; newline='' is required for CSV reading.
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes.append(row)   # each row is a dict with keys nodeId, label, name, type, zone
    return nodes

def format_seed_nodes_to_string(nodes: List[Dict[str, str]]) -> str:
    # Build a markdown table with a header and separator line.
    header = "| nodeId | label | name | type | zone |"
    sep = "|--------|-------|------|------|------|"
    rows = []
    for n in nodes:
        rows.append(f"| {n['nodeId']} | {n['label']} | {n['name']} | {n['type']} | {n['zone']} |")
    # Join all parts with newlines.
    return "\n".join([header, sep] + rows)

# ── Prompt template ───────────────────────────────────────────────────────────
# This is the system prompt sent to the LLM. It defines:
# - the output JSON schema (14 fields)
# - classification rules for the four rule types
# - entity linking instructions (station, sensor)
# - condition/action splitting heuristic
# - numeric threshold extraction rules
# - handling of special table layouts (threshold, maintenance, access, occupancy)
# - many numbered rules to enforce precise behaviour.
# The prompt is kept constant for every chunk; only the user prompt changes.

EXTRACTION_SYSTEM_PROMPT = """
You are an entity extractor for an industrial knowledge graph.
Read a Markdown chunk from a technical document and extract every operational rule into a strict JSON array.

Each rule MUST contain ALL 14 fields (use null for any that are missing):
  ruleId, class, station, sensor, sensorType, condition, action,
  severity, critHi, warnHi, critLo, warnLo, unit, source

Field rules:
- ruleId: set to null unless an explicit ID already appears in the text (e.g. "MAINT-01", "RULE-ST02-04").
- class: exactly one of: OperationalRule, ThresholdRule, AccessRule, MaintenanceRule.
  · ThresholdRule: a rule defined by numeric sensor limits (warn/crit bounds).
  · OperationalRule: a process rule with a condition and a required action but no sensor thresholds.
  · MaintenanceRule: a rule triggered by cumulative degradation (drift, stuck sensor, wear pattern) that prescribes a maintenance action (inspect, replace, calibrate, schedule). Use this class whenever the rule describes a degradation pattern → maintenance response, even if it also contains numeric drift limits.
  · AccessRule: a rule governing entry, authorisation, or role-based access to areas or systems.
- station: the component or zone this rule applies to. Use the EXACT name from the Known Seed Nodes list when a match exists. If the chunk contains a "## Section:" line, use that section name as the station for all rules in this chunk unless the rule text explicitly names a different station. If the chunk heading ends with the word "Thresholds" or "Rules", the station is the text before that word (e.g. heading "XYZ_Station Thresholds" → station = "XYZ_Station"). If a rule applies globally and is not tied to any specific station or zone, set station to null.
- sensor: the sensor involved. Use the EXACT name from the Known Seed Nodes list when a match exists. If no matching sensor is found in the seed list, construct the sensor name as '{station}_{sensorType}' using the EXACT station and sensorType values already determined for this rule (e.g. if station="ST02" and sensorType="TMP", sensor="ST02_TMP"). Never output a bare sensor type (e.g. "VIB", "TMP") when a seed-table name exists for that sensor. Never invent a sensor name that diverges from this pattern.
- sensorType: abbreviated form of the sensor name inferred from the text (e.g., TMP for temperature, PRS for pressure). Null if uncertain.
- condition: the trigger or condition described. For a prose rule, the text BEFORE the final colon or dash is the condition, the text AFTER is the action. Never put a maintenance statement (e.g. "inspect", "replace", "calibrate") into the condition field. action: the response or corrective measure. Never swap them.
  For prose rules that do NOT contain a colon or dash, the entire statement is the condition; the default action is "Inspect and notify maintenance". But if the text contains a dash "–" or "—", split the sentence at that dash: text before = condition, text after = action.
  Note: this is a heuristic; it will help but not cover every case.
- severity: one of MANDATORY, WARNING, CRITICAL, HIGH, MEDIUM. When a severity keyword (WARNING, CRITICAL, MANDATORY, HIGH, MEDIUM) appears anywhere in the rule text, you MUST set the severity field to that keyword. Null only if no severity keyword is present.
- Numeric thresholds (critHi, warnHi, critLo, warnLo): floats extracted from the text. Null if not present. When a prose rule states a specific numeric limit (e.g. "exceeds 26°C", "must not drop below 5 bar"), place that number in the appropriate threshold field: a value paired with "exceeds", "above", "greater than", ">" → warnHi or critHi; a value paired with "below", "less than", "drops under", "<" → warnLo or critLo. Use the severity keyword (CRITICAL/WARNING) to decide crit vs. warn. If only upper limits are given, leave lower limits as null — do NOT invent symmetric low values. If a rule states a parameter MUST remain between Lo and Hi (e.g. "must remain between 18 and 26°C"), set warnLo=Lo and warnHi=Hi; do NOT duplicate the same value into both critHi and warnHi. Only populate critHi/critLo when a SEPARATE CRITICAL threshold is explicitly stated.
- unit: the measurement unit as it appears in the text (e.g., °C, bar, %RH, m/s, persons). Null if absent.
- source: the document ID from the first heading of the chunk (e.g. 'SOP-001', 'SOP-002'). It must match the pattern 'SOP-NNN' or a similar document-level code. Do NOT use a section heading (e.g. "Operating Procedures"), a rule ID (e.g. "RULE-ST02-04"), or any other non-document string as the source. Scan the chunk headings and opening line; if a document-level code appears, you MUST populate source with it. Never leave source null when a document code is present.

RULES:
1. Only extract rules that are explicitly written in the text. Do NOT invent conditions, actions, or numeric thresholds — only record values that are literally present (e.g., "> 210°C", "MUST be logged"). You may ONLY create a rule for a sensor if that sensor is **explicitly mentioned in the chunk text** – you must never scan the seed table to generate rules for sensors that do not appear in the chunk. If the chunk only contains a generic heading like "Alarm Thresholds", do NOT generate rules for all possible sensors. If no rules exist in the chunk, return [].
2. If no numeric thresholds AND no explicit condition-action pair are present, return [] and DO NOT output any placeholder objects. If the chunk contains ONLY a document title, revision number, author name, or a high-level section heading without any rule text, you MUST return [].
   For example, a chunk that contains only a document header, a revision line, and a generic statement about the threshold hierarchy with absolutely no concrete numerical values or specific sensor names must yield EXACTLY [].
   Another example: a chunk whose only content is a structural table that merely lists station IDs, zones, and dependency descriptions without any measurable limits or actions yields exactly [] – do not invent a rule for it.
   This rule takes precedence over any other instruction: if you cannot SEE a specific numerical value AND a condition-action pair, you MUST output [].
3. Station assignment: if the chunk contains a "## Section:" line, use that section name for the station field of every rule in this chunk. Otherwise, if the chunk heading names a station or zone (or ends with "Thresholds"/"Rules"), derive the station from the heading. Only fall back to seed-node names when neither is present. If a rule applies to all areas with no specific station mentioned, set station to null.
4. For threshold tables: read column headers carefully before mapping values. The typical column order is CRIT_LO | WARN_LO | Nominal | WARN_HI | CRIT_HI. If a header cell is empty, infer its role from position between its neighbours — a blank cell between CRIT_LO and WARN_LONominal holds the WARN_LO value. Never swap high and low: critLo < warnLo < Nominal < warnHi < critHi.
   Example (made-up numbers, showing the pattern):
     | Sensor | Unit | CRIT_LO |      | WARN_LONominal | WARN_HI | CRIT_HI |
     | HTR    | °C   |  15.0   | 18.0 |     22.0       |  26.0   |  30.0   |
   Correct extraction: critLo=15.0, warnLo=18.0, nominal=22.0, warnHi=26.0, critHi=30.0
5. For prose rules (e.g. "RULE-ST02-04: CUR drift… monitor SPD" or "RULE-ACCESS-01: Entry to Zone X MUST be logged by the Supervisor role"): extract condition and action directly from the sentence. Authorisation, access, and maintenance rules follow the same extraction pattern as threshold rules. If a numeric limit appears in the prose (e.g. "if TMP exceeds 26°C"), also populate the corresponding threshold field.
6. If a chunk contains multiple explicit rule-IDs (e.g. "RULE-ACCESS-03: …" followed by "RULE-ACCESS-04: …"), you MUST extract EACH as a separate JSON object in the array. Never merge two rule-IDs into one object.
7. Tables that define role-based timing constraints are AccessRules. If a table has columns of the form "Role | … WARNING … (min) | … CRITICAL … (min)" (or similar severity-labelled timing columns), extract one AccessRule per data row: condition = the role and triggering event from that row, action = the required response within the time limit, warnHi = the WARNING time limit as a float, critHi = the CRITICAL time limit as a float (smaller deadline = stricter, so critHi ≤ warnHi is expected). Example with made-up values:
     | Role    | Max Ack — WARNING (min) | Max Ack — CRITICAL (min) |
     | TypeA   | 10                      | 5                        |
   → condition: "TypeA role receives WARNING alarm", action: "must acknowledge within limit", warnHi=10.0, critHi=5.0

8. Severity from column headers: If a table column header contains the text 'CRITICAL' (e.g. 'CRITICAL Response', 'CRITICAL Action'), treat cell values in that column as implying severity='CRITICAL'. If the cell text contains 'E-STOP' or 'EMERGENCY STOP', also set severity='CRITICAL'. If a header contains 'WARNING', set severity='WARNING'. Only apply this when no explicit severity keyword is already present in the rule text itself.
9. Rule splitting: If a single prose sentence or table row contains two distinct numeric limits each paired with a different severity label (e.g. "below 110 … WARNING" and "below 100 … CRITICAL"), extract TWO separate rule objects — one per limit+severity pair. Each object gets its own threshold field and severity. Do NOT merge them into a single object with both threshold values.
10. Occupancy and zone-capacity tables: If a table has columns like 'Zone'/'Area', 'Max Persons'/'Capacity', 'Enforcement'/'Action', extract one AccessRule per data row. The first column gives the zone name — you MUST use it as the station. Do NOT leave station null.
   Example (MADE-UP):
     | Zone             | Max Persons | Enforcement                  |
     | Production Area  | 12          | WARNING above 10; CRITICAL … |
   → station = "Production Area", condition = "WARNING above 10", critHi = 12.0, severity = "CRITICAL", unit = "persons".
11. Malformed table rows: If a table row does not contain a recognisable Rule ID, sensor name, numeric value, or condition-action pair — for example a row that contains only metadata labels like 'Station ID | Zone | Depends On | Description' — skip it entirely and do not generate a rule object for it.
12. Role-based timing constraints: If a table row lists separate time limits for WARNING and CRITICAL (e.g. columns "Max Ack – WARNING (min)" and "Max Ack – CRITICAL (min)"), produce TWO AccessRule objects per row:
    - Rule 1: severity = "WARNING", condition = "… role receives WARNING alarm", action = "must acknowledge within limit", warnHi = WARNING time (float).
    - Rule 2: severity = "CRITICAL", condition = "… role receives CRITICAL alarm", action = "must acknowledge within limit", critHi = CRITICAL time (float).
    Do NOT merge the two time limits into one object.

Return ONLY a valid JSON array. No markdown fences, no commentary.
""".strip()   # strip leading/trailing whitespace so it fits cleanly into the API call

def build_user_prompt(chunk_content: str, headings: list, seed_table: str) -> str:
    # Build the user message: seed table, chunk headings, and the chunk text itself.
    heading_str = ", ".join(headings) if headings else "none"
    return (
        f"## Known Seed Nodes\n{seed_table}\n\n"
        f"## Chunk Headings\n{heading_str}\n\n"
        f"## Chunk Text\n{chunk_content}\n\n"
        f"Extract all rules as a JSON array."
    )

# ── Deterministic ruleId post-processor ──────────────────────────────────────
# After the LLM returns rules with ruleId=null (or an explicit ID from text),
# we assign deterministic IDs based on rule class, station, sensor type, severity.
# Global counters (a dict) are passed between calls to avoid duplicate IDs
# across different chunks (e.g., RULE-ACC-01 should only appear once overall).

def _zone_abbreviation(station_name: str, seed_nodes: List[Dict[str, str]]) -> str:
    # Try to find a short abbreviation for a zone name using the seed node's nodeId.
    # E.g., "Production Area" might map to nodeId "N0002" → abbreviation "N000".
    # If not found, fall back to an acronym from the name itself.
    station_lower = station_name.lower().strip()
    for node in seed_nodes:
        if node.get("name", "").lower().strip() == station_lower:
            node_id = node.get("nodeId", "")
            # Keep up to 4 uppercase alphanumeric characters from the nodeId.
            prefix = re.sub(r'[^A-Z0-9]', '', node_id.upper())[:4]
            if prefix:
                return prefix
    # Fallback: clean the station name, take first 6 alphanumeric characters.
    return re.sub(r'[^A-Z0-9]', '', station_name.upper())[:6] or "ZONE"


def assign_rule_ids(
    rules: List[Dict[str, Any]],
    seed_nodes: Optional[List[Dict[str, str]]] = None,
    _global_counters: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Deterministically assigns ruleId to any rule where it is null.
    Called after LLM extraction so the LLM never has to construct IDs.
    seed_nodes is used to derive zone abbreviations from nodeIds.
    Pass _global_counters (a mutable dict) to share sequence state across chunks,
    preventing duplicate IDs like RULE-ACC-01 appearing in two different chunks.
    """
    _seed = seed_nodes or []
    # Use the passed mutable counters, or start fresh (should not happen in practice).
    counters: Dict[str, int] = _global_counters if _global_counters is not None else {}

    # Helper: increment a counter for a key (e.g., "ACC", "OP-ST01_FILLING")
    # and return a zero-padded two-digit sequence number.
    def _seq(key: str) -> str:
        counters[key] = counters.get(key, 0) + 1
        return f"{counters[key]:02d}"

    for rule in rules:
        # If the LLM already gave a ruleId (e.g., from text like "RULE-ST01-01"), skip.
        if rule.get("ruleId"):
            continue

        # Extract relevant fields for ID generation.
        cls = rule.get("class", "")
        station = (rule.get("station") or "").strip()
        sensor_type = (rule.get("sensorType") or "").strip()
        severity = (rule.get("severity") or "").strip()
        action = (rule.get("action") or "").strip()
        condition = (rule.get("condition") or "").strip()

        # Branch based on rule class.
        if cls == "ThresholdRule":
            if station and sensor_type:
                # e.g., RULE-THR-ST01_FILLING-TMP-CRIT
                # Replace spaces in station name with underscores.
                st = re.sub(r'\s+', '_', station)
                # Use first 4 uppercase chars of severity as a tag; if missing, use a sequence.
                sev_tag = severity[:4].upper() if severity else _seq(f'THR-{st}-{sensor_type}')
                rule["ruleId"] = f"RULE-THR-{st}-{sensor_type}-{sev_tag}"
            else:
                # If no station/sensor, it might be an anomaly definition (RULE-ANOM-...).
                # Look for AnomalyEvent::Type in the action string.
                m = re.search(r'AnomalyEvent::(\w+)', action)
                anom_type = m.group(1).upper() if m else _seq('ANOM')
                rule["ruleId"] = f"RULE-ANOM-{anom_type}"

        elif cls == "AccessRule":
            # Check if it's an acknowledgment rule (condition contains "acknowledgment").
            if "acknowledgment" in condition.lower():
                # Extract the role name (first word before "role").
                role_m = re.match(r'(\w+)\s+role', condition, re.IGNORECASE)
                prefix = role_m.group(1).upper()[:3] if role_m else "ROL"
                # Severity abbreviation: CRIT or WARN.
                sev_abbrev = "CRIT" if "CRITICAL" in severity.upper() else "WARN"
                rule["ruleId"] = f"RULE-ACK-{prefix}-{sev_abbrev}"
            elif station:
                # Occupancy rule: use zone abbreviation.
                rule["ruleId"] = f"RULE-OCC-{_zone_abbreviation(station, _seed)}"
            else:
                # Generic access rule (e.g., RULE-ACCESS-01).
                rule["ruleId"] = f"RULE-ACC-{_seq('ACC')}"

        elif cls == "MaintenanceRule":
            # e.g., RULE-MAINT-ST04_PACKAGING-01
            st = re.sub(r'\s+', '_', station) if station else "GEN"
            rule["ruleId"] = f"RULE-MAINT-{st}-{_seq(f'MAINT-{st}')}"

        elif cls == "OperationalRule":
            # e.g., RULE-OP-ST01_FILLING-01
            st = re.sub(r'\s+', '_', station) if station else "GEN"
            rule["ruleId"] = f"RULE-OP-{st}-{_seq(f'OP-{st}')}"

        else:
            # Fallback for any unrecognised class.
            rule["ruleId"] = f"RULE-MISC-{_seq('MISC')}"

    return rules


# ── JSON repair ───────────────────────────────────────────────────────────────
# Simple regex repair that removes trailing commas immediately before '}' or ']'.
# This fixes the most common LLM JSON syntax error without needing a full parser.

def _repair_json(raw: str) -> str:
    """Remove trailing commas before } or ] — the most common LLM JSON mistake."""
    return re.sub(r',\s*([\}\]])', r'\1', raw)


# ── LLM call ──────────────────────────────────────────────────────────────────
# Main function: loads seed nodes, builds prompts, calls the LLM, attempts JSON
# parsing with fallback repair, and then assigns rule IDs before returning.

def extract_rules_from_chunk(
    chunk_content: str,
    headings: List[str],
    seed_nodes_csv: str = "layers/data/seed_rules/dataset/kg_seeds/nodes_factory.csv",
    model_name: str = "qwen2.5-7b-instruct",
    temperature: float = 0.0,
    global_counters: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    # 1. Load and format seed nodes (optional but strongly recommended).
    if not os.path.exists(seed_nodes_csv):
        nodes = []
        # Provide a minimal table if the file is missing, so the prompt still works.
        seed_table = (
            "| nodeId | label | name | type | zone |\n"
            "|--------|-------|------|------|------|"
        )
    else:
        nodes = load_seed_nodes(seed_nodes_csv)
        seed_table = format_seed_nodes_to_string(nodes)

    # Build the full user prompt combining chunk content, headings, seed table.
    user_prompt = build_user_prompt(chunk_content, headings, seed_table)

    # 2. Call LLM (local server via OpenAI-compatible API).
    try:
        import openai   # local import; assume installed in environment
        client = openai.OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,   # 0.0 for deterministic output
        )
        raw = completion.choices[0].message.content

        # 3. Strip markdown fences (```json ... ```) if the LLM wrapped its output.
        def _strip_fences(text: str) -> str:
            text = text.strip()
            if text.startswith("```"):
                # Remove the opening fence (with optional language specifier).
                text = re.sub(r"^```(?:json)?", "", text)
                # Remove the closing fence.
                text = re.sub(r"```$", "", text)
            return text.strip()

        cleaned = _strip_fences(raw)

        # 4. Parse JSON with a fallback chain.
        parsed = None
        # Attempt 1: raw cleaned string.
        # Attempt 2: after repairing trailing commas.
        for attempt, candidate in enumerate([cleaned, _repair_json(cleaned)]):
            try:
                parsed = json.loads(candidate)
                break   # success, exit loop
            except json.JSONDecodeError:
                if attempt == 0:
                    continue   # try the repaired version
                # If even the repaired version fails, ask the LLM to fix its own output.
                fix_completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You output only valid JSON arrays. Fix the broken JSON below and return it, nothing else."},
                        {"role": "user", "content": candidate},
                    ],
                    temperature=0.0,
                )
                try:
                    parsed = json.loads(_strip_fences(fix_completion.choices[0].message.content))
                except json.JSONDecodeError as e2:
                    print(f"[Agent 2A] JSON repair failed: {e2}")

        # 5. Validate the parsed result format and assign rule IDs.
        if isinstance(parsed, list):
            return assign_rule_ids(parsed, nodes, _global_counters=global_counters)
        # Some LLMs might return an object with a "rules" key.
        if isinstance(parsed, dict) and "rules" in parsed:
            return assign_rule_ids(parsed["rules"], nodes, _global_counters=global_counters)
        return []
    except Exception as e:
        print(f"[Agent 2A] Extraction error: {e}")
        return []