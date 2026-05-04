# layer_2/agent_2a_tools.py — Agent 2A Entity Extraction Tool
#
# Purpose:
#   This module provides the core tool for Agent 2A: calling an LLM to extract
#   structured operational rules from a single text chunk.  The chunk comes from
#   Agent 1B's output (raw, un‑interpreted Markdown – no table repair, no
#   section injection).  The prompt is fully domain-agnostic; all domain knowledge
#   is injected via the seed node table.  Rule IDs are never created by the LLM —
#   they are assigned deterministically afterwards to avoid hallucination.

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
Read a text chunk from a technical document — it may be prose, a pipe-delimited table, a bulleted list, or any other format — and extract every operational rule into a strict JSON array.

Each rule MUST contain ALL 14 fields (use null for any that are missing):
  ruleId, class, station, sensor, sensorType, condition, action,
  severity, critHi, warnHi, critLo, warnLo, unit, source

Field rules:
- ruleId: set to null unless an explicit ID already appears in the text (e.g. "MAINT-01", "RULE-ST02-04").
- class: exactly one of: OperationalRule, ThresholdRule, AccessRule, MaintenanceRule.
  · ThresholdRule: a rule defined by numeric sensor limits (warn/crit bounds).
  · OperationalRule: a process rule with a condition and a required action but no sensor thresholds.
  · MaintenanceRule: a rule triggered by cumulative degradation (drift, stuck sensor, wear) that prescribes a maintenance action (inspect, replace, calibrate, schedule).
  · AccessRule: a rule governing entry, authorisation, or role-based access to areas or systems.
- station: the component or zone this rule applies to. Use the EXACT name from the Known Seed Nodes list when a match exists. Derive the station from the nearest heading, section title, or explicit name in the text. If a heading ends with "Thresholds" or "Rules", the station is the text before that word. If no station can be determined, set to null.
- sensor: the sensor involved. Use the EXACT name from the Known Seed Nodes list when a match exists. If not found in the seed list, construct the name as '{station}_{sensorType}'. Never output a bare sensor type (e.g. "VIB", "TMP") when a seed-table name exists. Never invent a name outside this pattern.
- sensorType: abbreviated form inferred from the text (e.g. TMP, PRS, VIB). Null if uncertain.
- condition: the trigger described. For any rule, the text that describes WHEN the rule fires is the condition. Never put a maintenance action (inspect, replace, calibrate) into condition. action: the response or corrective measure. Never swap them.
  Split heuristic: text before a colon ":" or em/en dash "–—" is the condition; text after is the action. If no separator, the whole statement is the condition; default action is "Inspect and notify maintenance".
- severity: one of MANDATORY, WARNING, CRITICAL, HIGH, MEDIUM. Set to the keyword that appears in the text. Null only if none present.
- Numeric thresholds (critHi, warnHi, critLo, warnLo): floats from the text. Null if not present. "exceeds / above / >" → warnHi or critHi. "below / less than / drops under / <" → warnLo or critLo. Use the severity keyword to decide crit vs. warn. If a parameter must remain between Lo and Hi, set warnLo=Lo and warnHi=Hi; only set critHi/critLo when a SEPARATE CRITICAL threshold is explicit.
- unit: measurement unit as written (°C, bar, %RH, m/s, persons). Null if absent.
- source: document-level code found anywhere in the chunk (e.g. 'SOP-001', 'SOP-002'). Must match a pattern like 'SOP-NNN'. Do NOT use section headings or rule IDs as source. Never leave null when a document code is present.

RULES:
1. Only extract rules explicitly written in the text. Do NOT invent anything. Only create a rule for a sensor explicitly mentioned in the chunk — never scan the seed table to generate rules for sensors absent from the text. Return [] if no rules exist.
2. If no numeric thresholds AND no explicit condition-action pair are present, return []. Chunks containing only a document title, revision line, author, or high-level section heading with no concrete values must yield exactly []. A structural table that lists only station IDs, zones, and dependency descriptions (no measurable limits, no actions) also yields []. This rule overrides all others: no specific value + no condition-action = [].
3. Station assignment: derive the station from the nearest heading or section title above the rule in the text. If the heading names a station or zone, use it. If no heading is present, infer the station from the rule text itself or from the seed node list. Set to null only when genuinely not determinable.
4. For threshold tables: read column headers carefully. Typical order: CRIT_LO | WARN_LO | Nominal | WARN_HI | CRIT_HI. Never swap high and low. If a header cell appears merged (e.g. "WARN_LONominal"), infer the intended column names from the surrounding headers and the numeric ordering. Never invent values; only extract numbers literally present.
5. For prose rules in any format (free text, bullets, numbered lists, labelled lines): extract condition and action directly. If a numeric limit appears (e.g. "if TMP exceeds 26°C"), also populate the corresponding threshold field.
6. If a chunk contains multiple explicit rule-IDs, extract EACH as a separate JSON object. Never merge two rule-IDs into one object.
7. Tables with role-based timing columns (e.g. "Role | Max Ack WARNING (min) | Max Ack CRITICAL (min)") are AccessRules. Extract one AccessRule per data row: condition = role + triggering event, action = required response, warnHi = WARNING time limit (float), critHi = CRITICAL time limit (float).
8. Severity from column headers: a column header containing 'CRITICAL' implies severity='CRITICAL' for cells in that column; 'WARNING' implies severity='WARNING'. 'E-STOP' or 'EMERGENCY STOP' in a cell → severity='CRITICAL'. Apply only when no explicit severity keyword is already in the rule text.
9. Rule splitting: a sentence or row with two numeric limits paired with different severity labels → extract TWO separate objects, one per limit+severity pair.
10. Occupancy/zone-capacity tables (columns: Zone/Area, Max Persons/Capacity, Enforcement/Action): extract one AccessRule per data row; use the zone name as station; never leave station null.
11. Skip table rows that contain only metadata labels with no Rule ID, sensor, numeric value, or condition-action pair.
12. Role-based timing: separate WARNING and CRITICAL time limits in one row → produce TWO AccessRule objects (one per severity). Do NOT merge them.

Return ONLY a valid JSON array. No markdown fences, no commentary.
""".strip()

def build_user_prompt(chunk_content: str, headings: list, seed_table: str) -> str:
    # Build the user message. Headings are included when available, otherwise omitted.
    parts = [f"## Known Seed Nodes\n{seed_table}"]
    if headings:
        parts.append(f"## Chunk Headings\n{', '.join(headings)}")
    parts.append(f"## Chunk Text\n{chunk_content}")
    parts.append("Extract all rules as a JSON array.")
    return "\n\n".join(parts)

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