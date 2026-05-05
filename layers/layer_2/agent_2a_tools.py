# layer_2/agent_2a_tools.py — Agent 2A: Entity Extractor
#
# Calls an LLM to extract structured operational rules from a single raw text chunk.
# The prompt is domain-agnostic: it describes the ontology schema (14 fields, 4 rule
# classes) but contains no factory-specific identifiers or ground-truth values.
#
# Rule IDs are intentionally left null — ID assignment is Agent 2C's responsibility
# (Ontology Alignment). The SBERT field-level evaluator does not match on ruleId.

import json
import os
import re
from typing import Any, Dict, List


# ── Seed node utilities ───────────────────────────────────────────────────────
# The seed node table is injected into the prompt as a reference so the LLM can
# link entity spans to known station/sensor names. Alignment to official node IDs
# is handled downstream by Agent 2C.

def load_seed_nodes(csv_path: str) -> List[Dict[str, str]]:
    import csv
    nodes = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes.append(row)
    return nodes


def format_seed_nodes_to_string(nodes: List[Dict[str, str]]) -> str:
    header = "| nodeId | label | name | type | zone |"
    sep    = "|--------|-------|------|------|------|"
    rows   = [f"| {n['nodeId']} | {n['label']} | {n['name']} | {n['type']} | {n['zone']} |"
              for n in nodes]
    return "\n".join([header, sep] + rows)


# ── System prompt (domain‑agnostic, fully schema‑driven) ─────────────────────

EXTRACTION_SYSTEM_PROMPT = """
You are an entity extractor for an industrial knowledge graph.
Read a text chunk from a technical document and extract every operational rule into a strict JSON array.

A rule is a logical statement connecting a condition (a state, trigger, or threshold violation) to an action (a required response, maintenance, or escalation). Rules may be written in free text, bullet points, or tables. **The chunk may contain corrupted Markdown tables (jumbled columns, repeated cell values, missing delimiters). Infer the underlying structure from context and extract the true rules. Do not skip content just because the table markup looks broken.**

Each rule MUST contain ALL 14 fields (use null for any that are genuinely absent):
  ruleId, class, station, sensor, sensorType, condition, action,
  severity, critHi, warnHi, critLo, warnLo, unit, source

Field definitions:
- ruleId: Always null. IDs are assigned by a downstream alignment agent. Never invent a rule ID.
- class: Exactly one of:
  · ThresholdRule     — defined by numeric upper/lower sensor limits.
  · OperationalRule   — a process rule with a condition and action, no strict numeric thresholds.
  · MaintenanceRule   — triggered by equipment degradation (drift, stuck sensor, wear) prescribing a physical intervention.
  · AccessRule        — governs personnel entry, occupancy limits, or role-based response times.
- station: The equipment station or zone name. Use the exact name as it appears in the text (e.g., "ST01_FILLING", "Production Area").
- sensor: The sensor identifier. Use the name as written in the text (e.g., "ST01_FILLING_TMP").
- sensorType: The abbreviated physical property (e.g., TMP, PRS, VIB, HUM, FLW, CUR, SPD, TEN, CNT) or null.
- condition: The exact text describing WHEN the rule fires or WHAT triggers it.
- action: The exact text describing WHAT must be done in response.
- severity: One of MANDATORY, WARNING, CRITICAL, HIGH, MEDIUM. Infer from explicit keywords (e.g., "emergency stop" → CRITICAL, "inspect within" → WARNING). If the source text contains extra qualifying words (e.g., "QC CRITICAL", "safety officer CRITICAL"), strip them and keep only the severity token.
- critHi, warnHi, critLo, warnLo: Numeric threshold values only, as raw floats (e.g., 26.0, not "26°C").
  · warnLo / warnHi map to warning-level boundaries.
  · critLo / critHi map to critical, halt, or emergency boundaries.
  · Lo = "drops below" direction; Hi = "exceeds" direction.
  · If a threshold is one‑sided, leave the opposite bound as null.
- unit: Measurement unit for the threshold values (e.g., °C, bar, %RH, m/s, L/min, A, N, pcs/min, persons, min).
- source: The document code found anywhere in the chunk (e.g., SOP-001, SOP-002).

**Critical extraction rules** (obey strictly):

1. **Threshold splitting**
   When a single source (table row, sentence) lists multiple threshold levels (e.g., WARN_HI / WARN_LO and CRIT_HI / CRIT_LO), you MUST create **separate rule objects** for each severity level.
   Example: A table row with columns CRIT_LO, WARN_LO, NOMINAL, WARN_HI, CRIT_HI yields **at least two rules**:
   · One with severity=WARNING, condition referencing the warning bounds, and only warnHi/warnLo values filled.
   · Another with severity=CRITICAL, condition referencing the critical bounds, and only critHi/critLo values filled.
   Never combine warning and critical thresholds into a single rule.

2. **Access / personnel rule extraction**
   Tables that define role‑zone permissions, maximum occupancies, or acknowledgement times contain implicit rules. Extract each row as an **AccessRule**:
   · For occupancy rows: station = zone name, condition = "occupancy exceeding X persons", action = enforcement text (e.g., "Alert security"), severity = WARNING or CRITICAL as stated.
   · For acknowledgement time rows: condition = "<role> must acknowledge within X minutes", action = none (or described escalation), severity = MANDATORY.
   · For role‑zone matrix rows: condition = "access by <role> to <zone>", action = "permitted" or "denied" (YES/NO), severity = null.
   Do not treat these tables as structural diagrams; they contain extractable rules.

3. **No threshold leakage**
   Numeric thresholds must come ONLY from the exact condition text of that rule. Do not import numbers from neighbouring sentences or other rules.

4. **Severity cleaning**
   Always normalise severity to the exact allowed tokens. If the text says "QC CRITICAL", output "CRITICAL". If it says "safety officer CRITICAL", output "CRITICAL". Do not include any role or qualifier.

5. **Independent rules only**
   A single text span can contain multiple independent rules. Extract each as a separate JSON object with its own complete 14 fields.

6. **Empty chunk**
   If the chunk contains no rules at all (e.g., a pure title page, table of contents, or completely non‑rule content), return an empty array [].

7. **Output format**
   Return ONLY a valid JSON array. No markdown fences, no extra commentary.
""".strip()


def build_user_prompt(chunk_content: str, headings: List[str], seed_table: str) -> str:
    parts = [f"## Known Seed Nodes\n{seed_table}"]
    if headings:
        parts.append(f"## Chunk Headings\n{', '.join(headings)}")
    parts.append(f"## Chunk Text\n{chunk_content}")
    parts.append("Extract all rules as a JSON array.")
    return "\n\n".join(parts)


# ── JSON parsing utilities ────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text)
    return text.strip()


def _repair_json(raw: str) -> str:
    # Remove trailing commas before } or ] — the most common LLM JSON mistake.
    return re.sub(r',\s*([\}\]])', r'\1', raw)


# ── Main extraction call ──────────────────────────────────────────────────────

def extract_rules_from_chunk(
    chunk_content: str,
    headings: List[str],
    seed_nodes_csv: str = "layers/data/seed_rules/dataset/kg_seeds/nodes_factory.csv",
    model_name: str = "qwen2.5-7b-instruct",
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    # Load seed nodes for entity linking reference in the prompt.
    if os.path.exists(seed_nodes_csv):
        nodes = load_seed_nodes(seed_nodes_csv)
        seed_table = format_seed_nodes_to_string(nodes)
    else:
        nodes = []
        seed_table = "| nodeId | label | name | type | zone |\n|--------|-------|------|------|------|"

    user_prompt = build_user_prompt(chunk_content, headings, seed_table)

    try:
        import openai
        client = openai.OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        raw = completion.choices[0].message.content
        cleaned = _strip_fences(raw)

        # Attempt direct parse, then trailing-comma repair.
        parsed = None
        for candidate in [cleaned, _repair_json(cleaned)]:
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue

        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "rules" in parsed:
            return parsed["rules"]
        return []

    except Exception as e:
        print(f"[Agent 2A] Extraction error: {e}")
        return []