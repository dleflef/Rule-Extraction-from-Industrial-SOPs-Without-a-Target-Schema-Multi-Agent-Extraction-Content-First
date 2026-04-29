import os
import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---------------------------------------------------------------------------
# 1. Pydantic Schema
# ---------------------------------------------------------------------------
class Triple(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    subject: str = Field(
        description="The source entity (e.g., a Rule ID or a Component)."
    )
    predicate: str = Field(
        description="The relationship connecting subject to object. Must be an ALLOWED PREDICATE."
    )
    object_: str = Field(
        alias="object",
        description="The target entity, literal value, or text span.",
    )

class RelationResult(BaseModel):
    triples: List[Triple] = Field(
        description="All extracted triples. Empty list if none found in this chunk."
    )

# ---------------------------------------------------------------------------
# 2. Strict LLM Setup
# ---------------------------------------------------------------------------
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")

llm = ChatOpenAI(
    base_url=LOCAL_LLM_URL,
    api_key="local-ignore",
    model="qwen2.5-7b-instruct",
    temperature=0.0,
    max_retries=2,
)

parser = JsonOutputParser(pydantic_object=RelationResult)

# ---------------------------------------------------------------------------
# 3. The Unified, Domain-Agnostic Prompt
# ---------------------------------------------------------------------------
UNIFIED_RELATION_PROMPT = """\
You are an expert Relation Extractor for an industrial Knowledge Graph.
Your STRICTLY LIMITED role is to identify semantic relationships between the provided entities and output them as (subject, predicate, object) triples.

━━━ ENTITY SPAN USAGE (CRITICAL) ━━━
For applies_to_station, applies_to_sensor, applies_to_zone, authorized_for, and monitors triples:
  - Use entity spans EXACTLY as they appear in the PRE-EXTRACTED ENTITIES list below.
  - If an entity is clearly named in the text but absent from the pre-extracted list, copy its name verbatim from the text.
  - Do NOT invent, normalise, or look up official identifiers. A downstream Ontology Alignment Agent will map these spans to the correct graph node IDs.

━━━ STRICT RULE REIFICATION (CRITICAL) ━━━
You MUST NOT use physical entities (like a sensor type abbreviation or a station name) as the subject for rule conditions or thresholds.
Use existing rule IDs from the text verbatim when they are present (e.g., "MAINT-01", "RULE-ST03-01", "RULE-CORR-01").
Only invent a new Rule ID (e.g., "RULE-01", "RULE-02") when the text contains no existing structured rule identifier.
BAD : (temperature sensor, has_crit_hi, 30.0)
GOOD: (RULE-01, has_crit_hi, 30.0), (RULE-01, applies_to_sensor, temperature sensor)

━━━ MANDATORY FIELDS — Every rule MUST have ALL applicable triples ━━━
For EVERY rule you detect, generate ALL of the following triples (omit only if truly absent from text):
  1. (RULE-XX, rdf:type,           <OperationalRule|ThresholdRule|MaintenanceRule|AccessRule>)
  2. (RULE-XX, applies_to_station, <official station ID>)    [use applies_to_zone for AccessRules]
  3. (RULE-XX, applies_to_sensor,  <official sensor ID>)     [if a specific sensor is mentioned — ALWAYS include]
  4. (RULE-XX, has_sensor_type,    <TMP|PRS|FLW|VIB|CUR|SPD|HUM|TEN|CNT|POS>)  [sensor abbrev.]
  5. (RULE-XX, has_condition,      <exact trigger condition or restriction text from document>)
  6. (RULE-XX, triggers_action,    <exact required response/action text from document>)
  7. (RULE-XX, has_severity,       <CRITICAL|WARNING|MANDATORY>)
     IMPORTANT: DO NOT guess or infer severity based on the action (e.g., do not assume an emergency stop is CRITICAL unless the word CRITICAL explicitly appears in the text). If the exact severity word is not present in the text chunk, omit the has_severity triple entirely.
  8. (RULE-XX, has_unit,           <unit string e.g. "°C", "L/min", "bar", "A", "mm/s", "%RH", "N", "pcs/min", "m/s", "persons", "min">)
  9. Threshold triples (numeric only — see below)
NOTE: For MaintenanceRule, has_severity accepts: CRITICAL, HIGH, MEDIUM, WARNING, MANDATORY.
A rule with fewer than 4 triples is INCOMPLETE — do not skip required fields.

⚠ CRITICAL WARNINGS:
  - triggers_action MUST be the actual response/action text (e.g. "Inspect bearings", "Alert security").
    NEVER output a severity word (CRITICAL, WARNING, MANDATORY) as the value for triggers_action.
  - has_condition MUST be extracted for ALL rule types — including AccessRule and MaintenanceRule.
    For AccessRules: has_condition = the occupancy limit, access restriction, or acknowledgment requirement.
  - applies_to_sensor MUST be included whenever a specific sensor code is mentioned or can be inferred
    from the section heading (e.g., section "UNIT_01_TEMP" → applies_to_sensor = UNIT_01_TEMP, copy verbatim).

━━━ RULE TYPE GUIDE ━━━
  OperationalRule  — If/then behaviour based on sensor readings (monitoring, speed reduction, notification).
  ThresholdRule    — Declares numeric limit values (critHi, warnHi, critLo, warnLo) for a sensor.
  MaintenanceRule  — Prescribes maintenance scheduling, inspection, or sensor/component replacement.
  AccessRule       — Governs personnel authorisation to enter a zone (role-based access control).

━━━ FEW-SHOT EXAMPLES (generic industrial context — do NOT copy these codes into real extractions) ━━━

EXAMPLE A — ThresholdRule (defines numeric sensor limits with four bounds):
  Source: "FILTER_01_FLOW: CRIT_LO=80 m3/h, WARN_LO=90 m3/h, WARN_HI=130 m3/h, CRIT_HI=145 m3/h. Action: Inspect pump. Severity: CRITICAL"
  Required triples:
    (RULE-01, rdf:type, ThresholdRule)
    (RULE-01, applies_to_station, FILTER_01)
    (RULE-01, applies_to_sensor, FILTER_01_FLOW)
    (RULE-01, has_sensor_type, FLW)
    (RULE-01, has_crit_lo, 80), (RULE-01, has_warn_lo, 90)
    (RULE-01, has_warn_hi, 130), (RULE-01, has_crit_hi, 145)
    (RULE-01, has_unit, m3/h)
    (RULE-01, triggers_action, Inspect pump)
    (RULE-01, has_severity, CRITICAL)

EXAMPLE B — MaintenanceRule (reactive maintenance task triggered by a sensor anomaly):
  Source: "REACTOR_02_VIB: Vibration reading frozen >12 consecutive samples — sensor failure. Action: Replace vibration sensor. Severity: CRITICAL"
  Required triples:
    (RULE-02, rdf:type, MaintenanceRule)
    (RULE-02, applies_to_station, REACTOR_02)
    (RULE-02, applies_to_sensor, REACTOR_02_VIB)
    (RULE-02, has_sensor_type, VIB)
    (RULE-02, has_condition, Vibration reading frozen >12 consecutive samples)
    (RULE-02, triggers_action, Replace vibration sensor)
    (RULE-02, has_severity, CRITICAL)
    (RULE-02, has_unit, mm/s)

EXAMPLE C — AccessRule (occupancy limit or zone access restriction):
  Source: "Chemical Lab A: max occupancy 5 persons. WARNING above 4. Action: Alert safety officer; enforce evacuation if CRITICAL."
  Required triples:
    (RULE-03, rdf:type, AccessRule)
    (RULE-03, applies_to_zone, Chemical Lab A)
    (RULE-03, has_condition, Occupancy WARNING above 4 persons; CRITICAL at max 5)
    (RULE-03, triggers_action, Alert safety officer; enforce evacuation if CRITICAL)
    (RULE-03, has_severity, WARNING)
    (RULE-03, has_unit, persons)

EXAMPLE D — AccessRule (time-based alarm acknowledgment rule):
  Source: "technician — WARNING alarm: must acknowledge within 20 minutes."
  Required triples:
    (RULE-04, rdf:type, AccessRule)
    (RULE-04, has_condition, technician role — WARNING alarm acknowledgment)
    (RULE-04, triggers_action, Must acknowledge within 20 minutes)
    (RULE-04, has_severity, MANDATORY)
    (RULE-04, has_unit, min)

EXAMPLE E — Tabular repeated-field format (MAINT-XX or similar IDs):
  Source: "MAINT-01, Sensor = VIB. MAINT-01, Station = PACK_01. MAINT-01, Trigger = VIB DRIFT >0.05 mm/s over 60 min. MAINT-01, Action = Inspect and lubricate bearing. MAINT-01, Priority = HIGH."
  Required triples:
    (MAINT-01, rdf:type, MaintenanceRule)
    (MAINT-01, applies_to_station, PACK_01)
    (MAINT-01, applies_to_sensor, VIB)
    (MAINT-01, has_sensor_type, VIB)
    (MAINT-01, has_condition, VIB DRIFT >0.05 mm/s over 60 min)
    (MAINT-01, triggers_action, Inspect and lubricate bearing)
    (MAINT-01, has_severity, HIGH)
  IMPORTANT: When a Trigger/Action/Priority cell contains multiple pieces merged together,
  separate them: Trigger → has_condition, Action → triggers_action, Priority → has_severity.
  Extract ONE triple set per unique rule ID — do not merge multiple rules into one.

KEY RULES FOR ThresholdRule:
  - Use rdf:type=ThresholdRule whenever the chunk defines CRIT_LO/WARN_LO/WARN_HI/CRIT_HI values.
  - Create ONE separate ThresholdRule with a unique RULE-XX ID for EACH distinct (station, sensor) pair.
    Do NOT merge multiple sensors or stations into a single rule.
  - applies_to_sensor is MANDATORY for every ThresholdRule. Infer it directly from the section heading.
  - Extract ALL FOUR numeric bounds: has_crit_lo, has_warn_lo, has_warn_hi, has_crit_hi.
    Do NOT confuse WARN_LO with WARN_HI or CRIT_LO with CRIT_HI.
  - Always include has_unit for the measurement unit.
  - has_condition should summarise all four threshold bounds as they appear in the text.

KEY RULES FOR MaintenanceRule:
  - Use rdf:type=MaintenanceRule when the action prescribes maintenance, inspection, replacement, scheduling, or corrective work.
  - A rule is MaintenanceRule (NOT OperationalRule) if the primary response is a maintenance or corrective task.
  - Always extract has_condition (the drift/stuck/threshold condition that triggers the maintenance).

KEY RULES FOR AccessRule:
  - Use rdf:type=AccessRule for occupancy limits, personnel authorisation rules, and alarm acknowledgment duties.
  - Always extract has_condition describing the specific restriction or acknowledgment requirement.
  - Use applies_to_zone (not applies_to_station) to identify the restricted area.
  - Use has_unit=persons for occupancy rules; has_unit=min for time-based acknowledgment rules.

━━━ THRESHOLD VALUES AND UNITS ━━━
For threshold predicates (has_crit_hi, has_warn_hi, has_crit_lo, has_warn_lo):
  - The object MUST be a number ONLY — strip all unit text from the value. Use the numeric values directly from the PRE-EXTRACTED ENTITIES if available.
  - ALSO generate a separate (RULE-XX, has_unit, <unit string>) triple for the same rule.

━━━ ALLOWED PREDICATES ━━━
Structural (Entity to Entity):
  monitors          -> Component monitors a Sensor.
  feeds_into        -> Component feeds into a downstream Component.
  authorized_for    -> Role is authorized to enter a Zone.

Rule-Attribute (Subject is ALWAYS a Rule ID like "RULE-01"):
  rdf:type          -> Rule class (see RULE TYPE GUIDE above).
  applies_to_station-> The station/component the rule governs (official ID).
  applies_to_sensor -> The sensor triggering the rule (official ID).
  applies_to_zone   -> The zone restricted by an access rule (official ID).
  has_sensor_type   -> Sensor type abbreviation (TMP, PRS, FLW, VIB, CUR, SPD, HUM, TEN, CNT, POS).
  has_condition     -> The trigger condition text (verbatim from document).
  triggers_action   -> The required response action text (verbatim from document).
  has_severity      -> CRITICAL, WARNING, or MANDATORY.
  has_crit_hi       -> Critical-high threshold (numeric only).
  has_warn_hi       -> Warning-high threshold (numeric only).
  has_crit_lo       -> Critical-low threshold (numeric only).
  has_warn_lo       -> Warning-low threshold (numeric only).
  has_unit          -> The measurement unit for threshold values.

TEXT CHUNK:
{text}

PRE-EXTRACTED ENTITIES (Use the exact spans, numeric_values, and units provided here):
{entities}

{format_instructions}
"""

_extraction_chain = (
    PromptTemplate(
        template=UNIFIED_RELATION_PROMPT,
        input_variables=["text", "entities"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )
    | llm
    | parser
)

# ---------------------------------------------------------------------------
# 5. Guardrail constants and helpers
# ---------------------------------------------------------------------------
_ALLOWED_RULE_CLASSES = {"OperationalRule", "ThresholdRule", "MaintenanceRule", "AccessRule"}
_ALLOWED_SEVERITIES   = {"CRITICAL", "HIGH", "MEDIUM", "WARNING", "MANDATORY"}
_THRESHOLD_PREDICATES = {"has_crit_hi", "has_warn_hi", "has_crit_lo", "has_warn_lo"}

# Narrow keywords — "inspect" is intentionally excluded because it appears in
# almost every OperationalRule action (e.g. "Inspect and notify maintenance").
_MAINT_KEYWORDS  = {"lubricate", "lubrication", "calibration", "sensor failure",
                    "preventive", "schedule maintenance", "replace sensor",
                    "replace ten sensor", "replace vibration sensor"}
# Alarm acknowledgment and access-control keywords → AccessRule
_ACCESS_KEYWORDS = {"acknowledge", "acknowledgment", "ack ", "access",
                    "authorized", "authorised", "unauthorized", "badge",
                    "restricted", "evacuation", "occupancy", "entry by"}

# Hallucinated placeholder values the LLM outputs when it cannot find real text
_HALLUCINATED_ACTIONS = {
    "no specific action mentioned", "not specified in text", "not specified",
    "no action specified", "action not specified", "no action", "none",
    "no specific action", "action not mentioned",
}
_HALLUCINATED_CONDITIONS = {
    "no specific condition mentioned", "not specified in text", "not specified",
    "condition not specified", "no condition specified", "none",
    "no specific condition", "condition not mentioned",
}


def _fix_json_str(s: str) -> str:
    """Apply standard JSON repairs: unquoted keys, trailing commas."""
    s = re.sub(
        r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
        lambda m: f' "{m.group(1)}":',
        s,
    )
    return re.sub(r',\s*([}\]])', r'\1', s)


def _repair_json_from_error(error_str: str) -> Optional[List[Dict]]:
    """
    Extract and repair malformed JSON from a LangChain parser error message.

    Handles two failure modes:
      1. Unquoted object keys: { subject: "x" } — fixed by _fix_json_str.
      2. Objects placed OUTSIDE the 'triples' array — the LLM closes the
         array after the first rule, then appends remaining rule objects at
         the top level of the outer object (invalid JSON). Detected and
         recovered by scanning for all flat triple objects in the raw string.

    Returns the list of raw triple dicts, or None if repair fails.
    """
    # --- Pass 1: standard whole-block repair ---
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', error_str, re.DOTALL)
    json_str = match.group(1).strip() if match else None

    if not json_str:
        m = re.search(r'(\{[\s\S]*\})', error_str)
        if m:
            json_str = m.group(1).strip()

    if json_str:
        try:
            data = json.loads(_fix_json_str(json_str))
            if isinstance(data, dict):
                triples = data.get("triples", [])
                if triples:
                    return triples
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # --- Pass 2: flat-object scan ---
    # The LLM sometimes closes the triples array after the first rule and
    # places subsequent triple objects at the top level of the outer object.
    # Those objects are syntactically invalid inside the outer dict (no key),
    # so Pass 1 fails. We scan the raw error string for ALL flat objects
    # {…} (no nested braces) and collect those that look like triples.
    all_triples: List[Dict] = []
    for obj_match in re.finditer(r'\{[^{}]+\}', error_str):
        candidate = obj_match.group()
        try:
            obj = json.loads(_fix_json_str(candidate))
            if isinstance(obj, dict) and "subject" in obj and "predicate" in obj:
                all_triples.append(obj)
        except Exception:
            pass

    if all_triples:
        return all_triples
    return None


def _format_and_apply_guardrails(
    raw_triples: List[Any],
    chunk_id: Any,
) -> List[Dict[str, str]]:
    """
    Shared post-processing applied to raw triple dicts regardless of whether
    they came from a clean LLM parse or a JSON repair fallback.

    Guardrail 1 — invalid rule classes → OperationalRule.
    Guardrail 2 — threshold values must be numeric only.
    Guardrail 3 — inject rdf:type for rules the LLM left unclassified,
                  with improved AccessRule detection via keyword matching.
    """
    formatted: List[Dict[str, str]] = []
    for t in raw_triples:
        if not isinstance(t, dict):
            continue

        subj    = t.get("subject", "")
        pred    = t.get("predicate", "")
        obj_val = t.get("object") or t.get("object_", "")

        # Guardrail 1 — invalid rule class → OperationalRule
        if pred == "rdf:type" and obj_val not in _ALLOWED_RULE_CLASSES:
            obj_val = "OperationalRule"

        # Guardrail 1b — pass HIGH/MEDIUM through for MaintenanceRule severity
        if pred == "has_severity" and obj_val.upper() in _ALLOWED_SEVERITIES:
            obj_val = obj_val.upper()

        # Guardrail 2
        if pred in _THRESHOLD_PREDICATES:
            m = re.search(r'-?\d+(?:\.\d+)?', str(obj_val))
            if m:
                obj_val = m.group()

        # Guardrail 2b — drop hallucinated action/condition placeholders
        if pred == "triggers_action" and obj_val.lower().strip() in _HALLUCINATED_ACTIONS:
            print(f"[Agent 2B Guardrail] Dropped hallucinated action for {subj}: '{obj_val}'")
            continue
        if pred == "has_condition" and obj_val.lower().strip() in _HALLUCINATED_CONDITIONS:
            print(f"[Agent 2B Guardrail] Dropped hallucinated condition for {subj}: '{obj_val}'")
            continue

        if subj and pred and obj_val:
            formatted.append({"subject": subj, "predicate": pred, "object": obj_val})

    # Guardrail 3 — infer rdf:type for unclassified rules
    rule_has_type:  Dict[str, bool] = {}
    rule_preds:     Dict[str, set]  = {}
    rule_actions:   Dict[str, str]  = {}
    rule_conditions: Dict[str, str] = {}

    for t in formatted:
        s, p, o = t["subject"], t["predicate"], t["object"]
        if s.upper().startswith("RULE-"):
            rule_preds.setdefault(s, set()).add(p)
            if p == "rdf:type":
                rule_has_type[s] = True
            if p == "triggers_action":
                rule_actions[s] = str(o).lower()
            if p == "has_condition":
                rule_conditions[s] = str(o).lower()

    injected: List[Dict[str, str]] = []
    for rule_id, preds in rule_preds.items():
        if rule_has_type.get(rule_id):
            continue
        action    = rule_actions.get(rule_id, "")
        condition = rule_conditions.get(rule_id, "")
        # Fix 2: require >= 2 threshold bounds to avoid misclassifying single-bound rules
        if len(_THRESHOLD_PREDICATES & preds) >= 2:
            inferred = "ThresholdRule"
        elif "applies_to_zone" in preds:
            inferred = "AccessRule"
        elif any(kw in action or kw in condition for kw in _ACCESS_KEYWORDS):
            inferred = "AccessRule"
        elif any(kw in action for kw in _MAINT_KEYWORDS):
            inferred = "MaintenanceRule"
        else:
            inferred = "OperationalRule"
        injected.append({"subject": rule_id, "predicate": "rdf:type", "object": inferred})
        print(f"[Agent 2B Guardrail] Injected rdf:type={inferred} for {rule_id} (chunk {chunk_id})")

    # Guardrail 3b — reclassify rules that already have a type but ≥2 threshold
    # bounds. The LLM sometimes assigns MaintenanceRule or OperationalRule to a
    # rule that is structurally a ThresholdRule (it defines numeric limit values).
    for t in formatted:
        if t["predicate"] == "rdf:type" and t["object"] in {"MaintenanceRule", "OperationalRule"}:
            rule_id = t["subject"]
            preds = rule_preds.get(rule_id, set())
            if len(_THRESHOLD_PREDICATES & preds) >= 2:
                print(f"[Agent 2B Guardrail] Reclassified {rule_id}: {t['object']} → ThresholdRule (has ≥2 threshold bounds)")
                t["object"] = "ThresholdRule"

    return formatted + injected


# ---------------------------------------------------------------------------
# 6. Entity-based threshold fallback
# ---------------------------------------------------------------------------
# When the LLM returns zero triples for a dense threshold table chunk, we parse
# threshold values directly from the pre-extracted entity spans and the chunk
# context heading.  This recovers ThresholdRule records that would otherwise be
# entirely lost as false negatives.

_THRESHOLD_FIELD_MAP = {
    "CRIT_LO":  "has_crit_lo",
    "WARN_LO":  "has_warn_lo",
    "WARN_HI":  "has_warn_hi",
    "CRIT_HI":  "has_crit_hi",
}

def _entity_threshold_fallback(
    chunk_text: str,
    entities: List[Dict],
) -> List[Dict[str, str]]:
    """
    Deterministic fallback: reconstruct threshold triples from entity spans
    produced by Agent 2A when the LLM produces zero relations.

    Expected entity span format (from Docling/Agent 1B structured tables):
      'FLW, CRIT_LO = 105.0', 'TMP, CRIT_HI = 30.0',
      'PRS, Unit = bar', 'TMP, CRITICAL Response = Inspect ...'
    Station is read from the [Context: <station> Thresholds] prefix in chunk_text.
    """
    # Extract station from context marker
    station_match = re.search(r'\[Context:\s*([A-Z0-9_]+)\s+Thresholds\]', chunk_text)
    if not station_match:
        return []
    station = station_match.group(1)

    # Group values by sensor type
    sensor_data: Dict[str, Dict] = {}

    for entity in entities:
        span = entity.get("span", "")

        # Pattern A: explicit threshold field — "SENSORTYPE, CRIT_LO = 105.0"
        # We explicitly exclude WARN_LONominal: that column is the operating nominal
        # value, not the WARN_LO alarm threshold.  The actual WARN_LO appears in
        # Agent 1B output as an empty-field span (Pattern B below).
        m_field = re.match(
            r'^([A-Z]+),\s*(CRIT_LO|WARN_LO|WARN_HI|CRIT_HI)(?!Nominal)\s*=\s*(-?[\d.]+)',
            span, re.IGNORECASE
        )
        if m_field:
            stype = m_field.group(1).upper()
            field_key = m_field.group(2).upper()
            value = m_field.group(3)
            sensor_data.setdefault(stype, {})[field_key] = value
            continue

        # Pattern B: empty field name — "FLW,  = 112.0"
        # Agent 1B sometimes outputs the WARN_LO value with a blank field header
        # because the WARN_LO and Nominal table columns were merged into one cell.
        m_empty = re.match(r'^([A-Z]+),\s{0,3}=\s*(-?[\d.]+)', span, re.IGNORECASE)
        if m_empty:
            stype = m_empty.group(1).upper()
            value = m_empty.group(2)
            # Only store as WARN_LO if it hasn't already been set by Pattern A
            sensor_data.setdefault(stype, {}).setdefault("WARN_LO", value)
            continue

        # Pattern C: unit — "SENSORTYPE, Unit = °C"
        m_unit = re.match(r'^([A-Z]+),\s*Unit\s*=\s*(.+)', span, re.IGNORECASE)
        if m_unit:
            stype = m_unit.group(1).upper()
            sensor_data.setdefault(stype, {})["unit"] = m_unit.group(2).strip()
            continue

        # Pattern D: response/action — "SENSORTYPE, CRITICAL Response = <action>"
        m_action = re.match(r'^([A-Z]+),\s*CRITICAL Response\s*=\s*(.+)', span, re.IGNORECASE)
        if m_action:
            stype = m_action.group(1).upper()
            sensor_data.setdefault(stype, {})["action"] = m_action.group(2).strip()
            continue

        # Pattern E: fallback using numeric_value on unambiguous keyword spans
        for kw in _THRESHOLD_FIELD_MAP:
            if kw in span.upper() and "NOMINAL" not in span.upper() and entity.get("numeric_value") is not None:
                stype_m = re.match(r'^([A-Z]+),', span)
                if stype_m:
                    stype = stype_m.group(1).upper()
                    sensor_data.setdefault(stype, {}).setdefault(kw, str(entity["numeric_value"]))

    triples: List[Dict[str, str]] = []
    for stype, fields in sensor_data.items():
        # Only emit a rule if we have at least two threshold bounds
        threshold_count = sum(1 for k in _THRESHOLD_FIELD_MAP if k in fields)
        if threshold_count < 2:
            continue

        rule_id = f"RULE-{station}-{stype}"
        triples += [
            {"subject": rule_id, "predicate": "rdf:type",            "object": "ThresholdRule"},
            {"subject": rule_id, "predicate": "applies_to_station",  "object": station},
            {"subject": rule_id, "predicate": "applies_to_sensor",   "object": f"{station}_{stype}"},
            {"subject": rule_id, "predicate": "has_sensor_type",     "object": stype},
        ]
        for field_key, pred_name in _THRESHOLD_FIELD_MAP.items():
            if field_key in fields:
                triples.append({"subject": rule_id, "predicate": pred_name, "object": fields[field_key]})
        if "unit" in fields:
            triples.append({"subject": rule_id, "predicate": "has_unit", "object": fields["unit"]})
        if "action" in fields:
            triples.append({"subject": rule_id, "predicate": "triggers_action", "object": fields["action"]})

        # Build condition summary from available bounds
        cond_parts = []
        if "CRIT_LO" in fields: cond_parts.append(f"CRIT_LO={fields['CRIT_LO']}")
        if "WARN_LO" in fields: cond_parts.append(f"WARN_LO={fields['WARN_LO']}")
        if "WARN_HI" in fields: cond_parts.append(f"WARN_HI={fields['WARN_HI']}")
        if "CRIT_HI" in fields: cond_parts.append(f"CRIT_HI={fields['CRIT_HI']}")
        if cond_parts:
            unit_str = f" {fields['unit']}" if "unit" in fields else ""
            condition = f"{stype} thresholds: {', '.join(cond_parts)}{unit_str}"
            triples.append({"subject": rule_id, "predicate": "has_condition", "object": condition})

        print(f"[Agent 2B Fallback] Built {len([t for t in triples if t['subject'] == rule_id])} "
              f"triples for {rule_id} from entity spans")

    return triples


# ---------------------------------------------------------------------------
# 7. Execution Function
# ---------------------------------------------------------------------------

def extract_relations_from_chunk(
    chunk_text: str,
    entities: List[Dict[str, str]],
    chunk_id: Any = "X",
) -> List[Dict[str, str]]:
    """Calls the LLM to extract relationships from a text chunk."""
    if not chunk_text or not chunk_text.strip() or not entities:
        return []

    try:
        entities_list = []
        for e in entities:
            entry = {"span": e["span"], "category": e.get("category", "")}
            if e.get("numeric_value") is not None:
                entry["numeric_value"] = e["numeric_value"]
            if e.get("unit"):
                entry["unit"] = e["unit"]
            entities_list.append(entry)
        entities_str = json.dumps(entities_list, ensure_ascii=False)

        result = _extraction_chain.invoke({
            "text":     chunk_text,
            "entities": entities_str,
        })

        if result is None:
            raw_triples = []
        else:
            raw_triples = result if isinstance(result, list) else (result.get("triples") or [])

        formatted = _format_and_apply_guardrails(raw_triples, chunk_id)

        # Fallback: if the LLM returned nothing for a chunk that has entity data,
        # attempt to reconstruct threshold triples deterministically from entity spans.
        if not formatted and entities:
            fallback = _entity_threshold_fallback(chunk_text, entities)
            if fallback:
                print(f"[Agent 2B Fallback] Using entity-based threshold triples for chunk {chunk_id} "
                      f"(LLM returned 0 triples, fallback produced {len(fallback)})")
                return _format_and_apply_guardrails(fallback, chunk_id)

        return formatted

    except Exception as e:
        # Attempt to salvage malformed JSON (e.g. unquoted keys) before giving up.
        repaired = _repair_json_from_error(str(e))
        if repaired is not None:
            print(f"[Agent 2B] Repaired malformed JSON for chunk {chunk_id} "
                  f"({len(repaired)} raw triples recovered).")
            return _format_and_apply_guardrails(repaired, chunk_id)
        print(f"[Agent 2B Error] Relation extraction failed for chunk {chunk_id}: {e}")
        return []
