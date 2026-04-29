import os
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---------------------------------------------------------------------------
# 1. Pydantic Schema (Expanded to capture thresholds)
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    span: str = Field(
        description="The exact word or phrase from the text, copied verbatim."
    )
    category: str = Field(
        description="Exactly one of: Component, Sensor, Parameter, AnomalyEvent, Zone, Role"
    )
    numeric_value: Optional[float] = Field(
        default=None, 
        description="If the entity is a Parameter containing a threshold/limit, extract the numeric value (e.g., 135.0)."
    )
    unit: Optional[str] = Field(
        default=None, 
        description="If the entity is a Parameter, extract its unit of measurement (e.g., '°C', 'L/min', 'bar', 'm/s')."
    )

class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(
        description="All extracted entities. Empty list if none found. Must include all fields for each entity."
    )

# ---------------------------------------------------------------------------
# 2. LLM Setup (Strictly qwen2.5-coder-7b-instruct at Temp 0.0)
# ---------------------------------------------------------------------------

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")

llm = ChatOpenAI(
    base_url=LOCAL_LLM_URL,
    api_key="local-ignore",
    model="qwen2.5-7b-instruct", # Corrected model name per notes
    temperature=0.0,                   # Enforced for baseline run
    max_retries=2,
)

parser = JsonOutputParser(pydantic_object=ExtractionResult)

# ---------------------------------------------------------------------------
# 3. Category Validation
# ---------------------------------------------------------------------------

_ALLOWED_CATEGORIES = {"Component", "Sensor", "Parameter", "AnomalyEvent", "Zone", "Role"}

_CATEGORY_FIX: Dict[str, str] = {
    "component": "Component", "station": "Component", "machine": "Component", "equipment": "Component", "unit": "Component", "system": "Component",
    "sensor": "Sensor", "device": "Sensor", "instrument": "Sensor",
    "parameter": "Parameter", "measurement": "Parameter", "threshold": "Parameter", "value": "Parameter",
    "anomalyevent": "AnomalyEvent", "anomaly event": "AnomalyEvent", "anomaly": "AnomalyEvent", "event": "AnomalyEvent", "alarm": "AnomalyEvent", "failure": "AnomalyEvent", "condition": "AnomalyEvent", "triggered condition": "AnomalyEvent", "action": "AnomalyEvent",
    "zone": "Zone", "location": "Zone", "area": "Zone", "room": "Zone",
    "role": "Role", "person": "Role", "personnel": "Role", "operator": "Role",
    # Severity/priority values (HIGH, MEDIUM, CRITICAL) from maintenance tables
    "priority": "AnomalyEvent", "severity": "AnomalyEvent", "level": "AnomalyEvent",
}

def _fix_category(raw: str) -> Optional[str]:
    """Returns the canonical category name, or None if unrecognisable."""
    if raw in _ALLOWED_CATEGORIES:
        return raw
    return _CATEGORY_FIX.get(raw.strip().lower())

# ---------------------------------------------------------------------------
# 4. The Domain-Agnostic Prompt
# ---------------------------------------------------------------------------

UNIFIED_NER_PROMPT = """\
You are an advanced Named Entity Recognition (NER) system for industrial Digital Twin technical documents.
Your task is to identify and extract meaningful entity spans from the provided text and classify each into exactly one of the six predefined ontological categories.

CATEGORIES & DEFINITIONS:
  Component    — A physical machine, unit, processing station, or system. 
                 IMPORTANT: Alphanumeric equipment codes are ALWAYS Component.
                 Examples: "pump motor", "SYS01_MIXER", "CONVEYOR_A" → ALL are Component, NEVER Zone.
  Sensor       — An instrument or measurement device. 
                 Sensor codes typically follow a pattern linking them to a Component (e.g., "SYS01_MIXER_TMP", "flow sensor").
  Parameter    — A physical measurement type, numeric value, or threshold constraint.
                 MUST include the numeric value and unit if present in the text (e.g., span: "135.0 L/min", numeric_value: 135.0, unit: "L/min").
  AnomalyEvent — A described failure mode, alarm condition, triggered event, or required response action. (e.g., "WARNING", "emergency stop", "exceeds nominal range")
  Zone         — A named HUMAN-readable physical area used for access control or layout. (e.g., "Main Warehouse", "Control Room", "Hazardous Storage")
                 Zone spans are plain human language names, NOT alphanumeric equipment codes.
  Role         — A job title, personnel type, or authorisation level. (e.g., "operator", "maintenance technician")

EXTRACTION RULES:
  1. EXACT SPAN: Copy the exact word or phrase from the text verbatim. Do not rephrase.
  2. STRUCTURAL HINTS: Extract full structured identifier codes verbatim (e.g., equipment IDs, sensor codes).
  3. NUMERIC FIELDS: For Parameters, always separate and extract the numeric_value and unit into their respective JSON fields.
  4. STRICT MAPPING: Do NOT extract spans that do not clearly fit one of the six categories.
  5. NO HALLUCINATIONS: Do NOT invent spans absent from the text.
  6. TABLE AWARENESS: Treat markdown table cell values and headers as potential entities.
  7. EXCLUDE METADATA: Do NOT extract document reference codes (e.g., "SOP-100"), revision numbers, or dates.

TEXT TO PROCESS:
{text}

{format_instructions}
"""

_extraction_chain = (
    PromptTemplate(
        template=UNIFIED_NER_PROMPT,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    | llm
    | parser
)

# ---------------------------------------------------------------------------
# 5. Execution Function
# ---------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# Post-LLM entity repair helpers
# ─────────────────────────────────────────────────────────────────────────────

_TOLERANCE_UNIT_RE = re.compile(r'^\s*±\s*[\d.]+\s*$')
_BARE_STYPE_RE     = re.compile(r'^([A-Z]{2,3})$')
_BARE_NUMERIC_RE   = re.compile(r'^-?[\d.]+(?:\s*±\s*[\d.]+)?$')


def _find_structured_span(stype: str, numeric_val: float, chunk_text: str) -> Optional[str]:
    """Find the full 'STYPE, FIELD = VALUE [± TOL]' span in chunk_text."""
    val_str = str(numeric_val)
    alt_str: Optional[str] = None
    try:
        if float(numeric_val) == int(float(numeric_val)):
            alt_str = str(int(float(numeric_val)))
    except (ValueError, TypeError):
        pass

    for v in ([val_str] + ([alt_str] if alt_str and alt_str != val_str else [])):
        v_esc = re.escape(v)
        # Named-field: "STYPE, FIELDNAME = VALUE [± TOL]"
        m = re.search(
            rf'\b{re.escape(stype)},\s*\w+\s*=\s*{v_esc}(?:\s*±\s*[\d.]+)?',
            chunk_text, re.IGNORECASE
        )
        if m:
            return m.group().rstrip('.')
        # Blank-field: "STYPE,  = VALUE" (WARN_LO written without a field label)
        m = re.search(
            rf'\b{re.escape(stype)},\s{{0,5}}=\s*{v_esc}',
            chunk_text
        )
        if m:
            return m.group().rstrip('.')
    return None


def _repair_entities(entities: List[Dict], chunk_text: str) -> List[Dict]:
    """
    Post-LLM repair for three systematic extraction bugs:

    1. Tolerance (±x.xx) parsed into the unit field — strip it.
    2. Unit bleeding on WARN_LONominal spans (e.g. HUM gets "°C" from TMP row)
       — override with the stype's declared unit from "STYPE, Unit = X" entities.
    3. Bare sensor-type ("VIB") or bare-numeric ("-1.0", "4.0 ± 0.5") spans
       — reconstruct the full "STYPE, FIELD = VALUE" span from the chunk text.
    """
    # Build stype→unit map from "STYPE, Unit = X" entities in this chunk
    stype_unit: Dict[str, str] = {}
    for ent in entities:
        m = re.match(r'^([A-Z]{2,3}),\s*Unit\s*=\s*(.+)', ent.get("span", ""), re.IGNORECASE)
        if m:
            stype_unit[m.group(1).upper()] = m.group(2).strip()

    repaired: List[Dict] = []
    for ent in entities:
        span    = ent.get("span", "")
        num_val = ent.get("numeric_value")
        unit    = ent.get("unit")

        # Fix 1: strip tolerance-as-unit (e.g. "± 1.2" → None)
        if unit and _TOLERANCE_UNIT_RE.match(str(unit)):
            unit = None

        # Fix 3: reconstruct bare sensor-type spans ("VIB" with numeric_value=0.12)
        if _BARE_STYPE_RE.match(span) and num_val is not None:
            found = _find_structured_span(span, float(num_val), chunk_text)
            if found:
                span = found

        # Fix 3b: reconstruct bare numeric spans ("-1.0", "4.0 ± 0.5", "55.0", …)
        elif _BARE_NUMERIC_RE.match(span):
            primary_m = re.match(r'^(-?[\d.]+)', span.strip())
            if primary_m:
                v_esc = re.escape(primary_m.group(1))
                m_found = re.search(
                    rf'([A-Z]{{2,3}}),\s*(?:\w+\s*)?=\s*{v_esc}(?:\s*±\s*[\d.]+)?',
                    chunk_text, re.IGNORECASE
                )
                if m_found:
                    span = m_found.group().rstrip('.')

        # Fix 2: for WARN_LONominal spans always use the stype's declared unit
        if "WARN_LONominal" in span:
            prefix_m = re.match(r'^([A-Z]{2,3}),', span)
            if prefix_m:
                stype = prefix_m.group(1).upper()
                if stype in stype_unit:
                    unit = stype_unit[stype]

        repaired.append({**ent, "span": span, "unit": unit})

    return repaired


def extract_entities_from_chunk(chunk_text: str) -> List[Dict[str, Any]]:
    """Calls the LLM to extract entities using the unified prompt."""
    if not chunk_text or not chunk_text.strip():
        return []

    try:
        result = _extraction_chain.invoke({"text": chunk_text})
        if result is None:
            return []
        raw_entities = result.get("entities") or []
    except Exception as e:
        print(f"[Agent 2A Error] LLM Extraction failed: {e}")
        return []

    validated = []
    # Normalize chunk text to handle minor whitespace anomalies during validation
    normalized_chunk = re.sub(r'\s+', ' ', chunk_text)

    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue

        span = (ent.get("span") or "").strip()
        cat  = _fix_category(ent.get("category", ""))

        if not span or not cat:
            if span:
                print(f"[Agent 2A Warning] Dropped entity due to unrecognised category: '{ent.get('category')}' (span='{span}')")
            continue

        # Reject hallucinations: exact or highly similar span must exist in text
        normalized_span = re.sub(r'\s+', ' ', span)
        if normalized_span not in normalized_chunk:
            print(f"[Agent 2A Warning] Dropped hallucinated entity (not in text): '{span}'")
            continue

        validated_entity = {
            "span": span,
            "category": cat,
            "numeric_value": ent.get("numeric_value"),
            "unit": ent.get("unit")
        }
        validated.append(validated_entity)

    return _repair_entities(validated, chunk_text)