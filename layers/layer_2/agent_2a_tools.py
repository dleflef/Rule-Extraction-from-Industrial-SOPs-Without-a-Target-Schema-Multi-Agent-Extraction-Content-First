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

    return validated