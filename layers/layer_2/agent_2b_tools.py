"""
Agent 2B Tools — Relation Extractor with Rule Reification

Architecture note:
  The professor's design requires (subject, predicate, object) triples.
  To preserve threshold values, severity, and conditions WITHOUT losing information,
  we use KG Reification: each operational rule becomes a Rule *node* (the hub) and
  all its attributes are expressed as separate typed triples radiating from it.

  Example for "VIB > 0.20 mm/s → Schedule maintenance":
    ("RULE-ST04-VIB-01", "rdf:type",          "OperationalRule")
    ("RULE-ST04-VIB-01", "applies_to_sensor",  "ST04_PACKAGING_VIB")
    ("RULE-ST04-VIB-01", "applies_to_station", "ST04_PACKAGING")
    ("RULE-ST04-VIB-01", "has_condition",      "VIB > 0.20 mm/s (WARNING)")
    ("RULE-ST04-VIB-01", "triggers_action",    "Schedule maintenance within 24 h")
    ("RULE-ST04-VIB-01", "has_severity",       "WARNING")
    ("RULE-ST04-VIB-01", "has_warn_hi",        "0.20")
    ("RULE-ST04-VIB-01", "has_crit_hi",        "0.35")
    ("RULE-ST04-VIB-01", "has_unit",           "mm/s")
    ("ST04_PACKAGING",   "monitors",           "ST04_PACKAGING_VIB")

  Agent 2C then aligns the entity objects (sensor, station, zone names) to official
  KG node IDs from the seed graph.
"""

import os
import json
from typing import Any, Dict, List
from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---------------------------------------------------------------------------
# 1. Pydantic Schema — Pydantic v2 requires ConfigDict for alias round-tripping
# ---------------------------------------------------------------------------
class Triple(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    subject: str = Field(
        description="The source entity: a Rule node ID (e.g. RULE-ST04-VIB-01) "
                    "or a Component/Station name for structural 'monitors' triples."
    )
    predicate: str = Field(
        description="Must be exactly one of the ALLOWED PREDICATES."
    )
    object_: str = Field(
        alias="object",
        description="The target: an entity name, a literal value, or a text string.",
    )

class RelationResult(BaseModel):
    triples: List[Triple] = Field(
        description="All extracted triples. Empty list if none found in this chunk."
    )

# ---------------------------------------------------------------------------
# 2. LLM Setup
# ---------------------------------------------------------------------------
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
PROMPT_MODE   = os.getenv("PROMPT_MODE", "graph_informed")

llm = ChatOpenAI(
    base_url=LOCAL_LLM_URL,
    api_key="local-ignore",
    model="qwen2.5-coder-7b-instruct",
    temperature=0.0,
    max_retries=2,
)

parser = JsonOutputParser(pydantic_object=RelationResult)

# ---------------------------------------------------------------------------
# 3. Shared predicate block (used in all prompts)
# ---------------------------------------------------------------------------
_PREDICATES = """━━━ CORE CONCEPT — RULE REIFICATION ━━━
Each rule found in the text MUST be represented as a RULE NODE that acts as the hub
of a "star" of triples.  This preserves threshold values, severity, and conditions
without information loss.

RULE NODE NAMING CONVENTION:
- If the text explicitly states a rule ID (e.g. RULE-ST01-01), use it exactly.
- Otherwise construct: RULE-{{STATION}}-{{SENSOR_TYPE}}-{{n}}  (e.g. RULE-ST04-VIB-01)
- For access / occupancy rules with no sensor: RULE-{{ZONE_CODE}}-ACC-{{n}}
- For rules with no identifiable station/sensor, use: RULE-{{CHUNK_ID}}-{{n}}
- Use consecutive n = 01, 02, … within each chunk.

━━━ ALLOWED PREDICATES ━━━
Structural (standard graph edges):
  monitors          → Component monitors a Sensor  (subject = station, object = sensor)
  feeds_into        → Component → downstream Component
  authorized_for    → Role → Zone

Rule-attribute (reification — subject is always a Rule node):
  rdf:type          → Rule node → rule class  (OperationalRule | ThresholdRule | MaintenanceRule | AccessRule)
  applies_to_station→ Rule node → station/component identifier
  applies_to_sensor → Rule node → sensor identifier
  applies_to_zone   → Rule node → zone name (for AccessRules only)
  has_sensor_type   → Rule node → sensor-type abbreviation (TMP | FLW | PRS | CUR | VIB | HUM | SPD | TEN | CNT | POS)
  has_condition     → Rule node → trigger condition text (preserve exactly)
  triggers_action   → Rule node → response action text (preserve exactly)
  has_severity      → Rule node → severity (CRITICAL | WARNING | MANDATORY | HIGH | MEDIUM)
  has_crit_hi       → Rule node → critical-high threshold (numeric string only, no units)
  has_warn_hi       → Rule node → warning-high threshold (numeric string only)
  has_crit_lo       → Rule node → critical-low threshold (numeric string only)
  has_warn_lo       → Rule node → warning-low threshold (numeric string only)
  has_unit          → Rule node → unit of measurement string"""

# ---------------------------------------------------------------------------
# 4A. Zero-Shot Prompt — no station/sensor enumeration
# ---------------------------------------------------------------------------
ZERO_SHOT_PROMPT = """You are an expert Relation Extractor for an industrial Knowledge Graph.
Extract semantic triples of the form (subject, predicate, object) from the text.

CURRENT CHUNK ID: {{chunk_id}}

{predicates}

PRE-EXTRACTED ENTITIES IN THIS CHUNK:
{{entities}}

TEXT TO PROCESS:
{{text}}

━━━ INSTRUCTIONS ━━━
1. For each rule, emit a complete star of triples with the Rule node as subject.
2. If no explicit rule ID is in the text and no station/sensor can be identified, use RULE-{{chunk_id}}-<n> (e.g. RULE-{{chunk_id}}-01).
3. Emit a structural (station, monitors, sensor) triple for every station/sensor pair found.
4. Threshold object values MUST be numeric strings only — no units (e.g. "128.0" not "128.0 L/min").
5. Preserve condition and action text EXACTLY as written.
6. Do NOT invent rules not in the text.
{{format_instructions}}
""".format(predicates=_PREDICATES)

# ---------------------------------------------------------------------------
# 4B. Graph-Informed Prompt — structural hints only (no listing of node IDs)
# ---------------------------------------------------------------------------
GRAPH_INFORMED_PROMPT = """You are an expert Relation Extractor for an industrial Knowledge Graph.
Extract semantic triples of the form (subject, predicate, object) from the text.

CURRENT CHUNK ID: {{chunk_id}}

{predicates}

━━━ SYSTEM CONTEXT (structural hints only) ━━━
Station identifiers follow the pattern ST<nn>_<FUNCTION> (e.g. ST01_FILLING).
Sensor identifiers follow the pattern <STATION>_<SENSORTYPE> (e.g. ST01_FILLING_TMP).
Use these patterns to construct entity names when the text implies a station/sensor.

PRE-EXTRACTED ENTITIES IN THIS CHUNK:
{{entities}}

TEXT TO PROCESS:
{{text}}

━━━ INSTRUCTIONS ━━━
1. For each rule, emit a complete star of triples with the Rule node as subject.
2. If no explicit rule ID is in the text and no station/sensor can be identified, use RULE-{{chunk_id}}-<n> (e.g. RULE-{{chunk_id}}-01).
3. Emit a structural (station, monitors, sensor) triple for every station/sensor pair found.
4. Threshold object values MUST be numeric strings only — no units (e.g. "128.0" not "128.0 L/min").
5. Preserve condition and action text EXACTLY as written.
6. Do NOT invent rules not in the text.
{{format_instructions}}
""".format(predicates=_PREDICATES)

# ---------------------------------------------------------------------------
# 4C. Table-Aware Prompt — for structured markdown tables (is_table=True)
# ---------------------------------------------------------------------------
TABLE_PROMPT = """You are an expert Relation Extractor for an industrial Knowledge Graph.
The following text is a structured markdown table from a technical document.
Each data row represents one rule with its attributes.

CURRENT CHUNK ID: {{chunk_id}}

{predicates}

PRE-EXTRACTED ENTITIES IN THIS CHUNK:
{{entities}}

TABLE TEXT:
{{text}}

━━━ INSTRUCTIONS ━━━
1. For EACH data row in the table, emit a complete star of triples with a Rule node as subject.
2. Map table column names to the allowed predicates (e.g. "CritHi" → has_crit_hi, "Station" → applies_to_station).
3. Threshold values MUST be numeric strings only — no units.
4. If the table has an explicit rule ID column, use those values for Rule node IDs.
5. If no explicit ID exists, use RULE-{{chunk_id}}-<n>.
6. Emit a structural (station, monitors, sensor) triple for every station/sensor pair.
{{format_instructions}}
""".format(predicates=_PREDICATES)

# ---------------------------------------------------------------------------
# 5. Build chains
# ---------------------------------------------------------------------------
def _make_chain(template: str):
    p = PromptTemplate(
        template=template,
        input_variables=["text", "entities", "chunk_id"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return p | llm | parser

_chains: Dict[str, Any] = {
    "zero_shot":      _make_chain(ZERO_SHOT_PROMPT),
    "graph_informed": _make_chain(GRAPH_INFORMED_PROMPT),
    "table":          _make_chain(TABLE_PROMPT),
}

# ---------------------------------------------------------------------------
# 6. Execution Function
# ---------------------------------------------------------------------------
def extract_relations_from_chunk(
    chunk_text: str,
    entities: List[Dict[str, str]],
    prompt_mode: str = PROMPT_MODE,
    is_table: bool = False,
    chunk_id: Any = "X",
) -> List[Dict[str, str]]:
    """Calls the LLM to extract reified rule triples from a text chunk.

    - Skips empty chunks but proceeds even when the entity list is sparse.
    - Table chunks always use the table-aware prompt.
    - chunk_id is embedded in the system context so the LLM can use it as a
      fallback prefix for rule IDs (RULE-<CHUNK_ID>-<n>).
    """
    if not chunk_text.strip():
        return []

    mode  = "table" if is_table else prompt_mode
    chain = _chains.get(mode, _chains["graph_informed"])

    try:
        entities_str = json.dumps([e["span"] for e in entities], ensure_ascii=False)
        result = chain.invoke({
            "text":     chunk_text,
            "entities": entities_str,
            "chunk_id": str(chunk_id),
        })
        formatted = []
        for t in result.get("triples", []):
            obj_val = t.get("object") or t.get("object_", "")
            formatted.append({
                "subject":   t.get("subject", ""),
                "predicate": t.get("predicate", ""),
                "object":    obj_val,
            })
        return formatted
    except Exception as e:
        print(f"[Agent 2B Warning] Relation extraction failed for chunk {chunk_id} (mode={mode}): {e}")
        return []
