import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---------------------------------------------------------------------------
# 1. Pydantic Schemas
# ---------------------------------------------------------------------------
class Entity(BaseModel):
    span: str = Field(description="The exact word or phrase from the text.")
    category: str = Field(description="Must be strictly one of: Component, Sensor, Parameter, AnomalyEvent, Zone, Role")

class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(description="A list of extracted entities. Empty list if none found.")

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

parser = JsonOutputParser(pydantic_object=ExtractionResult)

# ---------------------------------------------------------------------------
# 3A. Zero-Shot Prompt — no domain knowledge
# ---------------------------------------------------------------------------
ZERO_SHOT_PROMPT = """You are an expert Named Entity Recognition system.
Your task is to identify relevant spans of text and classify them into predefined ontological categories.

CATEGORIES TO EXTRACT:
- Component: A physical machine, unit, or system part.
- Sensor: A device or instrument that measures physical properties.
- Parameter: A physical measurement, condition, or threshold type (e.g., Temperature, Pressure).
- AnomalyEvent: A described failure, error, violation, or specific triggered condition.
- Zone: A physical area, room, or location.
- Role: A job title, personnel type, or authorization level.

TEXT TO PROCESS:
{text}

INSTRUCTIONS:
1. Extract entities based purely on the text provided.
2. Use the exact text span as it appears — do NOT normalize or rewrite it.
3. Do not invent entities that are not present in the text.
{format_instructions}
"""

# ---------------------------------------------------------------------------
# 3B. Graph-Informed Prompt — structural hints only, no enumeration of nodes
# ---------------------------------------------------------------------------
GRAPH_INFORMED_PROMPT = """You are an expert Named Entity Recognition system for an industrial manufacturing Knowledge Graph.
Your task is to identify relevant spans of text and classify them into predefined ontological categories.

SYSTEM CONTEXT (structural hints only):
- The facility has multiple physical stations and zones; station identifiers typically follow an uppercase pattern (e.g., ST<nn>_<FUNCTION>).
- Sensors are measurement devices attached to stations; their identifiers typically follow <STATION>_<SENSORTYPE>.
- Zones are named physical areas within the facility.
- Roles are personnel titles or authorization levels.

CATEGORIES TO EXTRACT:
- Component: A physical machine, unit, or station.
- Sensor: A device or instrument that measures physical properties.
- Parameter: A physical measurement, condition, or threshold type (e.g., Temperature, Pressure).
- AnomalyEvent: A described failure, error, violation, or specific triggered condition.
- Zone: A physical area, room, or location.
- Role: A job title, personnel type, or authorization level.

TEXT TO PROCESS:
{text}

INSTRUCTIONS:
1. Extract entities based on the text provided.
2. Use the exact text span as it appears — do NOT normalize or rewrite it. Mapping to official node IDs is handled downstream.
3. Do not invent entities that are not referenced in the text.
{format_instructions}
"""

# ---------------------------------------------------------------------------
# 3C. Table-Aware Prompt — for structured markdown tables (is_table=True)
# ---------------------------------------------------------------------------
TABLE_PROMPT = """You are an expert Named Entity Recognition system.
The following text is a structured markdown table from a technical document.
Extract entity spans from the table column headers and cell values.

CATEGORIES TO EXTRACT:
- Component: A physical station or equipment referenced in the table.
- Sensor: A sensor or measurement device referenced in the table.
- Parameter: A measured property or threshold type.
- AnomalyEvent: A described failure mode, alarm, or triggered condition.
- Zone: A physical area or location.
- Role: A personnel type or authorization level.

TABLE TEXT:
{text}

INSTRUCTIONS:
1. Extract each distinct entity found in the table cells or headers.
2. Use the exact span as written — do NOT normalize.
3. Do not invent entities not present in the table.
{format_instructions}
"""

# ---------------------------------------------------------------------------
# 4. Build chains
# ---------------------------------------------------------------------------
def _make_chain(template: str):
    p = PromptTemplate(
        template=template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return p | llm | parser

_chains: Dict[str, Any] = {
    "zero_shot":      _make_chain(ZERO_SHOT_PROMPT),
    "graph_informed": _make_chain(GRAPH_INFORMED_PROMPT),
    "table":          _make_chain(TABLE_PROMPT),
}

# ---------------------------------------------------------------------------
# 5. Execution Function
# ---------------------------------------------------------------------------
def extract_entities_from_chunk(
    chunk_text: str,
    prompt_mode: str = PROMPT_MODE,
    is_table: bool = False,
) -> List[Dict[str, str]]:
    """Calls the LLM to extract entities from a text chunk.

    Table chunks always use the table-aware prompt regardless of prompt_mode.
    """
    if not chunk_text.strip():
        return []

    mode  = "table" if is_table else prompt_mode
    chain = _chains.get(mode, _chains["graph_informed"])

    try:
        result = chain.invoke({"text": chunk_text})
        return result.get("entities", [])
    except Exception as e:
        print(f"[Agent 2A Warning] Extraction failed (mode={mode}): {e}")
        return []
