import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---------------------------------------------------------------------------
# 1. Pydantic Schemas (The "Notebook" Pattern)
# ---------------------------------------------------------------------------
class Entity(BaseModel):
    span: str = Field(description="The exact word or phrase from the text.")
    category: str = Field(description="Must be strictly one of: Component, Sensor, Parameter, AnomalyEvent, Zone, Role")

class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(description="A list of extracted entities. Empty list if none found.")

# ---------------------------------------------------------------------------
# 2. LLM Setup (Qwen2.5 as requested by Professor)
# ---------------------------------------------------------------------------
# Adjust the base_url to match your local inference server (LM Studio/Ollama)
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")

llm = ChatOpenAI(
    base_url=LOCAL_LLM_URL,
    api_key="local-ignore",
    model="qwen2.5-coder-7b-instruct",
    temperature=0.0, # Strictly 0.0 for baseline testing
    max_retries=2
)

parser = JsonOutputParser(pydantic_object=ExtractionResult)

# ---------------------------------------------------------------------------
# 3. Domain-Agnostic Zero-Shot Prompt
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 3B. Graph-Informed Prompt
# ---------------------------------------------------------------------------
GRAPH_INFORMED_PROMPT = """You are an expert Named Entity Recognition system mapping text to a structured Knowledge Graph.
Your task is to identify relevant spans of text, classify them into predefined ontological categories, and normalize their names based on known system constraints.

SYSTEM CONSTRAINTS:
- The system operates across 7 predefined zones (e.g., Production Area, Server Room, General Warehouse, Main Entrance, Chemical Storage, R&D Lab, Cafeteria).
- Components often represent specific stations (e.g., FILLING, SEALING, LABELLING, PACKAGING).
- Sensors measure specific physical parameters and are usually linked to a component (e.g., TMP for Temperature, PRS for Pressure, FLW for Flow, CUR for Current, VIB for Vibration).

CATEGORIES TO EXTRACT:
- Component: A physical machine, unit, or station (Output normalized if possible, e.g., ST01_FILLING).
- Sensor: A device/instrument that measures physical properties (Output normalized if possible, e.g., ST01_FILLING_TMP).
- Parameter: A physical measurement, condition, or threshold type (e.g., Temperature, Speed).
- AnomalyEvent: A described failure, error, violation, or specific triggered condition.
- Zone: A physical area, room, or location (Must map closely to the 7 predefined zones).
- Role: A job title, personnel type, or authorization level.

TEXT TO PROCESS:
{text}

INSTRUCTIONS:
1. Extract the entities from the text.
2. Where applicable, use the System Constraints to normalize the 'span' output so it aligns with strict system labeling rules. Do not invent entities that are not referenced in the text.
{format_instructions}
"""

prompt = PromptTemplate(
    template=GRAPH_INFORMED_PROMPT,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

"""
PROMPT_TEMPLATE = ###You are an expert Named Entity Recognition system. 
Your task is to identify relevant spans of text and classify them into predefined ontological categories.

CATEGORIES TO EXTRACT:
- Component: A physical machine, unit, or system part.
- Sensor: A device/instrument that measures physical properties.
- Parameter: A physical measurement, condition, or threshold type (e.g., Temperature, Speed).
- AnomalyEvent: A described failure, error, violation, or specific triggered condition.
- Zone: A physical area, room, or location.
- Role: A job title, personnel type, or authorization level.

TEXT TO PROCESS:
{text}

INSTRUCTIONS:
Extract the entities based purely on the text provided. Do not invent entities.
{format_instructions}
###

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
"""

extraction_chain = prompt | llm | parser

# ---------------------------------------------------------------------------
# 4. Synchronous Execution Function
# ---------------------------------------------------------------------------
def extract_entities_from_chunk(chunk_text: str) -> List[Dict[str, str]]:
    """Synchronously calls the LLM to extract entities."""
    try:
        result = extraction_chain.invoke({"text": chunk_text})
        return result.get("entities", [])
    except Exception as e:
        print(f"[Agent 2A Warning] Extraction failed for chunk: {e}")
        return []