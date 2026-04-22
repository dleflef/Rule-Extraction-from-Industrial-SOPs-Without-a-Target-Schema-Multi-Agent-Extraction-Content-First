import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---------------------------------------------------------------------------
# 1. Pydantic Schemas
# ---------------------------------------------------------------------------
class Triple(BaseModel):
    subject: str = Field(description="The source entity (must exactly match an extracted entity span).")
    predicate: str = Field(description="The relationship type connecting subject to object.")
    object_: str = Field(alias="object", description="The target entity or resulting action.")

class RelationResult(BaseModel):
    triples: List[Triple] = Field(description="A list of relationship triples. Empty list if none found.")

# ---------------------------------------------------------------------------
# 2. LLM Setup (Qwen2.5 as requested)
# ---------------------------------------------------------------------------
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")

llm = ChatOpenAI(
    base_url=LOCAL_LLM_URL,
    api_key="local-ignore",
    model="qwen2.5-coder-7b-instruct",
    temperature=0.0, # Strictly 0.0 for baseline testing
    max_retries=2
)

parser = JsonOutputParser(pydantic_object=RelationResult)

# ---------------------------------------------------------------------------
# 3. Domain-Agnostic Relation Prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are an expert Relation Extraction system for industrial knowledge graphs.
Your task is to identify semantic relationships between the provided entities based ONLY on the text.

ALLOWED PREDICATES (Use ONLY these):
- monitors: Connects a Component to a Sensor.
- feeds_into: Connects a Component to a downstream Component.
- depends_on: Connects a Component to an upstream system it requires.
- authorized_for: Connects a Role to a Zone.
- triggers: Connects an AnomalyEvent to an Action, Maintenance rule, or state change.
- part_of: Connects a Component or Sensor to a larger System or Zone it belongs to.
- controls: Connects a Component or System to a Parameter it manages.
- produces: Connects a Component to a specific output or result it generates.

TEXT TO PROCESS:
{text}

PRE-EXTRACTED ENTITIES IN THIS TEXT:
{entities}

INSTRUCTIONS:
1. Extract relationships forming (subject, predicate, object) triples.
2. The 'subject' MUST be one of the pre-extracted entities.
3. The 'predicate' MUST be exactly one of the ALLOWED PREDICATES.
4. The 'object' MUST also be one of the pre-extracted entities, EXCEPT when the predicate is 'triggers', where the object may be an action or state (e.g., "E-STOP", "Inspect") not present in the entity list.
5. If no valid relationships exist, return an empty list.
{format_instructions}
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["text", "entities"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

extraction_chain = prompt | llm | parser

# ---------------------------------------------------------------------------
# 4. Synchronous Execution Function
# ---------------------------------------------------------------------------
def extract_relations_from_chunk(chunk_text: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Synchronously calls the LLM to extract relations."""
    if not entities:
        return [] # Don't bother calling the LLM if there are no entities to connect
        
    try:
        # Convert entities to a readable string format for the prompt
        entities_str = json.dumps([e["span"] for e in entities], ensure_ascii=False)
        result = extraction_chain.invoke({
            "text": chunk_text,
            "entities": entities_str
        })
        
        # Format the output clearly
        formatted_triples = []
        for t in result.get("triples", []):
            formatted_triples.append({
                "subject": t.get("subject", ""),
                "predicate": t.get("predicate", ""),
                "object": t.get("object", "")
            })
        return formatted_triples
        
    except Exception as e:
        print(f"[Agent 2B Warning] Relation extraction failed for chunk: {e}")
        return []