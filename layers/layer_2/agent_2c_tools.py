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
class AlignedTriple(BaseModel):
    aligned_subject: str = Field(description="The exact official node name, or the original text if it's an Anomaly/Action.")
    subject_type: str = Field(description="The ontological category of the subject.")
    predicate: str = Field(description="The schema-compliant relationship.")
    aligned_object: str = Field(description="The exact official node name, or the original text if it's an Anomaly/Action.")
    object_type: str = Field(description="The ontological category of the object.")
    is_schema_compliant: bool = Field(description="True ONLY if the triple strictly obeys the structural rules and text context.")

class AlignmentResult(BaseModel):
    aligned_triples: List[AlignedTriple] = Field(description="List of aligned and verified triples.")

# ---------------------------------------------------------------------------
# 2. LLM Setup (Qwen2.5)
# ---------------------------------------------------------------------------
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")

llm = ChatOpenAI(
    base_url=LOCAL_LLM_URL,
    api_key="local-ignore",
    model="qwen2.5-coder-7b-instruct",
    temperature=0.0,
    max_retries=2
)

parser = JsonOutputParser(pydantic_object=AlignmentResult)

# ---------------------------------------------------------------------------
# 3. Ontology Alignment Prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are an expert Ontology Alignment Agent for a Digital Twin Knowledge Graph.
Your task is to map raw extracted relationship triples to official Knowledge Graph nodes and verify schema compliance.

OFFICIAL SEED NODES (Dictionary of "NodeName": "Category"):
{official_nodes}

SCHEMA CONGRUENCE RULES (MUST BE STRICTLY OBEYED):
1. 'monitors' MUST connect (Component -> monitors -> Sensor). If the raw object is a Parameter (e.g., 'TMP'), map it to the specific official Sensor (e.g., 'ST01_FILLING_TMP').
2. 'feeds_into' / 'depends_on' MUST connect (Component -> Component) or (Zone -> Component).
3. 'authorized_for' MUST connect (Role -> Zone).
4. 'triggers' MUST connect (AnomalyEvent -> Action). These categories rarely have official nodes, so keep their original text spans.
5. NEGATION RULE: Read the context text carefully. If the text says a role is NOT authorized (e.g., "= NO"), you MUST NOT output an 'authorized_for' relation. Mark `is_schema_compliant` as false, or drop it.

CONTEXT TEXT:
{text}

RAW TRIPLES TO ALIGN:
{raw_triples}

INSTRUCTIONS:
Align the subject and object to the exact official names from the list provided if they are Components, Sensors, Zones, or Roles. Check the context text to resolve ambiguities.
{format_instructions}
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["text", "raw_triples", "official_nodes"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

extraction_chain = prompt | llm | parser

# ---------------------------------------------------------------------------
# 4. Synchronous Execution Function
# ---------------------------------------------------------------------------
def align_relations_sync(chunk_text: str, raw_relations: List[Dict], official_nodes: Dict[str, str]) -> List[Dict]:
    """Synchronously calls the LLM to align relations to the seed graph."""
    if not raw_relations:
        return []
        
    try:
        nodes_str = json.dumps(official_nodes, ensure_ascii=False)
        triples_str = json.dumps(raw_relations, ensure_ascii=False)
        
        result = extraction_chain.invoke({
            "text": chunk_text,
            "raw_triples": triples_str,
            "official_nodes": nodes_str
        })
        
        # Filter out anything the LLM flagged as non-compliant (like the NO authorizations)
        valid_triples = []
        for t in result.get("aligned_triples", []):
            if t.get("is_schema_compliant", False):
                valid_triples.append({
                    "subject": t.get("aligned_subject"),
                    "subject_type": t.get("subject_type"),
                    "predicate": t.get("predicate"),
                    "object": t.get("aligned_object"),
                    "object_type": t.get("object_type")
                })
        return valid_triples
        
    except Exception as e:
        print(f"[Agent 2C Warning] Alignment failed for chunk: {e}")
        return []