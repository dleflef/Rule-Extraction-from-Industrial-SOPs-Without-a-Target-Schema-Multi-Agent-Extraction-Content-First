import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---------------------------------------------------------------------------
# 1. Pydantic Schema — mirrors the ground_truth.csv schema exactly
# ---------------------------------------------------------------------------
class ExtractedRule(BaseModel):
    ruleId: Optional[str] = Field(
        default=None,
        description="Rule identifier if explicitly stated in the text (e.g. RULE-ST01-01). Use null if not mentioned."
    )
    rule_class: str = Field(
        description="Exactly one of: OperationalRule, ThresholdRule, MaintenanceRule, AccessRule"
    )
    station: Optional[str] = Field(
        default=None,
        description="Station/component identifier. Normalize to official format when possible (e.g. ST01_FILLING, ST02_SEALING, SRV01_SERVERROOM). Null for rules with no station."
    )
    sensor: Optional[str] = Field(
        default=None,
        description="Full sensor identifier in STATION_TYPE format (e.g. ST01_FILLING_FLW). Null for access/occupancy rules."
    )
    sensorType: Optional[str] = Field(
        default=None,
        description="Sensor type abbreviation only (TMP, FLW, PRS, CUR, VIB, HUM, SPD, TEN, CNT, POS). Null if no sensor."
    )
    condition: Optional[str] = Field(
        default=None,
        description="The trigger condition string exactly as described in the text. Preserve numbers and units."
    )
    action: Optional[str] = Field(
        default=None,
        description="The response or corrective action string exactly as described in the text."
    )
    severity: Optional[str] = Field(
        default=None,
        description="Exactly one of: MANDATORY, WARNING, CRITICAL, HIGH, MEDIUM. Null if not specified."
    )
    critHi: Optional[float] = Field(
        default=None,
        description="Critical-high threshold — numeric value only, no unit string."
    )
    warnHi: Optional[float] = Field(
        default=None,
        description="Warning-high threshold — numeric value only, no unit string."
    )
    critLo: Optional[float] = Field(
        default=None,
        description="Critical-low threshold — numeric value only, no unit string."
    )
    warnLo: Optional[float] = Field(
        default=None,
        description="Warning-low threshold — numeric value only, no unit string."
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement (e.g. °C, bar, L/min, A, mm/s, %RH, N, pcs/min, m/s, persons, min). Null if none."
    )


class ExtractionResult(BaseModel):
    rules: List[ExtractedRule] = Field(
        description="All rules found in this text chunk. Empty list if no rules present."
    )


# ---------------------------------------------------------------------------
# 2. LLM Setup
# ---------------------------------------------------------------------------
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")

llm = ChatOpenAI(
    base_url=LOCAL_LLM_URL,
    api_key="local-ignore",
    model="qwen2.5-coder-7b-instruct",
    temperature=0.0,
    max_retries=2
)

parser = JsonOutputParser(pydantic_object=ExtractionResult)

# ---------------------------------------------------------------------------
# 3. Graph-Informed Rule Extraction Prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are an expert Rule Extraction system for industrial Knowledge Graphs.
Extract ALL structured operational rules from the text below and output them in the exact schema defined.

SYSTEM CONTEXT (use ONLY for normalizing entity names — do not invent rules):
- Known stations: ST01_FILLING, ST02_SEALING, ST03_LABELLING, ST04_PACKAGING,
  SRV01_SERVERROOM, WRH01_WAREHOUSE, CHM01_CHEMICALSTORAGE, RND01_RDLAB, CAF01_CAFETERIA
- Sensor naming convention: {{STATION}}_{{TYPE}} (e.g. ST01_FILLING_TMP)
- Sensor type codes: TMP=temperature, FLW=flow, PRS=pressure, CUR=current, VIB=vibration,
  HUM=humidity, SPD=speed, TEN=tension, CNT=count, POS=position
- Rule classes:
    OperationalRule  — if/then operational conditions with a triggered action
    ThresholdRule    — defines numeric alarm thresholds (critHi, warnHi, critLo, warnLo)
    MaintenanceRule  — maintenance triggers and maintenance actions
    AccessRule       — zone access, occupancy limits, personnel authorisation
- Severity values: CRITICAL, WARNING, MANDATORY, HIGH, MEDIUM

TEXT TO PROCESS:
{text}

EXTRACTION RULES:
1. Extract EVERY rule present — do not skip any.
2. Preserve condition and action text EXACTLY as written in the document (including numbers and units).
3. For ThresholdRules extract all four numeric values: critHi, warnHi, critLo, warnLo.
4. For AccessRules with no sensor leave sensor/sensorType/threshold fields as null.
5. Normalize station and sensor names to the official format from SYSTEM CONTEXT where possible.
6. Do NOT invent rules that are not stated in the text.
{format_instructions}
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

extraction_chain = prompt | llm | parser


# ---------------------------------------------------------------------------
# 4. Execution Function
# ---------------------------------------------------------------------------
def extract_rules_from_chunk(chunk_text: str) -> List[Dict[str, Any]]:
    """Calls the LLM to extract structured rule records from a text chunk."""
    if not chunk_text.strip():
        return []
    try:
        result = extraction_chain.invoke({"text": chunk_text})
        rules = []
        for r in result.get("rules", []):
            rules.append({
                "ruleId":     r.get("ruleId"),
                "rule_class": r.get("rule_class", "OperationalRule"),
                "station":    r.get("station"),
                "sensor":     r.get("sensor"),
                "sensorType": r.get("sensorType"),
                "condition":  r.get("condition"),
                "action":     r.get("action"),
                "severity":   r.get("severity"),
                "critHi":     r.get("critHi"),
                "warnHi":     r.get("warnHi"),
                "critLo":     r.get("critLo"),
                "warnLo":     r.get("warnLo"),
                "unit":       r.get("unit"),
            })
        return rules
    except Exception as e:
        print(f"[Agent 2B Warning] Rule extraction failed for chunk: {e}")
        return []
