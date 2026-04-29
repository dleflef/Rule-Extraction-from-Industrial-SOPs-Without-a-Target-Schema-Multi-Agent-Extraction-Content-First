import os
import re
import difflib
from typing import Dict, List, Optional, Tuple, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# 1. Strict LLM Setup
# ---------------------------------------------------------------------------
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")

_llm = ChatOpenAI(
    base_url=LOCAL_LLM_URL,
    api_key="local-ignore",
    model="qwen2.5-7b-instruct", # Fixed to mandated model
    temperature=0.0,                   # Enforced for baseline run
    max_retries=2,
)

# ---------------------------------------------------------------------------
# 2. The Unified Alignment Prompt
# ---------------------------------------------------------------------------
_ALIGNMENT_PROMPT = PromptTemplate(
    template="""You are an ontology alignment expert for an industrial Knowledge Graph.
Your task is to map a raw extracted term to the single best-matching official node identifier.

━━━ OFFICIAL NODE LIST ━━━
{node_list}

━━━ FEW-SHOT EXAMPLES ━━━
Raw term: "main water pump"
Answer: PMP_01_WATER

Raw term: "heat sensor on boiler"
Answer: BLR_02_TEMP

Raw term: "assembly area"
Answer: ZONE_ASSEMBLY

Raw term: "xyz_unknown_widget_999"
Answer: None

━━━ SURROUNDING CONTEXT ━━━
{context}

━━━ TASK ━━━
Raw term: "{raw_term}"

Instructions:
- Return ONLY the exact string from the official node list above that best semantically matches the raw term.
- Use the surrounding context to disambiguate.
- If no reasonable semantic match exists, return exactly: None
- Do NOT explain your reasoning. Do NOT add quotes or punctuation.
Answer:""",
    input_variables=["node_list", "raw_term", "context"],
)

_alignment_chain = _ALIGNMENT_PROMPT | _llm

# ---------------------------------------------------------------------------
# 3. Schema Definitions
# ---------------------------------------------------------------------------
_ENTITY_OBJECT_PREDICATES = {"applies_to_station", "applies_to_sensor", "applies_to_zone", "feeds_into", "authorized_for", "monitors"}
_STRUCTURAL_PREDICATES = {"monitors", "feeds_into", "authorized_for"}
_LITERAL_PREDICATES = {
    "rdf:type", "has_sensor_type", "has_condition", "triggers_action",
    "has_severity", "has_crit_hi", "has_warn_hi", "has_crit_lo", "has_warn_lo", "has_unit"
}

_ALL_ALLOWED_PREDICATES = _ENTITY_OBJECT_PREDICATES | _STRUCTURAL_PREDICATES | _LITERAL_PREDICATES

# This brilliantly enforces the structural schema requested by the professor
_PREDICATE_OBJ_TYPE: Dict[str, str] = {
    "applies_to_station": "Component", "applies_to_sensor": "Sensor", "applies_to_zone": "Zone",
    "authorized_for": "Zone", "feeds_into": "Component", "monitors": "Sensor",
}
_PREDICATE_SUBJ_TYPE: Dict[str, str] = {
    "monitors": "Component", "feeds_into": "Component", "authorized_for": "Role",
}

# ---------------------------------------------------------------------------
# 4. Alignment Algorithms
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(text).lower().replace(" ", "_"))

def _llm_align(raw: str, official_nodes: Dict[str, str], context: Optional[str] = None) -> Tuple[str, str]:
    """Calls the LLM as a last resort to find the best semantic match."""
    node_list = "\n".join(sorted(official_nodes.keys()))
    # Expanded context to allow LLM full visibility of the text chunk
    ctx_str = context if context else "(no surrounding context available)"
    
    try:
        response = _alignment_chain.invoke({"node_list": node_list, "raw_term": raw, "context": ctx_str})
        candidate = response.content.strip().strip("'").strip('"')
        
        if candidate.lower().startswith("answer:"):
            candidate = candidate[7:].strip()

        if candidate in official_nodes:
            return candidate, "llm_aligned"
            
        # Fallback if LLM gets chatty
        for node in official_nodes:
            if node in candidate:
                return node, "llm_aligned"
                
    except Exception as e:
        print(f"[Agent 2C Warning] LLM alignment failed for '{raw}': {e}")
    return raw, "unaligned"

def _fuzzy_align(
    raw: Optional[str],
    official_nodes: Dict[str, str],
    candidate_type: Optional[str] = None,
    context: Optional[str] = None,
    station_hint: Optional[str] = None,
) -> Tuple[str, str]:
    if not raw: return "", "unaligned"

    # Filter candidates strictly to the expected ontological type (enforces schema!)
    candidates = {k: v for k, v in official_nodes.items() if v == candidate_type} if candidate_type else official_nodes

    # 1. Exact Match
    if raw in candidates: return raw, "exact"

    # 2. Normalised Exact Match
    rn = _normalise(raw)
    for name in candidates:
        if _normalise(name) == rn: return name, "exact"

    # 3. Station-hinted direct lookup (new).
    # When the raw value is a short sensor-type abbreviation (e.g. "TMP", "PRS")
    # and we know the governing station, try <station>_<SENSORTYPE> first.  This
    # prevents short abbreviations from fuzzy-matching the wrong station's sensor.
    if candidate_type == "Sensor" and station_hint and len(raw) <= 5:
        direct = f"{station_hint}_{raw.upper()}"
        if direct in candidates:
            return direct, "exact"

    # 4a. Station-filtered Substring Containment (new).
    # When we have a station hint, try to find a match among sensors that belong
    # to that station before falling back to the full candidate pool.
    raw_clean = re.sub(r"[^a-z0-9]", "", raw.lower())
    if candidate_type == "Sensor" and station_hint:
        station_candidates = {k: v for k, v in candidates.items() if k.startswith(station_hint + "_")}
        for name in station_candidates:
            name_clean = re.sub(r"[^a-z0-9]", "", name.lower())
            if name_clean and (name_clean in raw_clean or raw_clean in name_clean):
                return name, "fuzzy"

    # 4b. General Substring Containment (original step 3)
    for name in candidates:
        name_clean = re.sub(r"[^a-z0-9]", "", name.lower())
        if name_clean and (name_clean in raw_clean or raw_clean in name_clean): return name, "fuzzy"

    # 5. Levenshtein Distance (65% similarity cutoff)
    close_matches = difflib.get_close_matches(raw.lower(), [c.lower() for c in candidates.keys()], n=1, cutoff=0.65)
    if close_matches:
        for name in candidates:
            if name.lower() == close_matches[0]: return name, "fuzzy"

    # 6. LLM Semantic Alignment (Final Fallback)
    return _llm_align(raw, candidates, context=context)

def _is_rule_node(subject: str) -> bool:
    """Checks if the subject is a reified Rule ID (fixed to include MAINT rules)."""
    subj_upper = str(subject).upper()
    return subj_upper.startswith("RULE-") or subj_upper.startswith("MAINT-")

# ---------------------------------------------------------------------------
# 5. Execution & Structural Validation
# ---------------------------------------------------------------------------
def align_triple(
    triple: Dict[str, str],
    official_nodes: Dict[str, str],
    context: Optional[str] = None,
    station_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Aligns entities and enforces structural congruence. Returns None if invalid."""
    predicate = triple.get("predicate", "")
    subject   = triple.get("subject", "")
    obj       = triple.get("object", "")

    if predicate not in _ALL_ALLOWED_PREDICATES:
        print(f"[Agent 2C] Dropping hallucinated predicate: '{predicate}'")
        return None

    aligned = dict(triple)
    aligned["_subject_alignment"] = "rule_node" if _is_rule_node(subject) else "n/a"
    aligned["_object_alignment"]  = "literal"

    # Align Objects
    if predicate in _ENTITY_OBJECT_PREDICATES:
        aligned["_raw_object"] = obj
        obj_type = _PREDICATE_OBJ_TYPE.get(predicate)
        obj_aligned, obj_status = _fuzzy_align(
            obj, official_nodes, candidate_type=obj_type, context=context,
            station_hint=station_hint if predicate == "applies_to_sensor" else None,
        )
        aligned["object"] = obj_aligned
        aligned["_object_alignment"] = obj_status

        if obj_status == "unaligned":
            print(f"[Agent 2C] Warning: could not align object '{obj}' for predicate '{predicate}'. Keeping raw value.")

    # Align Subjects (For Structural Edges)
    if predicate in _STRUCTURAL_PREDICATES:
        if not _is_rule_node(subject):
            subj_type = _PREDICATE_SUBJ_TYPE.get(predicate)
            subj_aligned, subj_status = _fuzzy_align(subject, official_nodes, candidate_type=subj_type, context=context)
            aligned["subject"] = subj_aligned
            aligned["_subject_alignment"] = subj_status

            if subj_status == "unaligned":
                print(f"[Agent 2C] Warning: could not align subject '{subject}' for predicate '{predicate}'. Keeping raw value.")

    return aligned


# Known station-ID pattern (uppercase alphanumeric segments separated by underscores)
_STATION_ID_RE = re.compile(r'\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+){1,3})\b')


def align_relations_for_chunk(
    chunk_text: str,
    raw_relations: List[Dict[str, str]],
    official_nodes: Dict[str, str],
) -> List[Dict[str, Any]]:
    if not raw_relations: return []
    context = chunk_text if chunk_text else None

    # Derive a station hint so sensor alignment prefers the correct station's sensor.
    # Priority 1 — an explicit applies_to_station / applies_to_zone triple in this chunk.
    # Priority 2 — a station ID found in the chunk text (e.g., "[Context: ST02_SEALING Thresholds]").
    station_hint: Optional[str] = None

    for t in raw_relations:
        if t.get("predicate") in ("applies_to_station", "applies_to_zone"):
            raw_station = t.get("object", "")
            if raw_station in official_nodes and official_nodes[raw_station] in ("Component", "Zone"):
                station_hint = raw_station
                break
            # Try normalised exact match
            rn = _normalise(raw_station)
            for node_id, node_type in official_nodes.items():
                if node_type in ("Component", "Zone") and _normalise(node_id) == rn:
                    station_hint = node_id
                    break
            if station_hint:
                break

    if not station_hint and context:
        for m in _STATION_ID_RE.finditer(context):
            candidate = m.group(1)
            if candidate in official_nodes and official_nodes[candidate] in ("Component", "Zone"):
                station_hint = candidate
                break

    validated_triples = []
    for t in raw_relations:
        aligned_t = align_triple(t, official_nodes, context=context, station_hint=station_hint)
        if aligned_t is not None:
            validated_triples.append(aligned_t)

    return validated_triples