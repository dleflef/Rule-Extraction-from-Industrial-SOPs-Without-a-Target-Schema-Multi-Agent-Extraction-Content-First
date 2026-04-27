"""
Agent 2C Tools — Ontology Alignment for Reified Rule Triples

Responsibility:
  Map the entity objects in each triple to their official KG seed node IDs.
  Rule node subjects (identifiable by the RULE- prefix) are kept as-is —
  they are NEW nodes that the extraction pipeline is creating.

  Aligns only these predicates' objects:
    applies_to_station, applies_to_sensor, applies_to_zone,
    monitors (both subject and object),
    feeds_into, authorized_for

  Literal-value predicates (has_condition, triggers_action, has_crit_hi, etc.)
  are passed through untouched.

Alignment strategy — 4 passes:
  1. Exact string match
  2. Normalised exact (lowercase, collapse special chars)
  3. Substring containment
  4. Few-shot LLM alignment via qwen2.5-coder-7b-instruct (temperature=0.0)
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# LLM Setup — mirrors Agent 2B
# ---------------------------------------------------------------------------

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")

_llm = ChatOpenAI(
    base_url=LOCAL_LLM_URL,
    api_key="local-ignore",
    model="qwen2.5-coder-7b-instruct",
    temperature=0.0,
    max_retries=2,
)

_ALIGNMENT_PROMPT = PromptTemplate(
    template="""You are an ontology alignment expert for an industrial Knowledge Graph.
Your task is to map a raw term to the single best-matching official node identifier.

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
- Use the surrounding context to disambiguate when the raw term is ambiguous.
- If no reasonable semantic match exists, return exactly: None
- Do NOT explain your reasoning.
- Do NOT add quotes, punctuation, or any other text.

Answer:""",
    input_variables=["node_list", "raw_term", "context"],
)

_alignment_chain = _ALIGNMENT_PROMPT | _llm

# ---------------------------------------------------------------------------
# Predicate classification sets
# ---------------------------------------------------------------------------

# Predicates whose object is an entity that should be aligned to the seed
_ENTITY_OBJECT_PREDICATES = {
    "applies_to_station",
    "applies_to_sensor",
    "applies_to_zone",
    "feeds_into",
    "authorized_for",
}

# Predicates where the subject is also an entity (structural edges)
_STRUCTURAL_PREDICATES = {"monitors", "feeds_into", "authorized_for"}

# Predicates whose object is a raw literal — never align these
_LITERAL_PREDICATES = {
    "rdf:type",
    "has_sensor_type",
    "has_condition",
    "triggers_action",
    "has_severity",
    "has_crit_hi",
    "has_warn_hi",
    "has_crit_lo",
    "has_warn_lo",
    "has_unit",
}

# Expected seed node type for each predicate's OBJECT side.
# Used to filter candidate nodes so the LLM sees a focused list.
_PREDICATE_OBJ_TYPE: Dict[str, str] = {
    "applies_to_station": "Component",
    "applies_to_sensor":  "Sensor",
    "applies_to_zone":    "Zone",
    "authorized_for":     "Zone",
    "feeds_into":         "Component",
    "monitors":           "Sensor",
}

# Expected seed node type for the SUBJECT side of structural predicates.
_PREDICATE_SUBJ_TYPE: Dict[str, str] = {
    "monitors":      "Component",
    "feeds_into":    "Component",
    "authorized_for": "Role",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", text.lower().replace(" ", "_"))


def _llm_align(
    raw: str,
    official_nodes: Dict[str, str],
    context: Optional[str] = None,
) -> Tuple[str, str]:
    """Calls the LLM to find the best semantic match from official_nodes."""
    node_list = "\n".join(sorted(official_nodes.keys()))
    ctx_str = context[:200] if context else "(no surrounding context available)"
    try:
        response = _alignment_chain.invoke({
            "node_list": node_list,
            "raw_term":  raw,
            "context":   ctx_str,
        })
        candidate = response.content.strip()
        if candidate and candidate != "None" and candidate in official_nodes:
            return candidate, "llm_aligned"
    except Exception as e:
        print(f"[Agent 2C Warning] LLM alignment failed for '{raw}': {e}")
    return raw, "unaligned"


def _fuzzy_align(
    raw: Optional[str],
    official_nodes: Dict[str, str],
    candidate_type: Optional[str] = None,
    context: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Returns (aligned_name, status).
    status: 'exact' | 'fuzzy' | 'llm_aligned' | 'unaligned'

    candidate_type: if provided, filters official_nodes to only nodes of that label
                    before each pass (falls back to all nodes if the filtered set is empty).
    context: surrounding sentence passed to the LLM on Pass 4 for disambiguation.
    """
    if not raw:
        return raw or "", "unaligned"

    # Build the candidate set — type-filtered when possible
    typed_nodes = (
        {k: v for k, v in official_nodes.items() if v == candidate_type}
        if candidate_type else {}
    )
    candidates = typed_nodes if typed_nodes else official_nodes

    # Pass 1 — exact match within the typed candidate set
    if raw in candidates:
        return raw, "exact"

    # Pass 2 — normalised exact
    rn = _normalise(raw)
    for name in candidates:
        if _normalise(name) == rn:
            return name, "exact"

    # Pass 3 — uppercase substring containment (e.g. "ST01_FILLING" ↔ "ST01 Filling Station")
    raw_up = raw.upper()
    matches = [
        name for name in candidates
        if name.upper() in raw_up or raw_up in name.upper()
    ]
    if matches:
        return min(matches, key=len), "fuzzy"

    # Pass 3B — separator-stripped containment
    # Handles CHM01_CHEMICALSTORAGE → "Chemical Storage" (cleaned: chemicalstorage ⊂ chm01chemicalstorage)
    # and RND01_RDLAB → "R&D Lab" (cleaned: rdlab ⊂ rnd01rdlab), etc.
    raw_clean = re.sub(r"[^a-z0-9]", "", raw.lower())
    fuzzy_matches = [
        name for name in candidates
        if (nc := re.sub(r"[^a-z0-9]", "", name.lower())) and nc in raw_clean
    ]
    if fuzzy_matches:
        return min(fuzzy_matches, key=len), "fuzzy"

    # Pass 4 — few-shot LLM alignment (use typed candidates for shorter, focused prompt)
    return _llm_align(raw, candidates, context=context)


def _is_rule_node(subject: str) -> bool:
    """Rule nodes start with RULE- — they are new nodes, not from the seed."""
    return subject.upper().startswith("RULE-")


# ---------------------------------------------------------------------------
# Main Alignment Function
# ---------------------------------------------------------------------------

def align_triple(
    triple: Dict[str, str],
    official_nodes: Dict[str, str],
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aligns entity references in a triple to official KG node IDs.
    Returns the triple enriched with alignment metadata.

    _raw_object is always preserved so alignment errors can be audited.
    context: a short surrounding sentence used to disambiguate LLM alignment.
    """
    predicate = triple.get("predicate", "")
    subject   = triple.get("subject", "")
    obj       = triple.get("object", "")

    aligned = dict(triple)
    aligned["_subject_alignment"] = "rule_node" if _is_rule_node(subject) else "n/a"
    aligned["_object_alignment"]  = "literal"

    # Align entity-object predicates
    if predicate in _ENTITY_OBJECT_PREDICATES:
        obj_type = _PREDICATE_OBJ_TYPE.get(predicate)
        aligned["_raw_object"] = obj  # preserve before any rewrite
        obj_aligned, obj_status = _fuzzy_align(
            obj, official_nodes, candidate_type=obj_type, context=context
        )
        aligned["object"] = obj_aligned
        aligned["_object_alignment"] = obj_status

    # Align both sides of structural predicates
    if predicate in _STRUCTURAL_PREDICATES:
        subj_type = _PREDICATE_SUBJ_TYPE.get(predicate)
        obj_type  = _PREDICATE_OBJ_TYPE.get(predicate)
        if not _is_rule_node(subject):
            subj_aligned, subj_status = _fuzzy_align(
                subject, official_nodes, candidate_type=subj_type, context=context
            )
            aligned["subject"] = subj_aligned
            aligned["_subject_alignment"] = subj_status
        aligned["_raw_object"] = obj
        obj_aligned, obj_status = _fuzzy_align(
            obj, official_nodes, candidate_type=obj_type, context=context
        )
        aligned["object"] = obj_aligned
        aligned["_object_alignment"] = obj_status

    return aligned


def align_relations_for_chunk(
    chunk_text: str,
    raw_relations: List[Dict[str, str]],
    official_nodes: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Aligns all triples in a single chunk, passing the chunk text as LLM context."""
    if not raw_relations:
        return []
    context = chunk_text[:300] if chunk_text else None
    return [align_triple(t, official_nodes, context=context) for t in raw_relations]
