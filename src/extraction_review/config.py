"""
Configuration for the knowledge graph extraction pipeline.

Configuration is loaded from configs/config.json via ResourceConfig.
The unified config contains ontology definitions, extraction settings, and validation rules.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from .json_util import create_union_schema as create_union_schema
from .json_util import get_extraction_schema as get_extraction_schema

logger = logging.getLogger(__name__)


# The name of the collection to use for storing extracted data.
EXTRACTED_DATA_COLLECTION: str = "kg-extraction"


class EntityType(BaseModel):
    """Definition of an entity type in the domain ontology."""

    name: str
    description: str


class RelationType(BaseModel):
    """Definition of a relation type in the domain ontology."""

    name: str
    description: str


class OntologyConfig(BaseModel):
    """Domain ontology configuration defining entity and relation types."""

    entity_types: list[EntityType] = []
    relation_types: list[RelationType] = []

    def get_entity_type_names(self) -> list[str]:
        """Return list of valid entity type names."""
        return [et.name for et in self.entity_types]

    def get_relation_type_names(self) -> list[str]:
        """Return list of valid relation type names."""
        return [rt.name for rt in self.relation_types]

    def build_extraction_prompt(self) -> str:
        """Build a system prompt that includes ontology definitions."""
        entity_list = "\n".join(
            f"- {et.name}: {et.description}" for et in self.entity_types
        )
        relation_list = "\n".join(
            f"- {rt.name}: {rt.description}" for rt in self.relation_types
        )
        return f"""Extract knowledge graph triples using the following ontology:

ENTITY TYPES:
{entity_list}

RELATION TYPES:
{relation_list}

For each entity, identify its type from the list above. For each relationship, use one of the defined relation types.
Include source context to support traceability."""


class ExtractSettings(BaseModel):
    """Extraction settings loaded from configs/config.json extract.settings."""

    extraction_mode: Literal["FAST", "BALANCED", "MULTIMODAL", "PREMIUM"]
    system_prompt: str | None = None
    citation_bbox: bool = False
    use_reasoning: bool = False
    cite_sources: bool = False
    confidence_scores: bool = False


class ExtractConfig(BaseModel):
    """Full extraction configuration with schema and settings."""

    json_schema: dict[str, Any]
    settings: ExtractSettings


class ClassifyRule(BaseModel):
    """Classification rule with type and description."""

    type: str
    description: str


class ClassifyParsingConfig(BaseModel):
    """Parsing config for Classify."""

    lang: str = Field(description="two-letter ISO 639 language code", default="en")
    max_pages: int | None = None
    target_pages: list[int] | None = None


class ClassifySettings(BaseModel):
    """Settings for document classification."""

    mode: Literal["FAST", "MULTIMODAL"] = "FAST"
    parsing_config: ClassifyParsingConfig = ClassifyParsingConfig()


class ClassifyConfig(BaseModel):
    """Classification configuration with rules and settings."""

    rules: list[ClassifyRule] = []
    settings: ClassifySettings = ClassifySettings()


class ValidationConfig(BaseModel):
    """Validation settings for extracted knowledge."""

    confidence_threshold: float = Field(
        default=0.7, description="Minimum confidence score for automatic acceptance"
    )
    require_source_context: bool = Field(
        default=True, description="Require source context for all extractions"
    )
    flag_low_confidence_for_review: bool = Field(
        default=True, description="Flag low-confidence extractions for human review"
    )


class ParseSettings(BaseModel):
    """Parsing settings for LlamaParse."""

    tier: Literal["fast", "agentic"] = "agentic"
    version: str = "latest"
    lang: str | None = Field(
        default=None, description="Two-letter ISO 639 language code"
    )
    max_pages: int | None = None


class ParseConfig(BaseModel):
    """Parse configuration for LlamaParse."""

    settings: ParseSettings = ParseSettings()


class JsonSchema(BaseModel):
    """Pydantic wrapper for a JSON schema loaded via ResourceConfig."""

    type: str = "object"
    properties: dict[str, Any] = {}
    required: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for APIs that expect JSON schema."""
        return self.model_dump(exclude_none=True)


class SplitCategory(BaseModel):
    """A category for document splitting."""

    name: str
    description: str


class SplittingStrategy(BaseModel):
    """Strategy for document splitting."""

    allow_uncategorized: bool = False


class SplitSettings(BaseModel):
    """Settings for document splitting."""

    splitting_strategy: SplittingStrategy = SplittingStrategy()


class SplitConfig(BaseModel):
    """Split configuration with categories and settings."""

    categories: list[SplitCategory] = []
    settings: SplitSettings = SplitSettings()


class Config(BaseModel):
    """Root configuration model for configs/config.json."""

    ontology: OntologyConfig = OntologyConfig()
    extract: ExtractConfig
    classify: ClassifyConfig = ClassifyConfig()
    validation: ValidationConfig = ValidationConfig()
    parse: ParseConfig = ParseConfig()
    split: SplitConfig = SplitConfig()
