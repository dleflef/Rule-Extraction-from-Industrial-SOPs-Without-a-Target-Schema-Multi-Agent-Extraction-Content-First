"""
Metadata workflow for the Knowledge Graph extraction pipeline.

Exposes configuration schema and ontology to the workflow editor interface.
"""

from typing import Annotated, Any

import jsonref
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.resource import ResourceConfig

from .config import (
    EXTRACTED_DATA_COLLECTION,
    ClassifyConfig,
    JsonSchema,
    OntologyConfig,
    ValidationConfig,
)


class MetadataResponse(StopEvent):
    """Response containing all configuration for the UI."""

    json_schema: dict[str, Any]
    extracted_data_collection: str
    ontology: dict[str, Any]
    document_types: list[str]
    validation: dict[str, Any]


class MetadataWorkflow(Workflow):
    """Provide extraction schema, ontology, and configuration to the workflow editor."""

    @step
    async def get_metadata(
        self,
        _: StartEvent,
        extraction_schema: Annotated[
            JsonSchema,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract.json_schema",
                label="Extraction Schema",
                description="JSON Schema defining the knowledge graph structure to extract",
            ),
        ],
        ontology_config: Annotated[
            OntologyConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="ontology",
                label="Domain Ontology",
                description="Entity and relation types for your domain",
            ),
        ],
        classify_config: Annotated[
            ClassifyConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="classify",
                label="Document Classification",
                description="Rules for classifying document types",
            ),
        ],
        validation_config: Annotated[
            ValidationConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="validation",
                label="Validation Rules",
                description="Confidence thresholds and validation settings",
            ),
        ],
    ) -> MetadataResponse:
        """Return configuration for the review interface."""
        schema_dict = extraction_schema.to_dict()
        json_schema = jsonref.replace_refs(schema_dict, proxies=False)

        # Build ontology summary
        ontology = {
            "entity_types": [
                {"name": et.name, "description": et.description}
                for et in ontology_config.entity_types
            ],
            "relation_types": [
                {"name": rt.name, "description": rt.description}
                for rt in ontology_config.relation_types
            ],
        }

        # Get document types from classification rules
        document_types = [rule.type for rule in classify_config.rules]

        return MetadataResponse(
            json_schema=json_schema,
            extracted_data_collection=EXTRACTED_DATA_COLLECTION,
            ontology=ontology,
            document_types=document_types,
            validation=validation_config.model_dump(),
        )


workflow = MetadataWorkflow(timeout=None)
