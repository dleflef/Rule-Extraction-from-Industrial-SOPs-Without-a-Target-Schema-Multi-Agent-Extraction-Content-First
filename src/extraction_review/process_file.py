"""
Knowledge Graph Extraction Workflow for Digital Twins.

Extracts entities and relationships from technical documents using a configurable
domain ontology. Supports any industrial domain through configuration.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from uuid import uuid4

from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData, InvalidExtractionData
from llama_cloud.types.classifier.classifier_rule_param import ClassifierRuleParam
from llama_cloud.types.file_query_params import Filter
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .clients import agent_name, get_llama_cloud_client, project_id
from .config import (
    EXTRACTED_DATA_COLLECTION,
    ClassifyConfig,
    ExtractConfig,
    OntologyConfig,
    ValidationConfig,
    get_extraction_schema,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Events
# =============================================================================


class FileEvent(StartEvent):
    """Input event with file to process."""

    file_id: str
    file_hash: str | None = None


class Status(Event):
    """Progress status event streamed to client."""

    level: Literal["info", "warning", "error"]
    message: str


class ClassifyJobStartedEvent(Event):
    """Classification job has been submitted."""

    pass


class ClassifiedEvent(Event):
    """Document has been classified."""

    document_type: str
    confidence: float
    reasoning: str | None = None


class ExtractJobStartedEvent(Event):
    """Extraction job has been submitted."""

    pass


class ExtractedEvent(Event):
    """Extraction completed successfully."""

    data: ExtractedData


class ExtractedInvalidEvent(Event):
    """Extraction completed but validation failed."""

    data: ExtractedData[dict[str, Any]]


class ValidationResult(BaseModel):
    """Result of validation for a single extraction."""

    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    flagged_for_review: bool = False


# =============================================================================
# Provenance Log Entry
# =============================================================================


class ProvenanceEntry(BaseModel):
    """Log entry for tracking extraction provenance - essential for scientific evaluation."""

    extraction_id: str
    timestamp: str
    file_id: str
    file_name: str | None = None
    document_type: str | None = None
    classification_confidence: float | None = None
    entity_count: int = 0
    relation_count: int = 0
    avg_extraction_confidence: float | None = None
    low_confidence_count: int = 0
    flagged_for_review: bool = False
    validation_issues: list[str] = Field(default_factory=list)


# =============================================================================
# Workflow State
# =============================================================================


class ExtractionState(BaseModel):
    """Workflow state persisted between steps."""

    file_id: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    document_type: str | None = None
    classification_confidence: float | None = None
    classify_job_id: str | None = None
    extract_job_id: str | None = None
    extraction_id: str = Field(default_factory=lambda: str(uuid4())[:8])


# =============================================================================
# Workflow
# =============================================================================


class KnowledgeGraphExtractionWorkflow(Workflow):
    """
    Extract knowledge graph triples from technical documents.

    Processes documents through classification, entity/relation extraction,
    and validation using a configurable domain ontology. All extractions
    include provenance logging for scientific traceability.
    """

    @step()
    async def start_classification(
        self,
        event: FileEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        classify_config: Annotated[
            ClassifyConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="classify",
                label="Document Classification",
                description="Rules for classifying document types (e.g., technical manual, maintenance report)",
            ),
        ],
    ) -> ClassifyJobStartedEvent:
        """Classify the document to determine its type for appropriate processing."""
        file_id = event.file_id
        logger.info(f"Starting KG extraction for file {file_id}")

        # Get file metadata
        files = await llama_cloud_client.files.query(
            filter=Filter(file_ids=[file_id])
        )
        file_metadata = files.items[0]
        filename = file_metadata.name

        ctx.write_event_to_stream(
            Status(level="info", message=f"Classifying document: {filename}")
        )

        # Build classification rules from config
        rules = [
            ClassifierRuleParam(type=rule.type, description=rule.description)
            for rule in classify_config.rules
        ]

        # Start classification job
        classify_job = await llama_cloud_client.classifier.jobs.create(
            file_ids=[file_id],
            rules=rules,
            mode=classify_config.settings.mode,
            parsing_configuration={
                "lang": classify_config.settings.parsing_config.lang,
                "max_pages": classify_config.settings.parsing_config.max_pages,
            },
        )

        file_hash = event.file_hash or file_metadata.external_file_id

        async with ctx.store.edit_state() as state:
            state.file_id = file_id
            state.filename = filename
            state.file_hash = file_hash
            state.classify_job_id = classify_job.id

        return ClassifyJobStartedEvent()

    @step()
    async def complete_classification(
        self,
        event: ClassifyJobStartedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
    ) -> ClassifiedEvent:
        """Wait for classification to complete and record the document type."""
        state = await ctx.store.get_state()

        await llama_cloud_client.classifier.wait_for_completion(state.classify_job_id)
        result = await llama_cloud_client.classifier.jobs.get_results(
            state.classify_job_id
        )

        # Get classification result
        item = result.items[0]
        if item.result is None:
            logger.warning(f"Classification failed for {state.filename}, using default")
            document_type = "unknown"
            confidence = 0.0
            reasoning = None
        else:
            document_type = item.result.type
            confidence = item.result.confidence
            reasoning = item.result.reasoning

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Document classified as '{document_type}' (confidence: {confidence:.2f})",
            )
        )

        async with ctx.store.edit_state() as state:
            state.document_type = document_type
            state.classification_confidence = confidence

        return ClassifiedEvent(
            document_type=document_type,
            confidence=confidence,
            reasoning=reasoning,
        )

    @step()
    async def start_extraction(
        self,
        event: ClassifiedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        ontology_config: Annotated[
            OntologyConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="ontology",
                label="Domain Ontology",
                description="Entity and relation types for your domain (fully configurable)",
            ),
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Extraction Settings",
                description="Configuration for knowledge extraction quality and features",
            ),
        ],
    ) -> ExtractJobStartedEvent:
        """Start knowledge extraction using the domain ontology."""
        state = await ctx.store.get_state()

        ctx.write_event_to_stream(
            Status(level="info", message=f"Extracting knowledge from {state.filename}")
        )

        # Build enhanced system prompt with ontology
        ontology_prompt = ontology_config.build_extraction_prompt()
        base_prompt = extract_config.settings.system_prompt or ""
        combined_prompt = f"{base_prompt}\n\n{ontology_prompt}"

        # Prepare extraction settings
        settings = extract_config.settings.model_dump()
        settings["system_prompt"] = combined_prompt

        # Start extraction job
        extract_job = await llama_cloud_client.extraction.run(
            config=settings,
            data_schema=extract_config.json_schema,
            file_id=state.file_id,
            project_id=project_id,
        )

        async with ctx.store.edit_state() as state:
            state.extract_job_id = extract_job.id

        return ExtractJobStartedEvent()

    @step()
    async def complete_extraction(
        self,
        event: ExtractJobStartedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Extraction Settings",
                description="Configuration for knowledge extraction quality and features",
            ),
        ],
        validation_config: Annotated[
            ValidationConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="validation",
                label="Validation Rules",
                description="Confidence thresholds and validation requirements",
            ),
        ],
    ) -> StopEvent:
        """Complete extraction, validate results, and save with provenance."""
        state = await ctx.store.get_state()

        # Wait for extraction
        await llama_cloud_client.extraction.jobs.wait_for_completion(
            state.extract_job_id
        )
        extracted_result = await llama_cloud_client.extraction.jobs.get_result(
            state.extract_job_id
        )
        extract_run = await llama_cloud_client.extraction.runs.get(
            run_id=extracted_result.run_id
        )

        # Parse and validate extraction result
        extracted_event: ExtractedEvent | ExtractedInvalidEvent
        try:
            schema_class = get_extraction_schema(extract_config.json_schema)
            data = ExtractedData.from_extraction_result(
                result=extract_run,
                schema=schema_class,
                file_name=state.filename,
                file_id=state.file_id,
                file_hash=state.file_hash,
            )
            extracted_event = ExtractedEvent(data=data)
        except InvalidExtractionData as e:
            logger.error(f"Extraction validation failed: {e}", exc_info=True)
            extracted_event = ExtractedInvalidEvent(data=e.invalid_item)

        # Stream extraction event to client
        ctx.write_event_to_stream(extracted_event)

        # Validate and compute metrics
        extracted_data = extracted_event.data
        data_dict = extracted_data.model_dump()

        validation_result = self._validate_extraction(
            data_dict, validation_config, extract_run
        )

        # Build provenance entry
        provenance = self._build_provenance(
            state, data_dict, validation_result, extract_run
        )

        # Add provenance to data
        data_dict["_provenance"] = provenance.model_dump()
        data_dict["_validation"] = validation_result.model_dump()

        # Remove past data for same file
        if extracted_data.file_hash is not None:
            delete_result = await llama_cloud_client.beta.agent_data.delete_by_query(
                deployment_name=agent_name or "_public",
                collection=EXTRACTED_DATA_COLLECTION,
                filter={"file_hash": {"eq": extracted_data.file_hash}},
            )
            if delete_result.deleted_count > 0:
                logger.info(
                    f"Removed {delete_result.deleted_count} existing record(s) for {state.filename}"
                )

        # Save to Agent Data
        item = await llama_cloud_client.beta.agent_data.agent_data(
            data=data_dict,
            deployment_name=agent_name or "_public",
            collection=EXTRACTED_DATA_COLLECTION,
        )

        entity_count = len(data_dict.get("data", {}).get("entities", []))
        relation_count = len(data_dict.get("data", {}).get("relations", []))

        status_msg = f"Extracted {entity_count} entities and {relation_count} relations"
        if validation_result.flagged_for_review:
            status_msg += " (flagged for review)"

        ctx.write_event_to_stream(Status(level="info", message=status_msg))

        logger.info(
            f"KG extraction complete for {state.filename}: "
            f"{entity_count} entities, {relation_count} relations, "
            f"provenance_id={provenance.extraction_id}"
        )

        return StopEvent(result=item.id)

    def _validate_extraction(
        self,
        data_dict: dict[str, Any],
        validation_config: ValidationConfig,
        extract_run: Any,
    ) -> ValidationResult:
        """Validate extraction results against configuration rules."""
        issues: list[str] = []
        flagged = False

        extracted = data_dict.get("data", {})
        entities = extracted.get("entities", [])
        relations = extracted.get("relations", [])

        # Check for empty extraction
        if not entities and not relations:
            issues.append("No entities or relations extracted")
            flagged = True

        # Check source context if required
        if validation_config.require_source_context:
            for i, entity in enumerate(entities):
                if not entity.get("source_context"):
                    issues.append(f"Entity '{entity.get('name', i)}' missing source context")

        # Check confidence scores from metadata
        field_metadata = {}
        if hasattr(extract_run, "extraction_metadata"):
            field_metadata = getattr(
                extract_run.extraction_metadata, "field_metadata", {}
            ) or {}

        low_confidence_fields = []
        if field_metadata:
            self._check_confidence_recursive(
                field_metadata,
                validation_config.confidence_threshold,
                low_confidence_fields,
                "",
            )

        if low_confidence_fields and validation_config.flag_low_confidence_for_review:
            flagged = True
            issues.append(
                f"{len(low_confidence_fields)} field(s) below confidence threshold"
            )

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            flagged_for_review=flagged,
        )

    def _check_confidence_recursive(
        self,
        metadata: dict[str, Any],
        threshold: float,
        low_confidence: list[str],
        path: str,
    ) -> None:
        """Recursively check confidence scores in field metadata."""
        for key, value in metadata.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                confidence = value.get("confidence")
                if confidence is not None and confidence < threshold:
                    low_confidence.append(current_path)
                # Recurse into nested structures
                self._check_confidence_recursive(
                    value, threshold, low_confidence, current_path
                )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._check_confidence_recursive(
                            item, threshold, low_confidence, f"{current_path}[{i}]"
                        )

    def _build_provenance(
        self,
        state: ExtractionState,
        data_dict: dict[str, Any],
        validation_result: ValidationResult,
        extract_run: Any,
    ) -> ProvenanceEntry:
        """Build provenance entry for scientific traceability."""
        extracted = data_dict.get("data", {})
        entities = extracted.get("entities", [])
        relations = extracted.get("relations", [])

        # Calculate average confidence if available
        avg_confidence = None
        low_confidence_count = 0
        if hasattr(extract_run, "extraction_metadata"):
            metadata = extract_run.extraction_metadata
            if hasattr(metadata, "field_metadata") and metadata.field_metadata:
                confidences = []
                self._collect_confidences(metadata.field_metadata, confidences)
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    low_confidence_count = sum(1 for c in confidences if c < 0.7)

        return ProvenanceEntry(
            extraction_id=state.extraction_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_id=state.file_id or "",
            file_name=state.filename,
            document_type=state.document_type,
            classification_confidence=state.classification_confidence,
            entity_count=len(entities),
            relation_count=len(relations),
            avg_extraction_confidence=avg_confidence,
            low_confidence_count=low_confidence_count,
            flagged_for_review=validation_result.flagged_for_review,
            validation_issues=validation_result.issues,
        )

    def _collect_confidences(
        self, metadata: dict[str, Any], confidences: list[float]
    ) -> None:
        """Collect all confidence scores from metadata."""
        for value in metadata.values():
            if isinstance(value, dict):
                conf = value.get("confidence")
                if conf is not None:
                    confidences.append(conf)
                self._collect_confidences(value, confidences)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._collect_confidences(item, confidences)


workflow = KnowledgeGraphExtractionWorkflow(timeout=None)

if __name__ == "__main__":
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main() -> None:
        file = await get_llama_cloud_client().files.create(
            file=Path("test.pdf").open("rb"),
            purpose="extract",
        )
        await workflow.run(start_event=FileEvent(file_id=file.id))

    asyncio.run(main())
