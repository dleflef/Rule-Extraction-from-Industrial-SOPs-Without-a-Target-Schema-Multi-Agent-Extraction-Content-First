"""
Tests for the Knowledge Graph Extraction workflow.

Tests the classification, extraction, and validation pipeline
for domain-agnostic knowledge graph construction.
"""

import json
import warnings
from pathlib import Path

import pytest
from extraction_review.clients import fake
from extraction_review.config import EXTRACTED_DATA_COLLECTION
from extraction_review.metadata_workflow import MetadataResponse
from extraction_review.metadata_workflow import workflow as metadata_workflow
from extraction_review.process_file import FileEvent
from extraction_review.process_file import workflow as process_file_workflow
from workflows.events import StartEvent


def get_extraction_schema() -> dict:
    """Load the extraction schema from the unified config file."""
    config_path = Path(__file__).parent.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text())
    return config["extract"]["json_schema"]


def get_ontology_config() -> dict:
    """Load the ontology configuration from the config file."""
    config_path = Path(__file__).parent.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text())
    return config["ontology"]


@pytest.mark.asyncio
async def test_process_file_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the full KG extraction workflow including classification and extraction."""
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")

    if fake is None:
        warnings.warn(
            "Skipping test because it cannot be mocked. "
            "Set `FAKE_LLAMA_CLOUD=true` in your environment to enable this test..."
        )
        return

    # Load test file into mock server
    file_id = fake.files.preload(path="tests/files/test.pdf")

    # Run the workflow
    result = await process_file_workflow.run(start_event=FileEvent(file_id=file_id))

    # Verify result is a valid agent data ID
    assert result is not None
    assert isinstance(result, str)
    assert len(result) == 7  # Agent data IDs are 7 characters


@pytest.mark.asyncio
async def test_metadata_workflow() -> None:
    """Test that metadata workflow returns ontology and schema configuration."""
    result = await metadata_workflow.run(start_event=StartEvent())

    assert isinstance(result, MetadataResponse)
    assert result.extracted_data_collection == EXTRACTED_DATA_COLLECTION
    assert result.json_schema == get_extraction_schema()

    # Verify ontology is included
    ontology = get_ontology_config()
    assert len(result.ontology["entity_types"]) == len(ontology["entity_types"])
    assert len(result.ontology["relation_types"]) == len(ontology["relation_types"])

    # Verify document types from classification rules
    assert len(result.document_types) > 0

    # Verify validation config is included
    assert "confidence_threshold" in result.validation
    assert "require_source_context" in result.validation
