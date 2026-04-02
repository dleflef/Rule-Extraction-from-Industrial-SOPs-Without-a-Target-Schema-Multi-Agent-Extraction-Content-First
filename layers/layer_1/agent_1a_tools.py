"""
layer_1/agent_1a_tools.py — Agent 1A Deterministic Tools

Implements the purely programmatic tools for Layer 1 ingestion.
No LLM calls are made here. The focus is exclusively on reading 
structured files, inferring schemas, and standardizing data formats.

Tool summary
────────────
  extract_schema(file_path)
      Reads headers + 5 rows, returns a schema dict.
      Handles .csv and .json; guarded against encoding / parse errors.

  standardize_data(file_path)
      Loads the file with Pandas, standardizes column headers 
      (lowercase, underscore separated), and returns a clean dictionary
      payload ready for Layer 2.
"""

import json
import os
from typing import Any, Dict

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Schema Extractor (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def extract_schema(file_path: str) -> Dict[str, Any]:
    """
    Stage 1 (Programmatic Schema Extraction).

    Reads ONLY the column headers and the first 5 rows of a structured
    file to generate a compact schema snapshot.

    Returns a dict with keys:
      file_type      : 'csv' or 'json'
      columns        : list of column / key names
      dtypes         : {col: dtype_str} (CSV only)
      sample_rows    : list of up to 5 row dicts
      total_columns  : int

    On failure, returns {"error": "<reason>"}.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, nrows=5)
            return {
                "file_type": "csv",
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_rows": df.head(5).to_dict(orient="records"),
                "total_columns": len(df.columns),
            }

        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                data = json.load(fh)

            if isinstance(data, list) and len(data) > 0:
                sample = data[:5]
                keys = list(sample[0].keys()) if isinstance(sample[0], dict) else []
            elif isinstance(data, dict):
                keys = list(data.keys())
                sample = [{k: (str(v)[:120] if not isinstance(v, (int, float, bool)) else v)
                           for k, v in data.items()}]
            else:
                return {"error": "Unsupported JSON structure: root must be a list or dict."}

            return {
                "file_type": "json",
                "columns": keys,
                "sample_rows": sample,
                "total_columns": len(keys),
            }

        else:
            return {"error": f"Unsupported file extension '{ext}'. Agent 1A handles .csv and .json only."}

    except Exception as exc:
        return {"error": f"Schema extraction failed: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Data Standardizer (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def standardize_data(file_path: str) -> Dict[str, Any]:
    """
    Stage 2 (Programmatic Data Standardization).

    Loads the full file and normalizes the structure so Layer 2 agents
    receive a consistent data format regardless of the source file type.
    
    Operations:
    1. Standardize column names (lowercase, replace spaces with underscores).
    2. Convert data into a standardized list-of-dicts format.
    3. Fill NaNs to prevent JSON serialization errors later in the pipeline.

    Returns a dict with keys:
        status           : 'success'
        total_rows       : int
        cleaned_columns  : list[str]
        data             : list[dict] (The normalized dataset)

    On failure, returns {"error": "<reason>"}.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in (".csv", ".json"):
        return {"error": f"Unsupported file extension '{ext}'."}

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:  # .json
            raw = pd.read_json(file_path)
            df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)

        # Standardize column headers
        df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
        
        # Handle NaN values to ensure safe JSON serialization for downstream agents
        df = df.fillna("")

        # Convert to a standard list of dictionaries
        normalized_records = df.to_dict(orient="records")

        return {
            "status": "success",
            "total_rows": len(df),
            "cleaned_columns": list(df.columns),
            "data": normalized_records,
        }

    except Exception as exc:
        return {"error": f"Data standardization failed: {exc}"}