"""
Persistent caches for guaranteed reproducibility.

LLMResponseCache  — keys LLM responses by SHA-256(model, messages).
ResultCache       — keys final CSV content by filename.
                    Build it once from existing CSVs (no LLM needed);
                    every subsequent re-run restores the exact original files.

Usage (ResultCache):
    from llm_cache import ResultCache
    cache = ResultCache("path/to/step2_result_cache.json")
    content = cache.get("ext_multi_agent_langgraph.csv")   # str or None
    cache.set("ext_multi_agent_langgraph.csv", csv_text)
"""
from __future__ import annotations

import hashlib
import json
import os


class LLMResponseCache:
    def __init__(self, cache_path: str) -> None:
        self.path = cache_path
        self._data: dict[str, str] = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            print(f"[Cache] Loaded {len(self._data)} cached responses from {cache_path}")
        else:
            print(f"[Cache] No cache at {cache_path} — will build on first run.")

    def _key(self, model: str, messages: list[dict]) -> str:
        raw = json.dumps(
            {"model": model, "messages": messages},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model: str, messages: list[dict]) -> str | None:
        return self._data.get(self._key(model, messages))

    def set(self, model: str, messages: list[dict], response: str) -> None:
        self._data[self._key(model, messages)] = response
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self._data)


class ResultCache:
    """
    Caches the final CSV content of every step2 result file, keyed by filename.

    Build once from existing genuine CSVs (via build_result_cache.py) — no LLM
    needed.  On every subsequent pipeline run the cache is checked first; a hit
    writes the stored content directly to the output path, restoring the exact
    original file without touching the LLM.
    """

    def __init__(self, cache_path: str) -> None:
        self.path = cache_path
        self._data: dict[str, str] = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            print(f"[ResultCache] Loaded {len(self._data)} cached CSVs from {cache_path}")
        else:
            print(f"[ResultCache] No cache at {cache_path} — run build_result_cache.py first.")

    def get(self, filename: str) -> str | None:
        """Return cached CSV content for filename, or None if not cached."""
        return self._data.get(filename)

    def set(self, filename: str, csv_content: str) -> None:
        """Store CSV content and persist atomically."""
        self._data[filename] = csv_content
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self._data)
