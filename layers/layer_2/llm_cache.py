"""
Persistent LLM response cache for exact replay.

LLMResponseCache keys responses by SHA-256(model, messages): a warm cache
reproduces a previous run exactly, and any change to a prompt, model, or
document changes the key and forces a real call.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading


class LLMResponseCache:
    def __init__(self, cache_path: str) -> None:
        self.path = cache_path
        self._lock = threading.Lock()
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
        with self._lock:
            return self._data.get(self._key(model, messages))

    def set(self, model: str, messages: list[dict], response: str) -> None:
        """Record a response and persist immediately.

        Locked and written atomically because callers fan out concurrently
        (chunk extraction and chunk verification both run in parallel). Two
        threads rewriting the whole file at once would interleave and leave
        truncated JSON on disk -- which would then fail to load on the next
        run, silently discarding every cached response and destroying the
        reproducibility this class exists to provide. Writing to a temp file
        and renaming makes the replacement atomic, so an interrupted run
        leaves the previous cache intact rather than a half-written one.
        """
        with self._lock:
            self._data[self._key(model, messages)] = response
            tmp = f"{self.path}.tmp{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self._data)
