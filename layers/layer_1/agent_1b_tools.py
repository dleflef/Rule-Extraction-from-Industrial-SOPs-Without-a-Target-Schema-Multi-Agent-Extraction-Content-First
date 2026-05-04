# layer_1/agent_1b_tools.py — Agent 1B Deterministic Tool (domain-agnostic, v2)
#
# Hybrid parsing strategy:
#   - Tables  → Docling native table objects → clean DataFrame text (bypasses MD export corruption)
#   - Prose   → Markdown chunking with heading / rule-ID boundary detection
#
# No factory names, station names, or domain vocabulary are used anywhere.
# All patterns are purely structural.

import html   # used to decode HTML entities (e.g., &amp;)
import os
import re
from typing import Any, Dict, List, Optional

try:
    from docling.document_converter import DocumentConverter
    _HAS_DOCLING = True     # Docling is available, can convert documents
except ImportError:
    _HAS_DOCLING = False    # will raise a clear error if parsing is attempted

# ── Constants ─────────────────────────────────────────────────────────────────
# Supported file extensions (lowercase). The parser fails early for anything else.
_SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".txt"}

# Minimum characters for a chunk to remain standalone; smaller chunks are merged.
_MIN_CHUNK_CHARS = 100
# Maximum characters before a chunk is forcibly split (to keep LLM context window small).
_MAX_CHUNK_CHARS = 3500

# Match any line that starts with a Markdown heading (#)
_HEADING_PATTERN = re.compile(r"^#")
# Match lines like "1.2.3 Description" or bullet-numbered items that structure the document.
_NUMBERED_ITEM_PATTERN = re.compile(r"^\s*[-*•]*\s*\d+(\.\d+)+\s+\w")
# Match rule IDs like "MAINT-02" or "RULE-ST01-01". Only used for chunk boundaries when NOT inside a table row.
_RULE_ID_PATTERN = re.compile(r"^[A-Z]{2,}-[A-Za-z0-9_-]+")
# Lines starting with a pipe character indicate a markdown table row.
_TABLE_LINE_PATTERN = re.compile(r"^\|")
# Detects the "--- | --- | ---" separator row emitted by _tables_to_structured_text.
# This helps identify a clean DataFrame-generated table inside the merged markdown.
_CLEAN_TABLE_SEP = re.compile(r"^-{3,}( \| -{3,})+$", re.MULTILINE)

# Detects merged threshold header cells like "WARN_LONominal" → should be split into "WARN_LO" and "Nominal".
_MERGED_THRESHOLD_HEADER = re.compile(r"^([A-Z]+(?:_[A-Z]+)*_(?:LO|HI))([A-Z][a-z]\w*)$")


def _fix_threshold_header_cells(cells: List[str]) -> List[str]:
    """
    Fix a merged threshold header like ['CRIT_LO', '', 'WARN_LONominal', 'WARN_HI', ...]
    → ['CRIT_LO', 'WARN_LO', 'Nominal', 'WARN_HI', ...].

    When a cell matches WARN_LONominal and the preceding cell is empty, the empty cell
    is replaced with the prefix (WARN_LO) and the merged cell with the suffix (Nominal).
    """
    # Make a copy so we don't mutate the original list in case of unexpected reuse.
    result = list(cells)
    # Scan the header row for the problematic merged cell.
    for i, cell in enumerate(result):
        m = _MERGED_THRESHOLD_HEADER.match(cell.strip())
        if m:
            # We only fix if the previous cell is empty; otherwise it might be a legitimate separate column.
            if i > 0 and not result[i - 1].strip():
                # Fill the blank cell with the prefix part (e.g., "WARN_LO")
                result[i - 1] = m.group(1)
                # Replace the merged cell with the suffix (e.g., "Nominal")
                result[i] = m.group(2)
            # Only one such merge per header row is expected, so break after fixing.
            break
    return result


def _is_chunk_boundary(line: str) -> bool:
    """Return True if `line` should start a new chunk."""
    # Headings and numbered items are always new chunk boundaries.
    if _HEADING_PATTERN.match(line) or _NUMBERED_ITEM_PATTERN.match(line):
        return True
    # Rule IDs are boundaries only in prose context.
    # Lines containing " | " are clean table data rows (e.g. "MAINT-01 | VIB | ...") — never split on those.
    if _RULE_ID_PATTERN.match(line) and " | " not in line:
        return True
    return False


def _sanitize_table_text(table_text: str) -> str:
    """
    Clean a pipe-delimited table string through three deterministic passes:
      1. Fix merged threshold header cells, e.g. WARN_LONominal → WARN_LO | Nominal,
         and remove the spurious empty column that accompanies the merge.
      2. Collapse consecutive identical adjacent cells in each data row — the symptom
         of Docling repeating a merged cell's value across all spanned columns.
      3. Remove columns that are empty in every data row.
    """
    # Split table into lines, ignoring blank lines.
    lines = [ln for ln in table_text.splitlines() if ln.strip()]
    # If there's only a header (or less), nothing to sanitize; return as-is.
    if len(lines) < 2:
        return table_text

    # Helper: parse a pipe-delimited row into a list of trimmed cells.
    def _parse_row(line: str) -> List[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    # Helper: check if a row is a separator line (e.g., "--- | --- | ---").
    def _is_sep(cells: List[str]) -> bool:
        return bool(cells) and all(re.match(r"^-+$", c) for c in cells if c)

    # Parse every line and mark which are separators.
    rows = [_parse_row(ln) for ln in lines]
    sep_flags = [_is_sep(r) for r in rows]

    # 1. Fix merged threshold header cells (only in the first non-separator row, i.e., the header)
    if rows and not sep_flags[0]:
        rows[0] = _fix_threshold_header_cells(rows[0])

    # 2. Collapse consecutive identical adjacent cells per non-separator row.
    # This removes duplicates caused by Docling's merged cell export.
    for i, (row, is_sep) in enumerate(zip(rows, sep_flags)):
        if is_sep or not row:
            continue  # skip separator rows and empty rows
        deduped = [row[0]]  # keep the first cell always
        for cell in row[1:]:
            # If this cell is non-empty and equal to the previous kept cell, skip it.
            if cell and cell == deduped[-1]:
                continue
            deduped.append(cell)
        rows[i] = deduped

    # 3. Pad to max width, then drop columns that are empty in every data row.
    # First, find the maximum number of columns across all rows.
    ncols = max((len(r) for r in rows), default=0)
    if ncols == 0:
        return table_text  # safety: no columns at all, return original
    # Pad shorter rows with empty strings so every row has ncols columns.
    for row in rows:
        while len(row) < ncols:
            row.append("")

    # Identify which indices correspond to non-separator rows (data rows).
    data_idx = [i for i, s in enumerate(sep_flags) if not s]
    # Determine which columns have at least one non-empty cell in any data row.
    keep = [c for c in range(ncols) if any(rows[i][c] for i in data_idx)]
    if not keep:
        return table_text  # no column has any content, give up

    # Keep only the desired columns in every row.
    rows = [[row[c] for c in keep] for row in rows]
    ncols = len(keep)
    # Rebuild separator rows to match the new column count (fill with "---").
    for i, is_sep in enumerate(sep_flags):
        if is_sep:
            rows[i] = ["---"] * ncols

    # Reconstruct the table string: rows joined by " | " and newlines.
    return "\n".join(" | ".join(row) for row in rows)


def _tables_to_structured_text(result) -> List[Optional[str]]:
    """
    Export each table in the Docling result as clean pipe-delimited text.
    Returns a list in document order; None means extraction failed for that table.
    Uses the native DataFrame API to bypass Docling's broken Markdown cell merging.
    """
    doc = result.document
    table_texts: List[Optional[str]] = []
    for table in doc.tables:
        try:
            # Get the table as a pandas DataFrame — this avoids the corrupted MD export.
            df = table.export_to_dataframe(doc=doc)
            if df is None or df.empty:
                table_texts.append(None)   # empty table, mark as missing
                continue
            lines = []
            # Build header row from DataFrame columns.
            cols = [str(c).strip().replace("\n", " ") for c in df.columns]
            lines.append(" | ".join(cols))
            # Build separator row (each column gets a "---" of minimum length 3).
            lines.append(" | ".join("-" * max(3, len(c)) for c in cols))
            # Build data rows.
            for _, row in df.iterrows():
                cells = [str(v).strip().replace("\n", " ") for v in row]
                lines.append(" | ".join(cells))
            table_texts.append("\n".join(lines))
        except Exception:
            table_texts.append(None)   # any error → treat as unextractable
    return table_texts


def _substitute_tables_in_markdown(markdown_text: str, table_texts: List[Optional[str]]) -> str:
    """
    Walk markdown line by line. Replace each pipe-prefixed table block with
    the corresponding clean DataFrame text when it contains no U+FFFD loss,
    otherwise fall back to the raw Markdown table. Both paths are then passed
    through _sanitize_table_text to remove structural artefacts.
    """
    # Helper: decides whether the clean DataFrame text is usable (no corruption).
    def _prefer_dataframe(clean: Optional[str]) -> bool:
        return clean is not None and "�" not in clean  # U+FFFD is the Unicode replacement character

    # Split the entire markdown into lines but keep the line endings for later reassembly.
    lines = markdown_text.splitlines(keepends=True)
    out: List[str] = []
    table_idx = 0               # which table in the list we are currently at
    in_table = False            # are we inside a table block?
    broken_buf: List[str] = []  # buffer for lines that belong to the current table block

    for line in lines:
        # Detect if this line is part of a markdown table (starts with pipe, allowing leading whitespace).
        is_pipe = bool(_TABLE_LINE_PATTERN.match(line.lstrip()))
        if is_pipe:
            if not in_table:
                # Start of a new table block.
                in_table = True
                broken_buf = []
            broken_buf.append(line)
        else:
            # We hit a non-table line. If we were inside a table block, flush it.
            if in_table:
                # Get the corresponding DataFrame text (if available)
                df_text = table_texts[table_idx] if table_idx < len(table_texts) else None
                # Use the raw markdown lines as fallback
                raw = "".join(broken_buf)
                # Choose the clean version if it passes the corruption check, else fallback.
                chosen = df_text if _prefer_dataframe(df_text) else raw
                # Append the (sanitized) chosen table representation to output.
                out.append(_sanitize_table_text(chosen) + "\n")
                table_idx += 1
                in_table = False
                broken_buf = []
            # Always append the non-table line as-is.
            out.append(line)

    # If document ended while inside a table, flush the last one.
    if in_table:
        df_text = table_texts[table_idx] if table_idx < len(table_texts) else None
        raw = "".join(broken_buf)
        chosen = df_text if _prefer_dataframe(df_text) else raw
        out.append(_sanitize_table_text(chosen) + "\n")

    # Combine all lines back into a single string.
    return "".join(out)


def _normalise(text: str) -> str:
    """Decode HTML entities and strip Markdown escaped underscores."""
    # Convert HTML entities like &amp; &lt; etc. to real characters.
    text = html.unescape(text)
    # Docling escapes underscores with a backslash; remove those backslashes.
    text = text.replace("\\_", "_")
    return text


def _merge_small_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge chunks shorter than _MIN_CHUNK_CHARS into the previous one.
    Never merges table chunks or rule-ID-headed chunks — those stay isolated.
    """
    if not chunks:
        return []

    # Helper: a chunk is a "rule‑ID chunk" if its first heading matches the rule ID pattern.
    def _is_rule_id_chunk(ch: Dict[str, Any]) -> bool:
        headings = ch["metadata"].get("headings", [])
        return bool(headings and _RULE_ID_PATTERN.match(headings[0]))

    # Helper: checks whether the chunk is a table (based on its recorded chunk_type).
    def _is_table_chunk(ch: Dict[str, Any]) -> bool:
        return ch["metadata"].get("chunk_type") == "table"

    # A chunk may be merged only if it is NOT a rule‑ID chunk and NOT a table chunk.
    def _mergeable(ch: Dict[str, Any]) -> bool:
        return not _is_rule_id_chunk(ch) and not _is_table_chunk(ch)

    # Combine two chunks into one, merging metadata fields.
    def _combine(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        combined = a["content"] + "\n" + b["content"]
        return {
            "chunk_id": b["chunk_id"],               # take the ID of the later chunk
            "content": combined,
            "metadata": {
                # Use headings from b if available, else from a.
                "headings": b["metadata"]["headings"] or a["metadata"]["headings"],
                # Merge page numbers (unique, sorted).
                "page_numbers": sorted(set(a["metadata"]["page_numbers"] +
                                           b["metadata"]["page_numbers"])),
                # If either was a table, mark combined as table.
                "is_table": a["metadata"]["is_table"] or b["metadata"]["is_table"],
                # Prefer the chunk_type of the later chunk; default to "prose".
                "chunk_type": (b["metadata"].get("chunk_type") or
                               a["metadata"].get("chunk_type", "prose")),
            },
            "char_count": len(combined),
        }

    merged = [chunks[0]]
    # Walk through chunks from left to right.
    for ch in chunks[1:]:
        prev = merged[-1]
        # If the previous chunk is small and both are mergeable, merge them.
        if prev["char_count"] < _MIN_CHUNK_CHARS and _mergeable(prev) and _mergeable(ch):
            merged.pop()
            merged.append(_combine(prev, ch))
        else:
            merged.append(ch)

    # Final safety check: if the last chunk is still small and can be merged with the one before it, do so.
    if (len(merged) >= 2
            and merged[-1]["char_count"] < _MIN_CHUNK_CHARS
            and _mergeable(merged[-1])
            and _mergeable(merged[-2])):
        last = merged.pop()
        prev = merged.pop()
        merged.append(_combine(prev, last))

    return merged


def _split_large_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Break chunks > _MAX_CHUNK_CHARS at paragraph breaks."""
    out = []
    for c in chunks:
        # If the chunk is small enough, keep it as-is.
        if c["char_count"] <= _MAX_CHUNK_CHARS:
            out.append(c)
            continue
        # Try to split on blank lines (paragraph boundaries).
        paragraphs = re.split(r"\n\s*\n", c["content"])
        headings = c["metadata"]["headings"]
        chunk_type = c["metadata"].get("chunk_type", "prose")
        # If there's only one paragraph (or it's one wall of text), brute-force split by character count.
        if len(paragraphs) <= 1:
            for i in range(0, len(c["content"]), _MAX_CHUNK_CHARS):
                seg = c["content"][i:i + _MAX_CHUNK_CHARS]
                out.append({
                    "content": seg,
                    "metadata": {
                        "headings": headings,
                        "page_numbers": c["metadata"]["page_numbers"],
                        "is_table": c["metadata"]["is_table"],
                        "chunk_type": chunk_type,
                    },
                    "char_count": len(seg),
                })
        else:
            # Split at each paragraph boundary, skipping empty paragraphs.
            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue
                out.append({
                    "content": p,
                    "metadata": {
                        "headings": headings,
                        "page_numbers": c["metadata"]["page_numbers"],
                        "is_table": c["metadata"]["is_table"],
                        "chunk_type": chunk_type,
                    },
                    "char_count": len(p),
                })
    return out


def parse_pdf_to_markdown_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Hybrid document parser: tables via Docling native objects, prose via Markdown chunking.

    Returns list of dicts with keys: chunk_id, content, metadata, char_count.
    metadata includes: headings, page_numbers, is_table, chunk_type ("table"|"prose").
    """
    # Early error if Docling is not installed.
    if not _HAS_DOCLING:
        raise ImportError("Docling is required. Install with: pip install docling")
    # Check file existence.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # Validate file extension.
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _SUPPORTED:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {sorted(_SUPPORTED)}")

    # 1. Convert document and extract clean table representations
    converter = DocumentConverter()
    result = converter.convert(file_path)
    # Produce clean pipe-delimited strings for every detected table.
    table_texts = _tables_to_structured_text(result)

    # 2. Export to Markdown, substitute broken table blocks, normalise text
    markdown_text = result.document.export_to_markdown()
    # Replace corrupted table blocks with the clean versions.
    markdown_text = _substitute_tables_in_markdown(markdown_text, table_texts)
    # Clean up HTML entities and escaped underscores.
    markdown_text = _normalise(markdown_text)

    # 3. Single-pass chunking by structural boundary lines
    lines = markdown_text.splitlines(keepends=True)  # keep line endings for exact reassembly
    chunks: List[Dict[str, Any]] = []
    buffer_lines: List[str] = []
    current_section: str = ""  # remembers the most recent non-rule-ID section heading

    # Inner function: finalize the current buffer into a chunk object and append to list.
    def flush_buffer() -> None:
        nonlocal current_section  # we read it, but do not modify it here (except when we could, but we don't)
        if not buffer_lines:
            return
        full = "".join(buffer_lines).strip()
        if not full:
            return  # discard empty chunks

        first_line = buffer_lines[0].strip()
        headings: List[str] = []
        # If the first line is a boundary, extract a clean heading from it.
        if _is_chunk_boundary(first_line):
            clean_h = first_line.lstrip("#-*• \t").strip()
            if clean_h:
                headings.append(clean_h)

        # Determine if this chunk is a rule ID that needs section context prepended.
        # A rule-ID chunk is a line that matches the rule ID pattern, is not a table row,
        # and is not already a heading or numbered item.
        is_rule_id_chunk = (
            _RULE_ID_PATTERN.match(first_line)
            and " | " not in first_line
            and not _HEADING_PATTERN.match(first_line)
            and not _NUMBERED_ITEM_PATTERN.match(first_line)
        )
        # Inject the current section context so the LLM knows which station the rule belongs to.
        if is_rule_id_chunk and current_section:
            full = f"## Section: {current_section}\n{full}"

        # Detect if the chunk is a table: either an original Markdown table (|---|) or a clean DataFrame table.
        has_md_table = "|---" in full or "| ---" in full or "|:--" in full
        has_clean_table = bool(_CLEAN_TABLE_SEP.search(full))
        is_table = has_md_table or has_clean_table
        chunk_type = "table" if is_table else "prose"

        # Append the built chunk (with temporary id 0; later reassigned).
        chunks.append({
            "chunk_id": 0,
            "content": full,
            "metadata": {
                "headings": headings,
                "page_numbers": [],
                "is_table": is_table,
                "chunk_type": chunk_type,
            },
            "char_count": len(full),
        })

    # Main loop over every line.
    for line in lines:
        stripped = line.strip()
        # If we encounter a boundary and the buffer already has content, flush the previous chunk.
        if _is_chunk_boundary(stripped) and buffer_lines:
            flush_buffer()
            # Start a new buffer with this boundary line.
            buffer_lines = [line]
            # Update the current section tracker, but only if this is NOT a rule-ID line.
            # We don't want rule IDs (like "RULE-ST01-01: ...") to become new section names.
            if not (_RULE_ID_PATTERN.match(stripped) and " | " not in stripped):
                section_text = stripped.lstrip("#-*•0123456789. \t").strip()
                if section_text:
                    current_section = section_text
        else:
            # Otherwise just add the line to the buffer.
            buffer_lines.append(line)
    # Don't forget the last chunk after the loop ends.
    flush_buffer()

    # 4. Post-process: merge small prose fragments, split oversized chunks
    merged = _merge_small_chunks(chunks)
    final = _split_large_chunks(merged)

    # Assign sequential chunk IDs from 0 upward.
    for idx, c in enumerate(final):
        c["chunk_id"] = idx

    return final