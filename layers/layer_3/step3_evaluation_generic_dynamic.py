"""step3_evaluation_generic_dynamic.py
==================================================
Fully schema-agnostic evaluator for step2_multi_agent_generic.py's output.

Category exact-match is structurally unwinnable in this setting: step2's
categories are discovered fresh per document (e.g. "EnvironmentalLimit",
"TriggeredResponse") and cannot be expected to equal a ground truth's own
fixed taxonomy (e.g. "OperationalRule", "ThresholdRule") -- not because
either is wrong, but because nothing ties their vocabularies together.
Per-domain --gt-*-field flags are equally unusable, since they require a
human to read each ground truth's columns before evaluation can run.

This evaluator therefore assumes NOTHING about either side's field names. A
ground truth row and a prediction row are each collapsed into one text blob
(every column's value, whatever it's called, joined together) and compared
holistically:
  - semantic similarity of the two blobs (SBERT cosine) -- this is what lets
    "EnvironmentalLimit ... TMP ... 175 195" line up with "ThresholdRule ...
    TMP ... 175.0 195.0" despite neither side knowing the other's vocabulary.
  - a numeric-overlap bonus: every number written anywhere in the GT row
    should appear somewhere in the matched prediction row, regardless of
    which field (or which JSON key inside "attributes") holds it. This is
    what actually rewards getting bound values right, since sentence
    embeddings alone are weak at distinguishing "175" from "195".
Only two columns are treated specially, because they're bookkeeping, not rule
content: an id column and any source/provenance-like column (a filename match
is not evidence of content match). Both are found STRUCTURALLY, not by a
column-name whitelist: the id column is whichever one has a distinct, short
value on (almost) every row -- the statistical signature of an identifier --
and a provenance column is whichever one holds the exact same value on every
row (unlike a real categorical content field such as category/severity, which
varies across several values). A small set of common names (id, ruleId...)
only nudges a close tie; it is never the sole mechanism, so a ground truth
whose id or source column is named something this file has never seen still
resolves correctly.

THE METRIC. F1_content, and only F1_content: Hungarian-assignment content
matching over the blended score above. One headline number, because reporting
several F1 variants side by side invites quoting whichever is highest.

Alongside it the evaluator reports an id-reuse DIAGNOSTIC -- what fraction of
the ground truth's identifiers appear verbatim among the predicted ids. It is
deliberately not expressed as a precision/recall/F1 triple, because it is a
property of the ground truth's annotation convention rather than of the system
under test: a ground truth whose ids were invented by an annotator (R01, M001)
scores zero no matter how good the extraction is, while one that reused codes
printed in the source document (OP-ST-001) scores well. Read it as "did this
annotator reuse printed identifiers", never as an accuracy figure.

WHAT F1_content DOES NOT MEASURE, established empirically by
step3_validity_checks.py rather than asserted here:
  - word order within a record (scrambling tokens costs ~0.005; the comparison
    behaves as a bag of tokens and numbers)
  - which record an attribute is bound to (permuting payloads costs ~0.085)
  - category-label correctness, which is unwinnable across vocabularies by the
    argument above and is priced accordingly
It does measure numeric correctness (corrupting values costs ~0.29) and it does
discriminate the right document from a wrong one (predictions scored against a
foreign ground truth fall to 0.02-0.09, worst pairing 0.25).

Usage -- identical for every domain, no per-GT configuration required:
    python step3_evaluation_generic_dynamic.py --pred <csv> --gt <ground_truth.csv>
    python step3_evaluation_generic_dynamic.py --pred-dir <dir> --gt <ground_truth.csv>
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util

_LAYERS_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.normpath(os.path.join(_LAYERS_DIR, ".."))
STEP3_RESULTS_DIR = os.path.join(_LAYERS_DIR, "step3_results")
os.makedirs(STEP3_RESULTS_DIR, exist_ok=True)

# A handful of common id-column names, used only as a small tiebreak hint on
# top of structural detection below -- never as the sole mechanism, so a
# ground truth using a column name never seen before (e.g. "identifier",
# "ref_no") still resolves correctly.
_ID_LIKE_NAMES = {"id", "ruleid", "rule_id"}
# Our own pipeline's fixed bookkeeping columns (this evaluator's own output
# contract, not a guess about an external file) -- always dropped from every
# prediction row.
# source_span is provenance, not content: it records WHICH line of the source
# document an item was taken from. No ground truth carries such a column, so
# every token it contributes is unmatchable by construction and can only dilute
# a content comparison -- the same reason the ground truth's own source column
# is dropped on the other side. The exclusion is symmetric and decided by what
# the column IS, not by what including it would do to a score.
_PRED_BOOKKEEPING = {"id", "source_file", "chunk_id", "source_span"}


# ── Free parameters ────────────────────────────────────────────────────────────
# Every constant this evaluator uses is declared here rather than written
# inline, because each one is a researcher degree of freedom: a number a human
# chose, which could in principle have been chosen by looking at results on the
# very corpus being scored. Collecting them in one place makes them countable
# and auditable, and every one is overridable from the command line so a
# reported figure can be shown to be stable across the range rather than to
# depend on one setting -- see --sensitivity, which sweeps the scoring
# parameters and reports whether the RANKING of systems changes at all.
#
# The defaults below are conventional starting points, not values fitted to any
# ground truth in this project. Treat any result that moves under --sensitivity
# as a result about the threshold, not about the system.
@dataclass
class EvalConfig:
    gt_id_field: str = ""             # "" = auto-detect structurally
    gt_exclude_fields: list[str] = field(default_factory=list)
    # scoring
    threshold: float = 0.6            # content-agreement needed to count a match
    semantic_weight: float = 0.6      # weight on SBERT cosine
    numeric_weight: float = 0.4       # weight on numeric overlap
    # id-column detection
    id_uniqueness_min: float = 0.9    # min distinct-value fraction to qualify as an id
    id_len_penalty: float = 200.0     # divisor penalising long values (ids are short)
    id_name_hint_bonus: float = 0.05  # nudge for a conventional id name; never decisive alone
    # provenance-column detection (see _detect_noise_fields)
    noise_distinct_floor: int = 6     # a document-label column has at most this many
    noise_distinct_frac: float = 0.1  # ...or this fraction of the rows, whichever is larger
    noise_max_value_len: int = 24     # document codes are short
    # Number extraction. A hyphen inside an asset tag reads as a minus sign, so
    # "GSH-401" yields -401 on both sides of the comparison; such values are 15%
    # of all numbers extracted from these ground truths. Counting them is the
    # default because they are genuine asset-identity evidence, and it is the
    # CONSERVATIVE setting: dropping them raises cleanroom +0.240 and
    # desalination +0.105 F1 (dev and biogas unchanged), because those two
    # ground truths encode tag hyphens differently from the documents
    # (ASCII "-" vs U+2011). Declared here so the choice is visible and
    # reversible rather than buried in a regex.
    drop_tag_derived_numbers: bool = False
    # Embedding backend. all-MiniLM-L6-v2 is English-only, which matches this
    # project's stated scope (all evaluated documents are English). Exposed so
    # that scope is a declared choice rather than a buried constant.
    sbert_model: str = "all-MiniLM-L6-v2"


def _detect_id_field(gt_df, cfg: "EvalConfig") -> str:
    """Structural detection, not a name whitelist: the id column is whichever
    one has a distinct, short value for (almost) every row -- the
    statistical signature of an identifier -- so a never-before-seen ground
    truth resolves correctly without needing its column name added to a
    list first. A known common name only nudges a close tie."""
    if cfg.gt_id_field:
        return cfg.gt_id_field
    n = len(gt_df)
    if n == 0 or len(gt_df.columns) == 0:
        return gt_df.columns[0] if len(gt_df.columns) else ""
    scored = []
    for c in gt_df.columns:
        vals = gt_df[c].astype(str).str.strip()
        nonempty = vals[vals != ""]
        if len(nonempty) == 0:
            continue
        uniqueness = nonempty.nunique() / len(nonempty)
        if uniqueness < cfg.id_uniqueness_min:
            continue
        avg_len = nonempty.str.len().mean()
        hint_bonus = cfg.id_name_hint_bonus if c.strip().lower() in _ID_LIKE_NAMES else 0.0
        scored.append((uniqueness - avg_len / cfg.id_len_penalty + hint_bonus, c))
    if not scored:
        return gt_df.columns[0]
    return max(scored, key=lambda t: t[0])[1]


def _detect_noise_fields(gt_df, cfg: "EvalConfig" = None) -> set[str]:
    """Structural detection of provenance/bookkeeping columns: a column that
    holds the SAME value for (almost) every row (e.g. a source-document tag
    repeated on every line) is bookkeeping, not distinguishing rule content
    -- unlike a real categorical content field (category, severity), which
    varies across several values even though it repeats. No name list
    involved, so this generalizes to a ground truth whose provenance column
    is called something this evaluator has never seen."""
    cfg = cfg or EvalConfig()
    n = len(gt_df)
    if n <= 1:
        return set()
    noise = set()
    for c in gt_df.columns:
        vals = gt_df[c].astype(str).str.strip()
        nonempty = vals[vals != ""]
        if len(nonempty) < 2:
            continue
        if nonempty.nunique() == 1:
            noise.add(c.strip().lower())
            continue
        # A column naming the SOURCE DOCUMENT of each row is provenance too,
        # even though it varies -- a corpus of four documents gives it four
        # values, so the constant-value test above misses it. It is recognised
        # structurally instead: few distinct values, each looking like a
        # document label rather than rule content (short, no spaces, and
        # carrying a digit, as document codes almost always do). Predictions
        # never carry such a column, so leaving it in the blob is a uniform
        # handicap on every ground-truth row -- it can only add noise to the
        # comparison, never signal. A genuine categorical content field
        # (severity, category) fails the test because its values are words.
        if nonempty.nunique() <= max(cfg.noise_distinct_floor,
                                     int(len(gt_df) * cfg.noise_distinct_frac)):
            v = nonempty.unique()
            if all(len(str(x)) <= cfg.noise_max_value_len and " " not in str(x)
                   and any(ch.isdigit() for ch in str(x)) for x in v):
                noise.add(c.strip().lower())
    return noise


# ── Hardware / SBERT ───────────────────────────────────────────────────────────
# Keyed by model name so overriding cfg.sbert_model actually takes effect
# rather than silently returning whichever model happened to load first.
_sbert_models: Dict[str, SentenceTransformer] = {}
_DEFAULT_SBERT = "all-MiniLM-L6-v2"


def get_sbert(name: str = _DEFAULT_SBERT) -> SentenceTransformer:
    if name not in _sbert_models:
        print(f"    Loading SBERT model ({name}) on device: [CPU]...")
        _sbert_models[name] = SentenceTransformer(name, device="cpu")
    return _sbert_models[name]


def _norm_str_id(s: Any) -> str:
    if pd.isna(s) or s is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def _extract_numbers(text: str, drop_tag_derived: bool = False) -> list[float]:
    """Numbers written anywhere in a blob. A hyphen inside an asset tag reads as
    a minus sign ("GSH-401" -> -401); drop_tag_derived removes those, see
    EvalConfig.drop_tag_derived_numbers for why it is off by default."""
    out = []
    for tok in _NUMBER_RE.findall(text or ""):
        try:
            v = float(tok)
        except ValueError:
            continue
        if drop_tag_derived and v < 0:
            continue
        out.append(v)
    return out


def _numeric_pairing(gt_numbers: list[float], pred_numbers: list[float]
                      ) -> List[Tuple[float, Optional[float], float]]:
    """For each GT number, the predicted number that scores it best, and that
    score. Returned rather than only averaged so the audit trail can show WHICH
    predicted value was credited against WHICH ground-truth value, and how
    close it had to be -- the per-number detail behind numeric_overlap."""
    out = []
    for g in gt_numbers:
        best_p, best_s = None, 0.0
        for p in pred_numbers:
            denom = max(abs(g), abs(p), 1e-9)
            s = max(0.0, 1.0 - abs(g - p) / denom)
            if s > best_s or best_p is None:
                best_p, best_s = p, s
        out.append((g, best_p, best_s))
    return out


def _numeric_overlap(gt_numbers: list[float], pred_numbers: list[float]) -> float:
    """Fraction of the GT row's own numbers that have a close match somewhere
    in the predicted row's numbers, tolerance-scored and averaged -- rewards
    capturing the right values regardless of which field held them."""
    if not gt_numbers:
        return 0.0
    if not pred_numbers:
        return 0.0
    pairing = _numeric_pairing(gt_numbers, pred_numbers)
    return sum(s for _, _, s in pairing) / len(pairing)


# ── Blob construction ──────────────────────────────────────────────────────────
def _gt_blob(row: Dict, id_field: str, exclude: set[str]) -> str:
    parts = []
    for k, v in row.items():
        if k == id_field or k.strip().lower() in exclude:
            continue
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return " | ".join(parts)


def _parse_attributes(raw: Any) -> dict:
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _pred_blob(row: Dict) -> str:
    parts = []
    for k, v in row.items():
        if k in _PRED_BOOKKEEPING or k == "attributes":
            continue
        if v not in (None, "") and not (isinstance(v, float) and pd.isna(v)):
            parts.append(str(v).strip())
    attributes = _parse_attributes(row.get("attributes", ""))
    for v in attributes.values():
        if v not in (None, ""):
            parts.append(str(v).strip())
    return " | ".join(parts)


def _batch_encode_texts(texts: List[str], model_name: str = _DEFAULT_SBERT) -> Dict[str, Any]:
    unique = list({t for t in texts if t})
    if not unique:
        return {}
    model = get_sbert(model_name)
    embs = model.encode(unique, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
    return {t: embs[i] for i, t in enumerate(unique)}


def _semantic_similarity(text_a: str, text_b: str, emb_cache: Dict[str, Any] = None,
                          model_name: str = _DEFAULT_SBERT) -> float:
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    model = get_sbert(model_name)
    emb_a = emb_cache.get(text_a) if emb_cache else None
    emb_b = emb_cache.get(text_b) if emb_cache else None
    if emb_a is None:
        emb_a = model.encode(text_a, convert_to_tensor=True)
    if emb_b is None:
        emb_b = model.encode(text_b, convert_to_tensor=True)
    return max(0.0, float(util.cos_sim(emb_a, emb_b).item()))


# ── Content agreement (holistic, no field-name alignment on either side) ──────
def content_agreement(cfg: EvalConfig, gt_blob: str, pred_blob: str,
                       gt_numbers: list[float], pred_numbers: list[float],
                       emb_cache: Dict[str, Any] = None) -> float:
    semantic = _semantic_similarity(gt_blob, pred_blob, emb_cache, cfg.sbert_model)
    if not gt_numbers:
        return semantic
    numeric = _numeric_overlap(gt_numbers, pred_numbers)
    return cfg.semantic_weight * semantic + cfg.numeric_weight * numeric


# ── Metrics ─────────────────────────────────────────────────────────────────────
def id_reuse_diagnostic(gt_rows: List[Dict], pred_rows: List[Dict], gt_id: str) -> Dict:
    """How many of the ground truth's identifiers appear verbatim among the
    predicted ids -- a property of the GROUND TRUTH's annotation convention, not
    a measure of the system.

    Deliberately NOT reported as precision/recall/F1. Expressed that way it sits
    beside F1_content looking like a stricter accuracy score, and reads as though
    the system's honest performance were near zero on the corpora where it is
    zero. What it actually distinguishes is whether an annotator reused codes
    printed in the source document (recoverable by any extractor) or invented
    row numbers (recoverable by none).

    Counted over DISTINCT normalised ids, and the distinct count is reported
    alongside, because a ground truth that leaves its id column blank on some
    rows has fewer identifiers than rows and the fraction would otherwise be
    read against the wrong denominator.
    """
    gt_ids = {_norm_str_id(r.get(gt_id, "")) for r in gt_rows if r.get(gt_id)} - {""}
    pred_ids = {_norm_str_id(r.get("id", "")) for r in pred_rows if r.get("id")} - {""}
    matched = len(gt_ids & pred_ids)
    return {
        "gt_ids_distinct": len(gt_ids),
        "gt_rows_total": len(gt_rows),
        "pred_ids_distinct": len(pred_ids),
        "gt_ids_found_verbatim": matched,
        "id_reuse_frac": round(matched / len(gt_ids), 3) if gt_ids else 0.0,
    }


@dataclass
class Assignment:
    """Everything the content metric looked at, kept rather than discarded.

    evaluate_content only needs `scores`, but a reader asking "why did THIS
    prediction count as a match for THAT ground-truth row" needs the two blobs
    that were compared and the two component scores that produced the decision.
    Keeping them here means the audit CSV is a printout of the very numbers the
    metric used, not a second implementation that could disagree with it.
    """
    gt_blobs: List[str]
    pred_blobs: List[str]
    gt_nums: List[List[float]]
    pred_nums: List[List[float]]
    pairs: List[Tuple[int, int]]      # (gt_index, pred_index), optimally assigned
    semantic: List[float]             # per pair
    numeric: List[float]              # per pair; -1.0 when the GT row has no numbers
    scores: List[float]               # per pair, the blended content agreement
    score_matrix: Any = None          # full gt x pred agreement, for runner-up reporting


def build_assignment(cfg: EvalConfig, gt_rows: List[Dict], pred_rows: List[Dict], gt_id: str,
                      exclude: set[str]) -> Assignment:
    """Optimally pair ground-truth rows with predicted rows and score each pair.

    Split out from evaluate_content because the assignment does not depend on
    cfg.threshold at all -- only the decision of which assigned pairs COUNT
    does. A threshold sweep can therefore reuse one call to this function
    instead of re-encoding every blob per threshold.
    """
    empty = Assignment([], [], [], [], [], [], [], [])
    if not gt_rows or not pred_rows:
        return empty
    gt_blobs = [_gt_blob(r, gt_id, exclude) for r in gt_rows]
    pred_blobs = [_pred_blob(r) for r in pred_rows]
    drop_tags = cfg.drop_tag_derived_numbers
    gt_nums = [_extract_numbers(b, drop_tags) for b in gt_blobs]
    pred_nums = [_extract_numbers(b, drop_tags) for b in pred_blobs]
    emb_cache = _batch_encode_texts(gt_blobs + pred_blobs, cfg.sbert_model)

    cost_matrix = np.zeros((len(gt_rows), len(pred_rows)))
    for i in range(len(gt_rows)):
        for j in range(len(pred_rows)):
            cost_matrix[i, j] = 1.0 - content_agreement(
                cfg, gt_blobs[i], pred_blobs[j], gt_nums[i], pred_nums[j], emb_cache)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pairs = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]
    semantic, numeric, scores = [], [], []
    for i, j in pairs:
        semantic.append(_semantic_similarity(gt_blobs[i], pred_blobs[j], emb_cache, cfg.sbert_model))
        numeric.append(_numeric_overlap(gt_nums[i], pred_nums[j]) if gt_nums[i] else -1.0)
        scores.append(1.0 - cost_matrix[i, j])
    return Assignment(gt_blobs, pred_blobs, gt_nums, pred_nums, pairs, semantic, numeric,
                      scores, 1.0 - cost_matrix)


def assignment_scores(cfg: EvalConfig, gt_rows: List[Dict], pred_rows: List[Dict], gt_id: str,
                       exclude: set[str]) -> List[float]:
    """Content-agreement score of every optimally-assigned (gt, pred) pair."""
    return build_assignment(cfg, gt_rows, pred_rows, gt_id, exclude).scores


def _prf_from_tp(tp: int, n_pred: int, n_gt: int) -> Tuple[float, float, float, int, int, int]:
    fp, fn = n_pred - tp, n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall, tp, fp, fn


def evaluate_content(cfg: EvalConfig, gt_rows: List[Dict], pred_rows: List[Dict], gt_id: str,
                      exclude: set[str], assignment: "Assignment" = None
                      ) -> Tuple[float, float, float, int, int, int]:
    if not gt_rows or not pred_rows:
        return 0.0, 0.0, 0.0, 0, len(pred_rows), len(gt_rows)
    a = assignment or build_assignment(cfg, gt_rows, pred_rows, gt_id, exclude)
    tp = sum(1 for s in a.scores if s >= cfg.threshold)
    return _prf_from_tp(tp, len(pred_rows), len(gt_rows))


# ── Audit trail: the per-record evidence behind F1_content ─────────────────────
# Everything below exists so that the headline number can be checked by reading a
# file instead of by trusting this code. A summary row saying "F1 = 0.62" is not
# reviewable: it does not say which ground-truth rule was considered found, which
# predicted record was credited for it, how similar the two actually were, or why
# the pair cleared the threshold. write_match_audit prints exactly that, one row
# per comparison decision, and write_f1_derivation prints the arithmetic that
# turns those decisions into precision, recall and F1.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _token_overlap(gt_blob: str, pred_blob: str) -> Tuple[float, str, str, str]:
    """A literal word-level view of the same two blobs.

    DIAGNOSTIC ONLY -- no term below enters the score. It is reported because
    "how are two records compared, word by word?" is the first question anyone
    asks of a semantic metric, and the honest answer ("not word by word, by
    embedding cosine") is only convincing next to the word-level picture it is
    being distinguished from. A pair with high cosine and near-zero shared
    vocabulary is exactly the case the semantic metric exists to catch -- and
    also exactly the case where it could be wrong, so it must be visible.
    """
    g, p = _tokens(gt_blob), _tokens(pred_blob)
    if not g and not p:
        return 0.0, "", "", ""
    # The three token lists are printed in full: the Jaccard is computed over
    # the complete sets, so truncating the lists would make the printed value
    # unreproducible from the columns printed beside it.
    jaccard = len(g & p) / len(g | p) if (g | p) else 0.0
    return (round(jaccard, 3), " ".join(sorted(g & p)),
            " ".join(sorted(g - p)), " ".join(sorted(p - g)))


def _fmt_nums(nums: List[float]) -> str:
    return " ".join(f"{n:g}" for n in nums)


def build_audit_rows(cfg: EvalConfig, gt_rows: List[Dict], pred_rows: List[Dict], gt_id: str,
                      a: "Assignment", file_name: str) -> pd.DataFrame:
    """One row per comparison decision: every assigned pair, then every
    ground-truth row and predicted record left unpaired."""
    paired_gt = {i for i, _ in a.pairs}
    paired_pred = {j for _, j in a.pairs}
    rows = []

    for idx in sorted(range(len(a.pairs)), key=lambda x: a.pairs[x]):
        i, j = a.pairs[idx]
        sem, num, score = a.semantic[idx], a.numeric[idx], a.scores[idx]
        counted = score >= cfg.threshold
        numeric_used = num >= 0.0
        if numeric_used:
            formula = (f"{cfg.semantic_weight}*{sem:.3f} + {cfg.numeric_weight}*{num:.3f} "
                       f"= {score:.3f}")
        else:   # GT row carries no numbers -> the numeric term is undefined, not zero
            formula = f"semantic only (GT row has no numbers) = {score:.3f}"
        detail = "; ".join(
            f"{g:g}->{'none' if p is None else format(p, 'g')} ({s:.3f})"
            for g, p, s in _numeric_pairing(a.gt_nums[i], a.pred_nums[j]))
        jac, shared, gt_only, pred_only = _token_overlap(a.gt_blobs[i], a.pred_blobs[j])
        # The runner-up: the best score this GT row could have got from any OTHER
        # record. Without it the audit shows only what the assignment chose and
        # gives no way to see whether it chose well -- a pair that beat its
        # nearest rival by 0.001 was effectively a coin toss and should be read
        # as one, however comfortably it cleared the threshold.
        alt_score, alt_row = "", ""
        if a.score_matrix is not None and a.score_matrix.shape[1] > 1:
            row_scores = a.score_matrix[i].copy()
            row_scores[j] = -np.inf
            k = int(np.argmax(row_scores))
            alt_score, alt_row = round(float(row_scores[k]), 4), k + 1
        rows.append({
            "file_name": file_name,
            "decision": "TP (counted as found)" if counted else "REJECTED (below threshold)",
            "counts_as": "TP" if counted else "FN for this GT row + FP for this record",
            "gt_row": i + 1,
            "gt_id": str(gt_rows[i].get(gt_id, "")),
            "pred_row": j + 1,
            "pred_id": str(pred_rows[j].get("id", "")),
            "combined_score": round(score, 4),
            "threshold": cfg.threshold,
            "score_formula": formula,
            "runner_up_score": alt_score,
            "runner_up_pred_row": alt_row,
            "margin_over_runner_up": ("" if alt_score == "" else round(score - alt_score, 4)),
            "semantic_cosine": round(sem, 4),
            "numeric_overlap": ("" if not numeric_used else round(num, 4)),
            "numeric_detail_gt_to_pred": detail,
            "gt_numbers": _fmt_nums(a.gt_nums[i]),
            "pred_numbers": _fmt_nums(a.pred_nums[j]),
            "word_overlap_jaccard_DIAGNOSTIC": jac,
            "shared_words_DIAGNOSTIC": shared,
            "gt_only_words_DIAGNOSTIC": gt_only,
            "pred_only_words_DIAGNOSTIC": pred_only,
            "gt_text_compared": a.gt_blobs[i],
            "pred_text_compared": a.pred_blobs[j],
        })

    for i in range(len(gt_rows)):
        if i in paired_gt:
            continue
        rows.append({
            "file_name": file_name,
            "decision": "UNPAIRED GT (fewer records than GT rows)",
            "counts_as": "FN",
            "gt_row": i + 1,
            "gt_id": str(gt_rows[i].get(gt_id, "")),
            "pred_row": "", "pred_id": "",
            "combined_score": "", "threshold": cfg.threshold,
            "score_formula": "no record left to assign",
            "runner_up_score": "", "runner_up_pred_row": "", "margin_over_runner_up": "",
            "semantic_cosine": "", "numeric_overlap": "",
            "numeric_detail_gt_to_pred": "",
            "gt_numbers": _fmt_nums(a.gt_nums[i]) if a.gt_nums else "",
            "pred_numbers": "",
            "word_overlap_jaccard_DIAGNOSTIC": "", "shared_words_DIAGNOSTIC": "",
            "gt_only_words_DIAGNOSTIC": "", "pred_only_words_DIAGNOSTIC": "",
            "gt_text_compared": a.gt_blobs[i] if a.gt_blobs else "",
            "pred_text_compared": "",
        })

    for j in range(len(pred_rows)):
        if j in paired_pred:
            continue
        rows.append({
            "file_name": file_name,
            "decision": "UNPAIRED RECORD (more records than GT rows)",
            "counts_as": "FP",
            "gt_row": "", "gt_id": "",
            "pred_row": j + 1,
            "pred_id": str(pred_rows[j].get("id", "")),
            "combined_score": "", "threshold": cfg.threshold,
            "score_formula": "no GT row left to assign",
            "runner_up_score": "", "runner_up_pred_row": "", "margin_over_runner_up": "",
            "semantic_cosine": "", "numeric_overlap": "",
            "numeric_detail_gt_to_pred": "",
            "gt_numbers": "",
            "pred_numbers": _fmt_nums(a.pred_nums[j]) if a.pred_nums else "",
            "word_overlap_jaccard_DIAGNOSTIC": "", "shared_words_DIAGNOSTIC": "",
            "gt_only_words_DIAGNOSTIC": "", "pred_only_words_DIAGNOSTIC": "",
            "gt_text_compared": "",
            "pred_text_compared": a.pred_blobs[j] if a.pred_blobs else "",
        })

    return pd.DataFrame(rows)


def f1_derivation_row(cfg: EvalConfig, file_name: str, n_gt: int, n_pred: int,
                       tp: int, fp: int, fn: int, f1: float, pr: float, re_: float) -> Dict:
    """The arithmetic from match decisions to F1, written out rather than implied."""
    # The assignment forms at most min(n_gt, n_pred) pairs, so when a run emits
    # more records than the ground truth has rows, the surplus are false
    # positives BY CONSTRUCTION -- correct or not, they had no row left to be
    # assigned to. That puts a hard ceiling on precision which has nothing to do
    # with extraction quality, and it must be reported next to the precision it
    # caps rather than left for a reader to derive.
    ceiling = min(n_gt, n_pred) / n_pred if n_pred else 0.0
    return {
        "file_name": file_name,
        "gt_rows_total": n_gt,
        "records_extracted": n_pred,
        "pairs_formed": min(n_gt, n_pred),
        "max_possible_precision": round(ceiling, 4),
        "forced_FP_from_row_count": max(0, n_pred - n_gt),
        "pairs_at_or_above_threshold_TP": tp,
        "FP_formula": f"records - TP = {n_pred} - {tp}",
        "FP": fp,
        "FN_formula": f"gt_rows - TP = {n_gt} - {tp}",
        "FN": fn,
        "precision_formula": f"TP/(TP+FP) = {tp}/({tp}+{fp})",
        "precision": round(pr, 4),
        "recall_formula": f"TP/(TP+FN) = {tp}/({tp}+{fn})",
        "recall": round(re_, 4),
        "f1_formula": f"2*P*R/(P+R) = 2*{pr:.4f}*{re_:.4f}/({pr:.4f}+{re_:.4f})",
        "f1_content": round(f1, 4),
        "threshold": cfg.threshold,
        "semantic_weight": cfg.semantic_weight,
        "numeric_weight": cfg.numeric_weight,
        "sbert_model": cfg.sbert_model,
    }


def write_report_rows(report_path: str, agg: pd.DataFrame, gt_path: str) -> None:
    """Append this corpus's rows to the cross-corpus comparison table.

    The per-corpus files exist because one evaluator invocation scores one ground
    truth, so a run over five corpora leaves five files that have to be opened and
    merged by hand before anything can be compared. This writes the comparison
    directly, with column names that say what they hold rather than which internal
    metric family they came from, and with the ground-truth row count included so
    a record count can be read against the target it is aiming at.

    Only the headline metric is carried, together with the ceiling that caps the
    precision beside it -- a precision of 0.58 against a ceiling of 0.65 says
    something entirely different from 0.58 against 1.0, and a summary that omits
    the ceiling invites the wrong one.
    """
    try:
        gt_rows = len(pd.read_csv(gt_path))
    except Exception:
        gt_rows = ""

    rows = []
    for _, r in agg.iterrows():
        corpus = re.sub(r"^ext_multi_agent_generic_|^ext_", "", str(r["condition"]))
        rows.append({
            "corpus":        corpus,
            "runs":          int(r["runs"]),
            "gt_rows":       gt_rows,
            "records_mean":  r.get("items_mean", ""),
            "records_std":   r.get("items_std", ""),
            "f1":            r.get("f1_mean", ""),
            "f1_std":        r.get("f1_std", ""),
            "f1_min":        r.get("f1_min", ""),
            "f1_max":        r.get("f1_max", ""),
            "precision":     r.get("pr_mean", ""),
            "max_precision": r.get("max_pr", ""),
            "recall":        r.get("re_mean", ""),
        })
    new = pd.DataFrame(rows)
    if os.path.exists(report_path):
        new = pd.concat([pd.read_csv(report_path), new], ignore_index=True)
    new = new.sort_values(by="f1", ascending=False)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    new.to_csv(report_path, index=False)


# ── Aggregation over repeated runs ─────────────────────────────────────────────
# Strips the run index and timestamp from a prediction filename so repeats of
# the same condition group together. Everything the grid varies deliberately
# (model, paradigm, seed) stays in the key; only what identifies one repeat is
# removed.
_RUN_SUFFIX = re.compile(
    r"(?:_run\d+)?(?:_\d{8}_\d{6})?(?:_run\d+)?\.csv$", re.IGNORECASE)


def condition_of(file_name: str) -> str:
    """Group key for repeats of one experimental condition."""
    return _RUN_SUFFIX.sub("", file_name)


def summarise_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated runs of each condition into mean +- std.

    A single run is one draw from a non-deterministic endpoint, so a lone F1
    cannot separate a real difference between conditions from endpoint noise.
    Reporting the mean with its standard deviation, and the number of runs
    behind it, makes that distinction visible: two conditions whose intervals
    overlap are tied, not ranked, however different their means look.
    """
    if df.empty:
        return df
    df = df.copy()
    df["condition"] = df["file_name"].map(condition_of)
    g = df.groupby("condition")
    out = pd.DataFrame({
        "runs":         g["content_f1"].size(),
        "f1_mean":      g["content_f1"].mean().round(3),
        "f1_std":       g["content_f1"].std(ddof=1).fillna(0.0).round(3),
        "f1_min":       g["content_f1"].min().round(3),
        "f1_max":       g["content_f1"].max().round(3),
        "pr_mean":      g["content_pr"].mean().round(3),
        "re_mean":      g["content_re"].mean().round(3),
        "items_mean":   g["total_extracted"].mean().round(1),
        "items_std":    g["total_extracted"].std(ddof=1).fillna(0.0).round(1),
        "max_pr":       g["max_possible_precision"].mean().round(3),
    }).sort_values("f1_mean", ascending=False)
    return out.reset_index()


def print_run_summary(agg: pd.DataFrame) -> None:
    if agg.empty:
        print("  nothing to aggregate")
        return
    print(f"\n  {'condition':<52}{'runs':>5}{'F1 mean+-std':>16}{'range':>16}{'items':>14}")
    for _, r in agg.iterrows():
        print(f"  {r['condition'][:50]:<52}{int(r['runs']):>5}"
              f"{r['f1_mean']:>9.3f}+-{r['f1_std']:<5.3f}"
              f"{r['f1_min']:>8.3f}-{r['f1_max']:<7.3f}"
              f"{r['items_mean']:>8.1f}+-{r['items_std']:<5.1f}")
    single = agg[agg["runs"] < 2]
    if len(single):
        print(f"\n  [warn] {len(single)} condition(s) have only ONE run, so their std is "
              f"unknown, not zero. Treat those numbers as provisional.")
    # A capped precision is not a measurement of quality, and reading it as one
    # is the single easiest misreading of this table.
    capped = agg[agg.get("max_pr", 1.0) < 0.999] if "max_pr" in agg else agg.iloc[0:0]
    for _, r in capped.iterrows():
        print(f"\n  [warn] {r['condition'][:50]}: this run emits more records than the ground "
              f"truth has rows, so precision CANNOT exceed {r['max_pr']:.3f} however correct "
              f"the surplus records are. Observed precision {r['pr_mean']:.3f} must be read "
              f"against that ceiling, not against 1.0.")


# ── Oracle gap: does selecting on the test set explain the result? ─────────────
def oracle_gap(agg: pd.DataFrame, reference: str) -> None:
    """Compare the best configuration chosen with full hindsight against a
    reference system.

    This is the quantitative form of the defence "an optimal a posteriori
    selection over the grid does not reach the pipeline's operating point, so
    the gain comes from the components rather than from having picked the right
    configuration on the test set". Stated it is a claim; computed it is a
    number, and a reviewer will compute it.

    Read it honestly in both directions. A LARGE positive gap supports the
    defence: no single configuration, even chosen after seeing the answers,
    matches the reference. A gap near zero or negative refutes it, and the
    defence must then be dropped rather than restated -- the oracle here is an
    upper bound on what test-set selection could ever have bought, so if the
    reference does not beat it, selection explains the result.

    The comparison is only meaningful when both sides were scored on the same
    ground truth with the same parameters, and it is reported against the mean
    across repeats, never a single lucky run.
    """
    if agg.empty:
        print("  oracle gap: nothing to compare")
        return
    ref_rows = agg[agg["condition"].str.contains(reference, case=False, regex=False)]
    if ref_rows.empty:
        print(f"  oracle gap: no condition matching {reference!r}; "
              f"available: {list(agg['condition'])[:6]}")
        return
    ref = ref_rows.iloc[0]
    others = agg[~agg["condition"].isin(ref_rows["condition"])]
    if others.empty:
        print("  oracle gap: nothing to compare the reference against")
        return
    best = others.loc[others["f1_mean"].idxmax()]
    gap = ref["f1_mean"] - best["f1_mean"]
    print(f"\n  ORACLE GAP")
    print(f"    best of {len(others)} configuration(s), chosen with hindsight:")
    print(f"      {best['condition'][:58]:<60} {best['f1_mean']:.3f} +- {best['f1_std']:.3f}")
    print(f"    reference system:")
    print(f"      {ref['condition'][:58]:<60} {ref['f1_mean']:.3f} +- {ref['f1_std']:.3f}")
    print(f"    gap = {gap:+.3f}")
    # a gap smaller than the noise it is measured against is not a gap
    noise = float(max(ref["f1_std"], best["f1_std"]))
    if gap <= 0:
        print(f"    -> The hindsight-selected configuration MATCHES OR BEATS the reference. "
              f"Test-set selection is sufficient to explain the result; do not claim "
              f"the components are what produced it.")
    elif gap <= noise:
        print(f"    -> The gap ({gap:.3f}) is within run-to-run noise ({noise:.3f}). "
              f"Report the two as indistinguishable rather than claiming a component gain.")
    else:
        print(f"    -> The gap ({gap:.3f}) exceeds run-to-run noise ({noise:.3f}), so no "
              f"single configuration reaches the reference even chosen after the fact. "
              f"This is the evidence for the components-not-selection argument.")


# ── Sensitivity analysis over the scoring free parameters ──────────────────────
def run_sensitivity(cfg: EvalConfig, gt_path: str, pred_paths: List[str],
                     thresholds: List[float], weights: List[float]) -> pd.DataFrame:
    """Re-score every prediction file at each (threshold, semantic_weight)
    combination and report how the RANKING of files changes.

    The point is not the absolute F1 at any one setting -- it is whether the
    ordering of systems survives the range. A ranking that holds across the
    sweep cannot be an artefact of the particular constants chosen; a ranking
    that flips is a finding about the threshold, and must be reported as such.

    Embeddings are computed once per (file, weight) pair and reused across all
    thresholds, since the assignment does not depend on the threshold.
    """
    gt_df = pd.read_csv(gt_path)
    gt_rows = gt_df.fillna("").to_dict("records")
    gt_id = _detect_id_field(gt_df, cfg)
    exclude = (_ID_LIKE_NAMES | _detect_noise_fields(gt_df, cfg)
               | {f.strip().lower() for f in cfg.gt_exclude_fields})

    rows = []
    for w in weights:
        sweep_cfg = EvalConfig(**{**cfg.__dict__, "semantic_weight": w, "numeric_weight": round(1.0 - w, 3)})
        for path in pred_paths:
            try:
                pred_rows = pd.read_csv(path).fillna("").to_dict("records")
            except Exception as e:
                print(f"    skipping {os.path.basename(path)}: {e}")
                continue
            scores = assignment_scores(sweep_cfg, gt_rows, pred_rows, gt_id, exclude)
            for t in thresholds:
                tp = sum(1 for s in scores if s >= t)
                f1, pr, rc, tp, fp, fn = _prf_from_tp(tp, len(pred_rows), len(gt_rows))
                rows.append({"file_name": os.path.basename(path),
                             "semantic_weight": w, "numeric_weight": round(1.0 - w, 3),
                             "threshold": t, "content_f1": round(f1, 3),
                             "content_pr": round(pr, 3), "content_re": round(rc, 3)})
    return pd.DataFrame(rows)


def summarise_sensitivity(df: pd.DataFrame) -> None:
    """Report rank stability: for each parameter setting, rank the files by
    content_f1, then show how much those ranks move across the whole sweep."""
    if df.empty:
        print("  no sensitivity rows produced")
        return
    df = df.copy()
    df["rank"] = df.groupby(["semantic_weight", "threshold"])["content_f1"] \
                    .rank(ascending=False, method="min")
    agg = df.groupby("file_name").agg(
        best_rank=("rank", "min"), worst_rank=("rank", "max"),
        mean_rank=("rank", "mean"), f1_min=("content_f1", "min"),
        f1_max=("content_f1", "max")).sort_values("mean_rank")
    print(f"\n  Rank stability across {df[['semantic_weight','threshold']].drop_duplicates().shape[0]} "
          f"parameter settings:\n")
    print(f"    {'file':<52} {'rank':>10}  {'F1 range':>14}")
    for name, r in agg.iterrows():
        rank = f"{int(r.best_rank)}" if r.best_rank == r.worst_rank else f"{int(r.best_rank)}-{int(r.worst_rank)}"
        print(f"    {name[:50]:<52} {rank:>10}  {r.f1_min:.3f}-{r.f1_max:.3f}")
    flips = (agg.best_rank != agg.worst_rank).sum()
    top = agg.index[0]
    top_always = bool(agg.loc[top, "worst_rank"] == 1)
    print(f"\n    files whose rank moves at all : {flips}/{len(agg)}")
    print(f"    best system is rank 1 at EVERY setting: {top_always}"
          f"{'' if top_always else '  <-- ranking depends on the parameters; report this'}")


# ── Main execution ──────────────────────────────────────────────────────────────
def run_evaluation(cfg: EvalConfig, gt_path: str, pred_path: str,
                    audit_dir: str = "", derivations: List[Dict] = None) -> dict:
    try:
        gt_df = pd.read_csv(gt_path)
        gt_rows = gt_df.fillna("").to_dict("records")
        pred_rows = pd.read_csv(pred_path).fillna("").to_dict("records")
    except Exception as e:
        print(f"    Error loading files for {os.path.basename(pred_path)}: {e}")
        return {}

    gt_id_field = _detect_id_field(gt_df, cfg)
    exclude = (_ID_LIKE_NAMES | _detect_noise_fields(gt_df, cfg)
               | {f.strip().lower() for f in cfg.gt_exclude_fields})

    # One assignment, used for the metric AND for the audit trail, so the file a
    # reader checks is a printout of the scored comparison rather than a re-run
    # of it that could differ.
    assignment = build_assignment(cfg, gt_rows, pred_rows, gt_id_field, exclude)
    f1_c, pr_c, re_c, tp_c, fp_c, fn_c = evaluate_content(
        cfg, gt_rows, pred_rows, gt_id_field, exclude, assignment)

    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(pred_path))[0]
        audit = build_audit_rows(cfg, gt_rows, pred_rows, gt_id_field, assignment,
                                 os.path.basename(pred_path))
        audit.to_csv(os.path.join(audit_dir, f"match_audit_{name}.csv"), index=False)
    if derivations is not None:
        derivations.append(f1_derivation_row(
            cfg, os.path.basename(pred_path), len(gt_rows), len(pred_rows),
            tp_c, fp_c, fn_c, f1_c, pr_c, re_c))

    idr = id_reuse_diagnostic(gt_rows, pred_rows, gt_id_field)
    n_gt, n_pred = len(gt_rows), len(pred_rows)
    return {
        "file_name": os.path.basename(pred_path),
        "gt_id_field_used": gt_id_field,
        "total_extracted": n_pred,
        "gt_rows_total": n_gt,
        "content_f1": round(f1_c, 3), "content_pr": round(pr_c, 3), "content_re": round(re_c, 3),
        "content_tp": tp_c, "content_fp": fp_c, "content_fn": fn_c,
        # Hard ceiling on precision imposed by the record count alone -- surplus
        # records have no GT row left to pair with and are false positives
        # whether or not they are correct. Reported beside the precision it caps.
        "max_possible_precision": round(min(n_gt, n_pred) / n_pred, 3) if n_pred else 0.0,
        "forced_fp_from_row_count": max(0, n_pred - n_gt),
        # DIAGNOSTIC, not a score: see id_reuse_diagnostic
        "gt_ids_found_verbatim": idr["gt_ids_found_verbatim"],
        "gt_ids_distinct": idr["gt_ids_distinct"],
        "id_reuse_frac": idr["id_reuse_frac"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schema-agnostic evaluation for step2_multi_agent_generic.py output "
                    "-- no per-domain field configuration required")
    parser.add_argument("--gt", required=True, help="Path to ground truth CSV.")
    parser.add_argument("--pred", default=None,
                         help="Path to a specific prediction CSV. If omitted, evaluates every CSV in --pred-dir.")
    parser.add_argument("--pred-dir", default=None,
                         help="Directory to batch-scan for prediction CSVs when --pred is omitted.")
    parser.add_argument("--gt-id-field", default="",
                         help="Override auto-detection of the ground truth's id column "
                              "(auto-detects 'id'/'ruleId'/'rule_id' case-insensitively).")
    parser.add_argument("--gt-exclude-fields", default="",
                         help="Extra comma-separated ground-truth columns to drop from the content "
                              "blob (id-like and source-like columns are already excluded automatically).")
    parser.add_argument("--threshold", type=float, default=0.6, help="Content-match acceptance threshold.")
    parser.add_argument("--semantic-weight", type=float, default=0.6)
    parser.add_argument("--numeric-weight", type=float, default=0.4)
    parser.add_argument("--sbert-model", default=_DEFAULT_SBERT,
                         help=f"Sentence-transformer used for semantic similarity (default: {_DEFAULT_SBERT}, English).")
    parser.add_argument("--id-uniqueness-min", type=float, default=0.9,
                         help="Min distinct-value fraction for a column to qualify as the id column.")
    parser.add_argument("--id-len-penalty", type=float, default=200.0,
                         help="Divisor penalising long values during id-column detection.")
    parser.add_argument("--id-name-hint-bonus", type=float, default=0.05,
                         help="Tiebreak nudge for conventional id column names; never decisive alone.")
    parser.add_argument("--noise-distinct-floor", type=int, default=6,
                         help="A provenance column has at most this many distinct values "
                              "(or --noise-distinct-frac of the rows, whichever is larger).")
    parser.add_argument("--noise-distinct-frac", type=float, default=0.1,
                         help="Row fraction alternative to --noise-distinct-floor.")
    parser.add_argument("--noise-max-value-len", type=int, default=24,
                         help="Max value length for a column to look like document labels.")
    parser.add_argument("--drop-tag-derived-numbers", action="store_true",
                         help="Ignore negative numbers produced by reading the hyphen in an "
                              "asset tag as a minus sign (GSH-401 -> -401). Off by default; "
                              "turning it on CHANGES reported figures.")
    parser.add_argument("--report", default="",
                         help="Append a one-row-per-condition summary to this CSV, using plain "
                              "column names (corpus, runs, gt_rows, records, f1, precision, "
                              "recall). Pass the SAME path for every corpus scored in a session "
                              "and the file becomes the comparison table across corpora -- which "
                              "the per-corpus files cannot be, since each holds one ground truth.")
    parser.add_argument("--label", default="",
                         help="Name for this corpus in --report and in output filenames. "
                              "Defaults to the condition name found in the predictions.")
    parser.add_argument("--aggregate", action="store_true",
                         help="After batch scoring, collapse repeated runs of each condition "
                              "into mean +- std (repeats are matched by stripping the _runN "
                              "and timestamp suffixes from the filename).")
    parser.add_argument("--oracle-gap", default="",
                         help="Substring identifying the reference system among the scored "
                              "conditions (e.g. 'multi_agent'). Reports best-of-the-rest "
                              "chosen with hindsight vs that reference, and whether the gap "
                              "exceeds run-to-run noise. Implies --aggregate.")
    parser.add_argument("--audit", action="store_true",
                         help="Also write the per-record evidence behind F1_content: one CSV per "
                              "prediction file listing every ground-truth row, the record it was "
                              "paired with, the two texts actually compared, the semantic and "
                              "numeric component scores, and whether the pair cleared the "
                              "threshold -- plus a per-corpus file showing the precision/recall/F1 "
                              "arithmetic those decisions produce.")
    parser.add_argument("--audit-dir", default="",
                         help="Where --audit writes (default: <step3_results>/match_audit/<label>).")
    parser.add_argument("--sensitivity", action="store_true",
                         help="Sweep the scoring parameters and report whether the RANKING of "
                              "prediction files is stable, instead of scoring once at the defaults.")
    parser.add_argument("--sens-thresholds", default="0.5,0.55,0.6,0.65,0.7,0.75,0.8",
                         help="Comma-separated thresholds for --sensitivity.")
    parser.add_argument("--sens-weights", default="0.4,0.5,0.6,0.7,0.8",
                         help="Comma-separated semantic weights for --sensitivity "
                              "(numeric weight is 1 - semantic).")
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        print(f"CRITICAL ERROR: Ground truth file not found at {args.gt}")
        raise SystemExit(1)

    cfg = EvalConfig(
        gt_id_field=args.gt_id_field,
        gt_exclude_fields=[f for f in args.gt_exclude_fields.split(",") if f.strip()],
        threshold=args.threshold,
        semantic_weight=args.semantic_weight,
        numeric_weight=args.numeric_weight,
        id_uniqueness_min=args.id_uniqueness_min,
        id_len_penalty=args.id_len_penalty,
        id_name_hint_bonus=args.id_name_hint_bonus,
        noise_distinct_floor=args.noise_distinct_floor,
        noise_distinct_frac=args.noise_distinct_frac,
        noise_max_value_len=args.noise_max_value_len,
        drop_tag_derived_numbers=args.drop_tag_derived_numbers,
        sbert_model=args.sbert_model,
    )

    print(f"\n{'='*65}")
    print("  Schema-Agnostic Evaluation (step2_multi_agent_generic.py)")
    print(f"  Ground Truth: {args.gt}")
    print(f"{'='*65}")
    get_sbert(cfg.sbert_model)

    if args.sensitivity:
        if args.pred:
            targets = [args.pred]
        else:
            if not args.pred_dir or not os.path.exists(args.pred_dir):
                print(f"Directory not found: {args.pred_dir}")
                raise SystemExit(1)
            targets = sorted(os.path.join(args.pred_dir, f) for f in os.listdir(args.pred_dir)
                             if f.endswith(".csv") and not f.endswith("_partial.csv"))
        ths = [float(x) for x in args.sens_thresholds.split(",") if x.strip()]
        ws  = [float(x) for x in args.sens_weights.split(",") if x.strip()]
        print(f"\nSensitivity sweep: {len(targets)} file(s) x {len(ths)} thresholds x {len(ws)} weightings")
        sens = run_sensitivity(cfg, args.gt, targets, ths, ws)
        summarise_sensitivity(sens)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(STEP3_RESULTS_DIR, f"sensitivity_{ts}.csv")
        sens.to_csv(out, index=False)
        print(f"\n  Full sweep written to: {out}")
        raise SystemExit(0)

    audit_label = re.sub(r"[^A-Za-z0-9]+", "_", args.label).strip("_") or "corpus"
    audit_dir = ""
    if args.audit:
        audit_dir = args.audit_dir or os.path.join(STEP3_RESULTS_DIR, "match_audit", audit_label)
    derivations: List[Dict] = [] if args.audit else None

    if args.pred:
        print(f"\nEvaluating single file: {args.pred}")
        metrics = run_evaluation(cfg, args.gt, args.pred, audit_dir, derivations)
        if derivations:
            os.makedirs(audit_dir, exist_ok=True)
            pd.DataFrame(derivations).to_csv(
                os.path.join(audit_dir, f"f1_derivation_{audit_label}.csv"), index=False)
            print(f"  Match audit + F1 derivation written to: {audit_dir}")
        if metrics:
            print(f"  (auto-detected GT id column: {metrics['gt_id_field_used']})")
            print("\n--- F1_content (Semantic + numeric-overlap assignment) ---")
            print(f"F1:        {metrics['content_f1']:.3f}")
            print(f"Precision: {metrics['content_pr']:.3f}")
            print(f"Recall:    {metrics['content_re']:.3f}")
            print(f"TP: {metrics['content_tp']} | FP: {metrics['content_fp']} | FN: {metrics['content_fn']}")
            if metrics["max_possible_precision"] < 0.999:
                print(f"\n[warn] {metrics['forced_fp_from_row_count']} record(s) had no ground-truth "
                      f"row left to pair with, so precision cannot exceed "
                      f"{metrics['max_possible_precision']:.3f} regardless of correctness.")
            print("\n--- id-reuse diagnostic (NOT an accuracy score) ---")
            print(f"{metrics['gt_ids_found_verbatim']}/{metrics['gt_ids_distinct']} of the ground "
                  f"truth's identifiers appear verbatim among the predicted ids "
                  f"({metrics['id_reuse_frac']:.3f}).")
            print("This reflects whether the annotator reused codes printed in the source")
            print("document, not how well the system extracted content.")
    else:
        pred_dir = args.pred_dir
        if not pred_dir or not os.path.exists(pred_dir):
            print(f"Directory not found: {pred_dir}")
            raise SystemExit(1)

        csv_files = [f for f in os.listdir(pred_dir) if f.endswith(".csv") and not f.endswith("_partial.csv")]
        if not csv_files:
            print(f"No prediction CSVs found in {pred_dir}")
            raise SystemExit(0)

        print(f"\nBatch Mode: Scanning {pred_dir} for results...")
        print(f"Found {len(csv_files)} files to evaluate.\n")

        all_metrics = []
        for i, file_name in enumerate(csv_files, 1):
            file_path = os.path.join(pred_dir, file_name)
            print(f"  [{i}/{len(csv_files)}] Evaluating {file_name}...")
            m = run_evaluation(cfg, args.gt, file_path, audit_dir, derivations)
            if m:
                all_metrics.append(m)

        if derivations:
            os.makedirs(audit_dir, exist_ok=True)
            deriv_path = os.path.join(audit_dir, f"f1_derivation_{audit_label}.csv")
            pd.DataFrame(derivations).to_csv(deriv_path, index=False)
            print(f"\n  Per-record match audit : {audit_dir}/match_audit_<run>.csv")
            print(f"  F1 arithmetic per run  : {deriv_path}")

        if all_metrics:
            summary_df = pd.DataFrame(all_metrics)
            ordered_cols = [
                "file_name", "total_extracted",
                "gt_rows_total", "max_possible_precision", "forced_fp_from_row_count",
                "content_f1", "content_pr", "content_re", "content_tp", "content_fp", "content_fn",
                "gt_ids_found_verbatim", "gt_ids_distinct", "id_reuse_frac",
            ]
            summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]
            summary_df = summary_df.sort_values(by="content_f1", ascending=False)

            ts = time.strftime("%Y%m%d_%H%M%S")
            # Name files after the corpus, not only the clock. A directory of
            # files distinguished only by timestamp cannot be read without
            # opening each one to find out which corpus it holds.
            label = args.label or re.sub(
                r"^ext_multi_agent_generic_|^ext_", "",
                condition_of(str(summary_df["file_name"].iloc[0])))
            safe = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or "corpus"
            out_path = os.path.join(STEP3_RESULTS_DIR, f"runs_{safe}_{ts}.csv")
            summary_df.to_csv(out_path, index=False)

            print(f"\n{'='*65}")
            print("Batch Evaluation Complete.")
            print(f"Summary report saved to: {out_path}")
            print(f"{'='*65}")

            if args.aggregate or args.oracle_gap:
                agg = summarise_runs(summary_df)
                print_run_summary(agg)
                agg_path = os.path.join(STEP3_RESULTS_DIR, f"summary_{safe}_{ts}.csv")
                agg.to_csv(agg_path, index=False)
                print(f"\n  Per-condition mean/std written to: {agg_path}")
                if args.report:
                    write_report_rows(args.report, agg, args.gt)
                    print(f"  Combined table updated: {args.report}")
                if args.oracle_gap:
                    oracle_gap(agg, args.oracle_gap)
            print(summary_df[["file_name", "content_f1", "content_pr", "content_re"]].head(10).to_string(index=False))
