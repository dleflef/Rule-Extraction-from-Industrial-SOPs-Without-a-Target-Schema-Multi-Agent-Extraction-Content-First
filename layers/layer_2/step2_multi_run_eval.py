"""
Multi-run evaluation and IAAS (Inter-Agent Agreement Score) computation
for the LangGraph multi-agent extraction pipeline.

Runs the pipeline N independent times at temperature 0.5 (no fixed seed)
to characterise LLM non-determinism, then computes:

  Extraction statistics (per run + aggregate):
    - F1_strict, F1_content, Precision, Recall  →  mean ± std, 95% CI

  IAAS metrics (across K runs):
    - Pairwise content agreement (mean ± std across all K*(K-1)/2 pairs)
    - Hallucination-proxy rate (rules present in < HALLUCINATION_FRAC of runs)
    - Consensus rule set (rules present in ≥ CONSENSUS_FRAC of runs)

Outputs:
    step2_results/multi_run/ext_multi_agent_run{i:02d}.csv   — per-run extractions
    step3_results/multi_run_stats.csv                        — F1 mean/std/CI95
    step3_results/iaas_report.json                           — IAAS metrics
    step3_results/consensus_rules.csv                        — consensus rule set

Usage:
    python3 step2_multi_run_eval.py                        # 20 runs at T=0.5, no seed
    python3 step2_multi_run_eval.py --runs 5
    python3 step2_multi_run_eval.py --temperature 0.5
    python3 step2_multi_run_eval.py --skip-runs            # load existing CSVs, recompute stats only
    python3 step2_multi_run_eval.py --seed-base 42 --force # reproducible from scratch (run i → seed 42+i)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ── Path setup ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_LAYERS_DIR   = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_LAYERS_DIR)

sys.path.insert(0, _SCRIPT_DIR)   # for step2_TEST
sys.path.insert(0, _LAYERS_DIR)   # for layer_3.step3_evaluation_rules

import Agentic_KnowledgeGraph_DigitalTwins.layers.layer_2.step2_multi_agent_baseline as _pipe                        # noqa: E402
from layer_3.step3_evaluation_rules import (             # noqa: E402
    run_evaluation,
    content_agreement,
    get_sbert,
    GROUND_TRUTH_FILE,
    _batch_encode_texts,
)
from sentence_transformers import util as _sbert_util  # noqa: E402

# ── Output directories ─────────────────────────────────────────────────────────
MULTI_RUN_DIR    = os.path.join(_SCRIPT_DIR,  "step2_results", "multi_run")
STEP3_RESULTS_DIR = os.path.join(_LAYERS_DIR, "step3_results")
os.makedirs(MULTI_RUN_DIR,     exist_ok=True)
os.makedirs(STEP3_RESULTS_DIR, exist_ok=True)

# ── IAAS / stability thresholds ────────────────────────────────────────────────
SIMILARITY_THRESHOLD  = 0.90   # SBERT cosine sim to consider two rules the same concept
HALLUCINATION_FRAC    = 0.30   # rules present in < 30% of runs → hallucination proxy
CONSENSUS_FRAC        = 0.70   # rules present in ≥ 70% of runs → consensus set
MULTI_RUN_TEMPERATURE = 0.5    # default temperature for non-determinism study


# ══════════════════════════════════════════════════════════════════════════════
#  Monkey-patches applied before each pipeline invocation
# ══════════════════════════════════════════════════════════════════════════════

def _make_patched_llm_call(temperature: float, seed: int | None = None, run_label: str | None = None):
    """
    Returns a drop-in replacement for step2_TEST.llm_call that uses the
    specified temperature. When seed is None, omits both the top-level seed
    parameter and the seed inside extra_body so each call draws a genuinely
    independent sample (original non-determinism study behaviour). When seed
    is an integer, passes it via extra_body.options.seed so the run is
    reproducible with the same LLM version.

    When run_label is provided, a per-run LLM response cache is loaded from
    multi_run/{run_label}_cache.json. Cache hits bypass the LLM entirely,
    guaranteeing identical per-run CSVs on every re-run after the first.
    """
    import time as _time
    from llm_cache import LLMResponseCache

    _cache: LLMResponseCache | None = None
    if run_label is not None:
        _cache_path = os.path.join(MULTI_RUN_DIR, f"{run_label}_cache.json")
        _cache = LLMResponseCache(_cache_path)

    def patched_llm_call(model: str, messages: list[dict], temperature: float = temperature) -> str:
        if model in _pipe.NO_SYSTEM_ROLE:
            messages = _pipe._merge_system_into_user(messages)
        if _cache is not None:
            hit = _cache.get(model, messages)
            if hit is not None:
                return hit
        delay = _pipe.RETRY_BASE_DELAY
        last_exc: Exception | None = None
        opts: dict = {"num_ctx": _pipe.LLM_NUM_CTX}
        if seed is not None:
            opts["seed"] = seed
        for attempt in range(1, _pipe.MAX_RETRIES + 1):
            try:
                resp = _pipe._client_for(model).chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=_pipe.MAX_OUTPUT_TOKENS,
                    extra_body={"options": opts},
                )
                text = resp.choices[0].message.content
                if _cache is not None:
                    _cache.set(model, messages, text)
                return text
            except Exception as exc:
                last_exc = exc
                if attempt < _pipe.MAX_RETRIES:
                    print(f"    retry {attempt}/{_pipe.MAX_RETRIES}: {exc}")
                    _time.sleep(delay)
                    delay *= 2
                else:
                    raise last_exc
        return ""

    return patched_llm_call


def _make_patched_save_node(run_label: str):
    """
    Returns a drop-in replacement for step2_TEST.save_node that writes to
    multi_run/ext_multi_agent_{run_label}.csv instead of the default path.
    """
    import csv as _csv

    def patched_save_node(state: dict) -> dict:
        rules    = state.get("normalized_rules", [])
        out_path = os.path.join(MULTI_RUN_DIR, f"ext_multi_agent_{run_label}.csv")
        extra    = sorted({k for r in rules for k in r} - set(_pipe.RULE_FIELDS))
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=_pipe.RULE_FIELDS + extra,
                                     extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rules)
        print(f"[Save] Wrote {len(rules)} rules → {out_path}")
        return {"output_path": out_path}

    return patched_save_node


# ══════════════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_multi(n_runs: int, temperature: float,
              seed_base: int | None = None, force: bool = False) -> list[str]:
    """
    Runs the pipeline n_runs times with the patched llm_call.

    seed_base: when set, run i gets seed=(seed_base + i), making the batch
               reproducible from scratch with the same LLM version.
               When None, no seed is passed (original non-determinism behaviour).
    force:     delete all existing run CSVs before starting so every run is
               regenerated, even if files already exist.
    """
    output_paths: list[str] = []
    manifest_entries: list[dict] = []

    if force:
        existing = [
            f for f in os.listdir(MULTI_RUN_DIR)
            if f.startswith("ext_multi_agent_run") and f.endswith(".csv")
        ]
        if existing:
            print(f"[Runner] --force: deleting {len(existing)} existing run CSVs …")
            for fname in existing:
                os.remove(os.path.join(MULTI_RUN_DIR, fname))

    for i in range(1, n_runs + 1):
        run_label = f"run{i:02d}"
        out_path  = os.path.join(MULTI_RUN_DIR, f"ext_multi_agent_{run_label}.csv")
        run_seed  = (seed_base + i) if seed_base is not None else None
        seed_tag  = f"seed={run_seed}" if run_seed is not None else "no seed"

        # Result cache — restores exact original CSV without any LLM call.
        _rc = _pipe._RESULT_CACHE
        _fname = f"ext_multi_agent_{run_label}.csv"
        if _rc is not None:
            _cached = _rc.get(_fname)
            if _cached is not None:
                with open(out_path, "w", encoding="utf-8", newline="") as _f:
                    _f.write(_cached)
                print(f"\n[ResultCache] {_fname} restored from cache.")
                output_paths.append(out_path)
                manifest_entries.append({"run": run_label, "seed": run_seed, "source": "result_cache"})
                continue

        if os.path.exists(out_path):
            print(f"\n[Runner] Run {i}/{n_runs} — found existing file, skipping: {out_path}")
            output_paths.append(out_path)
            manifest_entries.append({"run": run_label, "seed": run_seed, "source": "cached"})
            continue

        print(f"\n{'='*65}")
        print(f"  Multi-run pipeline  |  run {i}/{n_runs}  |  T={temperature}  |  {seed_tag}")
        print(f"{'='*65}")

        _pipe.llm_call   = _make_patched_llm_call(temperature, run_seed, run_label=run_label)
        _pipe.save_node  = _make_patched_save_node(run_label)

        _pipe.run_pipeline(force=True)  # force=True: save_node is monkey-patched to per-run path
        output_paths.append(out_path)
        manifest_entries.append({"run": run_label, "seed": run_seed, "source": "generated"})

    # Write run manifest so the exact seed config is always recoverable.
    manifest = {
        "seed_base":   seed_base,
        "temperature": temperature,
        "n_runs":      n_runs,
        "runs":        manifest_entries,
    }
    manifest_path = os.path.join(MULTI_RUN_DIR, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Runner] Manifest written → {manifest_path}  (seed_base={seed_base})")

    return output_paths


# ══════════════════════════════════════════════════════════════════════════════
#  F1 statistics aggregator
# ══════════════════════════════════════════════════════════════════════════════

def compute_f1_stats(csv_paths: list[str]) -> pd.DataFrame:
    """
    Evaluates each run CSV against the ground truth and returns a DataFrame
    with per-run rows plus a summary row (mean ± std, 95% CI).
    """
    metrics_cols = [
        "strict_f1", "strict_pr", "strict_re",
        "strict_tp", "strict_fp", "strict_fn",
        "content_f1", "content_pr", "content_re",
        "content_tp", "content_fp", "content_fn",
        "total_extracted",
    ]
    rows: list[dict] = []

    for path in csv_paths:
        m = run_evaluation(GROUND_TRUTH_FILE, path)
        if m:
            m["run"] = os.path.basename(path)
            rows.append(m)
        else:
            print(f"  [Warning] Evaluation failed for {path}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Aggregate summary row
    summary: dict[str, Any] = {"run": "AGGREGATE"}
    z = 1.96  # 95% CI multiplier
    n = len(df)
    for col in metrics_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna().astype(float)
        mu   = float(vals.mean())
        sd   = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        ci   = z * sd / (n ** 0.5) if n > 0 else 0.0
        summary[col]              = round(mu, 4)
        summary[f"{col}_std"]     = round(sd, 4)
        summary[f"{col}_ci95"]    = round(ci, 4)

    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  IAAS — Inter-Agent Agreement Score
# ══════════════════════════════════════════════════════════════════════════════

def _load_rules(csv_path: str) -> list[dict]:
    try:
        return pd.read_csv(csv_path).fillna("").to_dict("records")
    except Exception as e:
        print(f"  [Warning] Could not load {csv_path}: {e}")
        return []


def _rule_text(rule: dict) -> str:
    """Concatenate the semantically rich fields of a rule into a single string."""
    parts = [
        str(rule.get("class",      "") or ""),
        str(rule.get("station",    "") or ""),
        str(rule.get("sensor",     "") or ""),
        str(rule.get("condition",  "") or ""),
        str(rule.get("action",     "") or ""),
    ]
    return " | ".join(p for p in parts if p).strip()


def _pairwise_iaas(rules_i: list[dict], rules_j: list[dict],
                   emb_cache: dict) -> float:
    """
    Hungarian-matched pairwise agreement score between two rule sets.

    Score = (sum of matched content_agreement values) / max(|rules_i|, |rules_j|)

    Unmatched rules contribute 0, penalising sets with different cardinalities.
    Returns 0.0 if either set is empty.
    """
    if not rules_i or not rules_j:
        return 0.0

    ni, nj = len(rules_i), len(rules_j)
    cost = np.zeros((ni, nj), dtype=float)
    for a, r_i in enumerate(rules_i):
        for b, r_j in enumerate(rules_j):
            cost[a, b] = content_agreement(r_i, r_j, emb_cache)

    row_idx, col_idx = linear_sum_assignment(cost, maximize=True)
    matched_score = float(cost[row_idx, col_idx].sum())
    return matched_score / max(ni, nj)


def _embed_all_rules(all_rules: list[dict]) -> dict:
    """Build a shared SBERT embedding cache for all rule text fields."""
    texts: list[str] = []
    for r in all_rules:
        for field in ("condition", "action"):
            v = str(r.get(field) or "").strip()
            if v:
                texts.append(v)
    return _batch_encode_texts(texts)


def _cluster_rules(all_rules: list[dict]) -> list[list[dict]]:
    """
    Greedy clustering of rules by SBERT similarity on the concatenated
    rule text.  Two rules land in the same cluster if their cosine
    similarity exceeds SIMILARITY_THRESHOLD.

    Returns a list of clusters (each cluster = list of rule dicts).
    """
    sbert      = get_sbert()
    texts      = [_rule_text(r) for r in all_rules]
    unique_t   = list(dict.fromkeys(t for t in texts if t))
    if not unique_t:
        return [[r] for r in all_rules]

    embs = sbert.encode(unique_t, convert_to_tensor=True, show_progress_bar=False)
    text_to_emb = dict(zip(unique_t, embs))

    clusters:      list[list[dict]]  = []
    cluster_embs:  list[Any]         = []  # centroid embedding per cluster

    for rule, text in zip(all_rules, texts):
        emb = text_to_emb.get(text)
        if emb is None:
            clusters.append([rule])
            cluster_embs.append(None)
            continue

        best_sim   = -1.0
        best_idx   = -1
        for ci, c_emb in enumerate(cluster_embs):
            if c_emb is None:
                continue
            sim = float(_sbert_util.cos_sim(emb, c_emb).item())
            if sim > best_sim:
                best_sim = sim
                best_idx = ci

        if best_sim >= SIMILARITY_THRESHOLD:
            clusters[best_idx].append(rule)
            # Update centroid: mean of member embeddings
            member_texts  = [_rule_text(r) for r in clusters[best_idx]]
            member_embs   = [text_to_emb.get(t) for t in member_texts if text_to_emb.get(t) is not None]
            if member_embs:
                import torch
                cluster_embs[best_idx] = torch.stack(member_embs).mean(dim=0)
        else:
            clusters.append([rule])
            cluster_embs.append(emb)

    return clusters


def compute_iaas(csv_paths: list[str]) -> dict:
    """
    Computes all IAAS metrics across K pipeline runs.

    Returns a dict with:
      - iaas_mean / iaas_std          pairwise agreement
      - hallucination_rate            proxy: rules in < HALLUCINATION_FRAC of runs
      - consensus_rate                rules in ≥ CONSENSUS_FRAC of runs
      - n_clusters                    unique rule concepts found
      - n_consensus_rules             rules in consensus set
      - per_pair                      list of {run_i, run_j, iaas} for each pair
    """
    K = len(csv_paths)
    if K < 2:
        print("  [IAAS] Need at least 2 runs; skipping.")
        return {}

    print("\n[IAAS] Loading SBERT model …")
    get_sbert()

    run_rules: list[list[dict]] = [_load_rules(p) for p in csv_paths]
    run_names: list[str]        = [os.path.basename(p) for p in csv_paths]

    # Build shared embedding cache once
    all_rules_flat = [r for run in run_rules for r in run]
    print(f"[IAAS] Building embedding cache for {len(all_rules_flat)} rules …")
    emb_cache = _embed_all_rules(all_rules_flat)

    # ── Pairwise IAAS ─────────────────────────────────────────────────────────
    print("[IAAS] Computing pairwise agreement …")
    per_pair: list[dict] = []
    for (i, ri), (j, rj) in combinations(enumerate(run_rules), 2):
        score = _pairwise_iaas(ri, rj, emb_cache)
        per_pair.append({"run_i": run_names[i], "run_j": run_names[j], "iaas": round(score, 4)})
        print(f"    {run_names[i]} ↔ {run_names[j]}  IAAS = {score:.4f}")

    iaas_vals = [p["iaas"] for p in per_pair]
    iaas_mean = float(np.mean(iaas_vals))
    iaas_std  = float(np.std(iaas_vals, ddof=1)) if len(iaas_vals) > 1 else 0.0

    # ── Cluster rules across all runs ─────────────────────────────────────────
    print("[IAAS] Clustering rules across all runs …")
    clusters = _cluster_rules(all_rules_flat)
    n_clusters = len(clusters)
    print(f"[IAAS] Found {n_clusters} unique rule concepts across {K} runs.")

    # Tag each rule with its run index so the presence matrix can be built.
    # all_rules_flat contains references to the same dicts as run_rules, so
    # this mutation is visible in the cluster members too.
    for ri, run in enumerate(run_rules):
        for rule in run:
            rule["__run_idx__"] = ri

    # Binary presence matrix: rows = clusters, cols = runs
    presence = np.zeros((n_clusters, K), dtype=int)
    for ci, cluster in enumerate(clusters):
        for rule in cluster:
            ri = rule.get("__run_idx__", -1)
            if 0 <= ri < K:
                presence[ci, ri] = 1

    # ── Stability fractions per cluster ───────────────────────────────────────
    run_frac      = presence.sum(axis=1) / K          # shape (n_clusters,)
    n_hallu       = int((run_frac < HALLUCINATION_FRAC).sum())
    n_consensus   = int((run_frac >= CONSENSUS_FRAC).sum())
    hallu_rate    = n_hallu    / n_clusters if n_clusters else 0.0
    consensus_rate = n_consensus / n_clusters if n_clusters else 0.0

    print(f"[IAAS] Hallucination-proxy rate : {hallu_rate:.2%}  ({n_hallu}/{n_clusters} clusters)")
    print(f"[IAAS] Consensus rate           : {consensus_rate:.2%}  ({n_consensus}/{n_clusters} clusters)")

    # ── Consensus rule set ─────────────────────────────────────────────────────
    # Pick the most representative rule per consensus cluster (highest mean
    # content_agreement to all other rules in the same cluster).
    consensus_rules: list[dict] = []
    for ci, cluster in enumerate(clusters):
        if run_frac[ci] < CONSENSUS_FRAC:
            continue
        if len(cluster) == 1:
            rep = cluster[0]
        else:
            best_score, rep = -1.0, cluster[0]
            for r in cluster:
                score = float(np.mean([
                    content_agreement(r, other, emb_cache)
                    for other in cluster if other is not r
                ]))
                if score > best_score:
                    best_score, rep = score, r
        clean = {k: v for k, v in rep.items() if not k.startswith("__")}
        clean["run_frac"] = round(float(run_frac[ci]), 3)
        consensus_rules.append(clean)

    return {
        "iaas_mean":            round(iaas_mean, 4),
        "iaas_std":             round(iaas_std, 4),
        "hallucination_rate":   round(hallu_rate, 4),
        "consensus_rate":       round(consensus_rate, 4),
        "n_runs":               K,
        "n_clusters":           n_clusters,
        "n_consensus_rules":    n_consensus,
        "n_hallucination_rules": n_hallu,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "hallucination_frac":   HALLUCINATION_FRAC,
        "consensus_frac":       CONSENSUS_FRAC,
        "per_pair":             per_pair,
        "_consensus_rules":     consensus_rules,  # stored separately, not in JSON body
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Report writers
# ══════════════════════════════════════════════════════════════════════════════

def save_f1_stats(df: pd.DataFrame) -> None:
    out = os.path.join(STEP3_RESULTS_DIR, "multi_run_stats.csv")
    df.to_csv(out, index=False)
    print(f"\n[Stats] Saved F1 multi-run stats → {out}")

    # Pretty-print the aggregate row
    agg = df[df["run"] == "AGGREGATE"]
    if not agg.empty:
        row = agg.iloc[0]
        print(f"\n  F1_content : {row.get('content_f1', '—'):.4f}  "
              f"± {row.get('content_f1_std', '—'):.4f}  "
              f"(CI95 ±{row.get('content_f1_ci95', '—'):.4f})")
        print(f"  F1_strict  : {row.get('strict_f1', '—'):.4f}  "
              f"± {row.get('strict_f1_std', '—'):.4f}  "
              f"(CI95 ±{row.get('strict_f1_ci95', '—'):.4f})")


def save_iaas_report(iaas: dict) -> None:
    consensus_rules = iaas.pop("_consensus_rules", [])

    # JSON report
    out_json = os.path.join(STEP3_RESULTS_DIR, "iaas_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(iaas, f, indent=2, ensure_ascii=False)
    print(f"[IAAS] Saved IAAS report → {out_json}")

    # Consensus rules CSV
    if consensus_rules:
        fields = [k for k in consensus_rules[0] if not k.startswith("__")]
        out_csv = os.path.join(STEP3_RESULTS_DIR, "consensus_rules.csv")
        pd.DataFrame(consensus_rules)[fields].to_csv(out_csv, index=False)
        print(f"[IAAS] Saved {len(consensus_rules)} consensus rules → {out_csv}")

    # Pretty summary
    print(f"\n  IAAS           : {iaas['iaas_mean']:.4f} ± {iaas['iaas_std']:.4f}")
    print(f"  Hallucination  : {iaas['hallucination_rate']:.2%} "
          f"({iaas['n_hallucination_rules']}/{iaas['n_clusters']} rule concepts)")
    print(f"  Consensus set  : {iaas['consensus_rate']:.2%} "
          f"({iaas['n_consensus_rules']}/{iaas['n_clusters']} rule concepts)")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-run pipeline runner + IAAS evaluation"
    )
    parser.add_argument(
        "--runs", type=int, default=20,
        help="Number of independent pipeline runs (default: 20)",
    )
    parser.add_argument(
        "--temperature", type=float, default=MULTI_RUN_TEMPERATURE,
        help=f"LLM temperature for all calls (default: {MULTI_RUN_TEMPERATURE})",
    )
    parser.add_argument(
        "--skip-runs", action="store_true",
        help="Skip pipeline execution; load existing CSVs from multi_run/ and recompute stats",
    )
    parser.add_argument(
        "--seed-base", type=int, default=None,
        help="Base seed for reproducibility. Run i gets seed=(seed_base + i). "
             "Omit to use no seed (genuine non-determinism, original behaviour).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing run CSVs and regenerate all N runs from scratch.",
    )
    args = parser.parse_args()

    # ── Collect run CSVs ──────────────────────────────────────────────────────
    if args.skip_runs:
        csv_paths = sorted(
            os.path.join(MULTI_RUN_DIR, f)
            for f in os.listdir(MULTI_RUN_DIR)
            if f.startswith("ext_multi_agent_run") and f.endswith(".csv")
        )
        if not csv_paths:
            print(f"[Error] No run CSVs found in {MULTI_RUN_DIR}. Run without --skip-runs first.")
            sys.exit(1)
        print(f"[Runner] --skip-runs: found {len(csv_paths)} existing CSVs.")
    else:
        csv_paths = run_multi(args.runs, args.temperature, args.seed_base, args.force)

    # ── F1 statistics ─────────────────────────────────────────────────────────
    print(f"\n[Stats] Evaluating {len(csv_paths)} runs against ground truth …")
    f1_df = compute_f1_stats(csv_paths)
    if not f1_df.empty:
        save_f1_stats(f1_df)

    # ── IAAS ──────────────────────────────────────────────────────────────────
    print(f"\n[IAAS] Computing inter-run agreement across {len(csv_paths)} runs …")
    iaas = compute_iaas(csv_paths)
    if iaas:
        save_iaas_report(iaas)

    print("\n[Done] Multi-run evaluation complete.")


if __name__ == "__main__":
    main()
