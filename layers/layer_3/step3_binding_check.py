"""step3_binding_check.py
==================================================
Slot-binding verification on ALL four corpora, without inventing a field
mapping.

F1_content cannot tell whether a value sits in the right slot, and the direct
field-by-field verification is only possible on the development corpus, where
a syntactic column-name correspondence exists. This check closes most of the
remaining gap for the external corpora using only two things neither side has
to be told: the ground truth's OWN role columns (its low bound is whatever its
own low-bound column holds), and the internal structure of the predicted
records. No ground-truth column name is ever mapped to a predicted field name.

For every accepted (TP) pair from the existing match audit, and for every
populated role value in the ground-truth row (a low bound, a high bound, a
typed limit), the value is located inside the paired record:

    slot-bound   exactly one scalar field of the record holds the value
    in-text      the value appears only inside a free-text field
    ambiguous    more than one scalar field holds the value
    missing      the value appears nowhere in the record

If the extractor binds values to slots rather than scattering them, then for
each ground-truth role the same discovered field should receive the value in
(nearly) every pair, corpus-wide -- e.g. the biogas ground truth's bound_low
landing in the record field the document calls its low limit, in every matched
row. That corpus-level consistency is what is measured: it demonstrates slot
binding using the record's own vocabulary, which is the only vocabulary a
schema-agnostic extractor has.

Reads the frozen match-audit files and prediction CSVs; changes no score.
Writes step3_results/robustness/binding_check.csv (per-role summary) and
binding_check_detail.csv (one row per located value).

Usage:
    python3 step3_binding_check.py
"""

from __future__ import annotations

import glob
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_evaluation_generic_dynamic as E

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PRED_DIR = os.path.join(_PROJECT_ROOT, "layers", "layer_2", "step2_results_generic")
AUDIT_DIR = os.path.join(_PROJECT_ROOT, "layers", "step3_results", "match_audit")
OUT_DIR = os.path.join(_PROJECT_ROOT, "layers", "step3_results", "robustness")

# corpus -> ground truth path, role columns, optional role-qualifier column.
# These are GROUND-TRUTH-side declarations only: the ground truth's own schema
# says which of its columns is a low bound. Nothing here names a predicted
# field, so no cross-vocabulary mapping is introduced.
CORPORA = {
    "dev_production_line": {
        "gt": "data/dataset/kg_seed/ground_truth.csv",
        "roles": [("critLo", "critLo"), ("warnLo", "warnLo"),
                  ("warnHi", "warnHi"), ("critHi", "critHi")],
    },
    "external_test_biogas": {
        "gt": "data/external_test_biogas/ground_truth_biogas.csv",
        "roles": [("bound_low", "bound_low"), ("bound_high", "bound_high")],
    },
    "external_test_sulfuric_acid": {
        "gt": "data/external_test_sulfuric_acid/ground_truth_SA.csv",
        "roles": [("numeric_value", "numeric_value")],
        "qualifier": "unit",
    },
    "external_test_desalination": {
        "gt": "data/external_test_desalination/ground_truth_desalination.csv",
        "roles": [("min_bound", "min_bound"), ("max_bound", "max_bound"),
                  ("alert_low", "alert_low"), ("alert_high", "alert_high")],
    },
}

BOOKKEEPING = {"id", "source_file", "source_span"}


def _as_float(text) -> float | None:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _value_in_text(value: float, text: str) -> bool:
    """The value written as a standalone number anywhere in a text field."""
    pattern = re.compile(r"(?<![\d.])" + re.escape(f"{value:g}") + r"(?![\d])")
    return bool(pattern.search(text or ""))


def locate(value: float, record: pd.Series) -> tuple[str, str]:
    """Classify where a ground-truth value sits inside one predicted record."""
    scalar_hits, text_hits = [], []
    for col in record.index:
        if col in BOOKKEEPING:
            continue
        cell = str(record[col]).strip()
        if not cell:
            continue
        as_num = _as_float(cell)
        if as_num is not None:
            if as_num == value:
                scalar_hits.append(col)
        elif _value_in_text(value, cell):
            text_hits.append(col)
    if len(scalar_hits) == 1:
        return "slot-bound", scalar_hits[0]
    if len(scalar_hits) > 1:
        return "ambiguous", "|".join(scalar_hits)
    if text_hits:
        return "in-text", "|".join(text_hits)
    return "missing", ""


def check_corpus(tag: str, spec: dict) -> tuple[list[dict], list[dict]]:
    gt = pd.read_csv(os.path.join(_PROJECT_ROOT, spec["gt"]), dtype=str).fillna("")
    qualifier = spec.get("qualifier")
    detail, observations = [], []

    audits = sorted(glob.glob(os.path.join(AUDIT_DIR, tag, "match_audit_*.csv")))
    for audit_path in audits:
        stem = os.path.basename(audit_path)[len("match_audit_"):]
        pred_path = os.path.join(PRED_DIR, stem)
        if not os.path.exists(pred_path):
            print(f"  [skip] no prediction file for {stem}")
            continue
        pred = pd.read_csv(pred_path, dtype=str).fillna("")
        audit = pd.read_csv(audit_path)
        tps = audit[audit["decision"].astype(str).str.startswith("TP")]
        run = re.search(r"_run(\d+)_", stem)
        run_no = int(run.group(1)) if run else 0

        for _, row in tps.iterrows():
            g = gt.iloc[int(row["gt_row"]) - 1]
            p = pred.iloc[int(row["pred_row"]) - 1]
            for role, col in spec["roles"]:
                value = _as_float(g.get(col, ""))
                if value is None:
                    continue
                role_key = role
                if qualifier and str(g.get(qualifier, "")).strip():
                    role_key = f"{role}:{str(g[qualifier]).strip()}"
                outcome, field = locate(value, p)
                observations.append(
                    {"corpus": tag, "run": run_no, "role": role_key,
                     "outcome": outcome, "field": field})
                detail.append(
                    {"corpus": tag, "run": run_no, "gt_id": row["gt_id"],
                     "role": role_key, "value": value,
                     "outcome": outcome, "pred_field": field})
    return observations, detail


def summarise(observations: list[dict]) -> pd.DataFrame:
    rows = []
    df = pd.DataFrame(observations)
    for (corpus, role), grp in df.groupby(["corpus", "role"]):
        n = len(grp)
        bound = grp[grp["outcome"] == "slot-bound"]
        modal_field, modal_n = ("", 0)
        if len(bound):
            modal_field, modal_n = Counter(bound["field"]).most_common(1)[0]
        rows.append({
            "corpus": corpus, "role": role, "values_checked": n,
            "slot_bound": len(bound),
            "in_text": int((grp["outcome"] == "in-text").sum()),
            "ambiguous": int((grp["outcome"] == "ambiguous").sum()),
            "missing": int((grp["outcome"] == "missing").sum()),
            "modal_field": modal_field,
            "modal_field_share_of_slot_bound":
                round(modal_n / len(bound), 3) if len(bound) else "",
        })
    return pd.DataFrame(rows)


def slot_agreement(tag: str, spec: dict) -> list[dict]:
    """Direct slot correctness, wherever the two sides happen to name a numeric
    column the same way.

    The modal-field statistic above measures CONSISTENCY -- whether a given
    ground-truth role always lands in the same predicted field -- and a system
    that put every low bound in the high slot would score a perfect 1.0 on it,
    because it would do so consistently. That is not a check on correctness, and
    on corpora where no field correspondence exists it is the only check
    available. Where a correspondence does exist it costs nothing to ask the
    stronger question directly: does the value the annotation records under a
    name appear under THAT name in the record the metric paired it with?

    The correspondence is a string comparison on normalised column names, never
    a hand-written mapping, so this runs only on corpora where the two
    vocabularies coincide by themselves and is silently empty elsewhere.
    """
    gt_df = pd.read_csv(os.path.join(_PROJECT_ROOT, spec["gt"]))
    cfg = E.EvalConfig(gt_id_field=spec.get("id_field", ""))
    gt_rows = gt_df.fillna("").to_dict("records")
    gt_id = E._detect_id_field(gt_df, cfg)
    exclude = E._ID_LIKE_NAMES | E._detect_noise_fields(gt_df, cfg)

    def norm(c: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(c).lower())

    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(
            PRED_DIR, f"ext_multi_agent_generic_{tag}_run*.csv"))):
        pred = pd.read_csv(path).fillna("")
        shared = {norm(c) for c in gt_df.columns} & {norm(c) for c in pred.columns}
        gt_by_norm = {norm(c): c for c in gt_df.columns}
        pred_by_norm = {norm(c): c for c in pred.columns}
        numeric = [n for n in shared
                   if n not in {"id", "ruleid", "source", "category",
                                "sourcefile", "sourcespan"}
                   and any(_as_float(v) is not None
                           for v in gt_df[gt_by_norm[n]].fillna(""))]
        if not numeric:
            continue
        pred_rows = pred.to_dict("records")
        a = E.build_assignment(cfg, gt_rows, pred_rows, gt_id, exclude)
        for k, (i, j) in enumerate(a.pairs):
            if a.scores[k] < cfg.threshold:
                continue
            for n in numeric:
                gv = _as_float(gt_rows[i].get(gt_by_norm[n], ""))
                if gv is None:
                    continue
                # Two different failures must not be added together. A value the
                # extraction never produced is a RECALL miss; a value it produced
                # but filed elsewhere is a BINDING error. Conflating them would
                # report a recall problem as a binding problem and overstate the
                # very thing this check exists to measure, so presence anywhere
                # in the record is recorded first and binding is conditioned on
                # it.
                blob = " ".join(str(v) for v in pred_rows[j].values())
                anywhere = _value_in_text(gv, blob)
                pv = _as_float(pred_rows[j].get(pred_by_norm[n], ""))
                rows.append({"corpus": tag, "run": os.path.basename(path),
                             "field": gt_by_norm[n], "gt_value": gv,
                             "pred_value": pv,
                             "present_anywhere": anywhere,
                             "in_correct_slot": anywhere and pv is not None
                                                and abs(gv - pv) < 1e-9})
    return rows


def main() -> None:
    all_obs, all_detail = [], []
    for tag, spec in CORPORA.items():
        print(f"[binding] {tag}")
        obs, detail = check_corpus(tag, spec)
        all_obs.extend(obs)
        all_detail.extend(detail)

    if not all_obs:
        sys.exit("no observations produced -- are the audit files present?")

    summary = summarise(all_obs)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_summary = os.path.join(OUT_DIR, "binding_check.csv")
    out_detail = os.path.join(OUT_DIR, "binding_check_detail.csv")
    summary.to_csv(out_summary, index=False)
    pd.DataFrame(all_detail).to_csv(out_detail, index=False)

    print(summary.to_string(index=False))
    print(f"[binding] wrote {out_summary}")
    print(f"[binding] wrote {out_detail}")

    slots = [r for tag, spec in CORPORA.items() for r in slot_agreement(tag, spec)]
    if slots:
        sdf = pd.DataFrame(slots)
        agg = sdf.groupby("corpus").agg(
            gt_cells=("in_correct_slot", "size"),
            present_anywhere=("present_anywhere", "sum"),
            in_correct_slot=("in_correct_slot", "sum")).reset_index()
        # recall_frac: did the extraction produce the value at all.
        # binding_frac: OF THOSE it produced, how many sit in the slot the
        # annotation would put them in. Only the second is a binding figure.
        agg["recall_frac"] = (agg["present_anywhere"] / agg["gt_cells"]).round(3)
        agg["binding_frac"] = (agg["in_correct_slot"]
                               / agg["present_anywhere"].replace(0, pd.NA)).round(3)
        out_slot = os.path.join(OUT_DIR, "slot_agreement.csv")
        agg.to_csv(out_slot, index=False)
        print("\n[binding] direct slot correctness where the two vocabularies "
              "coincide by themselves (binding_frac is conditioned on the value "
              "having been extracted at all):")
        print(agg.to_string(index=False))
        print(f"[binding] wrote {out_slot}")
    else:
        print("\n[binding] no corpus shares a numeric column name with its "
              "predictions; direct slot correctness is unavailable.")


if __name__ == "__main__":
    main()
