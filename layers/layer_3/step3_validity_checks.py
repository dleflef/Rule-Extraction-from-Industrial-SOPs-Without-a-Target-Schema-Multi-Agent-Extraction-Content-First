"""step3_validity_checks.py
==================================================
Evidence that F1_content measures extraction quality rather than a similarity
floor -- produced as tables, not as an argument.

A reported F1 of 0.72-0.84 means nothing on its own. Sentence embeddings assign
a non-trivial cosine to any two strings drawn from the same genre, so a metric
built on them can post a respectable-looking score while being unable to tell a
correct extraction from an unrelated one. Two questions decide whether the
number is worth reporting, and both are empirical:

  1. Does the metric COLLAPSE when the predictions cannot possibly be right?
     Scoring a corpus's predictions against a DIFFERENT corpus's ground truth is
     the null case: nothing is correct, so a metric that measures content must
     go to ~0. One that stays high is measuring genre, not content.

  2. Does the metric LOSE points when the content is degraded in a specific,
     known way? Corrupting numeric quantities tests value sensitivity, while
     destroying word order with the vocabulary fixed tests order sensitivity.
     What each perturbation costs says WHICH part of a record the metric is
     actually reading -- a
     perturbation that costs nothing marks something the metric does not check,
     which is a limitation to declare rather than a result to hide.

A third table sweeps the acceptance threshold. The point is not to find a good
threshold -- it is to show where the reported operating point sits on the curve.
An operating point on a declining slope cannot have been chosen to flatter the
system; one sitting on a peak, or on a cliff edge, has to be reported as such.

Nothing here re-implements the metric: every figure comes from the evaluator's
own evaluate_content / assignment_scores, so these tables and the reported
results cannot drift apart.

Usage:
    python3 step3_validity_checks.py                 # all four corpora, all runs
    python3 step3_validity_checks.py --runs 1        # first run per corpus only
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_evaluation_generic_dynamic as E   # noqa: E402

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PRED_DIR = os.path.join(_PROJECT_ROOT, "layers", "layer_2", "step2_results_generic")
OUT_DIR = os.path.join(E.STEP3_RESULTS_DIR, "validity")

# tag : ground truth : id-field override ("" = auto-detect). Mirrors the corpus
# list in run_and_evaluate.sh; desalination overrides for the reason given there.
CORPORA = [
    ("dev_production_line",        "data/dataset/kg_seed/ground_truth.csv",                         ""),
    ("external_test_biogas",       "data/external_test_biogas/ground_truth_biogas.csv",             ""),
    ("external_test_sulfuric_acid", "data/external_test_sulfuric_acid/ground_truth_SA.csv",         ""),
    ("external_test_desalination", "data/external_test_desalination/ground_truth_desalination.csv", ""),
]

THRESHOLDS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
SEED = 42


# ── Perturbations ──────────────────────────────────────────────────────────────
# Each takes a prediction DataFrame and returns degraded prediction rows. They
# are deliberately narrow: each one damages ONE aspect of a record so that the
# score it costs is attributable to that aspect rather than to general noise.
# ATTRIBUTES holds a JSON object, so it is parsed and re-serialised rather than
# string-edited -- a perturbation that accidentally produced invalid JSON would
# blank the whole payload and overstate its own effect. A candidate number must
# also be detached from letters, digits, underscores and an immediately
# preceding hyphen. This preserves identifiers such as ST02_SEALING and DGT-101
# while still recognising standalone signed quantities such as -1.0 and values
# next to units such as 20°C or 35%RH.
_NUM_IN_TEXT = re.compile(
    r"(?<![A-Za-z0-9_-])-?(?:\d+(?:\.\d*)?|\.\d+)(?![A-Za-z0-9_])")
_PASSTHROUGH = ("id", "source_file", "source_span")


def _map_attributes(raw, fn):
    try:
        d = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return raw
    if not isinstance(d, dict):
        return raw

    def walk(value):
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if value is None:
            return value
        return fn(str(value))

    return json.dumps(walk(d), ensure_ascii=False)


def _attribute_leaf_strings(raw) -> List[str]:
    """Text leaves that the evaluator will append from an attributes object."""
    try:
        data = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict):
        return []

    out = []

    def visit(value):
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif value is not None:
            out.append(str(value))

    visit(data)
    return out


def _numbers_in_row(row: Dict) -> List[str]:
    """Standalone numeric quantities in exactly the content cells being tested."""
    out = []
    for column, value in row.items():
        if column in _PASSTHROUGH or value in (None, ""):
            continue
        texts = (_attribute_leaf_strings(value) if column == "attributes"
                 else [str(value)])
        for text in texts:
            out.extend(match.group(0) for match in _NUM_IN_TEXT.finditer(text))
    return out


def _replace_row_numbers(row: Dict, replacements: List[str]) -> Dict:
    """Replace a row's standalone quantities in deterministic blob order."""
    replacement_iter = iter(replacements)

    def replace_text(value):
        return _NUM_IN_TEXT.sub(lambda _match: next(replacement_iter), str(value))

    out = dict(row)
    for column, value in row.items():
        if column in _PASSTHROUGH or value in (None, ""):
            continue
        out[column] = (_map_attributes(value, replace_text)
                       if column == "attributes" else replace_text(value))
    return out


def _bumped_number(raw: str) -> str:
    """The deterministic x -> 3x + 7 corruption, without needless .0 suffixes."""
    return f"{float(raw) * 3.0 + 7.0:.12g}"


def corrupt_numbers(df: pd.DataFrame, rng: random.Random) -> List[Dict]:
    """Every standalone numeric quantity is changed; identifiers stay intact.

    Isolates the question "are the extracted VALUES verified, or only the words
    around them". A metric that barely notices has not checked a single bound.
    Digits inside asset tags (DGT-101) are left alone -- corrupting those would
    damage identity, not values, and confound the two.
    """
    rows = df.to_dict("records")
    return [_replace_row_numbers(row, [_bumped_number(v) for v in _numbers_in_row(row)])
            for row in rows]


def permute_payloads(df: pd.DataFrame, rng: random.Random) -> List[Dict]:
    """Numeric payloads permuted between compatible records.

    Every numeric occurrence in every content field is considered, including
    quantities embedded in prose. Records are grouped by payload length and
    rotated within their group, so the recipient keeps the same number of
    numeric positions but receives a different record's quantities. Identifier
    digits are excluded by _NUM_IN_TEXT.

    This is the binding test, and it is deliberately narrow. Permuting a record's
    ENTIRE content between records would be close to permuting whole records, and
    a set-matching metric is invariant to that by construction -- the multiset of
    blobs barely changes, so such a test measures the definition of set matching
    rather than any property of this metric. Moving only the quantities produces
    the failure that actually occurs in extraction: the right rule carrying the
    wrong bound.
    """
    rows = df.to_dict("records")
    payloads = [_numbers_in_row(row) for row in rows]
    groups = {}
    for index, payload in enumerate(payloads):
        if payload:
            groups.setdefault(len(payload), []).append(index)

    out = [dict(row) for row in rows]
    for indices in groups.values():
        if len(indices) < 2:
            continue
        rng.shuffle(indices)
        for position, target in enumerate(indices):
            donor = indices[(position + 1) % len(indices)]
            out[target] = _replace_row_numbers(rows[target], payloads[donor])
    return out


def reverse_bounds_within_record(df: pd.DataFrame, rng: random.Random) -> List[Dict]:
    """Reverse a record's own numeric quantities across their content positions.

    This is the safety-critical binding failure stated concretely: a row whose
    columns read (critical low, warning low, warning high, critical high) and
    hold (0.65, 0.75, 0.95, 1.05) becomes (1.05, 0.95, 0.75, 0.65), so the
    critical LOW bound is now reported as the critical HIGH bound and vice
    versa. Every value the document printed is still present and still on the
    correct rule; only the slot each occupies is wrong. A plant configured from
    the corrupted record would alarm at the wrong end of every range.

    It is a strictly harder test than permute_payloads, and tests a different
    thing. That perturbation moves quantities BETWEEN records, so the optimal
    assignment can recover by re-pairing; this one leaves the multiset of values
    in each record untouched, so re-pairing cannot help and no assignment-level
    recovery is available. Whatever this costs is what the agreement function
    itself can see about which slot a value occupies.

    Records carrying fewer than two numeric values are unchanged, since there is
    no within-record ordering to reverse. Unlike the former column-detection
    implementation, quantities embedded in prose are included, so the test is
    applied consistently to every corpus without assuming its induced schema.
    """
    rows = df.to_dict("records")
    out = []
    for row in rows:
        values = _numbers_in_row(row)
        out.append(_replace_row_numbers(row, list(reversed(values)))
                   if len(values) >= 2 else dict(row))
    return out


def permute_labels(df: pd.DataFrame, rng: random.Random) -> List[Dict]:
    """The reverse: the category label permuted between records, payloads stay
    put. Measures how much of the score the discovered label carries."""
    d = df.copy()
    if "category" in d:
        v = list(d["category"])
        rng.shuffle(v)
        d["category"] = v
    return d.to_dict("records")


def _rebuild_from_tokens(df: pd.DataFrame, shuffle: bool, rng: random.Random) -> List[Dict]:
    """Rewrite each record so its blob is exactly its own tokens, optionally in
    scrambled order.

    Shuffling words INSIDE each field does not test word order here: most fields
    hold a single token ("35", "C", "Temperature"), so there is no order in them
    to destroy. The order that exists is the order of the assembled blob, so the
    scramble has to be applied there -- to the same token multiset the metric
    would otherwise see.
    """
    rows = []
    for r in df.to_dict("records"):
        toks = E._pred_blob(r).replace(" | ", " ").split()
        if shuffle:
            rng.shuffle(toks)
        out = {c: "" for c in df.columns}
        for k in _PASSTHROUGH:
            if k in r:
                out[k] = r[k]
        out["category"] = " ".join(toks)      # whole blob in one content field
        out["attributes"] = "{}"
        rows.append(out)
    return rows


def word_salad(df: pd.DataFrame, rng: random.Random) -> List[Dict]:
    """Same tokens, destroyed order. Separates "the metric reads meaning" from
    "the metric counts shared words": a bag-of-words metric in disguise is
    untouched by this. Read it against tokens_in_order below, never alone --
    that control isolates the cost of the rewrite itself from the cost of the
    scramble."""
    return _rebuild_from_tokens(df, shuffle=True, rng=rng)


def tokens_in_order(df: pd.DataFrame, rng: random.Random) -> List[Dict]:
    """Control for word_salad: identical rewrite, original token order kept.
    Any F1 difference from the reported figure is the price of flattening the
    record, and only the gap between this and word_salad is word order."""
    return _rebuild_from_tokens(df, shuffle=False, rng=rng)


def pad_with_corpus_numbers(df: pd.DataFrame, rng: random.Random) -> List[Dict]:
    """Every record gains the numbers carried by OTHER records of the same file.

    No extracted content is changed, no value is made wrong, and no knowledge of
    the ground truth is used: the padding is drawn from the prediction file
    itself. Numbers are pooled the way the EVALUATOR reads them -- from the whole
    record, not from columns that happen to be numeric -- because a metric reads
    the flattened blob and padding must therefore be measured against the same
    view.

    This is the test the symmetric numeric term exists to pass. A metric scoring
    only "does the record contain the ground truth's numbers" prices surplus
    values at zero, so padding can only help it. A metric that also asks whether
    the record's own numbers are justified must charge for them. A score that
    RISES under this perturbation is not measuring numeric correctness, whatever
    it charges for corrupting a value.
    """
    rows = df.to_dict("records")
    per_record = [set(_numbers_in_row(row)) for row in rows]
    for i, r in enumerate(rows):
        borrowed = set()
        for j, nums in enumerate(per_record):
            if j != i:
                borrowed |= nums
        borrowed -= per_record[i]
        if borrowed:
            extra = sorted(borrowed)
            rng.shuffle(extra)
            r["_padding"] = " ".join(extra)
    return rows


PERTURBATIONS = [
    ("numbers_corrupted", corrupt_numbers),
    ("payloads_permuted", permute_payloads),
    ("padded_with_corpus_numbers", pad_with_corpus_numbers),
    ("bounds_reversed",   reverse_bounds_within_record),
    ("labels_permuted",   permute_labels),
    ("tokens_in_order",   tokens_in_order),
    ("word_salad",        word_salad),
]


# ── Corpus loading ─────────────────────────────────────────────────────────────
class Corpus:
    def __init__(self, tag: str, gt_rel: str, id_field: str, n_runs: int):
        self.tag = tag
        self.cfg = E.EvalConfig(gt_id_field=id_field)
        gt_df = pd.read_csv(os.path.join(_PROJECT_ROOT, gt_rel))
        self.gt_rows = gt_df.fillna("").to_dict("records")
        self.gt_id = E._detect_id_field(gt_df, self.cfg)
        self.exclude = E._ID_LIKE_NAMES | E._detect_noise_fields(gt_df, self.cfg)
        paths = sorted(glob.glob(os.path.join(
            PRED_DIR, f"ext_multi_agent_generic_{tag}_run*.csv")))
        self.pred_paths = paths[:n_runs] if n_runs else paths
        self.preds = [pd.read_csv(p).fillna("") for p in self.pred_paths]

    def f1(self, pred_rows) -> float:
        return E.evaluate_content(self.cfg, self.gt_rows, pred_rows,
                                  self.gt_id, self.exclude)[0]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ── Table 1: threshold curve ───────────────────────────────────────────────────
def threshold_curve(corpora: List[Corpus]) -> pd.DataFrame:
    """F1 at every threshold, reusing one assignment per run -- the pairing does
    not depend on the threshold, only the accept/reject decision does."""
    rows = []
    for c in corpora:
        per_t = {t: [] for t in THRESHOLDS}
        for df in c.preds:
            pred_rows = df.to_dict("records")
            scores = E.assignment_scores(c.cfg, c.gt_rows, pred_rows, c.gt_id, c.exclude)
            for t in THRESHOLDS:
                tp = sum(1 for s in scores if s >= t)
                per_t[t].append(E._prf_from_tp(tp, len(pred_rows), len(c.gt_rows))[0])
        row = {"corpus": c.tag, "runs": len(c.preds)}
        for t in THRESHOLDS:
            row[f"f1@{t}"] = round(_mean(per_t[t]), 3)
        reported = _mean(per_t[0.6])
        lo, hi = _mean(per_t[0.55]), _mean(per_t[0.65])
        row["max_change_per_0.05_step"] = round(max(abs(reported - lo), abs(reported - hi)), 3)
        # A STRICT peak is the suspicious shape: a threshold that scores better
        # than both its neighbours could have been chosen for that reason. A tie
        # with a neighbour is a plateau, which is the opposite finding -- the
        # score is insensitive to the choice there -- so the two must not be
        # reported under one label.
        row["operating_point_is_local_max"] = bool(reported > lo and reported > hi)
        row["operating_point_shape"] = (
            "peak" if reported > lo and reported > hi else
            "plateau" if abs(reported - lo) < 1e-9 or abs(reported - hi) < 1e-9 else
            "declining" if reported < lo else "rising")
        rows.append(row)
    return pd.DataFrame(rows)


# ── Table 2: null + perturbations ──────────────────────────────────────────────
def perturbation_table(corpora: List[Corpus], null_pairs: List[Dict]) -> pd.DataFrame:
    """Reported F1 beside the null case and each degraded variant."""
    rows = []
    for c in corpora:
        rng = random.Random(SEED)
        reported = [c.f1(df.to_dict("records")) for df in c.preds]

        # NULL: this corpus's predictions scored against every OTHER corpus's
        # ground truth. Nothing can be correct, so a content metric must go to 0.
        # Recorded per (predictions, ground truth) pair as well as pooled: a
        # pooled mean hides which pairing is responsible for the worst case, and
        # the worst case is what a reviewer will ask about.
        null = []
        for other in corpora:
            if other.tag == c.tag:
                continue
            per_pair = [E.evaluate_content(other.cfg, other.gt_rows,
                                           df.to_dict("records"),
                                           other.gt_id, other.exclude)[0]
                        for df in c.preds]
            null_pairs.append({
                "predictions_from": c.tag,
                "scored_against_gt": other.tag,
                "runs": len(per_pair),
                "f1_mean": round(_mean(per_pair), 3),
                "f1_max": round(max(per_pair), 3),
            })
            null.extend(per_pair)

        row = {
            "corpus": c.tag,
            "runs": len(c.preds),
            "f1_reported": round(_mean(reported), 3),
            "f1_reported_sd": round(_std(reported), 3),
            "f1_null_wrong_gt": round(_mean(null), 3),
            "f1_null_max": round(max(null), 3) if null else 0.0,
        }
        for name, fn in PERTURBATIONS:
            # Each perturbation draws from its OWN seeded stream, keyed on its
            # name. Sharing one generator across the list makes every figure
            # depend on the order and number of perturbations that ran before
            # it, so adding a new check silently moves the results of the
            # existing ones. Keying on the name makes each figure a property of
            # that perturbation alone, and reproducible in isolation.
            rng = random.Random(f"{SEED}:{name}")
            perturbed = [fn(df, rng) for df in c.preds]
            vals = [c.f1(rows) for rows in perturbed]
            changed = [
                sum(E._pred_blob(original) != E._pred_blob(damaged)
                    for original, damaged in zip(df.to_dict("records"), rows))
                for df, rows in zip(c.preds, perturbed)
            ]
            row[f"f1_{name}"] = round(_mean(vals), 3)
            row[f"drop_{name}"] = round(_mean(reported) - _mean(vals), 3)
            row[f"changed_records_{name}_mean"] = round(_mean(changed), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def print_verdict(pert: pd.DataFrame, curve: pd.DataFrame) -> None:
    print(f"\n{'='*78}\n  READING THESE TABLES\n{'='*78}")
    worst_null = pert["f1_null_max"].max()
    print(f"\n  NULL CASE (predictions vs the WRONG ground truth)")
    print(f"    highest F1 any corpus reached on a corpus it did not come from: {worst_null:.3f}")
    if worst_null < 0.15:
        print("    -> Well below every reported score. The metric is discriminating the")
        print("       right document from a wrong one, not scoring genre similarity.")
    else:
        print("    -> HIGH. A sizeable part of every reported score is reachable without")
        print("       being right. Report the null alongside the headline figure.")

    print(f"\n  PERTURBATIONS (mean F1 lost)")
    for name, _ in PERTURBATIONS:
        d = pert[f"drop_{name}"]
        if name == "tokens_in_order":
            note = "CONTROL for word_salad -- cost of the rewrite alone, not a check"
        elif name == "word_salad":
            # only the gap over the control is attributable to word order
            gap = (pert["f1_tokens_in_order"] - pert["f1_word_salad"]).mean()
            note = (f"word order alone costs {gap:+.3f} vs its control -- "
                    + ("order IS read" if gap > 0.05 else
                       "ORDER IS NOT READ: this behaves as a bag of tokens"))
        else:
            note = ("detectable F1 sensitivity" if d.mean() > 0.01 else
                    "no material F1 response at the operating threshold")
        print(f"    {name:<20} {d.mean():+.3f}  (per corpus {d.min():+.3f}..{d.max():+.3f})  {note}")

    print(f"\n  THRESHOLD (reported operating point = 0.60)")
    for _, r in curve.iterrows():
        flag = ""
        if r["operating_point_is_local_max"]:
            flag = "  <-- STRICT PEAK; justify the choice independently"
        elif r["max_change_per_0.05_step"] > 0.10:
            flag = "  <-- unstable: report the curve, not the single figure"
        print(f"    {r['corpus']:<30} F1 moves {r['max_change_per_0.05_step']:.3f} "
              f"per 0.05 step  [{r['operating_point_shape']}]{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=0,
                    help="Runs per corpus to use (0 = all available).")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'='*78}")
    print("  Step-3 validity checks: is F1_content measuring extraction quality?")
    print(f"{'='*78}")
    E.get_sbert(E._DEFAULT_SBERT)

    corpora = []
    for tag, gt_rel, idf in CORPORA:
        if not os.path.exists(os.path.join(_PROJECT_ROOT, gt_rel)):
            print(f"  [skip] {tag}: ground truth not found")
            continue
        c = Corpus(tag, gt_rel, idf, args.runs)
        if not c.preds:
            print(f"  [skip] {tag}: no prediction files in {PRED_DIR}")
            continue
        print(f"  {tag}: {len(c.gt_rows)} GT rows, {len(c.preds)} run(s)")
        corpora.append(c)
    if len(corpora) < 2:
        print("\nNeed at least two corpora -- the null case scores one corpus's "
              "predictions against another's ground truth.")
        raise SystemExit(1)

    print("\n  [1/2] threshold curve ...")
    curve = threshold_curve(corpora)
    print("  [2/2] null case + perturbations ...")
    null_pairs: List[Dict] = []
    pert = perturbation_table(corpora, null_pairs)

    curve_path = os.path.join(args.out_dir, "validity_threshold_curve.csv")
    pert_path = os.path.join(args.out_dir, "validity_perturbations.csv")
    null_path = os.path.join(args.out_dir, "validity_null_pairs.csv")
    curve.to_csv(curve_path, index=False)
    pert.to_csv(pert_path, index=False)
    pd.DataFrame(null_pairs).sort_values("f1_max", ascending=False).to_csv(null_path, index=False)

    pd.set_option("display.width", 250)
    print(f"\n{'='*78}\n  THRESHOLD CURVE\n{'='*78}")
    print(curve.to_string(index=False))
    print(f"\n{'='*78}\n  NULL CASE + PERTURBATIONS\n{'='*78}")
    cols = ["corpus", "f1_reported", "f1_null_wrong_gt", "f1_null_max"] + \
           [f"f1_{n}" for n, _ in PERTURBATIONS]
    print(pert[cols].to_string(index=False))
    print(f"\n{'='*78}\n  NULL CASE BY PAIRING (worst first)\n{'='*78}")
    print(pd.DataFrame(null_pairs).sort_values("f1_max", ascending=False)
          .head(6).to_string(index=False))
    print_verdict(pert, curve)
    print(f"\n  Written: {curve_path}")
    print(f"           {pert_path}")
    print(f"           {null_path}")
    print(f"  Perturbation seed: {SEED} (deterministic; re-running reproduces these figures)")
