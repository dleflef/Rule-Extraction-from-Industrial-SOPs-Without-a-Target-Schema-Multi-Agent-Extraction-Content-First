# External test corpus: reverse-osmosis desalination (high-pressure RO skids)

One SOP for a fictional high-pressure reverse-osmosis desalination skid.

## Construction

Same protocol as the other two external corpora (see
`../external_test_sulfuric_acid/README.md`):

- Generated 2026-08-06 by **DeepSeek-V4-Pro** — a different model family from
  the `gemma4:31b` the extraction pipeline runs on.
- **Column names and class vocabulary** were the generator's own, and the
  development corpus's field names were explicitly **forbidden**.
- **The domain was the generator's choice** from anything outside a five-item
  exclusion list.
- The prompt did **not** stop at the container format. It also mandated eight
  document features (a)–(h), required an identifier column carrying printed
  codes where the document prints them, and required the generator to state a
  granularity rule and follow it. See the verbatim prompt below; this is the
  most heavily specified of the three.
- Document and annotation were produced in the **same generation pass**, so the
  annotation is not an independent reader of the document.
- Generated **once** and kept **as delivered**; no regeneration, no discarding,
  no selection among candidates.
- **Not read before the first scoring run.**
- No repairs. The annotation is byte-identical to what was delivered.

## The generating prompt, verbatim

```text
You are producing a HELD-OUT TEST CORPUS for evaluating an information-extraction
system. The system will be scored on how well it recovers your ground truth from
your document. Your job is to write both, and to make them a fair, complete test.

## 1. Pick a domain

Choose ONE industrial process-control domain that is NOT in this list:
  - bottling / packaging production line
  - anaerobic digester / biogas plant
  - semiconductor cleanroom
  - hydrogen refuelling station
  - wind farm

Anything else is fine (water treatment, district heating, LNG regasification,
pharmaceutical freeze-drying, grain terminal, tunnel ventilation, brewery,
paper mill, ...). Also give me a ONE-WORD lowercase name for the corpus.

## 2. Write the SOP document

Plain text, roughly 150-250 lines. It must be a realistic Standard Operating
Procedure with its OWN identifier scheme (e.g. "WT-OP-014", "R-07", "M001") and
its OWN vocabulary for every concept.

DO NOT reuse these words as field/column names anywhere: station, sensor,
sensorType, critHi, warnHi, critLo, warnLo. Invent whatever terms your domain
would actually print. The point of this test is that the extractor has never
seen your vocabulary.

The document MUST contain all of the following:
  (a) Numbered/coded rules written as prose sentences.
  (b) At least three rules stated in prose with NO identifier of their own.
  (c) At least one DATA TABLE whose columns are different KINDS of information
      about each row (a unit, a lower bound, an upper bound, a required
      response, a deadline...). One row = one rule.
  (d) At least one CROSS-REFERENCE MATRIX: the row axis is a list of things of
      one kind (roles, phases, zones, modes...), the column axis is a list of
      things of another single kind, and every cell states what applies at that
      intersection. One cell = one rule.
  (e) A preamble / purpose / scope section that states NO rules at all.
  (f) Numeric limits with units, including at least one parameter carrying
      several tiered bounds (e.g. a low alarm, a low warning, a high warning,
      a high alarm).
  (g) At least one rule that cross-references another rule by its identifier.
  (h) At least one line that states TWO separate rules at once.

## 3. Write the ground truth CSV

One row per rule. Column names must come from YOUR document's vocabulary, not
from any convention I've given you. Include an id column carrying the
document's printed identifier where one exists, and a source column naming the
document.

Rules for the ground truth, in order of importance:

  1. COMPLETENESS. Every rule-bearing element in the document must appear as a
     row. Every cell of every matrix. Every row of every table. Every prose
     rule, including the ones with no printed identifier. Do not skip a table
     or a section because it feels minor or descriptive - if it states
     something that constrains operation, it gets a row. An incomplete ground
     truth silently punishes a correct extractor, which ruins the test.
  2. STATE YOUR GRANULARITY RULE explicitly at the top of your answer, in one
     or two sentences, and then follow it without exception.
  3. Nothing in the ground truth may appear that is not stated in the document.

## 4. Before you answer, check

  - Does every one of (a)-(h) actually appear in the document?
  - Count the rule-bearing elements in the document. Does the CSV have exactly
    that many rows? State both numbers.
  - Does the preamble contain zero rules, and zero ground-truth rows?
  - Are any of the forbidden words used as column names?

## 5. Output format

Reply with exactly three things:
  1. The corpus name and your granularity rule, plus the two counts.
  2. A fenced block labelled DOCUMENT containing the complete .txt content.
  3. A fenced block labelled GROUND_TRUTH containing the complete .csv content.
```

## Files

- `SOP_RO_V4_HighPressureROSkids.txt` — the procedure.
- `ground_truth_desalination.csv` — 26 rows, 12 columns
  (`source, identifier, equipment_or_role, measurement_or_phase, metric_unit,
  min_bound, max_bound, alert_low, alert_high, required_action, time_allowance,
  rule_text`).

## Conventions this annotation chose for itself

- This is the **only** corpus of the four whose identifiers are printed in the
  document (`OP-ST-001`), so it is the only one where identifier agreement is
  achievable at all.
- The annotation records an identifier **only where the document prints one**:
  `identifier` is populated on 11 of 26 rows (10 distinct codes).
- Bounds named `min_bound` / `max_bound`, with separate `alert_low` /
  `alert_high`.
- No rule-class column at all.

## Scoring note — no flags, and what that costs

This corpus is scored by the same generic evaluator as every other, with no
per-corpus settings:

    python3 layers/layer_3/step3_evaluation_generic_dynamic.py \
        --gt data/external_test_desalination/ground_truth_desalination.csv \
        --pred-dir <dir> --aggregate

Reported figure: **0.804 ± 0.018** (P 0.787, R 0.823, max P 0.956).

Structural id detection misfires here, and the figure absorbs it. `identifier`
is populated on only 11 of 26 rows, so it fails the 0.9 coverage gate, and
detection falls through to `rule_text` — distinct on almost every row — which
means the rule statement itself is treated as the identifier and dropped from
the ground-truth blob.

Pinning the column by hand (`--gt-id-field identifier`) scores the same
predictions at **0.850 ± 0.019**. That is *not* what the thesis reports. A
metric needing per-corpus configuration to produce its headline figure invites
the objection the metric exists to remove, so the 0.046 is left on the table and
declared instead (thesis §The Desalination Figure Is Depressed by a Detection Misfire, in Limitations).
