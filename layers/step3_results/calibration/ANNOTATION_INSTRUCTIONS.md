# Threshold calibration: annotation instructions

**Time required:** about one hour per annotator. **Annotators needed:** two,
working independently. Neither should be the author of the pipeline.

## What this establishes

The metric accepts a pair of rows as describing the same rule when their
agreement score reaches 0.6. Nothing currently shows that this boundary matches
where a person would draw it. These labels settle that.

## The task

Open `calibration_sheet.csv`. It has 60 items. Each item shows two texts:

- **`gt_text_compared`** (call it A): one rule as written in a reference
  annotation.
- **`pred_text_compared`** (call it B): one rule as produced by an extraction
  system.

For each item, answer one question in the `human_same_rule` column:

> **Do A and B state the same operational rule?**

Write `y` or `n`. Nothing else. Leave no blanks.

## How to decide

Judge the **constraint being described**, not the wording.

- Different field names, different category labels, different word order, and a
  different number of fields are all irrelevant. Ignore them.
- Say `y` if a plant operator following A and a plant operator following B would
  do the same thing in the same situation.
- Say `n` if they would not, including when B is about the right equipment but a
  different requirement, or when B states only a fragment of A and an operator
  could not act on it.
- If B carries extra detail that A omits, but the constraint A states is present
  and correct in B, say `y`.

Some items are deliberately near the boundary and will feel genuinely
uncertain. Make a call rather than leaving a blank; disagreement between the two
annotators on hard items is itself a result and is reported.

## Rules

1. **Do not open `calibration_key.csv`.** It holds the scores and decisions the
   sheet is designed to withhold. Opening it invalidates the exercise.
2. **Do not confer.** The two annotators label independently. Compare only after
   both sheets are finished.
3. Save each filled sheet under a distinct name, for example
   `calibration_sheet_annotator1.csv`.

## Producing the result

```zsh
# agreement between the two annotators: does the question have a stable answer?
python3 layers/layer_3/step3_calibration_sample.py agree \
        calibration_sheet_annotator1.csv calibration_sheet_annotator2.csv

# each annotator against the metric's tau = 0.6 decisions
python3 layers/layer_3/step3_calibration_sample.py score calibration_sheet_annotator1.csv
python3 layers/layer_3/step3_calibration_sample.py score calibration_sheet_annotator2.csv
```

`score` also reports how accurately each candidate threshold from 0.45 to 0.75
reproduces the human labels, which locates the reported operating point relative
to the human boundary.

## Reading the outcome

Report the inter-annotator kappa first. The metric cannot be expected to agree
with human judgment more closely than two humans agree with each other, so that
figure bounds everything else.

Then report each annotator's agreement with the tau = 0.6 decisions. Whatever it
comes to, report it. A result showing the threshold sits away from the human
boundary is a finding about the metric and belongs in the thesis exactly as much
as a favourable one does. The sample, the seed and the key are fixed in advance
so that this cannot be re-rolled.
