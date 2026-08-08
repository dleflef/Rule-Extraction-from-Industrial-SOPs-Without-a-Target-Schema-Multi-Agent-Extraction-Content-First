"""step3_audit_interventions.py
==================================================
Aggregate the extraction pipeline's audit-stage verdicts into one table.

The audit stage is the one pipeline component whose contribution the headline
metric cannot see: F1_content scores the saved rows, and the rows are saved
AFTER the auditor has kept, dropped, or corrected them. Whether the auditor
actually intervenes is therefore a separate empirical question, and this
script answers it from the audit_log_*.csv sidecars the pipeline writes --
one verdict row per audited record, straight from the auditor's reply.

Nothing here re-runs or re-judges anything: every figure is a count over
verdict rows the pipeline already wrote, so the table cannot drift from the
runs it describes. Runs are grouped by corpus label exactly as the evaluator
groups score files: by stripping the run index and timestamp from the name.

Runs made before the verdict sidecar existed left their audit outcomes only in
the runner's stdout. If such a transcript is present at RUNNER_LOG its per-chunk
lines ("K/N kept, C corrected") are parsed as a second batch, labelled
`_earlier_batch` -- but ONLY when it describes the same corpora as the current
sidecars, since a batch run over a different corpus set under different prompts
is not a replication of this one. No such transcript is retained in the
repository at present, so this path is normally inactive.

Writes step3_results/audit_interventions.csv.

Usage:
    python3 step3_audit_interventions.py
"""

import glob
import os
import re
import sys

import pandas as pd

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LOG_DIR = os.path.join(_PROJECT_ROOT, "layers", "layer_2", "step2_results_generic")
OUT_PATH = os.path.join(_PROJECT_ROOT, "layers", "step3_results",
                        "audit_interventions.csv")

_STEM = re.compile(r"audit_log_(?P<corpus>.+)_run(?P<run>\d+)_\d{8}_\d{6}\.csv$")

RUNNER_LOG = os.path.join(_PROJECT_ROOT, "logs", "final_run.log")
_BANNER = re.compile(r"^#{4,} (?P<tag>\S+) #{4,}$")
_AUDIT_LINE = re.compile(r"^\s*\[audit\] (?P<chunk>\S+): (?P<kept>\d+)/(?P<total>\d+) "
                         r"kept, (?P<corrected>\d+) corrected")
_RUN_BANNER = re.compile(r"^=+$")


def parse_runner_log(path: str) -> list[dict]:
    """Recover per-chunk audit counts from a runner transcript. Coarser than
    the sidecar (counts per chunk, no per-record verdicts), and exactly as
    trustworthy: these lines were printed by the same audit_chunk call that
    made the decisions."""
    rows: list[dict] = []
    corpus, run = "", 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            b = _BANNER.match(line.strip())
            if b:
                if b.group("tag") != corpus:
                    corpus, run = b.group("tag"), 0
                continue
            if _RUN_BANNER.match(line.strip()) and corpus:
                continue
            if "[Cache]" in line and corpus:
                # Each repeat announces its cache state once, at run start.
                run += 1
                continue
            m = _AUDIT_LINE.match(line)
            if m and corpus:
                rows.append({"corpus": f"{corpus}_earlier_batch",
                             "run": max(run, 1),
                             "chunk_id": m.group("chunk"),
                             "kept": int(m.group("kept")),
                             "total": int(m.group("total")),
                             "corrected": int(m.group("corrected"))})
    return rows


def main() -> None:
    paths = sorted(glob.glob(os.path.join(LOG_DIR, "**", "audit_log_*.csv"),
                             recursive=True))
    if not paths:
        sys.exit(f"no audit_log files under {LOG_DIR} -- the sidecar is only "
                 f"written by runs made after it was added, so older runs "
                 f"cannot be aggregated retroactively")

    frames = []
    for path in paths:
        m = _STEM.search(os.path.basename(path))
        if not m:
            continue
        df = pd.read_csv(path)
        df["corpus"] = m.group("corpus")
        df["run"] = int(m.group("run"))
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)

    summaries = []
    for corpus, g in rows.groupby("corpus", sort=True):
        corrected = g["corrected_fields"].fillna("").astype(str).str.strip() != ""
        override = g["chunk_audit_override"].astype(str).str.lower() == "true"
        n = len(g)
        n_dropped = int((g["outcome"] == "dropped").sum())
        n_corrected = int(corrected.sum())
        summaries.append({
            "corpus": corpus,
            "runs": g["run"].nunique(),
            "chunk_audits": g.groupby("run")["chunk_id"].nunique().sum(),
            "records_audited": n,
            "kept": int((g["outcome"] == "kept").sum()),
            "dropped": n_dropped,
            "corrected": n_corrected,
            "intervention_frac": round((n_dropped + n_corrected) / n, 4),
            "override_chunks": int(g[override].groupby("run")["chunk_id"]
                                   .nunique().sum()),
        })

    # The historical transcript is parsed only when it describes the SAME
    # corpora as the current sidecars. It was produced by an earlier prompt
    # revision over a corpus set that has since changed, and a batch scored
    # under different prompts on different documents is not a replication of
    # this one -- reporting the two side by side under one heading would
    # invite exactly that reading.
    current_corpora = set(rows["corpus"].unique())
    log_rows = pd.DataFrame(parse_runner_log(RUNNER_LOG)) if os.path.exists(RUNNER_LOG) else pd.DataFrame()
    if not log_rows.empty:
        log_corpora = {c.replace("_earlier_batch", "") for c in log_rows["corpus"].unique()}
        if not log_corpora <= current_corpora:
            print(f"[skip] {os.path.basename(RUNNER_LOG)} describes corpora "
                  f"{sorted(log_corpora - current_corpora)} that are no longer "
                  f"evaluated; its batch is not comparable and is excluded.")
            log_rows = pd.DataFrame()
    if not log_rows.empty:
        for corpus, g in log_rows.groupby("corpus", sort=True):
            n = int(g["total"].sum())
            n_dropped = int((g["total"] - g["kept"]).sum())
            n_corrected = int(g["corrected"].sum())
            summaries.append({
                "corpus": corpus,
                "runs": g["run"].nunique(),
                "chunk_audits": len(g),
                "records_audited": n,
                "kept": int(g["kept"].sum()),
                "dropped": n_dropped,
                "corrected": n_corrected,
                "intervention_frac": round((n_dropped + n_corrected) / n, 4),
                "override_chunks": 0,
            })
    out = pd.DataFrame(summaries)
    # One total per batch kind -- summing a sidecar batch and a transcript
    # batch into one row would average two different experiments.
    totals = []
    for label, part in [("ALL_sidecar_runs",
                         out[~out["corpus"].str.endswith("_earlier_batch")]),
                        ("ALL_earlier_batch",
                         out[out["corpus"].str.endswith("_earlier_batch")])]:
        if part.empty:
            continue
        totals.append({
            "corpus": label,
            "runs": int(part["runs"].sum()),
            "chunk_audits": int(part["chunk_audits"].sum()),
            "records_audited": int(part["records_audited"].sum()),
            "kept": int(part["kept"].sum()),
            "dropped": int(part["dropped"].sum()),
            "corrected": int(part["corrected"].sum()),
            "intervention_frac": round(
                (part["dropped"].sum() + part["corrected"].sum())
                / part["records_audited"].sum(), 4),
            "override_chunks": int(part["override_chunks"].sum()),
        })
    out = pd.concat([out, pd.DataFrame(totals)], ignore_index=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"[audit-interventions] {len(paths)} sidecar file(s)")
    for r in out.to_dict("records"):
        print(f"  {r['corpus']}: {r['records_audited']} audited over "
              f"{r['chunk_audits']} chunk audits -> {r['dropped']} dropped, "
              f"{r['corrected']} corrected ({r['intervention_frac']:.2%})")
    print(f"[audit-interventions] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
