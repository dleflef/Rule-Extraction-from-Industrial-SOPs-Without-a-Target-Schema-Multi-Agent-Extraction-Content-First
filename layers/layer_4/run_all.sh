#!/usr/bin/env bash
# Canonical layer_4 execution order — regenerates every CSV under
# step4_results/ and detection_results/ from scratch, deterministically.
#
# Prerequisites:
#   - Neo4j running (used by step4_populate, step4b_load_abox,
#     step5_detect, step5_sensitivity; the other scripts are standalone)
#   - the Layer-2 extraction CSV wired into step4_populate.py
set -euo pipefail
cd "$(dirname "$0")"

python3 step4_populate.py            # extracted rules -> Neo4j
python3 step4b_load_abox.py          # ABox load + GOVERNS_ABOX validation
python3 step5_detect.py              # Phase 2 scoring (guide Table 2)
python3 step5_holdout.py             # held-out injection (one-shot seed)
python3 step5_holdout.py --multiseed # 10 pre-committed seeds, pooled
python3 step5_sensitivity.py         # parameter grid + strictness + calibration
                                     # (after holdout: Part 3 checks the fresh
                                     #  injected stream, present even on a
                                     #  clean clone where it is gitignored)
python3 step5_baselines.py           # oracle / z-score bracketing
python3 step5_leakage_audit.py       # GT-leakage runtime audit

echo
echo "layer_4 suite complete — results in step4_results/ and detection_results/"
