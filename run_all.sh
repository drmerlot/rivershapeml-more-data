#!/usr/bin/env bash
# Reproduce every number in the memo (after ./get_data.sh). Prints to stdout; tables go to data/derived/.
set -euo pipefail
cd "$(dirname "$0")/src"
echo "== Silverway (opening example)";            python silverway.py
echo; echo "== How many new training rows";       python new_rows.py
echo; echo "== Where targets sit on the flow axis"; python target_flows.py
echo; echo "== Consistency and target certainty (300 sampled training sites; downloads per-site records)"
python method_agreement.py 300 1
echo; echo "== Noise floor vs model error";       python noise_floor.py
echo; echo "== Depth gap: instrument vs location"; python depth_gap.py
echo; echo "== Retrain experiment (bankfull width)"
python site_comids.py && python build_targets.py && python build_features.py && python retrain_experiment.py
