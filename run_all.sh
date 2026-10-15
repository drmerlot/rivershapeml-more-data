#!/usr/bin/env bash
# Reproduce the full analysis (after ./get_data.sh and ./setup_r.sh). Every table this writes is described in
# docs/data_model.yaml; `python src/data_model.py` checks the two against each other.
set -euo pipefail
cd "$(dirname "$0")/src"
echo "== B. Site selection";              python new_rows.py && python site_comids.py
echo; echo "== C. Targets (AHGestimation)"; python build_targets.py
echo; echo "== D. Data checks (300 sampled training sites; downloads per-site records)"
python method_agreement.py 300 1 && python noise_floor.py && python depth_gap.py
python silverway.py && python target_flows.py
echo; echo "== E. Model experiments (current model's features and settings)"
python retrain_riverml.py --target TW_bf && python retrain_riverml.py --target Y_bf && python train_eval.py
echo; echo "== F. Product";                    python predict_baseline.py && python predict_product.py && python build_product_compare.py
echo; echo "== Data model check";              python data_model.py
