#!/usr/bin/env bash
# Complete evaluation workflow of the revised manuscript. Run from revision_analysis/:
#   bash run_revision_analysis.sh            # everything that needs only the tiles and the weights
#   WITH_ORTHO=1 bash run_revision_analysis.sh   # also the steps that need the orthomosaic (wall-to-wall, Figure 1, Figure S1)
# Weights: ../weights/best/weights_<model>_fold<k>.pt (+ _final.pt) and ../weights/last/weights_<model>_fold<k>.pt
set -euo pipefail
PY=${PYTHON:-python}
BEST=${SESAME_WEIGHTS_BEST:-../weights/best}
LAST=${SESAME_WEIGHTS_LAST:-../weights/last}

# 1. data set, annotations and flight geometry
$PY flight_geometry.py
$PY dataset_audit.py
$PY annotation_shape.py

# 2. out-of-fold predictions (tile resized to 640 px; PAD = grey border in px)
for k in 0 1 2 3 4; do
  for setting in "pad16_last $LAST 16" "pad16_best $BEST 16" "nopad_last $LAST 0" "nopad_best $BEST 0"; do
    set -- $setting
    $PY predict_tiles.py yolo26n  $k out/$1 e2e  $2 $3
    $PY predict_tiles.py yolo26n  $k out/$1 nms  $2 $3
    $PY predict_tiles.py rtdetr-l $k out/$1 none $2 $3
  done
done
$PY make_native_sets.py

# 3. evaluation, cross-fitted thresholds, block counts, paired tests and bootstrap
for d in pad16_last pad16_best nopad_last nopad_best native_best native_last; do
  $PY evaluate_cv.py out/$d
done
for d in pad16_last pad16_best nopad_last nopad_best native_best; do
  $PY stats_compare.py out/$d
done

# 4. training logs, box-size bias, CPU benchmark, summary used for all numbers in the manuscript
$PY selection_bias.py
$PY box_size_bias.py
$PY benchmark_speed.py
if [ "${WITH_ORTHO:-0}" = "1" ]; then $PY valid_area.py; fi
$PY consolidate.py

# 5. wall-to-wall mapping with the final models (needs the orthomosaic)
if [ "${WITH_ORTHO:-0}" = "1" ]; then
  $PY wall_to_wall.py yolo26n  nms  $BEST/weights_yolo26n_final.pt  out/w2w
  $PY wall_to_wall.py yolo26n  e2e  $BEST/weights_yolo26n_final.pt  out/w2w
  $PY wall_to_wall.py rtdetr-l none $BEST/weights_rtdetr-l_final.pt out/w2w
  $PY w2w_summary.py
  $PY make_previews.py
  $PY fig1_study.py
  $PY fig5_w2w.py
  $PY fp_contact_sheet.py
fi

# 6. figures that need only the evaluation outputs
$PY figs_results.py
echo "done"
