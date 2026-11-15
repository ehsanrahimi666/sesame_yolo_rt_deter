#!/usr/bin/env bash
# Additional analyses of the second revision. Run from revision_analysis/ after run_revision_analysis.sh:
#   bash run_revision2_analysis.sh             # steps that need neither the orthomosaic nor the weights
#   WITH_ORTHO=1 bash run_revision2_analysis.sh   # also the steps that need the orthomosaic and the weights
# Without the orthomosaic, the raw detections in ../results/r2/ are used (see README, section 4).
set -euo pipefail
PY=${PYTHON:-python}
mkdir -p out/r2/mapval out/r2/w2w

if [ "${WITH_ORTHO:-0}" = "1" ]; then
  # validation of the complete mapping pipeline: fold-k model on strip k of the annotated area
  for k in 0 1 2 3 4; do
    $PY mapping_validation_predict.py yolo26n  nms  $k
    $PY mapping_validation_predict.py yolo26n  e2e  $k
    $PY mapping_validation_predict.py rtdetr-l none $k
  done
  # wall-to-wall mapping of the final models (raw detections before NMS)
  $PY wall_to_wall_raw.py yolo26n  nms
  $PY wall_to_wall_raw.py yolo26n  e2e
  $PY wall_to_wall_raw.py rtdetr-l none
else
  cp ../results/r2/mapval/raw_*.csv out/r2/mapval/
  cp ../results/r2/w2w/raw_*.csv ../results/r2/w2w/valid_frac_1m.npy out/r2/w2w/
fi
$PY mapping_validation_eval.py --stitch
$PY mapping_validation_eval.py
$PY wall_to_wall_stitch.py
$PY fig5_w2w.py
if [ "${WITH_ORTHO:-0}" = "1" ]; then
  $PY nonvegetated_detections.py        # needs the previews of make_previews.py
  $PY visual_check_sample.py
fi
$PY block_bootstrap.py
$PY fold0_sensitivity.py
$PY figS1_loss_curves.py
echo "done"
