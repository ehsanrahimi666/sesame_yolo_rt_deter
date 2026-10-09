# Automated detection and counting of sesame flower clusters in UAV imagery

Data, code, trained models, and results for:

> Rahimi, E., and C. Jung. Automated detection and counting of sesame (*Sesamum indicum*) flower clusters in unmanned
> aerial vehicle imagery: Comparing a lightweight YOLO26 model with a larger RT-DETR transformer. *Applications in Plant
> Sciences* (in revision).

The repository allows the complete analysis to be reproduced: preparation of the tiled data set and spatial folds,
five-fold spatially blocked cross-validation of YOLO26n and RT-DETR-l, the evaluation used in the manuscript
(identical inference protocol for all detectors, de-duplicated cluster-level metrics, cross-fitted confidence
thresholds, block-level counting accuracy, paired tests, and spatial bootstrap), the CPU benchmark, and the
wall-to-wall mapping of the field.

---

## Contents

```
sesame_yolo_rt_deter/
├── README.md                     this file
├── requirements.txt              Python environment with pinned package versions
├── CITATION.cff                  how to cite this repository
├── scripts/                      pipeline that produced the data set and the models
│   ├── 01_prepare_sesame_dataset.py       orthomosaic + annotations -> 512-px tiles, YOLO labels, spatial folds
│   ├── 02_train_yolo_colab.py             five-fold training + final model (Kaggle/Colab GPU); YOLO26n or RT-DETR-l
│   ├── 03_predict_wall_to_wall.py         wall-to-wall script of the first submission (YOLO26n)
│   └── 04_predict_wall_to_wall_rtdetr.py  wall-to-wall script of the first submission (RT-DETR-l)
├── gis/                          reference annotations and annotated area (EPSG:32652)
│   ├── sesame.shp                1201 flower clusters, outlined as (rotated) rectangles
│   └── Field_border.shp          annotated area (107.8 m2)
├── flight/                       DJI timestamp (MRK) file of the flight (231 images, 5 July 2026)
├── dataset/sesame_dataset_v1/    tiled data set used for training and evaluation
│   ├── images/                   216 RGB tiles, 512 x 512 px (2 mm per pixel)
│   ├── labels/                   YOLO labels (class xc yc w h, normalized)
│   ├── tiles.csv                 tile id, spatial fold, map origin of every tile
│   ├── folds.csv                 tiles and box instances per fold
│   ├── boxes.csv                 every box instance in tile pixels and map coordinates
│   ├── dataset_summary.txt       tile accounting and data set statistics
│   └── prep_config.json          settings of 01_prepare_sesame_dataset.py
├── yolo26_results/               YOLO26n training runs: runs/<run>/args.yaml (full configuration) and
│                                 results.csv (per-epoch log) for folds 0-4 and the final model; folds/ (example split)
├── Rt_DETER/                     RT-DETR-l training runs, same structure
├── revision_analysis/            evaluation workflow of the revised manuscript (CPU)
│   ├── lib/paths.py              input/output locations (relative to the repository; override with environment variables)
│   ├── lib/evalcore.py           matching, AP, cluster-level de-duplication, block counts
│   ├── run_revision_analysis.sh  runs the complete workflow in order
│   └── *.py                      individual steps (see below)
├── results/                      outputs used in the manuscript (see "Results files")
├── figures/                      Figures 1-5 and Figure S1 of the manuscript
└── weights/                      download location for the trained weights (see weights/README.md)
```

The trained weights of all models (best and last checkpoints of every fold model, and the final models) are provided
as assets of the GitHub release of this repository (see `weights/README.md`). The orthomosaic (25,000 x 20,999 pixels,
2 mm; about 0.8 GB) and the raw UAV images are not included because of their size; they are available from the
corresponding author on request. They are needed only for the wall-to-wall mapping, Figure 1, and Figure S1; all
cross-validation results can be reproduced from the tiles in `dataset/sesame_dataset_v1/`.

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The analysis was run with Python 3.13 on a CPU; training used one NVIDIA GPU in a Kaggle notebook. Both used
**Ultralytics 8.4.105**. The version matters: in 8.4.105, YOLO26 models use their one-to-one (NMS-free) head by
default, whereas more recent releases (e.g., 8.4.174) default to the one-to-many head followed by NMS. All analysis
scripts therefore select the head explicitly (`end2end=True` or `end2end=False`). For a CPU-only installation of
PyTorch, install `torch` and `torchvision` from `https://download.pytorch.org/whl/cpu` before the other requirements.

---

## 1. Data set preparation

`scripts/01_prepare_sesame_dataset.py` reads the orthomosaic (`odm_orthophoto.tif`, bands R, G, B, alpha),
`gis/Field_border.shp`, and `gis/sesame.shp`, and writes `dataset/sesame_dataset_v1/`:

* each annotated rectangle is converted to its axis-aligned bounding box;
* the annotated area is divided into five contiguous strips (4.24 m wide) along its east-west axis, which serve as the
  spatial folds;
* the orthomosaic is cut into 512 x 512-pixel tiles (1.024 m) on a 256-pixel grid (50% overlap); a tile is kept if at
  least 90% of its area lies inside the annotated area, all pixels are valid, and it lies entirely within one strip
  (tiles straddling a strip boundary are discarded, leaving a gap of at least 0.51 m between folds);
* a box clipped by a tile boundary is kept if at least half of its area remains inside the tile and it is at least 4
  pixels wide and high.

Result: 216 tiles (fold 0: 4, fold 1: 48, fold 2: 52, fold 3: 68, fold 4: 44), 903 annotated clusters in tiles, 2566 box
instances. Every candidate tile contained at least one cluster, so the data set has no flower-free tiles.

## 2. Training

`scripts/02_train_yolo_colab.py` runs on Kaggle or Google Colab with the zipped data set
(`sesame_dataset_v1.zip`). For each fold it trains on the other four folds and validates on the held-out fold; it then
trains a final model on all 216 tiles. The complete configuration of every run is stored in
`yolo26_results/runs/<run>/args.yaml` and `Rt_DETER/runs/<run>/args.yaml`; per-epoch logs are in `results.csv`.

| Setting | YOLO26n | RT-DETR-l |
|---|---|---|
| `CFG["MODELS"]` | `["yolo26n"]` | `["rtdetr-l"]` |
| `CFG["BATCH"]` / nominal batch | 16 / 64 | 8 / 64 |
| `CFG["PATIENCE"]` (early stopping) | 50 | 20 |
| Epochs (maximum) | 150 | 150 |
| Pretrained weights | `yolo26n.pt` (COCO) | `rtdetr-l.pt` (COCO) |
| Optimizer (automatic selection) | AdamW, lr 0.002, momentum 0.9, weight decay 5e-4 | same |
| Augmentation | mosaic (off in the last 10 epochs), rotation ±45°, scale ±30%, translation ±10%, horizontal and vertical flips (0.5), HSV 0.015/0.7/0.4 | same |
| Input size; seed | 640 px (tiles enlarged ×1.25); 0, deterministic | same |

The script selects the model class from the weights name (RT-DETR weights are loaded as an RT-DETR model
automatically). Install the pinned version before running it, because the script otherwise installs the newest
Ultralytics release:

```bash
pip install ultralytics==8.4.105
python scripts/02_train_yolo_colab.py      # edit CFG["MODELS"], CFG["BATCH"], CFG["PATIENCE"] as in the table
```

Training took 4.4-8.0 min per fold for YOLO26n and 10.3-22.0 min per fold for RT-DETR-l.

## 3. Evaluation workflow of the revised manuscript

All steps are in `revision_analysis/` and are run from that folder; outputs are written to `revision_analysis/out/`.
Download the weights first (see `weights/README.md`), so that they are found under `weights/best/` and
`weights/last/`. The complete workflow is

```bash
cd revision_analysis
bash run_revision_analysis.sh                 # steps that need only the tiles and the weights
WITH_ORTHO=1 bash run_revision_analysis.sh    # also wall-to-wall mapping, Figure 1 and Figure S1 (needs the orthomosaic)
```

or step by step:

1. **Data set, annotations, flight geometry**
   ```bash
   python flight_geometry.py      # flight lines and image spacing (MRK file)
   python dataset_audit.py        # fold composition and train-test separation (Table 2), reference box sizes
   python annotation_shape.py     # rotation and size of the drawn rectangles vs. their axis-aligned boxes
   python valid_area.py           # valid area of the orthomosaic and 1-m grid (needs the orthomosaic)
   ```
2. **Out-of-fold predictions** with an explicit inference protocol: each 512-px tile is resized to 640 px and surrounded
   by a grey (114) border of PAD pixels; PAD = 16 reproduces the padding of the Ultralytics validator for YOLO models.
   ```bash
   # predict_tiles.py <model> <fold> <outdir> <mode: e2e|nms|none> <weights_dir> <pad>
   for k in 0 1 2 3 4; do
     python predict_tiles.py yolo26n  $k out/pad16_last e2e  ../weights/last 16    # primary protocol
     python predict_tiles.py yolo26n  $k out/pad16_last nms  ../weights/last 16
     python predict_tiles.py rtdetr-l $k out/pad16_last none ../weights/last 16
   done
   ```
   Sensitivity settings: `out/pad16_best`, `out/nopad_best`, `out/nopad_last` (same commands with `../weights/best`
   and/or PAD 0); `python make_native_sets.py` builds `out/native_best` (each model's own validator preprocessing).
3. **Evaluation and statistics** (for each output folder)
   ```bash
   python evaluate_cv.py out/pad16_last     # tile- and cluster-level metrics, cross-fitted thresholds, block counts
   python stats_compare.py out/pad16_last   # paired t, exact Wilcoxon, corrected resampled t; spatial bootstrap
   ```
4. **Further analyses and the summary file** from which all numbers in the manuscript are taken
   ```bash
   python selection_bias.py      # best vs. last epochs from the training logs
   python box_size_bias.py       # ratio of predicted to reference box size, with and without the border
   python benchmark_speed.py     # CPU inference time and GFLOPs
   python consolidate.py         # -> out/summary.json
   ```
5. **Wall-to-wall mapping** with the final models (needs the orthomosaic)
   ```bash
   python wall_to_wall.py yolo26n  nms  ../weights/best/weights_yolo26n_final.pt  out/w2w
   python wall_to_wall.py yolo26n  e2e  ../weights/best/weights_yolo26n_final.pt  out/w2w
   python wall_to_wall.py rtdetr-l none ../weights/best/weights_rtdetr-l_final.pt out/w2w
   python w2w_summary.py         # totals at cross-validated, count-calibrated and common thresholds; 1-m grids
   ```
6. **Figures**: `make_previews.py` + `fig1_study.py` (Figure 1), `figs_results.py` (Figures 2-4), `fig5_w2w.py`
   (Figure 5), `fp_contact_sheet.py` (Figure S1).

### Reproducing the tables and figures

| Manuscript item | Produced by | Main output |
|---|---|---|
| Table 1 (acquisition and processing) | `flight_geometry.py`, `valid_area.py`; image metadata and photogrammetry report | `out/flight_geometry.json`, `out/valid_area.json` |
| Table 2 (fold composition) | `dataset_audit.py`, `evaluate_cv.py` | `out/fold_composition.csv`, `out/fold_separation.csv`, `out/pad16_last/eval_blocks.csv` |
| Table 3 (configurations, cost) | `yolo26_results/runs/*/args.yaml`, `Rt_DETER/runs/*/args.yaml`, `benchmark_speed.py` | `out/benchmark_cpu.json` |
| Table 4 (cross-validated accuracy) | `evaluate_cv.py`, `stats_compare.py`, `consolidate.py` | `out/pad16_last/*.csv`, `out/summary.json` |
| Table 5 (wall-to-wall) | `wall_to_wall.py`, `w2w_summary.py` | `out/w2w_summary.csv` |
| Figure 1 | `make_previews.py`, `fig1_study.py` | `figures/Figure1.png` |
| Figures 2-4 | `figs_results.py` | `figures/Figure2.png` - `Figure4.png` |
| Figure 5 | `w2w_summary.py`, `fig5_w2w.py` | `figures/Figure5.png` |
| Appendix S1, Tables S1-S5 | `evaluate_cv.py`, `stats_compare.py`, `box_size_bias.py`, `selection_bias.py`, `consolidate.py` | `out/*/eval_fold_metrics.csv`, `stats_pairs.csv`, `stats_bootstrap.csv`, `eval_thresholds.csv`, `out/final/selection_bias.csv` |
| Appendix S1, Figure S1 | `fp_contact_sheet.py` | `figures/FigureS1_consensus_false_positives.png` |

Neither the weights nor the orthomosaic are needed to reproduce the reported accuracy. Copy the folders
`results/pad16_last`, `pad16_best`, `nopad_last`, `nopad_best`, and `native_best` (out-of-fold predictions) and the
file `results/benchmark_cpu.json` to `revision_analysis/out/`, then run steps 1, 3, and 4 except `valid_area.py` and
`benchmark_speed.py`:

```bash
cd revision_analysis && mkdir -p out
cp -r ../results/pad16_last ../results/pad16_best ../results/nopad_last ../results/nopad_best ../results/native_best out/
cp ../results/benchmark_cpu.json out/
python flight_geometry.py && python dataset_audit.py && python annotation_shape.py
for d in pad16_last pad16_best nopad_last nopad_best native_best; do python evaluate_cv.py out/$d; python stats_compare.py out/$d; done
python selection_bias.py && python box_size_bias.py && python consolidate.py
```

This reproduces every value in `results/summary.json` and every evaluation file in `results/`. CPU times measured with
`benchmark_speed.py` depend on the hardware.

---

## Key definitions

* **Primary protocol:** tile resized to 640 px + 16-px grey border (672-px network input), last checkpoints, predictions
  retained down to a confidence of 0.001 for evaluation.
* **Inference modes of YOLO26n:** `e2e` = one-to-one head (NMS-free); `nms` = one-to-many head followed by NMS (IoU 0.7).
  RT-DETR-l is NMS-free by design (`none`).
* **Tile level:** AP@0.5 and AP@0.5:0.95 on the held-out tiles (101-point interpolation, as in Ultralytics); clusters in
  overlapping tiles are counted several times.
* **Cluster level (de-duplicated):** a prediction is kept only if its center lies in the central 256 x 256-px core of the
  tile that produced it (cores of neighboring tiles abut without overlap); remaining duplicates are removed by NMS
  (IoU 0.5); predictions are matched greedily, in order of decreasing confidence, to the unique reference boxes centered
  in the cores at IoU >= 0.5.
* **Cross-fitted threshold:** for each held-out fold, the confidence threshold that maximizes pooled cluster-level F1 on
  the out-of-fold predictions of the other four folds.
* **Blocks:** complete 2 x 2 groups of cores (1.024 x 1.024 m); every prediction and reference box is assigned to the
  core containing its center and thereby to that core's block.
* **Paired tests:** paired t-test, exact Wilcoxon signed-rank test, and corrected resampled t-test (Nadeau and Bengio,
  2003) over the five folds; **spatial bootstrap:** 2000 resamples of the 0.512-m cores within folds (seed 2026).

## Results files

| Path | Content |
|---|---|
| `results/pad16_last/` | primary protocol: out-of-fold predictions `preds_<config>_fold<k>.csv` (tile pixels, confidence >= 0.001), `eval_fold_metrics.csv`, `eval_thresholds.csv`, `eval_sweep.csv`, `eval_blocks.csv`, `stats_pairs.csv`, `stats_pooled.csv`, `stats_bootstrap.csv` |
| `results/pad16_best/`, `nopad_best/`, `nopad_last/`, `native_best/` | the same for the sensitivity settings |
| `results/w2w/` | wall-to-wall detections (`w2w_<config>.csv`, map coordinates in EPSG:32652, confidence >= 0.01) and run metadata |
| `results/w2w_summary.csv`, `w2w_agreement.json` | wall-to-wall totals, densities, box sizes, and agreement of 1-m cell densities |
| `results/summary.json` | all numbers reported in the manuscript |
| `results/*.csv`, `*.json` | fold composition and separation, annotation shape, box-size bias, CPU benchmark, training-log summary, valid area, flight geometry |

The files `cv_results.csv`, `cv_summary.csv`, `final_model_info.json`, and `fig_*.png` in `yolo26_results/` and
`Rt_DETER/` are outputs of the training script used for the first submission; they were computed with the
prediction function of the package (no border) and checkpoints selected on the held-out fold, and are superseded by
the evaluation in `revision_analysis/`.

## Citation and license

Please cite the article above when using the data, code, or models (see also `CITATION.cff`). The trained weights are
derived from Ultralytics models, which are distributed under the AGPL-3.0 license
(https://github.com/ultralytics/ultralytics).

Contact: Ehsan Rahimi (ehsanrahimi666@gmail.com)
