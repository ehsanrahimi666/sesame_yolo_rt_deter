"""Combine predictions into the 'validator default' setting: YOLO26n with the 16-px border (as in its Ultralytics
validator) and RT-DETR-l without a border (as in its validator), for best (and optionally last) checkpoints."""
import shutil
from pathlib import Path

for ckpt in ("best", "last"):
    yolo_dir, rt_dir, out = Path(f"out/pad16_{ckpt}"), Path(f"out/nopad_{ckpt}"), Path(f"out/native_{ckpt}")
    if not (yolo_dir.exists() and rt_dir.exists()):
        continue
    out.mkdir(parents=True, exist_ok=True)
    for f in yolo_dir.glob("preds_yolo26n-*_fold*.csv"):
        shutil.copy(f, out / f.name)
    for f in rt_dir.glob("preds_rtdetr-l_fold*.csv"):
        shutil.copy(f, out / f.name)
    print(f"{out}: {len(list(out.glob('preds_*.csv')))} prediction files")
