#!/usr/bin/env python
"""
Run one fold model on its held-out tiles with an explicit, model-independent inference protocol.

Protocol: each 512-px tile is resized to 640 px (bilinear) and surrounded by a constant grey (114)
border of PAD px; the network input is therefore (640 + 2*PAD) px. PAD = 16 reproduces the letterbox
padding of the Ultralytics validator (rect batches, pad = 0.5 stride); PAD = 0 is plain 640-px input.
YOLO26 head selection: mode 'e2e' = one-to-one head (NMS-free); 'nms' = one-to-many head + NMS (IoU 0.7).

Usage: python predict_tiles.py <model> <fold> <outdir> <mode: e2e|nms|none> <weights_dir> <pad> [chunk]
Output: <outdir>/preds_<tag>_fold<k>.csv with tile, fold, x0, y0, x1, y1 (tile px), score (conf >= 0.001)
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(DATA_DIR)

def worker(name, wpath, mode, pad, out_csv, stems):
    import cv2
    from ultralytics import YOLO, RTDETR
    M = RTDETR(wpath) if name.startswith("rtdetr") else YOLO(wpath)
    extra = {} if mode == "none" else {"end2end": mode == "e2e"}
    rows = []
    for stem in stems:
        im = cv2.imread(str(DATA / "images" / f"{stem}.png"))
        im = cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR)
        if pad: im = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        r = M.predict(source=im, imgsz=640 + 2 * pad, conf=0.001, device="cpu", verbose=False, max_det=300, **extra)[0]
        b = (r.boxes.xyxy.cpu().numpy() - pad) * 512 / 640; s = r.boxes.conf.cpu().numpy()
        rows += [(stem, *map(float, bb), float(ss)) for bb, ss in zip(b, s)]
    pd.DataFrame(rows, columns=["tile", "x0", "y0", "x1", "y1", "score"]).to_csv(out_csv, index=False)

if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        _, _, name, wpath, mode, pad, out_csv, *stems = sys.argv
        worker(name, wpath, mode, int(pad), out_csv, stems); sys.exit(0)
    name, k, outdir, mode, wdir, pad = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]).resolve(), sys.argv[4], Path(sys.argv[5]), sys.argv[6]
    cs = int(sys.argv[7]) if len(sys.argv) > 7 else 12
    outdir.mkdir(parents=True, exist_ok=True)
    tiles = pd.read_csv(DATA / "tiles.csv"); stems = [f[:-4] for f in tiles[tiles.fold == k].file_name]
    tag = name if mode == "none" else f"{name}-{mode}"
    parts = []
    for i in range(0, len(stems), cs):
        part = outdir / f"_part_{tag}_{k}_{i}.csv"
        subprocess.run([sys.executable, __file__, "--worker", name, str(wdir / f"weights_{name}_fold{k}.pt"), mode, pad, str(part), *stems[i:i + cs]], check=True)
        parts.append(pd.read_csv(part)); part.unlink()
    df = pd.concat(parts, ignore_index=True); df.insert(1, "fold", k)
    df.to_csv(outdir / f"preds_{tag}_fold{k}.csv", index=False)
    print(f"{tag} fold {k} pad {pad}: {len(stems)} tiles, {len(df)} predictions", flush=True)
