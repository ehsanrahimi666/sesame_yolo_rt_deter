#!/usr/bin/env python
"""
Wall-to-wall inference over the full orthomosaic for one model configuration (memory-safe, chunked).

Tiling: 512-px tiles with 64-px overlap (stride 448 px), identical for all configurations.
Each tile is resized to 640 px and given a 16-px grey border (network input 672 px), as in the cross-validation.
Every tile containing at least one valid (alpha > 0) pixel is processed; invalid pixels are left black.
Detections are kept at conf >= 0.01 so that any operating threshold can be applied afterwards;
a single global NMS (IoU 0.5) removes duplicates across tile seams. Because greedy NMS only lets a box
be suppressed by a higher-scoring box, filtering the NMS output at threshold t is identical to running
the whole pipeline at threshold t.

Usage: python wall_to_wall.py <model: yolo26n|rtdetr-l> <mode: e2e|nms|none> <weights> <outdir> [tiles_per_chunk]
Output: <outdir>/w2w_<tag>.csv  (x, y [m, EPSG:32652], x0, y0, x1, y1, score, width_cm, height_cm)
        <outdir>/w2w_<tag>_meta.json
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import sys, json, time, subprocess
from pathlib import Path
import numpy as np, pandas as pd

ORTHO = ORTHO_TIF
T, OVER = 512, 64
PAD = 16   # same inference protocol as the cross-validation (640-px content + 16-px grey border)

def grid(W, H):
    S = T - OVER
    cols = list(range(0, max(1, W - T + 1), S)); rows = list(range(0, max(1, H - T + 1), S))
    if cols[-1] + T < W: cols.append(W - T)
    if rows[-1] + T < H: rows.append(H - T)
    return [(r, c) for r in rows for c in cols]

def worker(name, mode, weights, out_csv, idx0, idx1):
    import rasterio, cv2
    from rasterio.windows import Window
    from ultralytics import YOLO, RTDETR
    M = RTDETR(weights) if name.startswith("rtdetr") else YOLO(weights)
    extra = {} if mode == "none" else {"end2end": mode == "e2e"}
    rows = []; n_done = 0
    with rasterio.open(ORTHO) as ds:
        tiles = grid(ds.width, ds.height)[idx0:idx1]
        batch, offs = [], []
        def flush():
            if not batch: return
            for (rr, cc), o in zip(offs, M.predict(source=batch, imgsz=640 + 2 * PAD, conf=0.01, device="cpu", verbose=False, max_det=300, **extra)):
                b = (o.boxes.xyxy.cpu().numpy() - PAD) * T / 640; s = o.boxes.conf.cpu().numpy()
                for bb, ss in zip(b, s):
                    rows.append((rr, cc, bb[0] + cc, bb[1] + rr, bb[2] + cc, bb[3] + rr, float(ss)))
            batch.clear(); offs.clear()
        for rr, cc in tiles:
            a = ds.read(4, window=Window(cc, rr, T, T))
            if (a > 0).sum() == 0: continue
            rgb = ds.read([1, 2, 3], window=Window(cc, rr, T, T))
            rgb[:, a == 0] = 0
            im = np.ascontiguousarray(np.transpose(rgb, (1, 2, 0))[..., ::-1])            # RGB -> BGR
            im = cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR)
            if PAD: im = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            batch.append(im); offs.append((rr, cc)); n_done += 1
            if len(batch) >= 4: flush()
        flush()
    pd.DataFrame(rows, columns=["row", "col", "px0", "py0", "px1", "py1", "score"]).to_csv(out_csv, index=False)
    print(json.dumps({"chunk": [idx0, idx1], "tiles_processed": n_done, "dets": len(rows)}), flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        _, _, name, mode, weights, out_csv, a, b = sys.argv
        worker(name, mode, weights, out_csv, int(a), int(b)); sys.exit(0)
    import rasterio, torch
    from torchvision.ops import nms
    name, mode, weights, outdir = sys.argv[1:5]; per = int(sys.argv[5]) if len(sys.argv) > 5 else 150
    outdir = Path(outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    tag = name if mode == "none" else f"{name}-{mode}"
    with rasterio.open(ORTHO) as ds:
        W, H, tr = ds.width, ds.height, ds.transform
    n = len(grid(W, H)); t0 = time.time(); parts = []; logs = []
    for i in range(0, n, per):
        part = outdir / f"_w2w_{tag}_{i}.csv"
        if not part.exists():
            r = subprocess.run([sys.executable, __file__, "--worker", name, mode, weights, str(part), str(i), str(min(i + per, n))], capture_output=True, text=True)
            logs += [l for l in r.stdout.splitlines() if l.startswith("{")]
            if r.returncode != 0: print(r.stderr[-2000:]); sys.exit(1)
        parts.append(pd.read_csv(part))
        print(f"{tag}: {min(i + per, n)}/{n} grid positions, {time.time() - t0:.0f} s", flush=True)
    D = pd.concat(parts, ignore_index=True)
    elapsed = time.time() - t0
    B = torch.tensor(D[["px0", "py0", "px1", "py1"]].values, dtype=torch.float64)
    S = torch.tensor(D.score.values, dtype=torch.float64)
    keep = nms(B, S, 0.5).numpy(); K = D.iloc[keep].copy()
    cx = (K.px0 + K.px1) / 2; cy = (K.py0 + K.py1) / 2
    K["x"] = tr.c + cx * tr.a; K["y"] = tr.f + cy * tr.e
    K["width_cm"] = (K.px1 - K.px0) * abs(tr.a) * 100; K["height_cm"] = (K.py1 - K.py0) * abs(tr.e) * 100
    K.to_csv(outdir / f"w2w_{tag}.csv", index=False)
    tiles_done = sum(json.loads(l)["tiles_processed"] for l in logs) if logs else None
    meta = dict(tag=tag, weights=weights, pad=PAD, grid_positions=n, tiles_processed=tiles_done, raw_dets=len(D), kept_after_nms=len(K),
                elapsed_s=elapsed, conf_floor=0.01, tile=T, overlap=OVER, nms_iou=0.5)
    (outdir / f"w2w_{tag}_meta.json").write_text(json.dumps(meta, indent=1)); print(json.dumps(meta))
    for p in outdir.glob(f"_w2w_{tag}_*.csv"): p.unlink()
