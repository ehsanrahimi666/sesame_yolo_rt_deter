"""Wall-to-wall mapping, step 1: raw detections of a final model on every grid tile of the orthomosaic.

Identical tiling and inference protocol to wall_to_wall.py (512-px tiles, stride 448 px, no-data pixels black, 640 px +
16-px grey border, confidence >= 0.01, at most 300 detections per tile), but the raw per-tile detections are written
before any NMS, so that seam-aware stitching (wall_to_wall_stitch.py) or global NMS alone can be applied afterwards.

Usage: python wall_to_wall_raw.py <yolo26n|rtdetr-l> <e2e|nms|none> [outdir]
Needs the orthomosaic (SESAME_ORTHO) and the final weights (weights/best/weights_<model>_final.pt).
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa
import json
import sys
import time
import pandas as pd
import rasterio
from stitch import grid
from tiles_io import read_tile, predict


def main():
    name, mode = sys.argv[1], sys.argv[2]
    outdir = sys.argv[3] if len(sys.argv) > 3 else "out/r2/w2w"
    os.makedirs(outdir, exist_ok=True)
    from ultralytics import YOLO, RTDETR
    w = str(WEIGHTS_BEST / f"weights_{name}_final.pt")
    M = RTDETR(w) if name.startswith("rtdetr") else YOLO(w)
    extra = {} if mode == "none" else {"end2end": mode == "e2e"}
    rows, batch, offs, n_done, t0 = [], [], [], 0, time.time()
    G = grid()
    with rasterio.open(ORTHO_TIF) as ds:
        for i, (rr, cc) in enumerate(G):
            im = read_tile(ds, rr, cc)
            if im is None:
                continue
            batch.append(im); offs.append((rr, cc)); n_done += 1
            if len(batch) >= 4:
                rows += predict(M, batch, offs, extra); batch, offs = [], []
            if i % 200 == 0:
                print(json.dumps({"grid_pos": i, "of": len(G), "tiles_done": n_done, "elapsed_s": round(time.time() - t0)}), flush=True)
        if batch:
            rows += predict(M, batch, offs, extra)
    pd.DataFrame(rows, columns=["row", "col", "px0", "py0", "px1", "py1", "score"]).to_csv(f"{outdir}/raw_{name}-{mode}.csv", index=False)
    print(json.dumps({"tag": f"{name}-{mode}", "grid_positions": len(G), "tiles_processed": n_done, "raw_dets": len(rows),
                      "elapsed_s": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
