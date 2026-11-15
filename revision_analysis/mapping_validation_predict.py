"""Validation of the mapping pipeline, step 1: out-of-fold run of the wall-to-wall pipeline over the annotated area.

For fold k, the fold-k model (last checkpoint; trained without strip k) is applied with the tiling and inference
protocol of the wall-to-wall mapping (512-px tiles on the full-orthomosaic grid, stride 448 px; no-data pixels black;
resized to 640 px with a 16-px grey border; confidence >= 0.01; at most 300 detections per tile) to every grid tile that
intersects strip k of the annotated area. Raw detections (before stitching and NMS) are written in full-orthomosaic
pixel coordinates.

Usage: python mapping_validation_predict.py <yolo26n|rtdetr-l> <e2e|nms|none> <fold> [outdir]
Needs the orthomosaic (SESAME_ORTHO) and the last-checkpoint fold weights (weights/last/).
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa
import json
import sys
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box
from stitch import T, X0, Y0, RX, RY, grid
from tiles_io import read_tile, predict


def edges():
    f = "out/dataset_audit.json" if os.path.exists("out/dataset_audit.json") else str(ROOT / "results" / "dataset_audit.json")
    return json.load(open(f))["edges"]


def main():
    name, mode, fold = sys.argv[1], sys.argv[2], int(sys.argv[3])
    outdir = sys.argv[4] if len(sys.argv) > 4 else "out/r2/mapval"
    os.makedirs(outdir, exist_ok=True)
    from ultralytics import YOLO, RTDETR
    w = str(WEIGHTS_LAST / f"weights_{name}_fold{fold}.pt")
    M = RTDETR(w) if name.startswith("rtdetr") else YOLO(w)
    extra = {} if mode == "none" else {"end2end": mode == "e2e"}
    field = gpd.read_file(GIS_DIR + "Field_border.shp").geometry.union_all()
    e = edges()
    strip = box(e[fold], field.bounds[1] - 1, e[fold + 1], field.bounds[3] + 1).intersection(field)
    tiles = [(r, c) for r, c in grid()
             if box(X0 + c * RX, Y0 - (r + T) * RY, X0 + (c + T) * RX, Y0 - r * RY).intersects(strip)]
    rows, batch, offs, n_done = [], [], [], 0
    with rasterio.open(ORTHO_TIF) as ds:
        for rr, cc in tiles:
            im = read_tile(ds, rr, cc)
            if im is None:
                continue
            batch.append(im); offs.append((rr, cc)); n_done += 1
            if len(batch) >= 4:
                rows += predict(M, batch, offs, extra); batch, offs = [], []
        if batch:
            rows += predict(M, batch, offs, extra)
    tag = f"{name}-{mode}"
    pd.DataFrame(rows, columns=["row", "col", "px0", "py0", "px1", "py1", "score"]).to_csv(f"{outdir}/raw_{tag}_fold{fold}.csv", index=False)
    print(json.dumps({"model": name, "mode": mode, "fold": fold, "grid_tiles": len(tiles), "tiles_processed": n_done, "raw_dets": len(rows)}))


if __name__ == "__main__":
    main()
