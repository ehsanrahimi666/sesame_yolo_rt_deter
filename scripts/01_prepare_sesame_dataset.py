#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_prepare_sesame_dataset.py   (YOLO bounding-box version)
==========================================================
LOCAL (Windows) dataset builder for sesame-flower DETECTION and COUNTING.

Your sesame.shp polygons are drawn as boxes, so every annotation is reduced to its
axis-aligned bounding box -- no segmentation masks are produced or needed.

Reads:
  odm_orthophoto.tif   4 bands (red, green, blue, alpha). Band 4 is used ONLY as the
                       validity mask and is NEVER written to the tiles.
  Field_border.shp     1 polygon = the annotated region
  sesame.shp           1201 flower boxes

Writes a ready-to-train Ultralytics YOLO dataset:
  <OUT>/images/*.png            512x512 3-band RGB tiles, zero nodata pixels
  <OUT>/labels/*.txt            YOLO format: "0 xc yc w h"  (normalised 0-1)
  <OUT>/tiles.csv               tile index + georeferencing + spatial fold id
  <OUT>/folds.csv               tiles / flowers per fold
  <OUT>/boxes.csv               every box in pixel + map coords (size stats)
  <OUT>/dataset_summary.txt     numbers for the Methods/Results sections
  <OUT>/prep_config.json
  ../sesame_dataset_v1.zip      <-- UPLOAD THIS TO GOOGLE DRIVE

Guarantees
----------
* No nodata / black pixels: a tile is kept only if alpha > 0 for >= VALID_FRAC of pixels.
* No spatial leakage: folds are contiguous strips along the field's long axis, and any
  tile whose footprint straddles a strip boundary is DISCARDED, so no validation tile
  shares a single pixel with a training tile.
* Edge flowers kept only if >= MIN_POLY_KEEP of their area survives the clip.

Install once:
    pip install rasterio geopandas shapely pillow numpy tqdm

Run:
    python 01_prepare_sesame_dataset.py
"""

import json
import shutil
import zipfile
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box as shp_box, Polygon
from shapely.ops import unary_union
from PIL import Image
from tqdm import tqdm

# =============================================================================
# CONFIG -- edit paths here only
# =============================================================================
CFG = {
    "ORTHO": r"D:\sdm\new papers\idea\DJI\Sesame\DJI_202607051136_008_Create-Area-Route7\RGB\odm_orthophoto\odm_orthophoto.tif",
    "FIELD": r"D:\sdm\new papers\idea\DJI\Sesame\DJI_202607051136_008_Create-Area-Route7\RGB\Field_border.shp",
    "ANNOT": r"D:\sdm\new papers\idea\DJI\Sesame\DJI_202607051136_008_Create-Area-Route7\RGB\sesame.shp",
    "OUT":   r"D:\sdm\new papers\idea\DJI\Sesame\dataset\sesame_dataset_v1",
    "ZIP_NAME": "sesame_dataset_v1.zip",
    "CLASS_NAME": "sesame_flower",

    # tiling
    "TILE": 512,              # px -> 512 * 0.002 m = 1.024 m on the ground
    "STRIDE": 256,            # px -> 50% overlap between neighbouring training tiles
    "N_FOLDS": 5,

    # tile acceptance
    "VALID_FRAC": 1.00,       # required fraction of alpha>0 pixels (1.00 = strictly no nodata)
    "MIN_FIELD_FRAC": 0.90,   # required fraction of tile inside Field_border
    "EMPTY_TILE_FRAC": 0.15,  # flower-free tiles kept, as a fraction of positive tiles

    # object acceptance
    "MIN_POLY_KEEP": 0.50,    # keep an edge-clipped flower if >= 50% of its area remains
    "MIN_BOX_PX": 4,

    "SEED": 0,
}


# =============================================================================
def load_vectors(path, crs):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS defined.")
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf.reset_index(drop=True)


def main():
    random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"])

    out = Path(CFG["OUT"])
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    ds = rasterio.open(CFG["ORTHO"])
    res_x, res_y = abs(ds.transform.a), abs(ds.transform.e)
    print(f"Orthomosaic : {ds.width} x {ds.height} px, {ds.count} bands, dtype={ds.dtypes[0]}")
    print(f"CRS         : {ds.crs}")
    print(f"Resolution  : {res_x*1000:.2f} mm x {res_y*1000:.2f} mm")
    if ds.count >= 4:
        print("Band 4      : used as validity mask only, NOT exported")

    field = load_vectors(CFG["FIELD"], ds.crs)
    annot = load_vectors(CFG["ANNOT"], ds.crs)
    field_geom = unary_union(field.geometry.values)
    print(f"Field       : {field_geom.area:.2f} m^2 ({field_geom.area/1e4:.4f} ha)")
    print(f"Annotations : {len(annot)} boxes")

    # every annotation reduced to its axis-aligned bounding box
    ann_boxes = [Polygon.from_bounds(*g.bounds) for g in annot.geometry.values]
    ann_areas = np.array([b.area for b in ann_boxes])
    sindex = gpd.GeoDataFrame(geometry=ann_boxes, crs=ds.crs).sindex

    inv = ~ds.transform
    minx, miny, maxx, maxy = field_geom.bounds

    # --- fold strips along the field's longer axis ----------------------------
    span_x, span_y = maxx - minx, maxy - miny
    axis = "x" if span_x >= span_y else "y"
    lo, hi = (minx, maxx) if axis == "x" else (miny, maxy)
    edges = np.linspace(lo, hi, CFG["N_FOLDS"] + 1)
    print(f"Fold axis   : {axis.upper()} ({span_x:.1f} m E-W x {span_y:.1f} m N-S), "
          f"{CFG['N_FOLDS']} strips of {(hi-lo)/CFG['N_FOLDS']:.2f} m")

    def fold_of_interval(a, b):
        ia = int(np.clip(np.searchsorted(edges, a, side="right") - 1, 0, CFG["N_FOLDS"] - 1))
        ib = int(np.clip(np.searchsorted(edges, b, side="right") - 1, 0, CFG["N_FOLDS"] - 1))
        return ia if ia == ib else None

    # --- tile grid ------------------------------------------------------------
    c0, r0 = inv * (minx, maxy)
    c1, r1 = inv * (maxx, miny)
    c0, r0 = max(int(np.floor(c0)), 0), max(int(np.floor(r0)), 0)
    c1, r1 = min(int(np.ceil(c1)), ds.width), min(int(np.ceil(r1)), ds.height)

    T, S = CFG["TILE"], CFG["STRIDE"]
    cols = list(range(c0, max(c0 + 1, c1 - T + 1), S))
    rows = list(range(r0, max(r0 + 1, r1 - T + 1), S))
    print(f"Tile grid   : {len(cols)} x {len(rows)} = {len(cols)*len(rows)} candidates "
          f"({T} px = {T*res_x:.3f} m, stride {S} px)")

    reject = {"outside_field": 0, "nodata": 0, "fold_buffer": 0, "empty_dropped": 0}
    positives, empties = [], []

    for rr in tqdm(rows, desc="scanning"):
        for cc in cols:
            if cc + T > ds.width or rr + T > ds.height:
                continue
            x0, y0 = ds.transform * (cc, rr)
            x1, y1 = ds.transform * (cc + T, rr + T)
            tgeom = shp_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

            ffrac = tgeom.intersection(field_geom).area / tgeom.area
            if ffrac < CFG["MIN_FIELD_FRAC"]:
                reject["outside_field"] += 1
                continue

            a, b = (tgeom.bounds[0], tgeom.bounds[2]) if axis == "x" else (tgeom.bounds[1], tgeom.bounds[3])
            fold = fold_of_interval(a, b)
            if fold is None:
                reject["fold_buffer"] += 1
                continue

            win = Window(cc, rr, T, T)
            if ds.count >= 4:
                if float((ds.read(4, window=win) > 0).mean()) < CFG["VALID_FRAC"]:
                    reject["nodata"] += 1
                    continue
            rgb = ds.read([1, 2, 3], window=win)
            if ds.count < 4 and float((rgb.sum(axis=0) > 0).mean()) < CFG["VALID_FRAC"]:
                reject["nodata"] += 1
                continue

            objs = []
            for i in sindex.query(tgeom):
                bpoly = ann_boxes[i]
                if not bpoly.intersects(tgeom):
                    continue
                clip = bpoly.intersection(tgeom)
                if clip.is_empty or clip.area / max(ann_areas[i], 1e-12) < CFG["MIN_POLY_KEEP"]:
                    continue
                bx0, by0, bx1, by1 = clip.bounds
                pc0, pr0 = inv * (bx0, by1)          # map maxy -> min row
                pc1, pr1 = inv * (bx1, by0)
                pc0, pc1 = np.clip([pc0 - cc, pc1 - cc], 0, T)
                pr0, pr1 = np.clip([pr0 - rr, pr1 - rr], 0, T)
                w, h = pc1 - pc0, pr1 - pr0
                if w < CFG["MIN_BOX_PX"] or h < CFG["MIN_BOX_PX"]:
                    continue
                objs.append({"src_id": int(i),
                             "px": [float(pc0), float(pr0), float(pc1), float(pr1)],
                             "map": [float(bx0), float(by0), float(bx1), float(by1)]})

            rec = {"col": cc, "row": rr, "fold": fold, "field_frac": round(ffrac, 4),
                   "origin": ds.transform * (cc, rr), "objs": objs, "rgb": rgb}
            (positives if objs else empties).append(rec)

    n_keep_empty = int(round(CFG["EMPTY_TILE_FRAC"] * len(positives)))
    random.shuffle(empties)
    reject["empty_dropped"] = max(0, len(empties) - n_keep_empty)
    kept = positives + empties[:n_keep_empty]
    kept.sort(key=lambda d: (d["row"], d["col"]))
    if not kept:
        raise SystemExit("No tiles survived. Lower MIN_FIELD_FRAC or TILE and re-run.")

    # --- write tiles + YOLO labels -------------------------------------------
    tiles_rows, box_rows = [], []
    n_boxes = 0
    for img_id, rec in enumerate(tqdm(kept, desc="writing"), start=1):
        arr = rec.pop("rgb")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            lo_, hi_ = np.percentile(arr, 1), np.percentile(arr, 99)
            arr = np.clip((arr - lo_) / max(hi_ - lo_, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        stem = f"tile_r{rec['row']:06d}_c{rec['col']:06d}"
        Image.fromarray(np.transpose(arr, (1, 2, 0))).save(
            out / "images" / f"{stem}.png", format="PNG", optimize=True)

        lines = []
        for o in rec["objs"]:
            x0, y0, x1, y1 = o["px"]
            xc, yc = (x0 + x1) / 2 / T, (y0 + y1) / 2 / T
            w, h = (x1 - x0) / T, (y1 - y0) / T
            lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            box_rows.append((stem, rec["fold"], o["src_id"],
                             *[f"{v:.2f}" for v in o["px"]],
                             *[f"{v:.4f}" for v in o["map"]],
                             f"{(x1-x0)*res_x*100:.2f}", f"{(y1-y0)*res_y*100:.2f}"))
            n_boxes += 1
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        xo, yo = rec["origin"]
        tiles_rows.append((img_id, f"{stem}.png", rec["row"], rec["col"], rec["fold"],
                           f"{xo:.4f}", f"{yo:.4f}", len(rec["objs"]), rec["field_frac"]))

    with open(out / "tiles.csv", "w", encoding="utf-8") as f:
        f.write("image_id,file_name,tile_row,tile_col,fold,x_origin,y_origin,n_flowers,field_frac\n")
        for t in tiles_rows:
            f.write(",".join(str(v) for v in t) + "\n")
    with open(out / "boxes.csv", "w", encoding="utf-8") as f:
        f.write("tile,fold,src_id,px_x0,px_y0,px_x1,px_y1,map_x0,map_y0,map_x1,map_y1,"
                "width_cm,height_cm\n")
        for b in box_rows:
            f.write(",".join(str(v) for v in b) + "\n")

    fold_stats = [(k,
                   sum(1 for t in tiles_rows if t[4] == k),
                   sum(t[7] for t in tiles_rows if t[4] == k)) for k in range(CFG["N_FOLDS"])]
    with open(out / "folds.csv", "w", encoding="utf-8") as f:
        f.write("fold,n_tiles,n_flowers\n")
        for k, nt, nf in fold_stats:
            f.write(f"{k},{nt},{nf}\n")

    # --- summary --------------------------------------------------------------
    bw = np.array([float(b[11]) for b in box_rows])
    bh = np.array([float(b[12]) for b in box_rows])
    bw_px, bh_px = bw / (res_x * 100), bh / (res_y * 100)
    diam_cm = (bw + bh) / 2

    with open(out / "prep_config.json", "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2)

    lines = [
        "SESAME FLOWER DATASET (YOLO bounding boxes) - PREPARATION SUMMARY",
        "=" * 64,
        f"Orthomosaic        : {ds.width} x {ds.height} px, {ds.count} bands",
        "                     band 4 = validity mask only, NOT exported",
        f"CRS                : {ds.crs}",
        f"GSD                : {res_x*1000:.2f} mm",
        f"Field border area  : {field_geom.area:.2f} m^2 ({field_geom.area/1e4:.4f} ha)",
        f"Source annotations : {len(annot)} boxes",
        "",
        f"Tile size          : {T} px = {T*res_x:.3f} m ; stride {S} px "
        f"({100*(1-S/T):.0f}% overlap)",
        f"Fold design        : {CFG['N_FOLDS']} contiguous strips along {axis.upper()}; "
        "straddling tiles discarded (zero leakage)",
        "",
        "TILE ACCOUNTING",
        f"  candidate positions       : {len(cols)*len(rows)}",
        f"  rejected, outside field   : {reject['outside_field']}",
        f"  rejected, nodata/black    : {reject['nodata']}",
        f"  rejected, fold buffer     : {reject['fold_buffer']}",
        f"  flower-free tiles dropped : {reject['empty_dropped']}",
        f"  KEPT with flowers         : {len(positives)}",
        f"  KEPT flower-free negatives: {min(n_keep_empty, len(empties))}",
        f"  KEPT TOTAL                : {len(tiles_rows)}",
        "",
        f"Box instances exported      : {n_boxes} (duplicated across overlapping tiles)",
        f"Mean flowers per tile       : {n_boxes/max(len(tiles_rows),1):.1f}",
        f"Flower density in field     : {len(annot)/field_geom.area:.2f} flowers m-2",
        "",
        "BOX SIZE",
        f"  width  : {bw.mean():.2f} cm mean ({bw_px.mean():.1f} px), "
        f"range {bw.min():.2f}-{bw.max():.2f} cm",
        f"  height : {bh.mean():.2f} cm mean ({bh_px.mean():.1f} px), "
        f"range {bh.min():.2f}-{bh.max():.2f} cm",
        f"  mean box diameter : {diam_cm.mean():.2f} cm (median {np.median(diam_cm):.2f})",
        f"  object share of tile area : {100*(bw_px*bh_px).mean()/(T*T):.3f}%",
        "",
        "FOLD COMPOSITION",
        "  fold  tiles  flowers",
    ]
    for k, nt, nf in fold_stats:
        lines.append(f"  {k:<5} {nt:<6} {nf}")
    lines += ["", "=" * 64]
    summary = "\n".join(lines)
    (out / "dataset_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)

    zip_path = out.parent / CFG["ZIP_NAME"]
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out.parent))
    print(f"\nZIP READY -> {zip_path}   ({zip_path.stat().st_size/1e6:.1f} MB)")
    print(f"Upload to Google Drive as:  MyDrive/sesame/{CFG['ZIP_NAME']}")
    ds.close()


if __name__ == "__main__":
    main()
