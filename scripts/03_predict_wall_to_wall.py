#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_predict_wall_to_wall.py   (YOLO version)
===========================================
Apply the trained YOLO detector across the WHOLE orthomosaic and produce a
georeferenced sesame-flower inventory.

Runs locally (CPU is fine, GPU faster) or on Colab if you upload the ortho.
Download weights_<model>_final.pt from the training run and point WEIGHTS at it.

Outputs (in OUT_DIR):
    sesame_flowers.gpkg / .shp / .csv   one record per flower:
                                       x, y, score, width_cm, height_cm, diameter_cm
    flower_boxes.gpkg                   detection boxes as polygons
    density_1m.tif                      flowers per 1 m cell (GeoTIFF, same CRS)
    fig_density.png                     density map
    fig_stats.png                       (a) size, (b) confidence, (c) spatial distribution
    fig_examples.png                    2x2 montage of predicted tiles with boxes
    prediction_report.txt               every number needed for the Results section

Install once:
    pip install ultralytics rasterio geopandas shapely pandas matplotlib tqdm
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import Point, box as shp_box
from shapely.ops import unary_union
import torch
from torchvision.ops import nms
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
CFG = {
    "ORTHO": r"D:\sdm\new papers\idea\DJI\Sesame\DJI_202607051136_008_Create-Area-Route7\RGB\odm_orthophoto\odm_orthophoto_rgb_masked.tif",
    "WEIGHTS": r"D:\sdm\new papers\idea\DJI\Sesame\yolo26_results\weights_yolo26n_final.pt",
    "OUT_DIR": r"D:\sdm\new papers\idea\DJI\Sesame\results\wall_to_wall",

    # optional: restrict inference to a polygon (None = whole orthomosaic)
    "CLIP_SHP": None,          # e.g. r"D:\...\RGB\Field_border.shp"

    "TILE": 512,               # must match the training tile size
    "IMGSZ": 640,              # must match training imgsz
    "OVERLAP": 64,             # px -> stride = TILE - OVERLAP
    "MIN_VALID_FRAC": 0.50,    # skip tiles that are mostly nodata/black
    "CONF": 0.50,              # use best_conf from script 02 / final_model_info.json
    "TILE_IOU": 0.60,          # within-tile NMS (ignored by end-to-end YOLO26)
    "NMS_IOU": 0.50,           # global NMS across tile seams
    "BATCH": 8,
    "DEVICE": "auto",          # "auto" | "cuda" | "cpu" | 0
    "GRID_M": 1.0,             # density cell size, metres
    "N_EXAMPLE_TILES": 4,
}
# =============================================================================


def main():
    from ultralytics import YOLO

    out = Path(CFG["OUT_DIR"]); out.mkdir(parents=True, exist_ok=True)
    if CFG["DEVICE"] == "auto":
        dev = 0 if torch.cuda.is_available() else "cpu"
    else:
        dev = CFG["DEVICE"]

    # pick up conf/imgsz written by script 02, if present
    info_p = Path(CFG["WEIGHTS"]).parent / "final_model_info.json"
    conf, imgsz = CFG["CONF"], CFG["IMGSZ"]
    if info_p.exists():
        info = json.loads(info_p.read_text())
        conf = float(info.get("best_conf", conf))
        imgsz = int(info.get("imgsz", imgsz))
        print(f"Using best_conf={conf:.2f}, imgsz={imgsz} from {info_p.name}")

    model = YOLO(CFG["WEIGHTS"])
    n_par = sum(p.numel() for p in model.model.parameters())
    print(f"Loaded {Path(CFG['WEIGHTS']).name}  ({n_par/1e6:.2f} M params) on device {dev}")

    ds = rasterio.open(CFG["ORTHO"])
    res_x, res_y = abs(ds.transform.a), abs(ds.transform.e)
    print(f"Ortho {ds.width}x{ds.height}, {ds.count} bands, GSD {res_x*1000:.2f} mm, CRS {ds.crs}")
    if ds.count >= 4:
        print("Band 4 used as validity mask only; the model sees RGB only.")

    clip_geom = None
    if CFG["CLIP_SHP"]:
        g = gpd.read_file(CFG["CLIP_SHP"]).to_crs(ds.crs)
        clip_geom = unary_union(g.geometry.values)

    T = CFG["TILE"]; S = T - CFG["OVERLAP"]
    cols = list(range(0, max(1, ds.width - T + 1), S))
    rows = list(range(0, max(1, ds.height - T + 1), S))
    if cols[-1] + T < ds.width:
        cols.append(ds.width - T)
    if rows[-1] + T < ds.height:
        rows.append(ds.height - T)
    print(f"Tiling: {len(cols)} x {len(rows)} = {len(cols)*len(rows)} tiles "
          f"({T} px = {T*res_x:.3f} m, {CFG['OVERLAP']} px overlap = {CFG['OVERLAP']*res_x:.2f} m)")

    all_box, all_score = [], []
    n_tiles = n_skip = n_raw = 0
    valid_px = 0
    batch_imgs, batch_off = [], []
    examples = []
    t0 = time.time()

    def flush():
        nonlocal n_raw
        if not batch_imgs:
            return
        outs = model.predict(source=batch_imgs, conf=conf, iou=CFG["TILE_IOU"],
                             imgsz=imgsz, device=dev, verbose=False)
        for (cc, rr), o, im in zip(batch_off, outs, batch_imgs):
            b = o.boxes.xyxy.cpu().numpy()
            s = o.boxes.conf.cpu().numpy()
            if len(s) == 0:
                continue
            n_raw += len(s)
            if len(examples) < CFG["N_EXAMPLE_TILES"] and len(s) >= 5:
                examples.append((im[..., ::-1].copy(), b.copy(), s.copy()))
            b[:, [0, 2]] += cc
            b[:, [1, 3]] += rr
            all_box.append(b); all_score.append(s)
        batch_imgs.clear(); batch_off.clear()

    for rr in tqdm(rows, desc="inference"):
        for cc in cols:
            if clip_geom is not None:
                x0, y0 = ds.transform * (cc, rr)
                x1, y1 = ds.transform * (cc + T, rr + T)
                if not shp_box(min(x0, x1), min(y0, y1),
                               max(x0, x1), max(y0, y1)).intersects(clip_geom):
                    n_skip += 1
                    continue
            win = Window(cc, rr, T, T)
            if ds.count >= 4:
                vf = float((ds.read(4, window=win) > 0).mean())
            else:
                vf = None
            rgb = ds.read([1, 2, 3], window=win)            # band 4 never enters the model
            if vf is None:
                vf = float((rgb.sum(axis=0) > 0).mean())
            if vf < CFG["MIN_VALID_FRAC"]:
                n_skip += 1
                continue
            n_tiles += 1
            valid_px += int(round(vf * T * T))
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb / max(rgb.max(), 1) * 255, 0, 255).astype(np.uint8)
            arr = np.transpose(rgb, (1, 2, 0))[..., ::-1]   # RGB -> BGR for Ultralytics
            batch_imgs.append(np.ascontiguousarray(arr))
            batch_off.append((cc, rr))
            if len(batch_imgs) >= CFG["BATCH"]:
                flush()
    flush()
    elapsed = (time.time() - t0) / 60

    if not all_box:
        raise SystemExit("No detections. Lower CONF or check the weights path.")

    boxes = torch.from_numpy(np.concatenate(all_box)).float()
    scores = torch.from_numpy(np.concatenate(all_score)).float()
    print(f"\nRaw detections: {len(scores)}  ->  global NMS (IoU {CFG['NMS_IOU']}) ...")
    keep = nms(boxes, scores, CFG["NMS_IOU"]).numpy()
    n_removed = len(scores) - len(keep)
    boxes = boxes.numpy()[keep]; scores = scores.numpy()[keep]
    n_final = len(scores)
    print(f"Kept {n_final} flowers ({n_removed} seam duplicates removed, "
          f"{100*n_removed/max(n_raw,1):.1f}%)")

    # ---- geometry ------------------------------------------------------------
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    X, Y = ds.transform * (cx, cy)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    w_cm = (boxes[:, 2] - boxes[:, 0]) * res_x * 100
    h_cm = (boxes[:, 3] - boxes[:, 1]) * res_y * 100
    diam_cm = (w_cm + h_cm) / 2.0
    box_area_cm2 = w_cm * h_cm

    df = pd.DataFrame({"flower_id": np.arange(1, n_final + 1), "x": X, "y": Y,
                       "score": scores, "width_cm": w_cm, "height_cm": h_cm,
                       "diameter_cm": diam_cm, "box_area_cm2": box_area_cm2})
    gpd.GeoDataFrame(df, geometry=[Point(a, b) for a, b in zip(X, Y)],
                     crs=ds.crs).to_file(out / "sesame_flowers.gpkg", driver="GPKG")
    gpd.GeoDataFrame(df, geometry=[Point(a, b) for a, b in zip(X, Y)],
                     crs=ds.crs).to_file(out / "sesame_flowers.shp")
    df.to_csv(out / "sesame_flowers.csv", index=False)

    polys = []
    for b in boxes:
        mx0, my0 = ds.transform * (b[0], b[3])
        mx1, my1 = ds.transform * (b[2], b[1])
        polys.append(shp_box(min(mx0, mx1), min(my0, my1), max(mx0, mx1), max(my0, my1)))
    gpd.GeoDataFrame(df, geometry=polys, crs=ds.crs).to_file(
        out / "flower_boxes.gpkg", driver="GPKG")

    # ---- density grid --------------------------------------------------------
    g = CFG["GRID_M"]
    xmin, xmax, ymin, ymax = X.min(), X.max(), Y.min(), Y.max()
    nx = max(1, int(np.ceil((xmax - xmin) / g))); ny = max(1, int(np.ceil((ymax - ymin) / g)))
    Hh, _, _ = np.histogram2d(X, Y, bins=[nx, ny],
                              range=[[xmin, xmin + nx * g], [ymin, ymin + ny * g]])
    grid = np.flipud(Hh.T).astype("float32")
    with rasterio.open(out / "density_1m.tif", "w", driver="GTiff", height=ny, width=nx,
                       count=1, dtype="float32", crs=ds.crs,
                       transform=from_origin(xmin, ymin + ny * g, g, g), nodata=0) as dst:
        dst.write(grid, 1)

    imaged_m2 = valid_px * res_x * res_y
    occ = grid[grid > 0]
    dens = n_final / max(imaged_m2, 1e-9)
    cover_pct = 100 * (box_area_cm2.sum() / 1e4) / max(imaged_m2, 1e-9)

    # ---- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    im = ax.imshow(grid, cmap="YlOrRd", origin="upper",
                   extent=[xmin, xmin + nx * g, ymin, ymin + ny * g])
    fig.colorbar(im, ax=ax, shrink=.85).set_label(f"flowers per {g:g} m cell")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.set_title(f"Detected sesame-flower density ({n_final:,} flowers)")
    fig.tight_layout(); fig.savefig(out / "fig_density.png", dpi=220); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.5))
    axs[0].hist(diam_cm, bins=50, color="#E4823A")
    axs[0].set_xlabel("flower diameter (cm)"); axs[0].set_ylabel("count")
    axs[0].set_title("(a) Detected flower size")
    axs[1].hist(scores, bins=50, color="#4C78A8")
    axs[1].set_xlabel("detection confidence"); axs[1].set_title("(b) Confidence")
    s = axs[2].scatter(X, Y, c=scores, s=1.2, cmap="viridis")
    axs[2].set_xlabel("Easting (m)"); axs[2].set_ylabel("Northing (m)")
    axs[2].set_title("(c) Spatial distribution"); axs[2].set_aspect("equal")
    fig.colorbar(s, ax=axs[2], shrink=.85, label="confidence")
    fig.tight_layout(); fig.savefig(out / "fig_stats.png", dpi=220); plt.close(fig)

    if examples:
        k = len(examples)
        fig, axs = plt.subplots(2, 2, figsize=(8.4, 8.4))
        for ax, (img, bb, ss) in zip(axs.ravel(), examples):
            pim = Image.fromarray(img)
            dr = ImageDraw.Draw(pim)
            for b, sc in zip(bb, ss):
                dr.rectangle(list(b), outline=(60, 120, 255), width=2)
            ax.imshow(pim); ax.axis("off")
            ax.set_title(f"{len(ss)} detections", fontsize=9)
        for ax in axs.ravel()[k:]:
            ax.axis("off")
        fig.suptitle("Wall-to-wall inference: predicted sesame flowers", y=.99)
        fig.tight_layout(); fig.savefig(out / "fig_examples.png", dpi=200); plt.close(fig)

    # ---- report --------------------------------------------------------------
    rep = f"""SESAME WALL-TO-WALL PREDICTION REPORT
{'='*62}
Model                  : {Path(CFG['WEIGHTS']).name}  ({n_par/1e6:.2f} M params)
Confidence threshold   : {conf:.2f}   imgsz {imgsz}
Orthomosaic            : {ds.width} x {ds.height} px, GSD {res_x*1000:.2f} mm, CRS {ds.crs}
Tiles processed        : {n_tiles} ({n_skip} skipped as nodata/outside clip)
Valid imaged area      : {imaged_m2:.1f} m2 ({imaged_m2/1e4:.4f} ha)
Inference time         : {elapsed:.1f} min on device {dev}

Raw detections         : {n_raw}
Removed by global NMS  : {n_removed} ({100*n_removed/max(n_raw,1):.1f}%)
FINAL FLOWER COUNT     : {n_final}

Density (imaged area)  : {dens:.3f} flowers m-2
Occupied {g:g} m cells      : {int((grid>0).sum())} of {grid.size}
  mean                 : {occ.mean() if occ.size else 0:.2f} flowers per cell
  max                  : {occ.max() if occ.size else 0:.0f} flowers per cell

Detected box size      : width  mean {w_cm.mean():.2f} cm (SD {w_cm.std(ddof=1):.2f})
                         height mean {h_cm.mean():.2f} cm (SD {h_cm.std(ddof=1):.2f})
Mean diameter          : {diam_cm.mean():.2f} cm, median {np.median(diam_cm):.2f} cm,
                         range {diam_cm.min():.2f}-{diam_cm.max():.2f} cm
Summed box area        : {box_area_cm2.sum()/1e4:.2f} m2  =  {cover_pct:.2f}% of imaged area

Confidence             : mean {scores.mean():.3f}, median {np.median(scores):.3f},
                         {100*(scores>0.5).mean():.1f}% > 0.5, {100*(scores>0.9).mean():.1f}% > 0.9

NOTE: box size reflects how the reference boxes were drawn, not the true corolla
      dimensions. Report it as detected box size, not as floral area, unless the
      annotation boxes were drawn tight to the flowers.
{'='*62}
"""
    (out / "prediction_report.txt").write_text(rep, encoding="utf-8")
    print("\n" + rep)
    print(f"All outputs -> {out}")
    ds.close()


if __name__ == "__main__":
    main()
