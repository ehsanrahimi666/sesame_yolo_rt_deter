"""Validation of the mapping pipeline, step 2: comparison of the out-of-fold maps with all reference clusters.

Per fold model: optional seam-aware stitching (--stitch), global NMS (IoU 0.5), detections kept if centred in strip k of
the annotated area, and thresholded at the fold's cross-fitted F1-optimal threshold (primary) or at the pooled CV
threshold. The pooled map is matched greedily (decreasing confidence, IoU >= 0.5) to all 1201 reference boxes; counts are
compared on the 1-m cells of the density grid that lie entirely inside the annotated area.

Usage: python mapping_validation_eval.py [--stitch] [indir]   (indir: raw_<config>_fold<k>.csv; default out/r2/mapval)
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa
import json
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import contains_xy
from shapely.geometry import box
from evalcore import greedy_match, count_stats  # noqa
from stitch import X0, Y0, RX, RY, keep_mask, global_nms

STITCH = "--stitch" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
IN = args[0] if args else "out/r2/mapval"
res = lambda f: f"out/{f}" if os.path.exists(f"out/{f}") else str(ROOT / "results" / f)
field = gpd.read_file(GIS_DIR + "Field_border.shp").geometry.union_all()
edges = json.load(open(res("dataset_audit.json")))["edges"]
G = np.array([g.bounds for g in gpd.read_file(GIS_DIR + "sesame.shp").geometry])
gx, gy = (G[:, 0] + G[:, 2]) / 2, (G[:, 1] + G[:, 3]) / 2
gin = contains_xy(field, gx, gy); G, gx, gy = G[gin], gx[gin], gy[gin]
gfold = np.clip(np.searchsorted(edges, gx, side="right") - 1, 0, 4)
TH = pd.read_csv(res("pad16_last/eval_thresholds.csv"))
POOLED = {"yolo26n-e2e": 0.15, "yolo26n-nms": 0.38, "rtdetr-l": 0.61}
cm, cmy = 500 * RX, 500 * RY                      # 1-m cells of the density maps (origin = orthomosaic origin)
ix0, ix1 = int(np.floor((field.bounds[0] - X0) / cm)), int(np.ceil((field.bounds[2] - X0) / cm))
iy0, iy1 = int(np.floor((Y0 - field.bounds[3]) / cmy)), int(np.ceil((Y0 - field.bounds[1]) / cmy))
cells = [(i, j) for i in range(ix0, ix1) for j in range(iy0, iy1)
         if box(X0 + i * cm, Y0 - (j + 1) * cmy, X0 + (i + 1) * cm, Y0 - j * cmy).within(field)]
cellset = {c: k for k, c in enumerate(cells)}


def cell_counts(x, y):
    i = np.floor((x - X0) / cm).astype(int); j = np.floor((Y0 - y) / cmy).astype(int)
    out = np.zeros(len(cells))
    for a, b in zip(i, j):
        k = cellset.get((a, b))
        if k is not None:
            out[k] += 1
    return out


ref_cells = cell_counts(gx, gy)


def load_map(tag):
    name, mode = ("rtdetr-l", "none") if tag == "rtdetr-l" else ("yolo26n", tag.split("-")[1])
    parts = []
    for k in range(5):
        D = pd.read_csv(f"{IN}/raw_{name}-{mode}_fold{k}.csv")
        if STITCH:
            D = D[keep_mask(D)]
        K = global_nms(D, 0.5)
        K["xmin"] = X0 + K.px0 * RX; K["xmax"] = X0 + K.px1 * RX; K["ymin"] = Y0 - K.py1 * RY; K["ymax"] = Y0 - K.py0 * RY
        K["x"] = (K.xmin + K.xmax) / 2; K["y"] = (K.ymin + K.ymax) / 2
        inside = contains_xy(field, K.x.values, K.y.values) & (K.x.values >= edges[k]) & (K.x.values < edges[k + 1])
        K = K[inside].copy(); K["fold"] = k
        K["thr_cf"] = float(TH[(TH.model == tag) & (TH.fold == k)].thr_F1_crossfit.iloc[0])
        parts.append(K)
    return pd.concat(parts, ignore_index=True)


rows, cellrows = [], []
for tag in ["yolo26n-e2e", "yolo26n-nms", "rtdetr-l"]:
    P = load_map(tag)
    for crit in ["cross-fitted", "pooled CV"]:
        Q = P[P.score >= (P.thr_cf if crit == "cross-fitted" else POOLED[tag])]
        tp = greedy_match(Q[["xmin", "ymin", "xmax", "ymax"]].values, Q.score.values, G, 0.5)
        TP = int(tp.sum()); FP = len(Q) - TP; FN = len(G) - TP
        Pp = TP / max(TP + FP, 1); Rr = TP / len(G); F = 2 * Pp * Rr / (Pp + Rr)
        pc = cell_counts(Q.x.values, Q.y.values); cs = count_stats(ref_cells, pc)
        rec = dict(model=tag, criterion=crit, n_ref=len(G), n_pred=len(Q), TP=TP, FP=FP, FN=FN, P=Pp, R=Rr, F1=F,
                   total_relerr_pct=100 * (len(Q) - len(G)) / len(G), FP_per_m2=FP / field.area,
                   cells=cs["n_units"], cell_MAE=cs["MAE"], cell_RMSE=cs["RMSE"], cell_bias_pct=cs["bias_pct"],
                   cell_R2_11=cs["R2_11"], cell_r2=cs["r2_pearson"], cell_ref_mean=ref_cells.mean(),
                   cell_ref_min=ref_cells.min(), cell_ref_max=ref_cells.max())
        for k in range(5):
            m = (Q.fold == k).values; ng = int((gfold == k).sum())
            rec[f"fold{k}_relerr_pct"] = 100 * (m.sum() - ng) / ng
        rows.append(rec)
        if crit == "cross-fitted":
            cellrows += [dict(model=tag, cell=c, ref=a, pred=b) for c, (a, b) in enumerate(zip(ref_cells, pc))]
sfx = "_stitched" if STITCH else ""
pd.DataFrame(rows).to_csv(f"{IN}/mapval_summary{sfx}.csv", index=False)
pd.DataFrame(cellrows).to_csv(f"{IN}/mapval_cells{sfx}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(pd.DataFrame(rows).round(3).T.to_string())
