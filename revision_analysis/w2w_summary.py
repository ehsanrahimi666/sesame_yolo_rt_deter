#!/usr/bin/env python
"""Wall-to-wall summary at cross-validated thresholds: totals, density, box size, agreement maps."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import json, sys, numpy as np, pandas as pd, rasterio, geopandas as gpd
from shapely.geometry import Point
VA = json.load(open("out/valid_area.json")); AREA = VA["valid_area_m2"]
S = json.load(open("out/summary.json"))
TAGS = [t for t in ["yolo26n-e2e", "yolo26n-nms", "rtdetr-l"] if __import__("os").path.exists(f"out/w2w/w2w_{t}.csv")]
B = GIS_DIR
field = gpd.read_file(B + "Field_border.shp").geometry.iloc[0]
frac = np.load("out/valid_frac_1m.npy"); x0, y0 = VA["grid_origin"]; cm = VA["grid_cell_m"]
rows, grids = [], {}
with rasterio.open(ORTHO_TIF) as ds:
    for t in TAGS:
        D = pd.read_csv(f"out/w2w/w2w_{t}.csv")
        valid = np.array([v[0] > 0 for v in ds.sample(list(zip(D.x, D.y)), indexes=4)])
        D = D[valid]
        thr = S["thresholds"][t]
        for crit, th in [("CV F1-optimal", thr["pooled_F1opt"]), ("CV count-calibrated", thr["pooled_countcal"]), ("common 0.25", 0.25), ("common 0.50", 0.50)]:
            K = D[D.score >= th]
            side = (K.width_cm + K.height_cm) / 2
            inf = np.array([field.contains(Point(a, b)) for a, b in zip(K.x, K.y)]) if crit == "CV F1-optimal" else None
            rec = dict(model=t, criterion=crit, threshold=th, n=len(K), density_m2=len(K) / AREA, box_side_median_cm=float(np.median(side)),
                       box_side_IQR=[float(np.percentile(side, 25)), float(np.percentile(side, 75))], box_cover_pct=float(100 * (K.width_cm * K.height_cm).sum() / 1e4 / AREA),
                       conf_median=float(K.score.median()))
            if inf is not None:
                rec["n_in_annotated_area"] = int(inf.sum())
                ix = np.floor((K.x - x0) / cm).astype(int); iy = np.floor((y0 - K.y) / cm).astype(int)
                G = np.zeros_like(frac); np.add.at(G, (iy.clip(0, frac.shape[0] - 1), ix.clip(0, frac.shape[1] - 1)), 1)
                grids[t] = G
                # CV-based correction of the total (count ratio at this threshold in out-of-fold predictions)
                rec["cv_count_ratio"] = thr["pooled_F1opt_ratio"]; rec["n_bias_corrected"] = len(K) / thr["pooled_F1opt_ratio"]
            rows.append(rec)
df = pd.DataFrame(rows); df.to_csv("out/w2w_summary.csv", index=False)
pd.set_option("display.width", 250); print(df.drop(columns=["box_side_IQR"]).round(3).to_string(index=False))
ok = frac >= 0.5
dens = {t: np.where(ok, grids[t] / np.maximum(frac, 1e-9), np.nan) for t in grids}
np.savez("out/w2w_grids.npz", frac=frac, **{t.replace("-", "_"): dens[t] for t in dens})
agree = {"n_cells_valid_ge_50pct": int(ok.sum())}
keymap = {("yolo26n-nms", "yolo26n-e2e"): "nms_e2e", ("rtdetr-l", "yolo26n-nms"): "nms_rt", ("rtdetr-l", "yolo26n-e2e"): "rt_e2e"}
for (a, b), key in keymap.items():
    if a in dens and b in dens:
        m = ok & ~np.isnan(dens[a]) & ~np.isnan(dens[b])
        r = float(np.corrcoef(dens[a][m], dens[b][m])[0, 1])
        md = float(np.nanmean(dens[a][m] - dens[b][m]))
        agree[key] = r; agree[key + "_mean_diff"] = md; agree[key + "_n_cells"] = int(m.sum())
        print(f"1-m cell agreement {a} vs {b}: r = {r:.3f}, n cells = {m.sum()}, mean diff = {md:.2f} clusters m-2")
json.dump(agree, open("out/w2w_agreement.json", "w"), indent=1)
print("cells with >=50% valid:", int(ok.sum()), "valid area", round(AREA, 1))
