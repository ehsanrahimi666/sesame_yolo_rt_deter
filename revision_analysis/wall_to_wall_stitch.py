"""Wall-to-wall mapping, step 2: seam-aware stitching (primary) or global NMS only (pipeline of the first revision),
global NMS (IoU 0.5), removal of detections centred on no-data pixels, totals at the cross-validated, count-calibrated,
and common thresholds, 1-m density grids, and agreement between configurations.

Usage: python wall_to_wall_stitch.py [indir]     (indir with raw_<model>-<mode>.csv; default out/r2/w2w)
Needs the orthomosaic (alpha band) for the no-data test and for the valid fraction of each 1-m cell.
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa
import json
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely import contains_xy
from stitch import X0, Y0, RX, RY, W, H, keep_mask, global_nms, to_map

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/r2/w2w"
THR = {"yolo26n-e2e": {"CV F1-optimal": 0.15, "CV count-calibrated": 0.24},
       "yolo26n-nms": {"CV F1-optimal": 0.38, "CV count-calibrated": 0.34},
       "rtdetr-l": {"CV F1-optimal": 0.61, "CV count-calibrated": 0.64}}
res = lambda f: f"out/{f}" if os.path.exists(f"out/{f}") else str(ROOT / "results" / f)
VALID_M2 = json.load(open(res("valid_area.json")))["valid_area_m2"]
field = gpd.read_file(GIS_DIR + "Field_border.shp").geometry.union_all()
CELL = 500                                                # 1-m cells (500 px)


def valid_frac_1m():
    f = f"{OUT}/valid_frac_1m.npy"
    if os.path.exists(f):
        return np.load(f)
    ny, nx = int(np.ceil(H / CELL)), int(np.ceil(W / CELL))
    frac = np.zeros((ny, nx))
    with rasterio.open(ORTHO_TIF) as ds:
        for i in range(ny):
            r0 = i * CELL; h = min(CELL, H - r0)
            a = ds.read(4, window=Window(0, r0, W, h)) > 0
            for j in range(nx):
                frac[i, j] = a[:, j * CELL:(j + 1) * CELL].sum() / (CELL * CELL)
    np.save(f, frac)
    return frac


def valid_center(K):
    """True where the centre pixel of a detection is a valid (alpha > 0) orthomosaic pixel."""
    if not os.path.exists(ORTHO_TIF):
        print("orthomosaic not found: the no-data test is skipped", flush=True)
        return np.ones(len(K), bool)
    cx = np.floor((K.px0.values + K.px1.values) / 2).astype(int).clip(0, W - 1)
    cy = np.floor((K.py0.values + K.py1.values) / 2).astype(int).clip(0, H - 1)
    ok = np.zeros(len(K), bool)
    with rasterio.open(ORTHO_TIF) as ds:
        for r0 in range(0, H, 2048):                      # read the alpha band in horizontal blocks
            sel = (cy >= r0) & (cy < r0 + 2048)
            if sel.any():
                a = ds.read(4, window=Window(0, r0, W, min(2048, H - r0)))
                ok[sel] = a[cy[sel] - r0, cx[sel]] > 0
    return ok


def process(tag, stitch=True):
    nm, mode = ("rtdetr-l", "none") if tag == "rtdetr-l" else ("yolo26n", tag.split("-")[1])
    D = pd.read_csv(f"{OUT}/raw_{nm}-{mode}.csv")
    if stitch:
        D = D[keep_mask(D)]
    K = global_nms(D, 0.5)
    v = valid_center(K)
    print(tag, "stitched" if stitch else "global NMS only", "detections centred on no-data pixels:", int((~v).sum()), flush=True)
    return to_map(K[v].copy())


if __name__ == "__main__":
    frac = valid_frac_1m(); ok = frac >= 0.5
    rows, grids = [], {}
    tags = [t for t in ["yolo26n-nms", "yolo26n-e2e", "rtdetr-l"]
            if os.path.exists(f"{OUT}/" + ("raw_rtdetr-l-none.csv" if t == "rtdetr-l" else f"raw_yolo26n-{t.split('-')[1]}.csv"))]
    for tag in tags:
        for stitch in [True, False]:
            K = process(tag, stitch)
            if stitch:
                K.to_csv(f"{OUT}/w2w_{tag}_stitched.csv", index=False)
            crit = dict(THR[tag]); crit["common 0.25"] = 0.25; crit["common 0.50"] = 0.50
            for c, t in crit.items():
                Q = K[K.score >= t]; side = (Q.width_cm + Q.height_cm) / 2
                rows.append(dict(model=tag, pipeline="stitched" if stitch else "global NMS only", criterion=c, threshold=t,
                                 n=len(Q), density_m2=len(Q) / VALID_M2, box_median_cm=side.median(), box_q1=side.quantile(0.25),
                                 box_q3=side.quantile(0.75), n_in_annotated_area=int(contains_xy(field, Q.x.values, Q.y.values).sum())))
                if stitch and c == "CV F1-optimal":
                    ix = np.floor((Q.x.values - X0) / (CELL * RX)).astype(int); iy = np.floor((Y0 - Q.y.values) / (CELL * RY)).astype(int)
                    Gd = np.zeros(frac.shape); np.add.at(Gd, (iy, ix), 1)
                    grids[tag] = np.where(ok, Gd / np.maximum(frac, 1e-9), np.nan)
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/w2w_summary_r2.csv", index=False)
    np.savez(f"{OUT}/w2w_grids_r2.npz", frac=frac, **{t.replace("-", "_"): g for t, g in grids.items()})
    agree = {"n_cells_valid_ge_50pct": int(ok.sum())}
    keys = list(grids)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = grids[keys[i]][ok], grids[keys[j]][ok]; m = np.isfinite(a) & np.isfinite(b)
            agree[f"{keys[i]}__{keys[j]}"] = dict(r=float(np.corrcoef(a[m], b[m])[0, 1]), mean_diff=float((b[m] - a[m]).mean()), n=int(m.sum()))
    for t, g in grids.items():
        v = g[ok]; v = v[np.isfinite(v)]
        agree[f"{t}_cell_density"] = dict(p50=float(np.median(v)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                                          max=float(v.max()), pct_above_17_2=float(100 * (v > 17.2).mean()), pct_above_19=float(100 * (v > 19).mean()))
    json.dump(agree, open(f"{OUT}/w2w_agreement_r2.json", "w"), indent=1)
    pd.set_option("display.width", 250); print(R.round(3).to_string(index=False)); print(json.dumps(agree, indent=1))
