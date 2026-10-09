#!/usr/bin/env python
"""
Cross-validated evaluation of YOLO26n (two inference modes) and RT-DETR-l on the five spatial folds.

Inputs : out/final/preds_<tag>_fold<k>.csv  (raw predictions, conf >= 0.001; from predict_fold.py)
         out/final/val_<tag>_fold<k>.json   (Ultralytics tile-level mAP)
Outputs: out/final/eval_fold_metrics.csv    one row per model x fold (all metrics)
         out/final/eval_thresholds.csv      cross-fitted thresholds
         out/final/eval_sweep.csv           pooled F1/recall/precision/count ratio vs threshold
         out/final/eval_blocks.csv          per-block counts (non-overlapping 1.024 m blocks)
         out/final/eval_core_units.pkl      per-core TP/FP/GT units for bootstrap
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import sys, json, pickle
sys.path.insert(0, "lib")
import numpy as np, pandas as pd, geopandas as gpd
from evalcore import *

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/pad16_best")
TAGS = ["yolo26n-e2e", "yolo26n-nms", "rtdetr-l"]
THR = np.round(np.arange(0.01, 1.00, 0.01), 2)
IOUS = np.round(np.arange(0.5, 0.96, 0.05), 2)
ANN = GIS_DIR + "sesame.shp"

tiles = load_tiles()
G_all = np.array([g.bounds for g in gpd.read_file(ANN).geometry])
gt_tile = {s: load_gt_tile(s) for s in tiles.stem}

rows, sweep_rows, thr_rows, block_rows = [], [], [], []
units = {}
for tag in TAGS:
    P = pd.concat([pd.read_csv(OUT / f"preds_{tag}_fold{k}.csv") for k in range(5)], ignore_index=True)
    # ---------------- tile level --------------------------------------------------------------
    tl = {}
    for k in range(5):
        S, TPd, ng, cnt = [], {i: [] for i in IOUS}, 0, []
        for stem in tiles[tiles.fold == k].stem:
            g = gt_tile[stem]; pp = P[P.tile == stem]
            b = pp[["x0", "y0", "x1", "y1"]].values; s = pp.score.values
            for i in IOUS:
                TPd[i].append(greedy_match(b, s, g, i))
            S.append(s); ng += len(g); cnt.append((stem, len(g), s))
        TPd = {i: np.concatenate(v) for i, v in TPd.items()}
        tl[k] = dict(S=np.concatenate(S), TP=TPd[0.5], TPd=TPd, n=ng, cnt=cnt)
    # ---------------- cluster level (de-duplicated) ---------------------------------------------
    cl = {}
    for k in range(5):
        c = cluster_level(P[P.fold == k], tiles, G_all, k)
        c["TP"] = {iou: greedy_match(c["B"], c["S"], c["G"], iou) for iou in IOUS}
        cl[k] = c
    # pooled sweep (all folds) for figures
    for t in THR:
        Sx = np.concatenate([cl[k]["S"] for k in range(5)]); Tx = np.concatenate([cl[k]["TP"][0.5] for k in range(5)])
        n = sum(len(cl[k]["G"]) for k in range(5)); m = prf_at(Sx, Tx, n, t)
        sweep_rows.append(dict(model=tag, thr=t, P=m["P"], R=m["R"], F1=m["F1"], count_ratio=m["n_pred"] / n))
    # ---------------- per fold ------------------------------------------------------------------
    for k in range(5):
        others = [j for j in range(5) if j != k]
        So = np.concatenate([cl[j]["S"] for j in others]); To = np.concatenate([cl[j]["TP"][0.5] for j in others])
        no = sum(len(cl[j]["G"]) for j in others)
        f1s = np.array([prf_at(So, To, no, t)["F1"] for t in THR])
        t_f1 = float(THR[int(np.argmax(f1s))])
        npred = np.array([(So >= t).sum() for t in THR]); t_cnt = float(THR[int(np.argmin(np.abs(npred - no)))])
        # oracle (test-fold optimised, as in the original submission) for reference
        f1k = np.array([prf_at(cl[k]["S"], cl[k]["TP"][0.5], len(cl[k]["G"]), t)["F1"] for t in THR])
        t_or = float(THR[int(np.argmax(f1k))])
        thr_rows.append(dict(model=tag, fold=k, thr_F1_crossfit=t_f1, thr_count_crossfit=t_cnt, thr_F1_oracle=t_or))
        c = cl[k]; ng = len(c["G"])
        rec = dict(model=tag, fold=k, n_tiles=int((tiles.fold == k).sum()), tile_instances=tl[k]["n"], clusters=ng,
                   core_area_m2=c["core_area"],
                   AP50_tile=ap_101(tl[k]["S"], tl[k]["TP"], tl[k]["n"]),
                   AP50_95_tile=float(np.mean([ap_101(tl[k]["S"], tl[k]["TPd"][i], tl[k]["n"]) for i in IOUS])),
                   AP50_cluster=ap_101(c["S"], c["TP"][0.5], ng),
                   AP50_95_cluster=float(np.mean([ap_101(c["S"], c["TP"][i], ng) for i in IOUS])),
                   thr=t_f1)
        m = prf_at(c["S"], c["TP"][0.5], ng, t_f1); rec.update({f"{kk}_cluster": vv for kk, vv in m.items()})
        mt = prf_at(tl[k]["S"], tl[k]["TP"], tl[k]["n"], t_f1); rec.update({f"{kk}_tile": vv for kk, vv in mt.items()})
        mo = prf_at(c["S"], c["TP"][0.5], ng, t_or); rec.update(F1_cluster_oracle=mo["F1"])
        # counting on non-overlapping 1.024-m blocks
        tr_, pr_ = block_counts(c["B"], c["S"], c["G"], c["cores"], t_f1)
        cs = count_stats(tr_, pr_); rec.update({f"blk_{kk}": vv for kk, vv in cs.items()})
        for a, b in zip(tr_, pr_):
            block_rows.append(dict(model=tag, fold=k, true=a, pred=b))
        # fold (strip) totals at the cross-fitted F1 threshold and the count-calibrated threshold
        rec["total_true"] = ng; rec["total_pred_F1thr"] = int((c["S"] >= t_f1).sum())
        rec["total_relerr_F1thr_pct"] = 100 * (rec["total_pred_F1thr"] - ng) / ng
        rec["thr_count"] = t_cnt; rec["total_pred_cntthr"] = int((c["S"] >= t_cnt).sum())
        rec["total_relerr_cntthr_pct"] = 100 * (rec["total_pred_cntthr"] - ng) / ng
        rows.append(rec)
        # per-core units for bootstrap (cluster level, IoU 0.5, at t_f1)
        units[(tag, k)] = dict(B=c["B"], S=c["S"], TP=c["TP"][0.5], G=c["G"], cores=c["cores"], thr=t_f1)
    print(tag, "done", flush=True)

df = pd.DataFrame(rows); df.to_csv(OUT / "eval_fold_metrics.csv", index=False)
pd.DataFrame(thr_rows).to_csv(OUT / "eval_thresholds.csv", index=False)
pd.DataFrame(sweep_rows).to_csv(OUT / "eval_sweep.csv", index=False)
pd.DataFrame(block_rows).to_csv(OUT / "eval_blocks.csv", index=False)
pickle.dump(units, open(OUT / "eval_core_units.pkl", "wb"))
cols = ["model", "fold", "clusters", "AP50_tile", "AP50_95_tile", "AP50_cluster", "AP50_95_cluster", "thr", "P_cluster", "R_cluster", "F1_cluster", "F1_cluster_oracle", "blk_R2_11", "blk_r2_pearson", "blk_MAE", "blk_bias_pct", "total_relerr_F1thr_pct"]
pd.set_option("display.width", 250)
print(df[cols].round(3).to_string(index=False))
print(df.groupby("model")[cols[2:]].agg(["mean", "std"]).round(3).T.to_string())
