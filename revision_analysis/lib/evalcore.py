"""
Core evaluation utilities for the sesame flower-cluster detection study.

Two evaluation levels are supported:
  * tile level   - the conventional object-detection protocol on the (50 %-overlapping) 512-px tiles;
  * cluster level - a de-duplicated protocol in map coordinates: each prediction is kept only if its
                    centre lies in the central 256 x 256-px core of the tile that produced it (cores of
                    neighbouring tiles abut without overlapping), so every annotated cluster and every
                    prediction is counted once.
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from paths import *  # noqa

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms

RX, RY = 0.0019999573967093604, 0.0019999711682572143   # orthomosaic pixel size (m)
TILE, CORE0, CORE1 = 512, 128, 384                         # tile size and core window (px)
DATA = Path(DATA_DIR)


def load_tiles():
    t = pd.read_csv(DATA / "tiles.csv")
    t["stem"] = t.file_name.str[:-4]
    return t


def load_gt_tile(stem):
    p = DATA / "labels" / f"{stem}.txt"
    txt = p.read_text().strip() if p.exists() else ""
    if not txt:
        return np.zeros((0, 4))
    a = np.array([list(map(float, l.split()[1:5])) for l in txt.splitlines()])
    xc, yc, w, h = a.T * TILE
    return np.c_[xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def iou_matrix(a, b):
    """IoU between boxes a (n,4) and b (m,4) in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-12)


def greedy_match(pred, scores, gt, iou_thr=0.5):
    """Greedy score-ordered matching. Returns TP flag per prediction (in the input order)."""
    order = np.argsort(-scores, kind="stable")
    tp = np.zeros(len(pred), bool)
    if len(gt) == 0 or len(pred) == 0:
        return tp
    M = iou_matrix(pred[order], gt)
    used = np.zeros(len(gt), bool)
    for r, i in enumerate(order):
        cand = np.where(~used & (M[r] >= iou_thr))[0]
        if len(cand):
            j = cand[np.argmax(M[r, cand])]
            used[j] = True; tp[i] = True
    return tp


def ap_101(scores, tp, n_gt):
    """101-point interpolated AP (same integration as Ultralytics / COCO)."""
    if n_gt == 0:
        return np.nan
    o = np.argsort(-scores, kind="stable"); t = tp[o].astype(float)
    ctp = np.cumsum(t); cfp = np.cumsum(1 - t)
    rec = ctp / n_gt; prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def prf_at(scores, tp, n_gt, thr):
    k = scores >= thr
    TP = int(tp[k].sum()); FP = int(k.sum() - TP); FN = int(n_gt - TP)
    P = TP / (TP + FP) if TP + FP else 0.0; R = TP / n_gt if n_gt else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(TP=TP, FP=FP, FN=FN, P=P, R=R, F1=F, n_pred=int(k.sum()), n_gt=int(n_gt))


# ----------------------------------------------------------------------------------------------
def cluster_level(preds, tiles, gt_unique_map, fold, nms_iou=0.5):
    """
    De-duplicated evaluation for one fold.
    preds: DataFrame(tile, x0, y0, x1, y1, score) in tile pixels for the fold's held-out tiles.
    gt_unique_map: (N,4) unique annotation boxes in map coords (x0, y0, x1, y1) with y up.
    Returns dict with arrays for predictions (map boxes, scores) and GT inside the core region.
    """
    tf = tiles[tiles.fold == fold].set_index("stem")
    # core rectangles in map coords
    cx0 = tf.x_origin.values + CORE0 * RX; cx1 = tf.x_origin.values + CORE1 * RX
    cy1 = tf.y_origin.values - CORE0 * RY; cy0 = tf.y_origin.values - CORE1 * RY
    def in_core(X, Y):
        m = (X[:, None] >= cx0[None]) & (X[:, None] < cx1[None]) & (Y[:, None] > cy0[None]) & (Y[:, None] <= cy1[None])
        return m.any(1)
    # predictions: keep those centred in their own tile core, convert to map coords
    p = preds.copy()
    pcx = (p.x0 + p.x1) / 2; pcy = (p.y0 + p.y1) / 2
    p = p[(pcx >= CORE0) & (pcx < CORE1) & (pcy >= CORE0) & (pcy < CORE1)]
    xo = tf.loc[p.tile, "x_origin"].values; yo = tf.loc[p.tile, "y_origin"].values
    B = np.c_[xo + p.x0.values * RX, yo - p.y1.values * RY, xo + p.x1.values * RX, yo - p.y0.values * RY]
    S = p.score.values.astype(float)
    if len(B):
        keep = nms(torch.tensor(B, dtype=torch.float64), torch.tensor(S, dtype=torch.float64), nms_iou).numpy()
        B, S = B[keep], S[keep]
    # GT: unique annotations centred in the core region
    gx = (gt_unique_map[:, 0] + gt_unique_map[:, 2]) / 2; gy = (gt_unique_map[:, 1] + gt_unique_map[:, 3]) / 2
    G = gt_unique_map[in_core(gx, gy)]
    core_area = len(tf) * (CORE1 - CORE0) ** 2 * RX * RY
    return dict(B=B, S=S, G=G, core_area=core_area, cores=np.c_[cx0, cy0, cx1, cy1])


def block_counts(B, S, G, cores, thr, block=1.024):
    """Counts of predictions (score>=thr) and GT per complete 2x2-core block (1.024 m).

    Cores lie on a regular 0.512-m lattice (constant offset). Each core gets integer lattice indices; blocks are
    2 x 2 groups of cores anchored at even lattice positions. Every prediction or reference box is assigned to the
    core that contains its centre (same containment rule as in cluster_level), and thereby to that core's block.
    """
    from collections import Counter
    step = (CORE1 - CORE0) * RX
    ix = np.round(cores[:, 0] / step).astype(int); iy = np.round(cores[:, 1] / step).astype(int)
    bx0 = ix.min(); by0 = iy.min()
    key_of_core = [((a - bx0) // 2, (b - by0) // 2) for a, b in zip(ix, iy)]
    ncore = Counter(key_of_core)
    full = sorted(k for k, n in ncore.items() if n == 4)

    def core_index(X, Y):
        m = (X[:, None] >= cores[None, :, 0]) & (X[:, None] < cores[None, :, 2]) & (Y[:, None] > cores[None, :, 1]) & (Y[:, None] <= cores[None, :, 3])
        out = np.full(len(X), -1); hit = m.any(1); out[hit] = m[hit].argmax(1)
        return out

    keep = S >= thr
    pi = core_index((B[keep, 0] + B[keep, 2]) / 2, (B[keep, 1] + B[keep, 3]) / 2)
    gi = core_index((G[:, 0] + G[:, 2]) / 2, (G[:, 1] + G[:, 3]) / 2)
    cp = Counter(key_of_core[i] for i in pi if i >= 0)
    cg = Counter(key_of_core[i] for i in gi if i >= 0)
    return np.array([cg.get(k, 0) for k in full], float), np.array([cp.get(k, 0) for k in full], float)

def count_stats(true, pred):
    if len(true) == 0:   # fold without complete blocks
        return dict(n_units=0, MAE=np.nan, RMSE=np.nan, bias_pct=0.0, R2_11=np.nan, r2_pearson=np.nan,
                    true_total=0.0, pred_total=0.0)
    err = pred - true
    sst = ((true - true.mean()) ** 2).sum()
    r2_11 = 1 - (err ** 2).sum() / sst if sst > 0 else np.nan
    r = np.corrcoef(true, pred)[0, 1] if len(true) > 2 and true.std() > 0 and pred.std() > 0 else np.nan
    return dict(n_units=len(true), MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err ** 2).mean())),
                bias_pct=float(100 * err.sum() / max(true.sum(), 1)), R2_11=float(r2_11), r2_pearson=float(r ** 2),
                true_total=float(true.sum()), pred_total=float(pred.sum()))
