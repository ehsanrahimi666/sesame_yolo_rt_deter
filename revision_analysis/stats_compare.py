#!/usr/bin/env python
"""
Fold-wise paired comparisons and stratified spatial bootstrap for the three detector configurations.
Usage: python stats_compare.py <eval_dir>   (reads eval_fold_metrics.csv and eval_core_units.pkl)
Writes <eval_dir>/stats_pairs.csv, <eval_dir>/stats_pooled.csv, <eval_dir>/stats_bootstrap.csv
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import sys, pickle, itertools
sys.path.insert(0, "lib")
import numpy as np, pandas as pd
from scipy import stats
from evalcore import ap_101, CORE0, CORE1, RX

D = sys.argv[1] if len(sys.argv) > 1 else "out/pad16_last"
df = pd.read_csv(f"{D}/eval_fold_metrics.csv"); U = pickle.load(open(f"{D}/eval_core_units.pkl", "rb"))
TAGS = ["yolo26n-e2e", "yolo26n-nms", "rtdetr-l"]
METRICS = ["AP50_tile", "AP50_95_tile", "AP50_cluster", "AP50_95_cluster", "P_cluster", "R_cluster", "F1_cluster", "blk_MAE", "blk_R2_11"]
k = 5; ratio = 1 / (k - 1)          # n_test / n_train for 5-fold CV (Nadeau & Bengio, 2003)
rows = []
for a, b in [("yolo26n-nms", "yolo26n-e2e"), ("rtdetr-l", "yolo26n-e2e"), ("rtdetr-l", "yolo26n-nms")]:
    for m in METRICS:
        x = df[df.model == a].sort_values("fold")[m].values; y = df[df.model == b].sort_values("fold")[m].values
        ok = ~np.isnan(x) & ~np.isnan(y); d = x[ok] - y[ok]; n = len(d)
        md, sd = d.mean(), d.std(ddof=1)
        tcrit = stats.t.ppf(0.975, n - 1)
        p_t = stats.ttest_rel(x[ok], y[ok]).pvalue
        p_w = stats.wilcoxon(x[ok], y[ok], method="exact").pvalue if np.any(d != 0) else 1.0
        t_nb = md / np.sqrt((1 / n + ratio) * sd ** 2) if sd > 0 else np.nan
        p_nb = 2 * stats.t.sf(abs(t_nb), n - 1) if sd > 0 else np.nan
        rows.append(dict(A=a, B=b, metric=m, n_folds=n, mean_A=x[ok].mean(), mean_B=y[ok].mean(), diff=md, sd_diff=sd,
                         ci_lo=md - tcrit * sd / np.sqrt(n), ci_hi=md + tcrit * sd / np.sqrt(n), folds_A_better=int(((d < 0) if m.startswith("blk_MAE") else (d > 0)).sum()),
                         p_paired_t=p_t, p_wilcoxon=p_w, p_corrected_t=p_nb))
pairs = pd.DataFrame(rows); pairs.to_csv(f"{D}/stats_pairs.csv", index=False)

# ---- pooled (micro) metrics and core-level units ------------------------------------------------
def to_units(u):
    """Assign detections and GT to 0.512-m cores (index into u['cores'])."""
    c = u["cores"]
    def idx(X, Y):
        m = (X[:, None] >= c[None, :, 0]) & (X[:, None] < c[None, :, 2]) & (Y[:, None] > c[None, :, 1]) & (Y[:, None] <= c[None, :, 3])
        out = np.full(len(X), -1); hit = m.any(1); out[hit] = m[hit].argmax(1); return out
    di = idx((u["B"][:, 0] + u["B"][:, 2]) / 2, (u["B"][:, 1] + u["B"][:, 3]) / 2)
    gi = idx((u["G"][:, 0] + u["G"][:, 2]) / 2, (u["G"][:, 1] + u["G"][:, 3]) / 2)
    return di, gi
pooled, units = [], {}
for t in TAGS:
    TP = FP = FN = 0; Sall, Tall, ngall = [], [], 0
    for f in range(5):
        u = U[(t, f)]; keep = u["S"] >= u["thr"]
        tp = int(u["TP"][keep].sum()); TP += tp; FP += int(keep.sum()) - tp; FN += len(u["G"]) - tp
        Sall.append(u["S"]); Tall.append(u["TP"]); ngall += len(u["G"])
        units[(t, f)] = to_units(u)
    P = TP / (TP + FP); R = TP / (TP + FN)
    pooled.append(dict(model=t, TP=TP, FP=FP, FN=FN, precision=P, recall=R, F1=2 * P * R / (P + R),
                       AP50_cluster_pooled=ap_101(np.concatenate(Sall), np.concatenate(Tall), ngall), n_clusters=ngall))
pooled = pd.DataFrame(pooled); pooled.to_csv(f"{D}/stats_pooled.csv", index=False)

# ---- stratified bootstrap over cores (spatial units of 0.512 m), 2000 replicates -----------------
rng = np.random.default_rng(2026)
ncores = {f: len(U[(TAGS[0], f)]["cores"]) for f in range(5)}
B = 2000; res = {t: {"F1": [], "R": [], "P": [], "AP50": []} for t in TAGS}
pre = {}
for t in TAGS:
    for f in range(5):
        u = U[(t, f)]; di, gi = units[(t, f)]
        above = u["S"] >= u["thr"]
        pre[(t, f)] = dict(di=di, gi=gi, S=u["S"], TP=u["TP"], above=above)
for r in range(B):
    draw = {f: rng.integers(0, ncores[f], ncores[f]) for f in range(5)}
    for t in TAGS:
        Ss, Ts, Ab, ng = [], [], [], 0
        for f in range(5):
            p = pre[(t, f)]; cnt = np.bincount(draw[f], minlength=ncores[f])
            w_det = np.where(p["di"] >= 0, cnt[np.clip(p["di"], 0, None)], 0)
            w_gt = np.where(p["gi"] >= 0, cnt[np.clip(p["gi"], 0, None)], 0)
            Ss.append(np.repeat(p["S"], w_det)); Ts.append(np.repeat(p["TP"], w_det)); Ab.append(np.repeat(p["above"], w_det)); ng += w_gt.sum()
        S_ = np.concatenate(Ss); T_ = np.concatenate(Ts); A_ = np.concatenate(Ab)
        tp = (T_ & A_).sum(); fp = A_.sum() - tp; fn = ng - tp
        P = tp / max(tp + fp, 1); R = tp / max(ng, 1)
        res[t]["F1"].append(2 * P * R / max(P + R, 1e-12)); res[t]["R"].append(R); res[t]["P"].append(P)
        res[t]["AP50"].append(ap_101(S_, T_, ng))
brow = []
for a, b in [("yolo26n-nms", "yolo26n-e2e"), ("rtdetr-l", "yolo26n-e2e"), ("rtdetr-l", "yolo26n-nms")]:
    for m in ["AP50", "F1", "R", "P"]:
        d = np.array(res[a][m]) - np.array(res[b][m])
        brow.append(dict(A=a, B=b, metric=m + "_cluster_pooled", diff_median=np.median(d), ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                         p_two_sided=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
for t in TAGS:
    for m in ["AP50", "F1", "R", "P"]:
        v = np.array(res[t][m]); brow.append(dict(A=t, B="", metric=m + "_cluster_pooled", diff_median=np.median(v), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5), p_two_sided=np.nan))
boot = pd.DataFrame(brow); boot.to_csv(f"{D}/stats_bootstrap.csv", index=False)
pd.set_option("display.width", 250)
print(pairs.round(4).to_string(index=False)); print(pooled.round(4).to_string(index=False)); print(boot.round(4).to_string(index=False))
