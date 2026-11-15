"""Sensitivity analysis: primary protocol re-evaluated without fold 0 (4 tiles, 18 clusters in cores).
Thresholds are re-cross-fitted among folds 1-4 (each fold's threshold maximizes pooled F1 on the other three)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa
import numpy as np, pandas as pd, geopandas as gpd
from scipy import stats
from evalcore import *   # noqa
from pathlib import Path
P_DIR = Path("out/pad16_last") if os.path.exists("out/pad16_last/preds_yolo26n-nms_fold0.csv") else ROOT / "results" / "pad16_last"
TAGS = ["yolo26n-e2e", "yolo26n-nms", "rtdetr-l"]; THR = np.round(np.arange(0.01, 1.00, 0.01), 2)
FOLDS = [1, 2, 3, 4]
tiles = load_tiles(); G_all = np.array([g.bounds for g in gpd.read_file(GIS_DIR + "sesame.shp").geometry])
U, fold_rows, pooled_rows, blocks = {}, [], [], {}
for tag in TAGS:
    P = pd.concat([pd.read_csv(P_DIR / f"preds_{tag}_fold{k}.csv") for k in range(5)], ignore_index=True)
    cl = {}
    for k in FOLDS:
        c = cluster_level(P[P.fold == k], tiles, G_all, k); c["TP"] = greedy_match(c["B"], c["S"], c["G"], 0.5); cl[k] = c
    TP = FP = FN = 0; Sall, Tall, ng_all, bt, bp = [], [], 0, [], []
    for k in FOLDS:
        oth = [j for j in FOLDS if j != k]
        So = np.concatenate([cl[j]["S"] for j in oth]); To = np.concatenate([cl[j]["TP"] for j in oth]); no = sum(len(cl[j]["G"]) for j in oth)
        t = float(THR[int(np.argmax([prf_at(So, To, no, x)["F1"] for x in THR]))])
        c = cl[k]; m = prf_at(c["S"], c["TP"], len(c["G"]), t)
        a, b = block_counts(c["B"], c["S"], c["G"], c["cores"], t); bt.append(a); bp.append(b)
        cs = count_stats(a, b)
        fold_rows.append(dict(model=tag, fold=k, thr=t, P=m["P"], R=m["R"], F1=m["F1"], AP50=ap_101(c["S"], c["TP"], len(c["G"])), blk_MAE=cs["MAE"]))
        TP += m["TP"]; FP += m["FP"]; FN += m["FN"]; Sall.append(c["S"]); Tall.append(c["TP"]); ng_all += len(c["G"])
        U[(tag, k)] = dict(c=c, thr=t)
    Pp = TP / (TP + FP); Rr = TP / (TP + FN); bt = np.concatenate(bt); bp = np.concatenate(bp); cs = count_stats(bt, bp)
    pooled_rows.append(dict(model=tag, n=ng_all, TP=TP, FP=FP, FN=FN, P=Pp, R=Rr, F1=2 * Pp * Rr / (Pp + Rr),
                            AP50=ap_101(np.concatenate(Sall), np.concatenate(Tall), ng_all), blk_MAE=cs["MAE"], blk_bias=cs["bias_pct"], n_blocks=cs["n_units"]))
FR = pd.DataFrame(fold_rows); PR = pd.DataFrame(pooled_rows)
# paired tests on four folds (same fold models; correction ratio n_test/n_train = 1/4 as in the 5-fold design)
rows = []
for a, b in [("yolo26n-nms", "yolo26n-e2e"), ("rtdetr-l", "yolo26n-e2e"), ("rtdetr-l", "yolo26n-nms")]:
    for m in ["AP50", "P", "R", "F1", "blk_MAE"]:
        x = FR[FR.model == a].sort_values("fold")[m].values; y = FR[FR.model == b].sort_values("fold")[m].values
        d = x - y; n = len(d); sd = d.std(ddof=1)
        t_nb = d.mean() / np.sqrt((1 / n + 0.25) * sd ** 2)
        rows.append(dict(A=a, B=b, metric=m, diff=d.mean(), p_t=stats.ttest_rel(x, y).pvalue,
                         p_wilcoxon=stats.wilcoxon(x, y, method="exact").pvalue, p_corrected_t=2 * stats.t.sf(abs(t_nb), n - 1)))
TT = pd.DataFrame(rows)
# stratified core bootstrap of pooled differences (folds 1-4), 2000 replicates, seed 2026
def core_idx(u):
    c = u["cores"]
    def idx(X, Y):
        m = (X[:, None] >= c[None, :, 0]) & (X[:, None] < c[None, :, 2]) & (Y[:, None] > c[None, :, 1]) & (Y[:, None] <= c[None, :, 3])
        o = np.full(len(X), -1); h = m.any(1); o[h] = m[h].argmax(1); return o
    return idx((u["B"][:, 0] + u["B"][:, 2]) / 2, (u["B"][:, 1] + u["B"][:, 3]) / 2), idx((u["G"][:, 0] + u["G"][:, 2]) / 2, (u["G"][:, 1] + u["G"][:, 3]) / 2)
pre = {}
for t in TAGS:
    for f in FOLDS:
        u = U[(t, f)]["c"]; di, gi = core_idx(u); pre[(t, f)] = dict(di=di, gi=gi, S=u["S"], TP=u["TP"], above=u["S"] >= U[(t, f)]["thr"], n=len(u["cores"]))
rng = np.random.default_rng(2026); res = {t: {"F1": [], "AP50": [], "P": [], "R": []} for t in TAGS}
for r in range(2000):
    draw = {f: rng.integers(0, pre[(TAGS[0], f)]["n"], pre[(TAGS[0], f)]["n"]) for f in FOLDS}
    for t in TAGS:
        Ss, Ts, Ab, ng = [], [], [], 0
        for f in FOLDS:
            p = pre[(t, f)]; cnt = np.bincount(draw[f], minlength=p["n"])
            wd = np.where(p["di"] >= 0, cnt[np.clip(p["di"], 0, None)], 0); wg = np.where(p["gi"] >= 0, cnt[np.clip(p["gi"], 0, None)], 0)
            Ss.append(np.repeat(p["S"], wd)); Ts.append(np.repeat(p["TP"], wd)); Ab.append(np.repeat(p["above"], wd)); ng += wg.sum()
        S_, T_, A_ = np.concatenate(Ss), np.concatenate(Ts), np.concatenate(Ab)
        tp = (T_ & A_).sum(); fp = A_.sum() - tp; P_ = tp / max(tp + fp, 1); R_ = tp / max(ng, 1)
        res[t]["F1"].append(2 * P_ * R_ / max(P_ + R_, 1e-12)); res[t]["P"].append(P_); res[t]["R"].append(R_); res[t]["AP50"].append(ap_101(S_, T_, ng))
brow = []
for a, b in [("yolo26n-nms", "yolo26n-e2e"), ("rtdetr-l", "yolo26n-e2e"), ("rtdetr-l", "yolo26n-nms")]:
    for m in ["AP50", "F1", "P", "R"]:
        d = np.array(res[a][m]) - np.array(res[b][m]); obs = PR[PR.model == a][m].iloc[0] - PR[PR.model == b][m].iloc[0]
        brow.append(dict(A=a, B=b, metric=m, obs=obs, lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
BT = pd.DataFrame(brow)
pd.set_option("display.width", 250)
for name, df in [("pooled_folds1-4", PR), ("fold_metrics", FR), ("paired_tests_4folds", TT), ("bootstrap_diff_folds1-4", BT)]:
    os.makedirs("out/r2", exist_ok=True); df.to_csv(f"out/r2/fold0excl_{name}.csv", index=False); print("==", name); print(df.round(4).to_string(index=False))
