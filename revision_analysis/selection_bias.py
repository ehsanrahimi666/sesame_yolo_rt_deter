"""Quantify optimism from selecting the checkpoint (best epoch) on the held-out fold.
For each fold k, the 'leave-fold-out' epoch is the median best epoch of the other four folds
(clipped to the last epoch actually run); its held-out metrics are read from results.csv."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import pandas as pd, numpy as np
B = RUNS_DIR
out = []
for model, d in [("yolo26n-e2e", "yolo26_results/runs/yolo26n_fold{}/results.csv"), ("rtdetr-l", "Rt_DETER/runs/rtdetr-l_fold{}/results.csv")]:
    R = {}
    for k in range(5):
        r = pd.read_csv(B + d.format(k)); r.columns = [c.strip() for c in r.columns]; R[k] = r
    best = {k: int(R[k].loc[R[k]["metrics/mAP50-95(B)"].idxmax(), "epoch"]) for k in range(5)}
    for k in range(5):
        r = R[k]; e_lfo = int(np.median([best[j] for j in range(5) if j != k])); e_use = min(e_lfo, int(r.epoch.max()))
        rb = r[r.epoch == best[k]].iloc[0]; rl = r[r.epoch == e_use].iloc[0]; rlast = r.iloc[-1]
        out.append(dict(model=model, fold=k, best_epoch=best[k], lfo_epoch=e_lfo, used_epoch=e_use, last_epoch=int(r.epoch.max()),
                        mAP50_best=rb["metrics/mAP50(B)"], mAP50_lfo=rl["metrics/mAP50(B)"], mAP50_last=rlast["metrics/mAP50(B)"],
                        mAP5095_best=rb["metrics/mAP50-95(B)"], mAP5095_lfo=rl["metrics/mAP50-95(B)"], mAP5095_last=rlast["metrics/mAP50-95(B)"]))
df = pd.DataFrame(out); df.to_csv("out/final/selection_bias.csv", index=False)
pd.set_option("display.width", 250); print(df.round(3).to_string(index=False))
g = df.groupby("model")[["mAP50_best", "mAP50_lfo", "mAP50_last", "mAP5095_best", "mAP5095_lfo", "mAP5095_last"]].mean().round(3); print(g)
