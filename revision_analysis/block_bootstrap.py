"""95% confidence intervals of the counting statistics (Table 4): percentile intervals from 2000 bootstrap resamples of
the 1.024-m blocks within folds (seed 2026); the same resamples are used for all configurations.

Usage: python block_bootstrap.py [setting_dir]      (default: out/pad16_last, or results/pad16_last if absent)
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa
import sys
import numpy as np
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else ("out/pad16_last" if os.path.exists("out/pad16_last/eval_blocks.csv") else str(ROOT / "results" / "pad16_last"))
bl = pd.read_csv(f"{D}/eval_blocks.csv")
rng = np.random.default_rng(2026); B = 2000


def stats(t, p):
    e = p - t; sst = ((t - t.mean()) ** 2).sum()
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), bias=100 * e.sum() / t.sum(), R2=1 - (e ** 2).sum() / sst)


folds = sorted(bl.fold.unique())
idx = {f: np.where(bl[bl.model == "yolo26n-nms"].fold.values == f)[0] for f in folds}
draws = [np.concatenate([rng.choice(idx[f], len(idx[f]), replace=True) for f in folds]) for _ in range(B)]
out = []
for m in ["yolo26n-e2e", "yolo26n-nms", "rtdetr-l"]:
    d = bl[bl.model == m].reset_index(drop=True); t, p = d.true.values, d.pred.values
    obs = stats(t, p); bs = pd.DataFrame([stats(t[i], p[i]) for i in draws])
    for k in obs:
        out.append(dict(model=m, metric=k, obs=obs[k], lo=np.percentile(bs[k], 2.5), hi=np.percentile(bs[k], 97.5)))
os.makedirs("out/r2", exist_ok=True)
R = pd.DataFrame(out); R.to_csv("out/r2/block_bootstrap_ci.csv", index=False)
print(R.round(3).to_string(index=False))
