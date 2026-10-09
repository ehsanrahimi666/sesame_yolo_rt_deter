"""Size ratio of matched predicted boxes to reference boxes (tile level, IoU>=0.3 matching, conf>=0.25)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import sys; sys.path.insert(0, "lib")
import numpy as np, pandas as pd
from evalcore import *
tiles = load_tiles(); gt = {s: load_gt_tile(s) for s in tiles.stem}
rows = []
for setting, d in [("no border, best", "out/nopad_best"), ("16-px border, best", "out/pad16_best"), ("no border, last", "out/nopad_last"), ("16-px border, last", "out/pad16_last")]:
    for tag in ["yolo26n-e2e", "yolo26n-nms", "rtdetr-l"]:
        P = pd.concat([pd.read_csv(f"{d}/preds_{tag}_fold{k}.csv") for k in range(5)])
        P = P[P.score >= 0.25]
        ratios = []
        for stem, g in gt.items():
            if len(g) == 0: continue
            pp = P[P.tile == stem]; b = pp[["x0", "y0", "x1", "y1"]].values
            if len(b) == 0: continue
            M = iou_matrix(b, g)
            for i in range(len(b)):
                j = M[i].argmax()
                if M[i, j] >= 0.3:
                    sp = ((b[i, 2] - b[i, 0]) + (b[i, 3] - b[i, 1])) / 2; sg = ((g[j, 2] - g[j, 0]) + (g[j, 3] - g[j, 1])) / 2
                    ratios.append(sp / sg)
        r = np.array(ratios)
        rows.append(dict(setting=setting, model=tag, n=len(r), median_ratio=np.median(r), mean_ratio=r.mean(), q25=np.percentile(r, 25), q75=np.percentile(r, 75)))
df = pd.DataFrame(rows); df.to_csv("out/box_size_bias.csv", index=False); print(df.round(3).to_string(index=False))
