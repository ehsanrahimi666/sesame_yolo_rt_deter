"""In-sample check of the final models inside the annotated polygon (models were trained on these labels)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import sys; sys.path.insert(0, "lib")
import json, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
from evalcore import greedy_match
B = GIS_DIR
field = gpd.read_file(B + "Field_border.shp").geometry.iloc[0]
G = np.array([g.bounds for g in gpd.read_file(B + "sesame.shp").geometry])
inner = field.buffer(-0.15)  # avoid edge effects: evaluate 15 cm inside the polygon
gin = np.array([inner.contains(Point((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)) for g in G])
S = json.load(open("out/summary.json"))
for t in [x for x in ["yolo26n-e2e", "yolo26n-nms", "rtdetr-l"] if __import__("os").path.exists(f"out/w2w/w2w_{x}.csv")]:
    D = pd.read_csv(f"out/w2w/w2w_{t}.csv"); th = S["thresholds"][t]["pooled_F1opt"]
    D = D[D.score >= th]
    din = np.array([inner.contains(Point(a, b)) for a, b in zip(D.x, D.y)])
    D = D[din]
    # map boxes (x0,y0,x1,y1) from pixel columns
    tr = (480919.25018164766, 0.0019999573967093604, 4044525.1466317894, 0.0019999711682572143)
    Bm = np.c_[tr[0] + D.px0 * tr[1], tr[2] - D.py1 * tr[3], tr[0] + D.px1 * tr[1], tr[2] - D.py0 * tr[3]]
    tp = greedy_match(Bm, D.score.values, G[gin], 0.5)
    print(f"{t}: thr {th}; inner-polygon area {inner.area:.1f} m2; reference {gin.sum()}; detections {len(D)}; TP {tp.sum()}; FP {len(D) - tp.sum()}; FN {gin.sum() - tp.sum()}; P {tp.mean():.3f}; R {tp.sum() / gin.sum():.3f}")
