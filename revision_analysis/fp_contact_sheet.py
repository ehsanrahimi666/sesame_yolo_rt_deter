"""Contact sheet of locations where YOLO26n+NMS and RT-DETR-l both produced a false positive (primary protocol)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import sys, pickle; sys.path.insert(0, "lib")
import numpy as np, rasterio, matplotlib, geopandas as gpd
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rasterio.windows import Window
from evalcore import iou_matrix
U = pickle.load(open("out/pad16_last/eval_core_units.pkl", "rb"))
G = np.array([g.bounds for g in gpd.read_file(GIS_DIR + "sesame.shp").geometry])
locs = []
for f in range(5):
    a = U[("yolo26n-nms", f)]; b = U[("rtdetr-l", f)]
    ka = a["S"] >= a["thr"]; kb = b["S"] >= b["thr"]
    fpA = a["B"][ka][~a["TP"][ka]]; fpB = b["B"][kb][~b["TP"][kb]]; sA = a["S"][ka][~a["TP"][ka]]
    if len(fpA) and len(fpB):
        M = iou_matrix(fpA, fpB)
        for i in np.where(M.max(1) >= 0.5)[0]:
            locs.append((f, fpA[i], fpB[M[i].argmax()], sA[i]))
print(len(locs))
tr = (480919.25018164766, 0.0019999573967093604, 4044525.1466317894, 0.0019999711682572143)
n = len(locs); cols = 9; rows_ = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows_, cols, figsize=(cols * 1.1, rows_ * 1.1))
with rasterio.open(ORTHO_TIF) as ds:
    for ax, (f, ba, bb, s) in zip(axs.ravel(), locs):
        cx, cy = (ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2
        col = int((cx - tr[0]) / tr[1]); row = int((tr[2] - cy) / tr[3]); R = 75
        im = ds.read([1, 2, 3], window=Window(col - R, row - R, 2 * R, 2 * R)).transpose(1, 2, 0)
        ax.imshow(im, extent=[0, 2 * R, 2 * R, 0])
        for box_, c in [(ba, "#1baf7a"), (bb, "#eb6834")]:
            x0 = (box_[0] - tr[0]) / tr[1] - (col - R); x1 = (box_[2] - tr[0]) / tr[1] - (col - R)
            y0 = (tr[2] - box_[3]) / tr[3] - (row - R); y1 = (tr[2] - box_[1]) / tr[3] - (row - R)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=c, lw=0.8))
        for g in G:
            if abs((g[0] + g[2]) / 2 - cx) < 0.2 and abs((g[1] + g[3]) / 2 - cy) < 0.2:
                x0 = (g[0] - tr[0]) / tr[1] - (col - R); x1 = (g[2] - tr[0]) / tr[1] - (col - R)
                y0 = (tr[2] - g[3]) / tr[3] - (row - R); y1 = (tr[2] - g[1]) / tr[3] - (row - R)
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ffd400", lw=0.8, ls=(0, (2, 1))))
        ax.set_xlim(0, 2 * R); ax.set_ylim(2 * R, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"f{f} {s:.2f}", fontsize=5, pad=1)
for ax in axs.ravel()[n:]: ax.axis("off")
fig.tight_layout(pad=0.2); fig.savefig(FIGS_DIR + "FigureS1_consensus_false_positives.png", dpi=250); print("saved")
