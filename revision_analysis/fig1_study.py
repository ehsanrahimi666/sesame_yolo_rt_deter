"""Figure 1: study system, annotation unit and spatially blocked folds."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import os, json, numpy as np, pandas as pd, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MPoly
from shapely.ops import unary_union
from shapely.geometry import box
from PIL import Image
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6})
W = PREVIEW_DIR
B = GIS_DIR
D = DATA_DIR
FOLD_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
RX, RY = 0.0019999573967093604, 0.0019999711682572143
X0, Y0 = 480919.25018164766, 4044525.1466317894
field = gpd.read_file(B + "Field_border.shp").geometry.iloc[0]
tiles = pd.read_csv(D + "tiles.csv"); tiles["stem"] = tiles.file_name.str[:-4]
edges = np.linspace(field.bounds[0], field.bounds[2], 6)

fig = plt.figure(figsize=(7.2, 6.4))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.30)
top = outer[0].subgridspec(1, 2, width_ratios=[1, 1.62], wspace=0.17)
bot = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.34)
# (a) overview ---------------------------------------------------------------------------------
ax = fig.add_subplot(top[0])
im = Image.open(W + "ortho_1cm.jpg"); w, h = im.size; s = 0.009999786983546802
ax.imshow(np.asarray(im.reduce(2)), extent=[0, w * s, 0, h * s])
fx, fy = field.exterior.xy
ax.plot(np.array(fx) - X0, np.array(fy) - (Y0 - h * s), color="white", lw=1.6)
ax.plot(np.array(fx) - X0, np.array(fy) - (Y0 - h * s), color="#2a78d6", lw=0.9)
ax.text(np.mean(fx) - X0 - 2, np.min(fy) - (Y0 - h * s) - 2.6, "Annotated area\n(107.8 m²)", color="white", ha="center", va="top", fontsize=7, weight="bold")
ax.plot([3, 13], [2.5, 2.5], color="white", lw=2.5); ax.text(8, 3.4, "10 m", color="white", ha="center", fontsize=7, weight="bold")
ax.annotate("", xy=(46, 38.5), xytext=(46, 34.0), arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2)); ax.text(46, 38.9, "N", color="white", ha="center", va="bottom", fontsize=8, weight="bold")
ax.set_xlim(0, w * s); ax.set_ylim(0, h * s)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("(A) Orthomosaic, 5 July 2026", loc="left", fontsize=8)
# (b) folds -----------------------------------------------------------------------------------
ax = fig.add_subplot(top[1])
meta = json.load(open(W + "preview_meta.json")); bx0, by0, bx1, by1 = meta["field_window_bounds"]
fim = Image.open(W + "field_5mm.jpg")
ax.imshow(np.asarray(fim.reduce(2)), extent=[bx0 - edges[0], bx1 - edges[0], by0 - by0, by1 - by0])
ax.plot(np.array(fx) - edges[0], np.array(fy) - by0, color="white", lw=1.0)
for k in range(5):
    g = unary_union([box(r.x_origin, r.y_origin - 512 * RY, r.x_origin + 512 * RX, r.y_origin) for r in tiles[tiles.fold == k].itertuples()])
    geoms = getattr(g, "geoms", [g])
    for gg in geoms:
        xx, yy = gg.exterior.xy
        ax.add_patch(MPoly(np.c_[np.array(xx) - edges[0], np.array(yy) - by0], closed=True, fc=FOLD_COL[k], ec=FOLD_COL[k], alpha=0.45, lw=0.8))
    ax.text((edges[k] + edges[k + 1]) / 2 - edges[0], 12.25, f"Fold {k}", ha="center", va="center", fontsize=7, color="black", weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=FOLD_COL[k], lw=1.0))
for e in edges:
    ax.axvline(e - edges[0], color="white", lw=0.8, ls=(0, (3, 2)))
ex = tiles[tiles.stem == "tile_r018275_c008547"].iloc[0]
ax.add_patch(Rectangle((ex.x_origin - edges[0], ex.y_origin - 512 * RY - by0), 512 * RX, 512 * RY, fill=False, ec="white", lw=1.4))
ax.text(ex.x_origin - edges[0] + 1.15, ex.y_origin - by0 - 0.15, "C", color="white", fontsize=7, weight="bold", va="top")
ax.set_xlim(bx0 - edges[0], bx1 - edges[0]); ax.set_ylim(0, 13.0)
ax.set_xlabel("Easting relative to western field edge (m)", labelpad=1); ax.set_ylabel("Northing (m)", labelpad=1)
ax.tick_params(length=2, pad=1)
ax.set_title("(B) Annotated area and five spatial folds (shaded = retained tiles)", loc="left", fontsize=8)
# (c) example tile with reference boxes ----------------------------------------------------------
ax = fig.add_subplot(bot[0])
tim = Image.open(D + f"images/{ex.stem}.png").convert("RGB"); ax.imshow(tim, extent=[0, 512, 512, 0])
for line in open(D + f"labels/{ex.stem}.txt").read().splitlines():
    _, xc, yc, ww, hh = map(float, line.split())
    ax.add_patch(Rectangle(((xc - ww / 2) * 512, (yc - hh / 2) * 512), ww * 512, hh * 512, fill=False, ec="#ffd400", lw=0.9))
ax.add_patch(Rectangle((128, 128), 256, 256, fill=False, ec="white", lw=0.9, ls=(0, (3, 2))))
ax.plot([20, 120], [490, 490], color="white", lw=2); ax.text(70, 478, "20 cm", color="white", ha="center", fontsize=7, weight="bold")
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("(C) Tile with reference boxes;\ndashed = evaluation core", loc="left", fontsize=8)
# (d) close-up of flower clusters from the full-resolution orthomosaic ------------------------------
import rasterio
from rasterio.windows import Window
from matplotlib.patches import Polygon as MPoly2
ax = fig.add_subplot(bot[1])
ann_d = gpd.read_file(B + "sesame.shp")
g42 = ann_d.geometry.iloc[42]; cxd, cyd = g42.centroid.x + 0.03, g42.centroid.y + 0.03
half = 0.22                                                     # 44 x 44 cm window
with rasterio.open(ORTHO_TIF) as ds:
    r_off = 0
    c0 = int(round((cxd - half - X0) / RX)); r0 = int(round((Y0 - (cyd + half)) / RY)) - r_off
    n = int(round(2 * half / RX))
    a = ds.read([1, 2, 3], window=Window(c0, r0, n, n))
wx0 = X0 + c0 * RX; wy1 = Y0 - (r0 + r_off) * RY
ax.imshow(np.transpose(a, (1, 2, 0)), extent=[0, n * RX * 100, n * RY * 100, 0], interpolation="lanczos")
win = box(wx0, wy1 - n * RY, wx0 + n * RX, wy1)
for gg in ann_d.geometry:
    if gg.intersects(win):
        xy = np.array(gg.exterior.coords)
        ax.add_patch(MPoly2(np.c_[(xy[:, 0] - wx0) * 100, (wy1 - xy[:, 1]) * 100], closed=True, fill=False, ec="#ffd400", lw=1.1))
bx0, by0, bx1, by1 = ann_d.geometry.iloc[43].bounds
ax.add_patch(Rectangle(((bx0 - wx0) * 100, (wy1 - by1) * 100), (bx1 - bx0) * 100, (by1 - by0) * 100, fill=False, ec="white", lw=1.0, ls=(0, (2.5, 1.5))))
ax.set_xlim(0, n * RX * 100); ax.set_ylim(n * RY * 100, 0)
ax.plot([2.0, 7.0], [41.5, 41.5], color="white", lw=2); ax.text(4.5, 40.2, "5 cm", color="white", ha="center", fontsize=7, weight="bold")
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("(D) Flower clusters at\nshoot tips", loc="left", fontsize=8)
# (e) reference box size vs single-corolla length --------------------------------------------------
ax = fig.add_subplot(bot[2])
ann = gpd.read_file(B + "sesame.shp"); bb = np.array([g.bounds for g in ann.geometry])
side = ((bb[:, 2] - bb[:, 0]) + (bb[:, 3] - bb[:, 1])) / 2 * 100
# side length of the rectangles as drawn (mean of the four edges; the rectangles were rotated)
drawn = np.array([np.linalg.norm(np.diff(np.array(g.exterior.coords)[:5], axis=0), axis=1).mean() for g in ann.geometry]) * 100
ax.axvspan(2.0, 3.5, color="#52514e", alpha=0.18, lw=0)
ax.text(2.75, 0.97, "single\ncorolla\nlength", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color="#52514e")
bins = np.arange(2, 20.5, 0.5)
ax.hist(side, bins=bins, color="#2a78d6", ec="white", lw=0.4, label=f"axis-aligned box\n(median {np.median(side):.1f} cm)")
ax.hist(drawn, bins=bins, histtype="step", color="black", lw=0.9, label=f"rectangle as drawn\n(median {np.median(drawn):.1f} cm)")
ax.legend(frameon=False, fontsize=6.2, loc="upper right", handlelength=1.2, borderaxespad=0.2)
ax.set_xlabel("Size, (width + height)/2 (cm)", labelpad=1); ax.set_ylabel("Number of clusters", labelpad=1)
ax.set_xlim(0, 20); ax.tick_params(length=2, pad=1)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.set_title(f"(E) Size of the {len(side)}\nreference annotations", loc="left", fontsize=8)
fig.savefig(FIGS_DIR + "Figure1.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
fig.savefig(FIGS_DIR + "Figure1.tif", dpi=600, bbox_inches="tight", pad_inches=0.03, pil_kwargs={"compression": "tiff_lzw"})
print(ex.fold, ex.stem)
