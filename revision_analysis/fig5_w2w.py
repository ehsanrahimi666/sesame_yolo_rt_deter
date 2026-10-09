"""Figure 5: wall-to-wall density maps of detected flower clusters (1-m grid) and their agreement."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import json
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
B = GIS_DIR
VA = json.load(open("out/valid_area.json")); S = json.load(open("out/summary.json"))
AG = json.load(open("out/w2w_agreement.json"))
z = np.load("out/w2w_grids.npz")
frac = z["frac"]; nms = z["yolo26n_nms"]; rt = z["rtdetr_l"]
x0, y0 = VA["grid_origin"]; cm = VA["grid_cell_m"]; ny, nx = frac.shape
ext = [0, nx * cm, 0, ny * cm]
field = gpd.read_file(B + "Field_border.shp").geometry.iloc[0]
fx, fy = field.exterior.xy
fx = np.array(fx) - x0; fy = np.array(fy) - (y0 - ny * cm)

# single-hue sequential ramp (light to dark green) and a diverging ramp with a neutral grey midpoint
seq = LinearSegmentedColormap.from_list("seq", ["#f2f7ee", "#b9dca0", "#6cb25a", "#2e7d32", "#123d16"])
seq.set_bad("white")
div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#a9c8ee", "#e9e8e4", "#f4b48f", "#c94f13"])
div.set_bad("white")
vmax = float(np.nanpercentile(np.concatenate([nms[~np.isnan(nms)], rt[~np.isnan(rt)]]), 99))
tN = S["thresholds"]["yolo26n-nms"]["pooled_F1opt"]; tR = S["thresholds"]["rtdetr-l"]["pooled_F1opt"]

fig = plt.figure(figsize=(7.2, 5.9))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.30)


def mapax(pos, arr, cmap, norm=None, vmin=None, vmax=None, title="", cbl="", ylab=True):
    ax = fig.add_subplot(pos)
    im = ax.imshow(arr, extent=ext, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
    ax.plot(fx, fy, color="black", lw=0.8)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)", labelpad=1)
    if ylab: ax.set_ylabel("Northing (m)", labelpad=1)
    ax.set_title(title, loc="left", fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label(cbl, fontsize=6.5); cb.ax.tick_params(labelsize=6.3, length=2)
    return ax


mapax(gs[0, 0], nms, seq, vmin=0, vmax=vmax, title=f"(A) YOLO26n + NMS (threshold {tN:.2f})", cbl="Clusters m$^{-2}$")
mapax(gs[0, 1], rt, seq, vmin=0, vmax=vmax, title=f"(B) RT-DETR-l (threshold {tR:.2f})", cbl="Clusters m$^{-2}$", ylab=False)
d = rt - nms
lim = float(np.nanpercentile(np.abs(d), 99))
mapax(gs[1, 0], d, div, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), title="(C) Difference (RT-DETR-l − YOLO26n + NMS)",
      cbl="Clusters m$^{-2}$")
ax = fig.add_subplot(gs[1, 1])
m = ~np.isnan(nms) & ~np.isnan(rt)
ax.scatter(nms[m], rt[m], s=7, color="#2e7d32", edgecolor="white", lw=0.25, alpha=0.85)
hi = max(np.nanmax(nms[m]), np.nanmax(rt[m])) * 1.05
ax.plot([0, hi], [0, hi], color="black", lw=0.6, ls=(0, (3, 2)))
ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
ax.set_xlabel("YOLO26n + NMS (clusters m$^{-2}$)", labelpad=1); ax.set_ylabel("RT-DETR-l (clusters m$^{-2}$)", labelpad=1)
ax.text(0.04, 0.96, f"r = {AG['nms_rt']:.2f}\nn = {AG['nms_rt_n_cells']} cells\nmean difference = {AG['nms_rt_mean_diff']:+.2f} m$^{{-2}}$",
        transform=ax.transAxes, va="top", fontsize=6.5)
ax.set_title("(D) Agreement of 1-m cell densities", loc="left", fontsize=7.5)
ax.grid(color="#e4e3dd", lw=0.5)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.savefig(FIGS_DIR + "Figure5.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
print("ok", vmax, lim, AG)
