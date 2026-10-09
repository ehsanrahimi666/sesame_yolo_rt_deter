"""Figures 2-4 (cross-validated comparison, sensitivity, thresholds and counting)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import sys, json, numpy as np, pandas as pd, matplotlib
sys.path.insert(0, "lib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evalcore import count_stats
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False})
TAGS = ["yolo26n-e2e", "yolo26n-nms", "rtdetr-l"]
LAB = {"yolo26n-e2e": "YOLO26n\nNMS-free", "yolo26n-nms": "YOLO26n\n+ NMS", "rtdetr-l": "RT-DETR-l"}
LAB1 = {"yolo26n-e2e": "YOLO26n (NMS-free head)", "yolo26n-nms": "YOLO26n (one-to-many head + NMS)", "rtdetr-l": "RT-DETR-l"}
COL = {"yolo26n-e2e": "#2a78d6", "yolo26n-nms": "#1baf7a", "rtdetr-l": "#eb6834"}
MK = {"yolo26n-e2e": "o", "yolo26n-nms": "s", "rtdetr-l": "^"}
P = "out/pad16_last"
df = pd.read_csv(f"{P}/eval_fold_metrics.csv"); pooled = pd.read_csv(f"{P}/stats_pooled.csv")
OUT = FIGS_DIR

# ---------------- Figure 2: fold-wise paired comparison -------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.35))
panels = [("AP50_tile", "(A) AP@0.5 (tile level)", (0.70, 1.0)), ("AP50_95_tile", "(B) AP@0.5:0.95 (tile level)", (0.25, 0.65)),
          ("F1_cluster", "(C) F1 (cluster level)", (0.70, 1.0)), ("blk_MAE", "(D) Counting MAE\n(clusters per 1.024-m block)", (0.3, 1.7))]
x = np.arange(3)
for ax, (m, title, yl) in zip(axs, panels):
    labs = []
    for k in range(5):
        v = [df[(df.model == t) & (df.fold == k)][m].values[0] for t in TAGS]
        if np.any(np.isnan(v)): continue
        ax.plot(x, v, color="#b9b8b2", lw=0.7, zorder=1)
        labs.append([v[2], str(k)])
    # spread fold labels vertically so that they do not overlap (minimum gap = 4% of the axis range)
    labs.sort(); gap = 0.04 * (yl[1] - yl[0])
    for i in range(1, len(labs)):
        labs[i][0] = max(labs[i][0], labs[i - 1][0] + gap)
    for yv, k in labs:
        ax.text(2.12, yv, k, fontsize=5.8, color="#52514e", va="center")
    for i, t in enumerate(TAGS):
        v = df[df.model == t][m].dropna().values
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt=MK[t], color=COL[t], ms=6, mec="white", mew=0.6, capsize=2.5, lw=1.0, zorder=3)
        if m == "F1_cluster":
            ax.plot(i + 0.22, pooled[pooled.model == t].F1.values[0], marker="D", color="black", ms=3.2, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([LAB[t] for t in TAGS], fontsize=6.3); ax.set_xlim(-0.4, 2.45)
    ax.set_ylim(*yl); ax.set_title(title, loc="left", fontsize=7.5); ax.grid(axis="y", color="#e4e3dd", lw=0.5)
axs[2].plot([], [], marker="D", color="black", ls="", ms=3.2, label="pooled over folds")
axs[2].legend(frameon=False, fontsize=6, loc="lower left", handletextpad=0.2)
fig.text(0.5, -0.02, "Grey lines: individual spatial folds (numbers = fold); colored symbols: mean ± SD over folds", ha="center", fontsize=6.3, color="#52514e")
fig.tight_layout(w_pad=0.6); fig.savefig(OUT + "Figure2.png", dpi=400, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

# ---------------- Figure 3: sensitivity of the comparison ------------------------------------------
B = RUNS_DIR
fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.55, wspace=0.36)
for j, (t, d, ttl) in enumerate([("yolo26n-e2e", "yolo26_results/runs/yolo26n_fold{}/results.csv", "(A) YOLO26n (NMS-free head)"),
                                  ("rtdetr-l", "Rt_DETER/runs/rtdetr-l_fold{}/results.csv", "(B) RT-DETR-l")]):
    ax = fig.add_subplot(gs[0, j if j == 0 else 1:3] if False else gs[0, [0, 1][j] if j == 0 else 1])
    for k in range(5):
        r = pd.read_csv(B + d.format(k)); r.columns = [c.strip() for c in r.columns]
        ax.plot(r.epoch, r["metrics/mAP50(B)"], lw=0.8, color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"][k], label=f"fold {k}")
        be = r.loc[r["metrics/mAP50-95(B)"].idxmax()]
        ax.plot(be.epoch, be["metrics/mAP50(B)"], "o", ms=3.5, mfc="white", mec="black", mew=0.7, zorder=3)
        ax.plot(r.epoch.iloc[-1], r["metrics/mAP50(B)"].iloc[-1], "x", ms=4, color="black", mew=0.8, zorder=3)
    ax.set_ylim(0, 1); ax.set_xlabel("Epoch", labelpad=1); ax.set_ylabel("Held-out AP@0.5", labelpad=1)
    ax.set_title(ttl, loc="left", fontsize=7.5); ax.grid(color="#e4e3dd", lw=0.5)
    if j == 0:
        ax.plot([], [], "o", ms=3.5, mfc="white", mec="black", label="best (selected) epoch"); ax.plot([], [], "x", color="black", ms=4, label="last epoch")
        ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="lower right", handlelength=1.0, borderaxespad=0.1, columnspacing=0.6)
# (c) predicted / reference box size
bs = pd.read_csv("out/box_size_bias.csv")
ax = fig.add_subplot(gs[0, 2])
sets = ["no border, best", "16-px border, best", "no border, last", "16-px border, last"]
for i, t in enumerate(TAGS):
    sub = bs[bs.model == t].set_index("setting").loc[sets]
    xx = np.arange(4) + (i - 1) * 0.22
    ax.errorbar(xx, sub.median_ratio, yerr=[sub.median_ratio - sub.q25, sub.q75 - sub.median_ratio], fmt=MK[t], color=COL[t], ms=4.5, mec="white", mew=0.5, capsize=1.5, lw=0.8)
ax.axhline(1, color="black", lw=0.6, ls=(0, (3, 2)))
ax.set_xticks(range(4)); ax.set_xticklabels(["no\nborder", "16-px\nborder", "no\nborder", "16-px\nborder"], fontsize=6.3)
ax.text(0.5, -0.36, "best ckpt", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.3); ax.text(2.5, -0.36, "last ckpt", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.3)
ax.set_ylabel("Box-size ratio (predicted/reference)", labelpad=1); ax.set_ylim(0.8, 1.4)
ax.set_title("(C) Box-size bias", loc="left", fontsize=7.5); ax.grid(axis="y", color="#e4e3dd", lw=0.5)
# (d) AP50 under evaluation settings and (e) F1
SETS = [("Validator\ndefault", "out/native_best"), ("No border,\nbest ckpt", "out/nopad_best"), ("16-px border,\nbest ckpt", "out/pad16_best"),
        ("No border,\nlast ckpt", "out/nopad_last"), ("16-px border,\nlast ckpt\n(primary)", "out/pad16_last")]
for j, (m, ttl, yl) in enumerate([("AP50_tile", "(D) AP@0.5 (tile level) under five evaluation settings", (0.5, 1.0)),
                                 ("F1_cluster", "(E) F1 (cluster level) under the same settings", (0.5, 1.0))]):
    ax = fig.add_subplot(gs[1, :2] if j == 0 else gs[1, 2])
    if j == 0:
        for i, t in enumerate(TAGS):
            mu, sd = [], []
            for name, d in SETS:
                e = pd.read_csv(f"{d}/eval_fold_metrics.csv"); v = e[e.model == t][m].values; mu.append(v.mean()); sd.append(v.std(ddof=1))
            xx = np.arange(len(SETS)) + (i - 1) * 0.2
            ax.errorbar(xx, mu, yerr=sd, fmt=MK[t], color=COL[t], ms=5, mec="white", mew=0.5, capsize=2, lw=0.9, label=LAB1[t])
        ax.set_xticks(range(len(SETS))); ax.set_xticklabels([s[0] for s in SETS], fontsize=6.3)
        ax.axvspan(3.6, 4.4, color="#e4e3dd", alpha=0.6, lw=0)
        ax.legend(frameon=False, fontsize=6.2, loc="lower left", ncol=1)
    else:
        for i, t in enumerate(TAGS):
            mu, sd = [], []
            for name, d in [SETS[1], SETS[4]]:
                e = pd.read_csv(f"{d}/eval_fold_metrics.csv"); v = e[e.model == t][m].values; mu.append(v.mean()); sd.append(v.std(ddof=1))
            xx = np.arange(2) + (i - 1) * 0.2
            ax.errorbar(xx, mu, yerr=sd, fmt=MK[t], color=COL[t], ms=5, mec="white", mew=0.5, capsize=2, lw=0.9)
        ax.set_xticks(range(2)); ax.set_xticklabels(["No border,\nbest ckpt", "16-px border,\nlast ckpt\n(primary)"], fontsize=6.3)
        ax.axvspan(0.6, 1.4, color="#e4e3dd", alpha=0.6, lw=0)
        ttl = "(E) F1 (cluster level)"
    ax.set_ylim(*yl); ax.set_ylabel("AP@0.5" if j == 0 else "F1", labelpad=1); ax.set_title(ttl, loc="left", fontsize=7.5); ax.grid(axis="y", color="#e4e3dd", lw=0.5)
fig.savefig(OUT + "Figure3.png", dpi=400, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

# ---------------- Figure 4: thresholds and counting -----------------------------------------------
sw = pd.read_csv(f"{P}/eval_sweep.csv"); th = pd.read_csv(f"{P}/eval_thresholds.csv"); bl = pd.read_csv(f"{P}/eval_blocks.csv")
fig = plt.figure(figsize=(7.2, 4.3))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.55, wspace=0.38)
ax1 = fig.add_subplot(gs[0, 0:2] if False else gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
for t in TAGS:
    s = sw[sw.model == t]
    ax1.plot(s.thr, s.F1, color=COL[t], lw=1.2, label=LAB1[t]); ax2.plot(s.thr, s.count_ratio, color=COL[t], lw=1.2)
    lo, hi = th[th.model == t].thr_F1_crossfit.min(), th[th.model == t].thr_F1_crossfit.max()
    for ax in (ax1, ax2): ax.axvspan(lo, hi + 0.001, color=COL[t], alpha=0.18, lw=0)
ax1.set_xlabel("Confidence threshold", labelpad=1); ax1.set_ylabel("F1 (cluster level, pooled)", labelpad=1); ax1.set_ylim(0, 1); ax1.set_xlim(0, 1)
ax1.set_title("(A) F1 versus threshold", loc="left", fontsize=7.5); ax1.grid(color="#e4e3dd", lw=0.5)
ax2.axhline(1, color="black", lw=0.6, ls=(0, (3, 2)))
ax2.set_xlabel("Confidence threshold", labelpad=1); ax2.set_ylabel("Detected / reference clusters", labelpad=1); ax2.set_ylim(0, 2.5); ax2.set_xlim(0, 1)
ax2.set_title("(B) Count ratio versus threshold", loc="left", fontsize=7.5); ax2.grid(color="#e4e3dd", lw=0.5)
axl = fig.add_subplot(gs[0, 2]); axl.axis("off")
for t in TAGS: axl.plot([], [], color=COL[t], lw=1.5, label=LAB1[t])
axl.fill_between([], [], color="#b9b8b2", alpha=0.5, label="range of cross-fitted\nthresholds (5 folds)")
axl.legend(frameon=False, fontsize=6.4, loc="center left")
for i, t in enumerate(TAGS):
    ax = fig.add_subplot(gs[1, i]); g = bl[bl.model == t]; cs = count_stats(g.true.values, g.pred.values)
    jit = (np.random.default_rng(i).random(len(g)) - 0.5) * 0.25
    ax.scatter(g.true + jit, g.pred + jit[::-1], s=12, marker=MK[t], color=COL[t], edgecolor="white", lw=0.3, alpha=0.9)
    ax.plot([0, 20], [0, 20], color="black", lw=0.6, ls=(0, (3, 2))); ax.set_xlim(3, 19); ax.set_ylim(3, 19); ax.set_aspect("equal")
    ax.text(0.04, 0.96, f"MAE = {cs['MAE']:.2f}\nbias = {cs['bias_pct']:+.1f}%\nR² (1:1) = {cs['R2_11']:.2f}", transform=ax.transAxes, va="top", fontsize=6.3)
    ax.set_xlabel("Reference clusters per block", labelpad=1)
    if i == 0: ax.set_ylabel("Detected clusters per block", labelpad=1)
    ax.set_title(f"({'CDE'[i]}) {LAB1[t].replace(' (one-to-many head + NMS)', ' + NMS').replace(' (NMS-free head)', ', NMS-free')}", loc="left", fontsize=7.5); ax.grid(color="#e4e3dd", lw=0.5)
fig.savefig(OUT + "Figure4.png", dpi=400, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
print("ok")
