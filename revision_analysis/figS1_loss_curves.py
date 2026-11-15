"""Figure S1: epoch-wise training and held-out loss of every fold model, with best and last epochs."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})
TR, HO, GRID, INK2 = "#2a78d6", "#eb6834", "#e4e3dd", "#52514e"
M = [("YOLO26n", "yolo26_results/runs/yolo26n_fold{}/results.csv", ["box_loss", "cls_loss", "l1_loss"]),
     ("RT-DETR-l", "Rt_DETER/runs/rtdetr-l_fold{}/results.csv", ["giou_loss", "cls_loss", "l1_loss"])]
fig, axes = plt.subplots(2, 5, figsize=(7.1, 3.6), sharey="row")
rows = []
for i, (name, pat, comp) in enumerate(M):
    for k in range(5):
        r = pd.read_csv(RUNS_DIR + pat.format(k)); r.columns = [c.strip() for c in r.columns]
        tr = r[[f"train/{c}" for c in comp]].sum(axis=1, min_count=3); ho = r[[f"val/{c}" for c in comp]].sum(axis=1, min_count=3)
        e_best = int(r.loc[r["metrics/mAP50-95(B)"].idxmax(), "epoch"]); e_last = int(r.epoch.max())
        ax = axes[i, k]
        ax.plot(r.epoch, tr, color=TR, lw=1.0, label="Training loss")
        ax.plot(r.epoch, ho, color=HO, lw=1.0, ls=(0, (3, 1.2)), label="Held-out loss")
        yb = ho[r.epoch == e_best].iloc[0]; yl = ho[r.epoch == e_last].iloc[0]
        ax.plot(e_best, yb, "o", ms=4.2, mfc="white", mec="#0b0b0b", mew=0.8, zorder=5, label="Best checkpoint")
        ax.plot(e_last, yl, "x", ms=4.6, color="#0b0b0b", mew=0.9, zorder=5, label="Last checkpoint")
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(f"({'ABCDEFGHIJ'[i * 5 + k]}) {name}, fold {k}", loc="left", fontsize=6.6)
        ax.set_xlim(-4, 156 if i == 0 else 84); ax.set_xticks([0, 50, 100, 150] if i == 0 else [0, 25, 50, 75])
        if i == 1: ax.set_xlabel("Epoch")
        if k == 0: ax.set_ylabel(f"Total loss ({' + '.join(c.replace('_loss', '') for c in comp)})", fontsize=6.0)
        rows.append(dict(model=name, fold=k, best_epoch=e_best, last_epoch=e_last, train_first=tr.iloc[0], train_last=tr.iloc[-1],
                         heldout_min=ho.min(), heldout_min_epoch=int(r.epoch[ho.idxmin()]), heldout_best=yb, heldout_last=yl))
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.8, w_pad=1.3)
fig.savefig(FIGS_DIR + "FigureS1_loss_curves.png", dpi=600, bbox_inches="tight")
fig.savefig(FIGS_DIR + "FigureS1_loss_curves.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
df = pd.DataFrame(rows); os.makedirs("out/r2", exist_ok=True); df.to_csv("out/r2/loss_summary.csv", index=False); print(df.round(3).to_string(index=False))
