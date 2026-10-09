#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_train_yolo_colab.py
======================
YOLO26 detection + counting for sesame flowers, with 5-fold SPATIALLY BLOCKED
cross-validation. Runs unchanged on Google Colab and Kaggle.

MODEL
-----
YOLO26n: the current Ultralytics generation, end-to-end / NMS-free, nano scale.
Weights download automatically on first use. Five folds take roughly 15-25 min on a
T4 for a dataset this size. If you later want a within-YOLO26 scale ablation, add
"yolo26s" to MODELS -- every table and figure below adapts automatically.

OUTPUTS (in OUT_DIR, mirrored to Drive on Colab)
    cv_results.csv          every fold: mAP50, mAP50-95, P, R, F1, counting stats
    cv_summary.csv          mean +/- SD across folds
    model_comparison.csv    accuracy, params, FPS, best confidence threshold
    fig_accuracy.png        mAP@0.5 and mAP@0.5:0.95 with individual folds
    fig_precision_recall.png
    fig_fold_heatmap.png    mAP50 by spatial fold (the spatial-variation figure)
    fig_tradeoff.png        accuracy vs speed, bubble area = parameters
    fig_counts.png          predicted vs annotated flowers per tile
    fig_threshold.png       F1 and counting error vs confidence threshold
    fig_examples.png        example validation tiles, GT (green) vs prediction (blue)
    runs/                   Ultralytics native plots (PR curve, confusion matrix, curves)
    weights_yolo26n_fold*.pt
    weights_yolo26n_final.pt    trained on ALL tiles -> feed this to script 03
"""

import io
import json
import time
import shutil
import zipfile
import contextlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
CFG = {
    "ZIP_NAME": "sesame_dataset_v1.zip",
    "DRIVE_SUBDIR": "sesame",              # Colab: MyDrive/<DRIVE_SUBDIR>/<ZIP_NAME>

    "MODELS": ["yolo26n"],                 # YOLO26 only; add "yolo26s" for a scale ablation
    "N_FOLDS": 5,
    "EPOCHS": 150,
    "PATIENCE": 50,                        # generous: see the early-stopping lesson
    "IMGSZ": 640,                          # 512 px tiles are upscaled -> helps small objects
    "BATCH": 16,
    "SEED": 0,

    "PRED_CONF": 0.01,                     # predict low, threshold afterwards in numpy
    "PRED_IOU": 0.60,
    "SCORE_THR": 0.50,                     # reporting operating point
    "IOU_MATCH": 0.50,

    "TRAIN_FINAL_MODEL": True,             # retrain best model on all tiles
    "FINAL_MODEL": None,                   # None = pick the best mAP50; or e.g. "yolo26n"
}
# =============================================================================


def setup():
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"],
                       check=True)
    if Path("/kaggle/input").exists():
        env = "kaggle"
        work, out, drive_out = Path("/kaggle/working/data"), Path("/kaggle/working/results"), None
        src = next((p for p in Path("/kaggle/input").rglob(CFG["ZIP_NAME"])), None)
        if src is None:
            hit = next((p for p in Path("/kaggle/input").rglob("tiles.csv")), None)
            if hit is not None:
                out.mkdir(parents=True, exist_ok=True)
                return env, hit.parent, out, drive_out
    else:
        env = "colab"
        try:
            from google.colab import drive
            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive")
        except Exception:
            pass
        work, out = Path("/content/data"), Path("/content/results")
        drive_out = Path("/content/drive/MyDrive") / CFG["DRIVE_SUBDIR"] / "results"
        src = Path("/content/drive/MyDrive") / CFG["DRIVE_SUBDIR"] / CFG["ZIP_NAME"]

    if src is None or not Path(src).exists():
        raise FileNotFoundError(f"{CFG['ZIP_NAME']} not found (looked at {src}).")
    work.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    root = work / Path(CFG["ZIP_NAME"]).stem
    if not (root / "tiles.csv").exists():
        print(f"Unzipping {src} ...")
        with zipfile.ZipFile(src) as z:
            z.extractall(work)
        if not (root / "tiles.csv").exists():
            root = next(work.rglob("tiles.csv")).parent
    return env, root, out, drive_out


def read_tiles(root):
    rows = []
    with open(root / "tiles.csv", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split(",")
        for line in f:
            v = line.rstrip("\n").split(",")
            rows.append(dict(zip(hdr, v)))
    for r in rows:
        r["fold"] = int(r["fold"])
        r["n_flowers"] = int(r["n_flowers"])
        r["path"] = str((root / "images" / r["file_name"]).resolve())
        r["label"] = str((root / "labels" / (r["file_name"][:-4] + ".txt")).resolve())
    return rows


def write_fold_yaml(root, tiles, fold, out_dir, class_name="sesame_flower"):
    d = out_dir / "folds" / f"fold{fold}"
    d.mkdir(parents=True, exist_ok=True)
    tr = [t["path"] for t in tiles if t["fold"] != fold]
    va = [t["path"] for t in tiles if t["fold"] == fold]
    (d / "train.txt").write_text("\n".join(tr), encoding="utf-8")
    (d / "val.txt").write_text("\n".join(va), encoding="utf-8")
    y = d / "data.yaml"
    y.write_text(
        f"path: {root.resolve()}\n"
        f"train: {(d/'train.txt').resolve()}\n"
        f"val: {(d/'val.txt').resolve()}\n"
        f"nc: 1\nnames:\n  0: {class_name}\n", encoding="utf-8")
    return y, tr, va


def load_gt(label_path, imgw, imgh):
    p = Path(label_path)
    if not p.exists() or not p.read_text().strip():
        return np.zeros((0, 4), np.float32)
    b = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        xc, yc, w, h = (float(v) for v in parts[1:5])
        b.append([(xc - w / 2) * imgw, (yc - h / 2) * imgh,
                  (xc + w / 2) * imgw, (yc + h / 2) * imgh])
    return np.array(b, np.float32).reshape(-1, 4)


def match_pr(store, thr, iou_thr):
    tp = fp = fn = 0
    for d in store.values():
        keep = d["scores"] >= thr
        pb = d["boxes"][keep][np.argsort(-d["scores"][keep])]
        gb = d["gt"]
        used = np.zeros(len(gb), bool)
        for b in pb:
            best, best_iou = -1, iou_thr
            for gi, g in enumerate(gb):
                if used[gi]:
                    continue
                iw = max(0.0, min(b[2], g[2]) - max(b[0], g[0]))
                ih = max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
                inter = iw * ih
                if inter <= 0:
                    continue
                iou = inter / ((b[2]-b[0])*(b[3]-b[1]) + (g[2]-g[0])*(g[3]-g[1]) - inter)
                if iou >= best_iou:
                    best, best_iou = gi, iou
            if best >= 0:
                used[best] = True; tp += 1
            else:
                fp += 1
        fn += int((~used).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def count_stats(store, thr):
    pred = np.array([int((d["scores"] >= thr).sum()) for d in store.values()], float)
    true = np.array([len(d["gt"]) for d in store.values()], float)
    err = pred - true
    sst = float(((true - true.mean()) ** 2).sum())
    return {"count_mae": float(np.abs(err).mean()),
            "count_rmse": float(np.sqrt((err ** 2).mean())),
            "count_bias_pct": float(100 * err.sum() / max(true.sum(), 1)),
            "count_r2": float(1 - ((true - pred) ** 2).sum() / sst) if sst > 0 else np.nan,
            "pred_total": float(pred.sum()), "true_total": float(true.sum()),
            "_pred": pred, "_true": true}


# =============================================================================
def main():
    from ultralytics import YOLO
    import torch

    env, root, out_dir, drive_out = setup()
    dev = 0 if torch.cuda.is_available() else "cpu"
    print(f"env={env}  data={root}  device={dev}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tiles = read_tiles(root)
    folds = sorted({t["fold"] for t in tiles})
    print(f"{len(tiles)} tiles, {sum(t['n_flowers'] for t in tiles)} flower boxes")
    for k in folds:
        sub = [t for t in tiles if t["fold"] == k]
        print(f"  fold {k}: {len(sub):>4} tiles, {sum(t['n_flowers'] for t in sub):>5} flowers")

    yamls = {k: write_fold_yaml(root, tiles, k, out_dir)[0] for k in folds}
    rows, stores, speeds, params = [], {}, {}, {}

    for name in CFG["MODELS"]:
        print(f"\n{'='*62}\nMODEL: {name}\n{'='*62}")
        for k in folds:
            t0 = time.time()
            model = YOLO(f"{name}.pt")
            params[name] = sum(p.numel() for p in model.model.parameters())
            res = model.train(
                data=str(yamls[k]), epochs=CFG["EPOCHS"], imgsz=CFG["IMGSZ"],
                batch=CFG["BATCH"], patience=CFG["PATIENCE"], seed=CFG["SEED"],
                project=str(out_dir / "runs"), name=f"{name}_fold{k}", exist_ok=True,
                pretrained=True, plots=True, verbose=False, device=dev, val=True,
                degrees=45.0, flipud=0.5, fliplr=0.5, scale=0.3,
                mosaic=1.0, close_mosaic=10)
            best_w = Path(res.save_dir) / "weights" / "best.pt"
            model = YOLO(str(best_w))
            shutil.copy(best_w, out_dir / f"weights_{name}_fold{k}.pt")

            with contextlib.redirect_stdout(io.StringIO()):
                mv = model.val(data=str(yamls[k]), imgsz=CFG["IMGSZ"], device=dev, verbose=False)
            m = {"model": name, "fold": k,
                 "mAP50": float(mv.box.map50), "mAP50_95": float(mv.box.map),
                 "P_ultra": float(mv.box.mp), "R_ultra": float(mv.box.mr)}

            va = [t for t in tiles if t["fold"] == k]
            store, inf_ms = {}, []
            for r in model.predict(source=[t["path"] for t in va], conf=CFG["PRED_CONF"],
                                   iou=CFG["PRED_IOU"], imgsz=CFG["IMGSZ"], device=dev,
                                   verbose=False, stream=True):
                stem = Path(r.path).name
                tt = next(x for x in va if x["file_name"] == stem)
                h, w = r.orig_shape
                store[stem] = {"boxes": r.boxes.xyxy.cpu().numpy(),
                               "scores": r.boxes.conf.cpu().numpy(),
                               "gt": load_gt(tt["label"], w, h)}
                inf_ms.append(float(r.speed.get("inference", np.nan)))
            speeds.setdefault(name, []).append(float(np.nanmean(inf_ms)))
            stores[(name, k)] = store

            m.update(match_pr(store, CFG["SCORE_THR"], CFG["IOU_MATCH"]))
            cs = count_stats(store, CFG["SCORE_THR"])
            m.update({kk: vv for kk, vv in cs.items() if not kk.startswith("_")})
            m["fps"] = 1000.0 / max(np.nanmean(inf_ms), 1e-6)
            m["params_M"] = params[name] / 1e6
            m["train_min"] = (time.time() - t0) / 60
            m["n_train"] = sum(1 for t in tiles if t["fold"] != k)
            m["n_val"] = len(va)
            rows.append(m)
            print(f"  fold {k}: mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50_95']:.4f}  "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
                  f"countMAE={m['count_mae']:.2f} R2={m['count_r2']:.3f}  "
                  f"({m['train_min']:.1f} min)")
            del model
            torch.cuda.empty_cache()

    # ---- tables --------------------------------------------------------------
    keys = [k for k in rows[0] if k not in ("model", "fold")]
    with open(out_dir / "cv_results.csv", "w", encoding="utf-8") as f:
        f.write("model,fold," + ",".join(keys) + "\n")
        for r in rows:
            f.write(f"{r['model']},{r['fold']}," +
                    ",".join(f"{r[k]:.6f}" for k in keys) + "\n")
    with open(out_dir / "cv_summary.csv", "w", encoding="utf-8") as f:
        f.write("model,metric,mean,sd\n")
        for name in CFG["MODELS"]:
            sub = [r for r in rows if r["model"] == name]
            for k in keys:
                v = np.array([r[k] for r in sub], float)
                f.write(f"{name},{k},{np.nanmean(v):.6f},{np.nanstd(v, ddof=1):.6f}\n")

    # best confidence threshold per model, by mean F1 across folds
    thrs = np.arange(0.05, 0.96, 0.05)
    best_thr, sweep = {}, {}
    for name in CFG["MODELS"]:
        f1s = [np.mean([match_pr(stores[(name, k)], t, CFG["IOU_MATCH"])["f1"] for k in folds])
               for t in thrs]
        maes = [np.mean([count_stats(stores[(name, k)], t)["count_mae"] for k in folds])
                for t in thrs]
        sweep[name] = (np.array(f1s), np.array(maes))
        best_thr[name] = float(thrs[int(np.argmax(f1s))])

    with open(out_dir / "model_comparison.csv", "w", encoding="utf-8") as f:
        f.write("model,params_M,mAP50_mean,mAP50_sd,mAP50_95_mean,precision_mean,recall_mean,"
                "f1_mean,count_mae_mean,count_r2_mean,fps,best_conf\n")
        for name in CFG["MODELS"]:
            sub = [r for r in rows if r["model"] == name]
            g = lambda k: np.nanmean([r[k] for r in sub])
            f.write(f"{name},{params[name]/1e6:.3f},{g('mAP50'):.4f},"
                    f"{np.nanstd([r['mAP50'] for r in sub], ddof=1):.4f},{g('mAP50_95'):.4f},"
                    f"{g('precision'):.4f},{g('recall'):.4f},{g('f1'):.4f},"
                    f"{g('count_mae'):.3f},{g('count_r2'):.4f},"
                    f"{1000/np.nanmean(speeds[name]):.1f},{best_thr[name]:.2f}\n")

    print("\n--- CROSS-VALIDATED SUMMARY ------------------------------------")
    for name in CFG["MODELS"]:
        sub = [r for r in rows if r["model"] == name]
        print(f"  {name:<10} mAP50 {np.mean([r['mAP50'] for r in sub]):.4f} "
              f"+/- {np.std([r['mAP50'] for r in sub], ddof=1):.4f}   "
              f"F1 {np.mean([r['f1'] for r in sub]):.3f}   "
              f"count R2 {np.mean([r['count_r2'] for r in sub]):.3f}   "
              f"{1000/np.nanmean(speeds[name]):.1f} FPS   best conf {best_thr[name]:.2f}")

    # ---- figures -------------------------------------------------------------
    M = CFG["MODELS"]
    x = np.arange(len(M))

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, key, lab in zip(axs, ("mAP50", "mAP50_95"), ("(a) mAP@0.5", "(b) mAP@0.5:0.95")):
        mu = [np.mean([r[key] for r in rows if r["model"] == n]) for n in M]
        sd = [np.std([r[key] for r in rows if r["model"] == n], ddof=1) for n in M]
        ax.bar(x, mu, yerr=sd, capsize=4, color="#4C78A8", alpha=.85)
        for i, n in enumerate(M):
            ax.scatter(np.full(len(folds), i), [r[key] for r in rows if r["model"] == n],
                       color="#E45756", s=16, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(M); ax.set_title(lab)
        ax.set_ylabel(key); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "fig_accuracy.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.9 * len(M) + 2.2, 3.6))
    w = 0.35
    for j, (key, col, lab) in enumerate([("precision", "#4C78A8", "Precision"),
                                         ("recall", "#E4823A", "Recall")]):
        mu = [np.mean([r[key] for r in rows if r["model"] == n]) for n in M]
        sd = [np.std([r[key] for r in rows if r["model"] == n], ddof=1) for n in M]
        ax.bar(x + (j - .5) * w, mu, w, yerr=sd, capsize=4, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(M); ax.set_ylim(0, 1)
    ax.set_ylabel(f"score (mean +/- SD, IoU {CFG['IOU_MATCH']})")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "fig_precision_recall.png", dpi=220); plt.close(fig)

    H = np.array([[next(r["mAP50"] for r in rows if r["model"] == n and r["fold"] == k)
                   for k in folds] for n in M])
    fig, ax = plt.subplots(figsize=(1.3 * len(folds) + 3, 0.75 * len(M) + 2.2))
    im = ax.imshow(H, cmap="RdYlGn", aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H[i,j]:.3f}", ha="center", va="center", fontsize=9, weight="bold")
    ax.set_xticks(range(len(folds))); ax.set_xticklabels([f"Fold {k}" for k in folds])
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M)
    ax.set_title("mAP@0.5 by model and spatial fold")
    fig.colorbar(im, ax=ax, shrink=.85, label="mAP@0.5")
    fig.tight_layout(); fig.savefig(out_dir / "fig_fold_heatmap.png", dpi=220); plt.close(fig)
    if len(M) > 1:
        sd_models = float(H.std(axis=0, ddof=1).mean())   # spread across models, same fold
        sd_folds = float(H.std(axis=1, ddof=1).mean())    # spread across folds, same model
        print(f"\nVariation among models (mean SD) : {sd_models:.4f}")
        print(f"Variation among folds  (mean SD) : {sd_folds:.4f}")
        print("  -> " + ("spatial position dominates" if sd_folds > sd_models
                         else "architecture dominates"))

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for i, n in enumerate(M):
        sub = [r for r in rows if r["model"] == n]
        mu, sd = np.mean([r["mAP50"] for r in sub]), np.std([r["mAP50"] for r in sub], ddof=1)
        fps = 1000 / np.nanmean(speeds[n])
        ax.errorbar(fps, mu, yerr=sd, fmt="none", ecolor="grey", capsize=3, zorder=1)
        ax.scatter(fps, mu, s=120 + 400 * params[n] / 1e7, zorder=2,
                   color=plt.cm.viridis(i / max(len(M) - 1, 1)), edgecolor="k", lw=.6)
        ax.annotate(f"{n}\n{params[n]/1e6:.2f} M", (fps, mu), textcoords="offset points",
                    xytext=(8, 8), fontsize=8)
    ax.set_xlabel("inference speed (FPS)"); ax.set_ylabel("mAP@0.5 (mean +/- SD)")
    ax.set_title("Accuracy-efficiency trade-off (bubble area ~ parameters)")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(out_dir / "fig_tradeoff.png", dpi=220); plt.close(fig)

    fig, axs = plt.subplots(1, len(M), figsize=(4.2 * len(M), 4.0), squeeze=False)
    for i, n in enumerate(M):
        pr = np.concatenate([count_stats(stores[(n, k)], best_thr[n])["_pred"] for k in folds])
        tr = np.concatenate([count_stats(stores[(n, k)], best_thr[n])["_true"] for k in folds])
        ax = axs[0][i]
        ax.scatter(tr, pr, s=16, alpha=.55, color="#4C78A8", edgecolor="none")
        lim = [0, max(tr.max(), pr.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
        r2 = 1 - ((tr - pr) ** 2).sum() / ((tr - tr.mean()) ** 2).sum()
        ax.set_title(f"{n}  (R2 = {r2:.3f}, conf {best_thr[n]:.2f})")
        ax.set_xlabel("annotated flowers per tile"); ax.set_ylabel("detected flowers per tile")
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "fig_counts.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax2 = ax.twinx()
    for i, n in enumerate(M):
        f1s, maes = sweep[n]
        ax.plot(thrs, f1s, "-o", ms=3.5, label=f"{n} F1",
                color=plt.cm.viridis(i / max(len(M) - 1, 1)))
        ax2.plot(thrs, maes, "--s", ms=3.5, alpha=.7,
                 color=plt.cm.viridis(i / max(len(M) - 1, 1)))
        ax.axvline(best_thr[n], color="grey", ls=":", lw=1)
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("F1 (solid)")
    ax2.set_ylabel("counting MAE, flowers/tile (dashed)")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("Operating point")
    fig.tight_layout(); fig.savefig(out_dir / "fig_threshold.png", dpi=220); plt.close(fig)

    # example tiles: GT green, prediction blue
    from PIL import Image, ImageDraw
    n0 = M[0]
    picks = sorted(stores[(n0, folds[0])].items(),
                   key=lambda kv: -len(kv[1]["gt"]))[:4]
    if picks:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        for ax, (stem, d) in zip(axs.ravel(), picks):
            im = Image.open(root / "images" / stem).convert("RGB")
            dr = ImageDraw.Draw(im)
            for g in d["gt"]:
                dr.rectangle(list(g), outline=(60, 220, 60), width=2)
            for b, s in zip(d["boxes"], d["scores"]):
                if s >= best_thr[n0]:
                    dr.rectangle(list(b), outline=(60, 120, 255), width=2)
            ax.imshow(im); ax.axis("off")
            ax.set_title(f"{len(d['gt'])} annotated / "
                         f"{int((d['scores']>=best_thr[n0]).sum())} detected", fontsize=9)
        fig.suptitle(f"{n0}: annotated (green) vs detected (blue)", y=.99)
        fig.tight_layout(); fig.savefig(out_dir / "fig_examples.png", dpi=200); plt.close(fig)

    # ---- final model on all tiles -------------------------------------------
    if CFG["TRAIN_FINAL_MODEL"]:
        pick = CFG["FINAL_MODEL"] or max(
            M, key=lambda n: np.mean([r["mAP50"] for r in rows if r["model"] == n]))
        print(f"\n=== FINAL MODEL: {pick} trained on all {len(tiles)} tiles ===")
        d = out_dir / "folds" / "all"; d.mkdir(parents=True, exist_ok=True)
        allp = [t["path"] for t in tiles]
        (d / "train.txt").write_text("\n".join(allp), encoding="utf-8")
        (d / "val.txt").write_text("\n".join(allp[: max(8, len(allp) // 10)]), encoding="utf-8")
        (d / "data.yaml").write_text(
            f"path: {root.resolve()}\ntrain: {(d/'train.txt').resolve()}\n"
            f"val: {(d/'val.txt').resolve()}\nnc: 1\nnames:\n  0: sesame_flower\n",
            encoding="utf-8")
        model = YOLO(f"{pick}.pt")
        res = model.train(data=str(d / "data.yaml"), epochs=CFG["EPOCHS"], imgsz=CFG["IMGSZ"],
                          batch=CFG["BATCH"], patience=CFG["PATIENCE"], seed=CFG["SEED"],
                          project=str(out_dir / "runs"), name=f"{pick}_final", exist_ok=True,
                          pretrained=True, plots=True, verbose=False, device=dev, val=True,
                          degrees=45.0, flipud=0.5, fliplr=0.5, scale=0.3,
                          mosaic=1.0, close_mosaic=10)
        shutil.copy(Path(res.save_dir) / "weights" / "best.pt",
                    out_dir / f"weights_{pick}_final.pt")
        (out_dir / "final_model_info.json").write_text(json.dumps(
            {"model": pick, "weights": f"weights_{pick}_final.pt",
             "best_conf": best_thr[pick], "imgsz": CFG["IMGSZ"],
             "params_M": params[pick] / 1e6}, indent=2), encoding="utf-8")
        print(f"Saved weights_{pick}_final.pt   (use conf={best_thr[pick]:.2f} in script 03)")

    if drive_out is not None:
        drive_out.mkdir(parents=True, exist_ok=True)
        for p in out_dir.rglob("*"):
            if p.is_file() and p.stat().st_size < 200e6:
                tgt = drive_out / p.relative_to(out_dir)
                tgt.parent.mkdir(parents=True, exist_ok=True)
                tgt.write_bytes(p.read_bytes())
        print(f"Copied results to {drive_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
