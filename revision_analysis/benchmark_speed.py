#!/usr/bin/env python
"""CPU inference benchmark (batch 1) for the three configurations under the evaluation protocol
(512-px tile -> 640 px + 16-px border = 672-px input). Reports mean and SD per tile over 100 tiles."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import time, json, platform, numpy as np, cv2, torch, glob, os
import ultralytics
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import get_flops
torch.set_num_threads(os.cpu_count())
imgs = sorted(glob.glob(DATA_DIR + "images/*.png"))
rng = np.random.default_rng(0); sel = [imgs[i] for i in rng.choice(len(imgs), 100, replace=False)]
def prep(p):
    im = cv2.resize(cv2.imread(p), (640, 640), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(im, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=(114, 114, 114))
X = [prep(p) for p in sel]
out = {"cpu": platform.processor() or "", "torch": torch.__version__, "ultralytics": ultralytics.__version__, "threads": torch.get_num_threads()}
try:
    out["cpu_model"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
except Exception: pass
for tag, w, cls, extra in [("yolo26n-e2e", "weights_yolo26n_final.pt", YOLO, {"end2end": True}),
                          ("yolo26n-nms", "weights_yolo26n_final.pt", YOLO, {"end2end": False}),
                          ("rtdetr-l", "weights_rtdetr-l_final.pt", RTDETR, {})]:
    M = cls(str(WEIGHTS_BEST / w))
    nparam = sum(p.numel() for p in M.model.parameters())
    gf = get_flops(M.model, imgsz=672)
    for x in X[:10]:
        M.predict(source=x, imgsz=672, conf=0.25, device="cpu", verbose=False, **extra)
    tt, pre, inf, post = [], [], [], []
    for x in X:
        t0 = time.perf_counter(); r = M.predict(source=x, imgsz=672, conf=0.25, device="cpu", verbose=False, **extra)[0]; tt.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
    out[tag] = dict(params=int(nparam), GFLOPs_672=round(gf, 2), total_ms_mean=float(np.mean(tt)), total_ms_sd=float(np.std(tt, ddof=1)),
                    inference_ms_mean=float(np.mean(inf)), preprocess_ms_mean=float(np.mean(pre)), postprocess_ms_mean=float(np.mean(post)),
                    tiles_per_s=1000 / float(np.mean(tt)))
    print(tag, json.dumps(out[tag]), flush=True)
json.dump(out, open("out/benchmark_cpu.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}))
