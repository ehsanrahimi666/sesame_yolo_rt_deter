"""Detections on non-vegetated ground (bare soil and plastic mulch) away from the crop canopy.
Vegetation is mapped from the 1-cm orthomosaic preview with the excess-green index (ExG = 2g - r - b of
chromatic coordinates) and Otsu's threshold; detections of the wall-to-wall maps (final models, cross-validated
thresholds) are counted where their centre lies on non-vegetated ground at least d cm from any vegetation pixel."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa
import numpy as np, pandas as pd, json
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
B = PREVIEW_DIR                       # ortho_1cm.jpg and ortho_1cm_alpha.png from make_previews.py
X0, Y0 = 480919.25018164766, 4044525.1466317894; PX = 0.0019999573967093604 * 5; PY = 0.0019999711682572143 * 5
rgb = np.asarray(Image.open(B + "ortho_1cm.jpg").convert("RGB")).astype(np.float64)
valid = np.asarray(Image.open(B + "ortho_1cm_alpha.png")) > 0
s = rgb.sum(2) + 1e-9; r, g, b = rgb[..., 0] / s, rgb[..., 1] / s, rgb[..., 2] / s
exg = 2 * g - r - b
v = exg[valid]
# Otsu threshold
hist, edges = np.histogram(v, bins=512); c = (edges[:-1] + edges[1:]) / 2
w0 = np.cumsum(hist); w1 = w0[-1] - w0; m0 = np.cumsum(hist * c) / np.maximum(w0, 1); m1 = (np.sum(hist * c) - np.cumsum(hist * c)) / np.maximum(w1, 1)
t_otsu = c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]
veg = (exg > t_otsu) & valid
veg = ndi.binary_opening(veg, iterations=1)                # remove isolated specks
dist = ndi.distance_transform_edt(~veg)                    # cm (1 px = 1 cm) to nearest vegetation pixel
dark = rgb.mean(2) < 60                                     # black polyethylene mulch
res = {"exg_otsu": float(t_otsu), "valid_m2": float(valid.sum() * PX * PY), "veg_frac": float(veg.sum() / valid.sum())}
import sys
STITCHED = "--global-nms-only" not in sys.argv      # default: maps after seam-aware stitching
dets = {t: pd.read_csv((f"out/r2/w2w/w2w_{t}_stitched.csv") if STITCHED else str(ROOT / "results" / "w2w" / f"w2w_{t}.csv"))
        for t in ["yolo26n-nms", "rtdetr-l", "yolo26n-e2e"]}
THR = {"yolo26n-nms": 0.38, "rtdetr-l": 0.61, "yolo26n-e2e": 0.15}
rows = []
for d in [5, 10, 20]:
    zone = valid & ~veg & (dist >= d)
    area = zone.sum() * PX * PY
    for t, D in dets.items():
        K = D[D.score >= THR[t]]
        cc = np.floor((K.x.values - X0) / PX).astype(int); rr = np.floor((Y0 - K.y.values) / PY).astype(int)
        ok = (rr >= 0) & (rr < zone.shape[0]) & (cc >= 0) & (cc < zone.shape[1])
        inz = np.zeros(len(K), bool); inz[ok] = zone[rr[ok], cc[ok]]
        on_mulch = np.zeros(len(K), bool); on_mulch[ok] = dark[rr[ok], cc[ok]]
        rows.append(dict(min_dist_cm=d, zone_m2=area, zone_frac_valid=zone.sum() / valid.sum(), model=t, n_in_zone=int(inz.sum()),
                         per_m2=inz.sum() / area, n_total=len(K), dens_total=len(K) / res["valid_m2"], pct_of_total=100 * inz.sum() / len(K),
                         n_in_zone_mulch=int((inz & on_mulch).sum())))
os.makedirs("out/r2", exist_ok=True)
R = pd.DataFrame(rows); R.to_csv("out/r2/" + ("nonvegetated_detections.csv" if STITCHED else "nonvegetated_detections_global_nms_only.csv"), index=False)
np.save("out/r2/veg_mask_1cm.npy", veg); np.save("out/r2/dist_veg_1cm.npy", dist.astype(np.float32))
print(json.dumps(res)); pd.set_option("display.width", 220); print(R.round(4).to_string(index=False))
# quick-look image of the 10-cm zone
zone = valid & ~veg & (dist >= 10)
q = (rgb * 0.6).astype(np.uint8); q[zone] = [255, 0, 255]
Image.fromarray(q).resize((1000, 840)).save("out/r2/nonvegetated_zone_preview.jpg", quality=85)
