"""Visual check of map detections outside the annotated area: random sample of YOLO26n + NMS detections
(stitched map, threshold 0.38) and of detections on bare ground (>= 10 cm from vegetation)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa
import sys, json
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.windows import Window
from shapely import contains_xy
from PIL import Image, ImageDraw, ImageFont
from stitch import X0, Y0, RX, RY
os.makedirs("out/r2", exist_ok=True)
field = gpd.read_file(GIS_DIR + "Field_border.shp").geometry.union_all()
veg = np.load("out/r2/veg_mask_1cm.npy"); dist = np.load("out/r2/dist_veg_1cm.npy")       # from nonvegetated_detections.py
K = pd.read_csv("out/r2/w2w/w2w_yolo26n-nms_stitched.csv"); K = K[K.score >= 0.38].reset_index(drop=True)
out_ann = ~contains_xy(field, K.x.values, K.y.values)
c1 = np.floor((K.x.values - X0) / (RX * 5)).astype(int); r1 = np.floor((Y0 - K.y.values) / (RY * 5)).astype(int)
bare = (~veg[r1, c1]) & (dist[r1, c1] >= 10)
pop_rand = np.where(out_ann)[0]; pop_bare = np.where(out_ann & bare)[0]
rs = np.random.default_rng(2026).choice(pop_rand, 100, replace=False)
bs = np.random.default_rng(2027).choice(np.setdiff1d(pop_bare, rs), 20, replace=False)
S = pd.concat([K.loc[rs].assign(sample="random"), K.loc[bs].assign(sample="bare ground")]).reset_index().rename(columns={"index": "det_id"})
S.insert(0, "chip", np.arange(1, len(S) + 1))
S.to_csv("out/r2/visual_check_sample.csv", index=False)
print("population outside annotated area:", len(pop_rand), "bare-ground (>=10 cm):", len(pop_bare), "of", len(K))
H = 75; SC = 3   # 30 x 30 cm window, upscaled x3
ds = rasterio.open(ORTHO_TIF)
chips = []
for r in S.itertuples():
    cxp = (r.px0 + r.px1) / 2; cyp = (r.py0 + r.py1) / 2
    win = Window(int(cxp - H), int(cyp - H), 2 * H, 2 * H)
    a = ds.read([1, 2, 3], window=win, boundless=True, fill_value=0)
    im = Image.fromarray(np.transpose(a, (1, 2, 0))).resize((2 * H * SC, 2 * H * SC), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    x0 = (r.px0 - (int(cxp - H))) * SC; x1 = (r.px1 - (int(cxp - H))) * SC; y0 = (r.py0 - int(cyp - H)) * SC; y1 = (r.py1 - int(cyp - H)) * SC
    L = 22; x0 -= 6; y0 -= 6; x1 += 6; y1 += 6
    for col, wd in [((255, 255, 255), 9), ((255, 0, 255), 5)]:
        for (px, py, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
            d.line([(px, py), (px + dx * L, py)], fill=col, width=wd); d.line([(px, py), (px, py + dy * L)], fill=col, width=wd)
    chips.append(im)
try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26); fsm = ImageFont.truetype("DejaVuSans.ttf", 22)
except Exception: font = fsm = ImageFont.load_default()
cw = 2 * H * SC; pad = 40; cols, rowsn = 4, 5
pages = []
for p0 in range(0, len(chips), cols * rowsn):
    page = Image.new("RGB", (cols * (cw + 20) + 20, rowsn * (cw + pad + 10) + 90), "white"); d = ImageDraw.Draw(page)
    d.text((20, 20), "Is the object inside the magenta corners a flower cluster on a sesame plant?  Window 30 x 30 cm.", fill="black", font=fsm)
    for k, im in enumerate(chips[p0:p0 + cols * rowsn]):
        i = p0 + k; x = 20 + (k % cols) * (cw + 20); y = 70 + (k // cols) * (cw + pad + 10)
        d.text((x, y), f"{i + 1}", fill="black", font=font)
        page.paste(im, (x, y + pad))
    pages.append(page)
pages[0].save("out/r2/visual_check_sheet.pdf", save_all=True, append_images=pages[1:], resolution=150)
print("pages", len(pages))
