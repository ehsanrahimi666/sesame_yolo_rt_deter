"""Exact valid (alpha > 0) area of the orthomosaic and a 1-m validity grid."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import numpy as np, rasterio, json
from rasterio.windows import Window
with rasterio.open(ORTHO_TIF) as ds:
    tr = ds.transform; n = 0
    for _, w in ds.block_windows(4):
        n += int((ds.read(4, window=w) > 0).sum())
    area = n * abs(tr.a) * abs(tr.e)
    # 1-m validity fraction grid (500 x 500 px blocks at 2 mm)
    W, H = ds.width, ds.height; g = 500
    nx, ny = int(np.ceil(W / g)), int(np.ceil(H / g)); frac = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            a = ds.read(4, window=Window(i * g, j * g, min(g, W - i * g), min(g, H - j * g)))
            frac[j, i] = (a > 0).mean() * (a.size / (g * g))
np.save("out/valid_frac_1m.npy", frac)
json.dump({"valid_pixels": n, "valid_area_m2": area, "grid_origin": [tr.c, tr.f], "grid_cell_px": g, "grid_cell_m": g * abs(tr.a)}, open("out/valid_area.json", "w"), indent=1)
print(n, round(area, 2), frac.shape, round(frac.sum() * (g * abs(tr.a)) * (g * abs(tr.e)), 2))
