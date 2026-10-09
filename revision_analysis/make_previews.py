"""Low-resolution previews of the orthomosaic used in Figure 1 (1-cm overview and 5-mm view of the annotated area)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from PIL import Image

out = PREVIEW_DIR
_o.makedirs(out, exist_ok=True)
with rasterio.open(ORTHO_TIF) as ds:
    print(ds.width, ds.height, ds.count, ds.dtypes, ds.transform)
    f = 5                                              # 2 mm -> 1 cm
    H, W = ds.height // f, ds.width // f
    arr = ds.read([1, 2, 3, 4], out_shape=(4, H, W), resampling=Resampling.average)
    rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.uint8)
    Image.fromarray(rgb).save(out + "ortho_1cm.jpg", quality=92)
    alpha = arr[3]
    Image.fromarray((alpha > 0).astype(np.uint8) * 255).save(out + "ortho_1cm_alpha.png")
    t = ds.transform
    meta = {"full_transform": list(t)[:6], "decim": f, "W": W, "H": H, "valid_px_frac_1cm": float((alpha > 0).mean()),
            "valid_area_m2_1cm": float((alpha > 0).sum() * (0.002 * f) ** 2)}
    # annotated area at 5 mm
    bounds = [480926.0, 4044482.0, 480948.5, 4044495.0]
    win = from_bounds(*bounds, t)
    f2 = 2.5
    hh, ww = int(win.height / f2), int(win.width / f2)
    a2 = ds.read([1, 2, 3, 4], window=win, out_shape=(4, hh, ww), resampling=Resampling.average)
    Image.fromarray(np.transpose(a2[:3], (1, 2, 0)).astype(np.uint8)).save(out + "field_5mm.jpg", quality=92)
    meta["field_window_bounds"] = bounds
    meta["field_5mm_shape"] = [hh, ww]
    json.dump(meta, open(out + "preview_meta.json", "w"), indent=1)
    print(meta)
