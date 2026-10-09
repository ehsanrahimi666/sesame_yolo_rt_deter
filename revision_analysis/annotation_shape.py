"""Shape of the reference annotations: rotated rectangles as drawn versus their axis-aligned envelopes."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import json
import numpy as np
import geopandas as gpd

g = gpd.read_file(GIS_DIR + "sesame.shp")
rot, side, env = [], [], []
for geom in g.geometry:
    c = np.array(geom.exterior.coords)[:4]
    e = np.roll(c, -1, axis=0) - c
    L = np.linalg.norm(e, axis=1)
    o = np.degrees(np.arctan2(e[0][1], e[0][0])) % 90
    rot.append(min(o, 90 - o))                      # deviation of the rectangle from the image axes (0-45 degrees)
    side.append(L.mean() * 100)                     # (width + height)/2 of the drawn rectangle, cm
    b = geom.bounds
    env.append(((b[2] - b[0]) + (b[3] - b[1])) / 2 * 100)   # (width + height)/2 of the axis-aligned box, cm
rot, side, env = map(np.array, (rot, side, env))
out = {"n": int(len(g)), "drawn_side_mean_cm": float(side.mean()), "drawn_side_median_cm": float(np.median(side)),
       "drawn_side_iqr": [float(x) for x in np.percentile(side, [25, 75])], "rotation_median_deg": float(np.median(rot)),
       "rotation_iqr": [float(x) for x in np.percentile(rot, [25, 75])], "env_over_drawn_median": float(np.median(env / side)),
       "env_median_cm": float(np.median(env))}
json.dump(out, open("out/annotation_shape.json", "w"), indent=1)
print(out)
