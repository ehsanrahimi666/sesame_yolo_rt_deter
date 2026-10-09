"""Derive flight geometry (line spacing, along-track spacing, effective overlaps) from the DJI MRK file."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import re, numpy as np, json
from pyproj import Transformer
mrk = MRK_FILE
rows = []
for line in open(mrk):
    m = re.search(r'^(\d+)\s+([\d.]+).*?([\d.]+),Lat\s+([\d.]+),Lon\s+([\d.]+),Ellh', line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
rows = np.array(rows)
tr = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
E, N = tr.transform(rows[:, 3], rows[:, 2])
t = rows[:, 1]
print("n images", len(rows), "duration s", t[-1]-t[0])
d = np.hypot(np.diff(E), np.diff(N)); dt = np.diff(t)
print("consecutive spacing (m): median %.2f, IQR %.2f-%.2f" % (np.median(d), *np.percentile(d, [25, 75])))
print("interval (s): median %.2f" % np.median(dt))
# headings of consecutive moves
hd = np.degrees(np.arctan2(np.diff(E), np.diff(N))) % 360
# identify straight lines: group consecutive segments with similar heading
lines = []; cur = [0]
for i in range(1, len(rows)):
    if i >= 2 and min(abs(hd[i-1]-hd[i-2]), 360-abs(hd[i-1]-hd[i-2])) > 30:
        lines.append(cur); cur = [i]
    else:
        cur.append(i)
lines.append(cur)
long_lines = [l for l in lines if len(l) >= 5]
print("n lines (>=5 imgs):", len(long_lines), [len(l) for l in long_lines])
# line direction and spacing between adjacent lines (perpendicular distance between line centroids)
dirs = []
cents = []
for l in long_lines:
    xy = np.c_[E[l], N[l]]
    c = xy.mean(0); u, s, vt = np.linalg.svd(xy - c); dirs.append(vt[0]); cents.append(c)
dirs = np.array(dirs); cents = np.array(cents)
mdir = dirs[0] * np.sign(dirs @ dirs[0])[:, None]; mdir = mdir.mean(0); mdir /= np.linalg.norm(mdir)
perp = np.array([-mdir[1], mdir[0]])
offs = np.sort(cents @ perp)
sp = np.diff(offs)
print("line azimuth (deg from N): %.1f" % (np.degrees(np.arctan2(mdir[0], mdir[1])) % 180))
print("line spacing (m):", np.round(sp, 2), "median %.2f" % np.median(sp))
# along-track spacing within lines
at = []
for l in long_lines:
    at += list(np.hypot(np.diff(E[l]), np.diff(N[l])))
at = np.array(at)
print("along-track spacing within lines: median %.2f m (IQR %.2f-%.2f)" % (np.median(at), *np.percentile(at, [25, 75])))
# footprint at camera-to-surface distance: GSD from ODM average 1.178 mm; nominal from relative altitude 5.46 m
pix = 17.3e-3/5280; f = 12.29e-3
for label, gsd in [("ground (5.46 m AGL take-off ref)", 5.46*pix/f), ("ODM avg surface GSD", 1.178e-3)]:
    W = 5280*gsd; H = 3956*gsd
    fo = 1 - np.median(at)/H; so = 1 - np.median(sp)/W
    print(f"{label}: GSD {gsd*1000:.3f} mm, footprint {W:.2f} x {H:.2f} m, forward overlap {fo*100:.0f}%, side overlap {so*100:.0f}%")
json.dump({"n_images": int(len(rows)), "duration_s": float(t[-1]-t[0]), "median_interval_s": float(np.median(dt)),
           "n_lines": len(long_lines), "line_spacing_m_median": float(np.median(sp)), "along_track_m_median": float(np.median(at))},
          open("out/flight_geometry.json", "w"), indent=1)
