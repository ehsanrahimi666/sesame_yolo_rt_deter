"""Audit of the tiled dataset: fold geometry, composition, unique flowers, separation between folds."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import json, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
B = GIS_DIR
D = DATA_DIR
field = gpd.read_file(B + "Field_border.shp"); ann = gpd.read_file(B + "sesame.shp")
print("CRS", field.crs, ann.crs)
fg = unary_union(field.geometry.values)
print("field area %.2f m2, bounds %s" % (fg.area, np.round(fg.bounds, 2)))
# annotation boxes (axis-aligned bounds of each polygon)
bb = np.array([g.bounds for g in ann.geometry])
w = (bb[:, 2] - bb[:, 0]) * 100; h = (bb[:, 3] - bb[:, 1]) * 100
d = (w + h) / 2
print("unique annotations:", len(ann))
print("width cm mean %.2f sd %.2f median %.2f; height mean %.2f sd %.2f median %.2f" % (w.mean(), w.std(ddof=1), np.median(w), h.mean(), h.std(ddof=1), np.median(h)))
print("mean side (w+h)/2: mean %.2f sd %.2f median %.2f, IQR %.2f-%.2f, range %.2f-%.2f cm" % (d.mean(), d.std(ddof=1), np.median(d), *np.percentile(d, [25, 75]), d.min(), d.max()))
print("in pixels at 2 mm: mean side %.1f px; median %.1f" % (d.mean()*5, np.median(d)*5))
print("box area cm2 mean %.1f median %.1f" % ((w*h).mean(), np.median(w*h)))
# fold strips: as in 01_prepare script, along longer axis of field bounds
minx, miny, maxx, maxy = fg.bounds
edges = np.linspace(minx, maxx, 6)
print("strip edges (E):", np.round(edges, 3), "strip width %.3f m" % (edges[1]-edges[0]))
tiles = pd.read_csv(D + "tiles.csv"); boxes = pd.read_csv(D + "boxes.csv")
T = 512 * 0.0019999573967093604  # tile size in m (x); y res 0.0019999711682572143
tiles["x1"] = tiles.x_origin + 512 * 0.0019999573967093604
tiles["y1"] = tiles.y_origin - 512 * 0.0019999711682572143
tiles["geom"] = [box(r.x_origin, r.y1, r.x1, r.y_origin) for r in tiles.itertuples()]
comp = []
for k in range(5):
    sub = tiles[tiles.fold == k]; bsub = boxes[boxes.fold == k]
    strip = box(edges[k], miny, edges[k+1], maxy).intersection(fg)
    cover = unary_union(list(sub.geom))
    comp.append(dict(fold=k, strip_area_m2=round(strip.area, 2), n_tiles=len(sub), box_instances=len(bsub),
                     unique_flowers=bsub.src_id.nunique(), tile_union_area_m2=round(cover.area, 2),
                     ann_in_strip=int(sum(strip.contains(g.centroid) for g in ann.geometry)),
                     mean_inst_per_tile=round(len(bsub)/max(len(sub),1), 2)))
comp = pd.DataFrame(comp); print(comp.to_string(index=False))
print("total unique flowers in tiles:", boxes.src_id.nunique(), "of", len(ann))
# flowers whose src_id appears in more than one fold?
mf = boxes.groupby("src_id").fold.nunique(); print("src_ids in >1 fold:", int((mf > 1).sum()))
# instances per unique flower
print("instances per unique flower: mean %.2f, max %d" % (boxes.groupby('src_id').size().mean(), boxes.groupby('src_id').size().max()))
# separation between each validation tile and nearest training tile (edge-to-edge)
from shapely import distance
res = []
for k in range(5):
    va = tiles[tiles.fold == k]; tr = tiles[tiles.fold != k]
    trU = unary_union(list(tr.geom))
    dd = np.array([g.distance(trU) for g in va.geom])
    res.append(dict(fold=k, min_sep_m=dd.min(), median_sep_m=np.median(dd), max_sep_m=dd.max(), n_adjacent=int((dd < 0.01).sum())))
sep = pd.DataFrame(res); print(sep.round(3).to_string(index=False))
alld = []
for k in range(5):
    va = tiles[tiles.fold == k]; tr = tiles[tiles.fold != k]; trU = unary_union(list(tr.geom))
    alld += [g.distance(trU) for g in va.geom]
alld = np.array(alld); print("ALL val tiles: min %.3f median %.3f max %.3f m; share within 0.5 m: %.2f" % (alld.min(), np.median(alld), alld.max(), (alld < 0.5).mean()))
# gap between strips: distance between union of tiles of adjacent folds
for k in range(4):
    a = unary_union(list(tiles[tiles.fold == k].geom)); b = unary_union(list(tiles[tiles.fold == k+1].geom))
    print(f"gap fold{k}-fold{k+1}: {a.distance(b):.3f} m")
tiles.drop(columns="geom").to_csv("out/tiles_geo.csv", index=False)
comp.to_csv("out/fold_composition.csv", index=False); sep.to_csv("out/fold_separation.csv", index=False)
json.dump({"edges": list(edges), "field_area": fg.area, "n_ann": len(ann), "box_side_mean_cm": d.mean(), "box_side_median_cm": float(np.median(d)),
           "box_w_mean": w.mean(), "box_h_mean": h.mean(), "box_side_sd": d.std(ddof=1), "box_side_iqr": list(np.percentile(d, [25, 75]))}, open("out/dataset_audit.json", "w"), indent=1)
