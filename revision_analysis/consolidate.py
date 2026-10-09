"""Consolidate all numbers used in the manuscript into out/summary.json (single source of truth)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "lib")); from paths import *  # noqa

import sys, json; sys.path.insert(0, "lib")
import numpy as np, pandas as pd
from evalcore import count_stats
S = {}
TAGS = ["yolo26n-e2e", "yolo26n-nms", "rtdetr-l"]
def r(x, n=3): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)
P = "out/pad16_last"
df = pd.read_csv(f"{P}/eval_fold_metrics.csv")
S["primary_per_fold"] = df.round(4).to_dict(orient="records")
mets = ["AP50_tile", "AP50_95_tile", "AP50_cluster", "AP50_95_cluster", "P_cluster", "R_cluster", "F1_cluster", "thr", "blk_MAE", "blk_R2_11", "blk_r2_pearson", "blk_bias_pct", "total_relerr_F1thr_pct"]
S["primary_mean_sd"] = {t: {m: [r(df[df.model == t][m].mean()), r(df[df.model == t][m].std(ddof=1))] for m in mets} for t in TAGS}
S["primary_pooled"] = pd.read_csv(f"{P}/stats_pooled.csv").round(4).to_dict(orient="records")
S["primary_pairs"] = pd.read_csv(f"{P}/stats_pairs.csv").round(4).to_dict(orient="records")
S["primary_boot"] = pd.read_csv(f"{P}/stats_bootstrap.csv").round(4).to_dict(orient="records")
bl = pd.read_csv(f"{P}/eval_blocks.csv")
S["primary_blocks"] = {t: {k: r(v, 3) for k, v in count_stats(bl[bl.model == t].true.values, bl[bl.model == t].pred.values).items()} for t in TAGS}
S["primary_blocks"]["n_blocks"] = int((bl.model == TAGS[0]).sum()); S["primary_blocks"]["true_mean"] = r(bl[bl.model == TAGS[0]].true.mean(), 2); S["primary_blocks"]["true_sd"] = r(bl[bl.model == TAGS[0]].true.std(ddof=1), 2)
th = pd.read_csv(f"{P}/eval_thresholds.csv")
S["thresholds"] = {t: {"crossfit_F1_range": [r(th[th.model == t].thr_F1_crossfit.min(), 2), r(th[th.model == t].thr_F1_crossfit.max(), 2)],
                       "crossfit_count_range": [r(th[th.model == t].thr_count_crossfit.min(), 2), r(th[th.model == t].thr_count_crossfit.max(), 2)],
                       "oracle_range": [r(th[th.model == t].thr_F1_oracle.min(), 2), r(th[th.model == t].thr_F1_oracle.max(), 2)]} for t in TAGS}
sw = pd.read_csv(f"{P}/eval_sweep.csv")
for t in TAGS:
    s = sw[sw.model == t]; i = s.F1.idxmax(); j = (s.count_ratio - 1).abs().idxmin()
    S["thresholds"][t]["pooled_F1opt"] = r(s.loc[i, "thr"], 2); S["thresholds"][t]["pooled_F1opt_F1"] = r(s.loc[i, "F1"]); S["thresholds"][t]["pooled_F1opt_P"] = r(s.loc[i, "P"]); S["thresholds"][t]["pooled_F1opt_R"] = r(s.loc[i, "R"])
    S["thresholds"][t]["pooled_F1opt_ratio"] = r(s.loc[i, "count_ratio"]); S["thresholds"][t]["pooled_countcal"] = r(s.loc[j, "thr"], 2)
    # flatness: thresholds within 0.01 F1 of max
    near = s[s.F1 >= s.F1.max() - 0.01].thr; S["thresholds"][t]["F1_within_0.01_range"] = [r(near.min(), 2), r(near.max(), 2)]
# FP density
core_area = 216 * (256 * 0.0019999573967093604) * (256 * 0.0019999711682572143)
S["core_area_m2"] = r(core_area, 2)
pooled = pd.read_csv(f"{P}/stats_pooled.csv")
S["fp_per_m2"] = {row.model: r(row.FP / core_area, 2) for row in pooled.itertuples()}
S["clusters_per_m2_cores"] = r(pooled.n_clusters.iloc[0] / core_area, 2)
# sensitivity table
sens = {}
for name, d in [("validator_default_best", "out/native_best"), ("noborder_best", "out/nopad_best"), ("border_best", "out/pad16_best"), ("noborder_last", "out/nopad_last"), ("border_last", "out/pad16_last")]:
    e = pd.read_csv(f"{d}/eval_fold_metrics.csv")
    sens[name] = {t: {m: [r(e[e.model == t][m].mean()), r(e[e.model == t][m].std(ddof=1))] for m in ["AP50_tile", "AP50_95_tile", "AP50_cluster", "F1_cluster", "P_cluster", "R_cluster", "blk_MAE"]} for t in TAGS}
S["sensitivity"] = sens
S["box_size_bias"] = pd.read_csv("out/box_size_bias.csv").round(3).to_dict(orient="records")
S["selection_bias_logs"] = pd.read_csv("out/final/selection_bias.csv").round(4).to_dict(orient="records")
S["benchmark"] = json.load(open("out/benchmark_cpu.json"))
S["fold_composition"] = pd.read_csv("out/fold_composition.csv").to_dict(orient="records")
S["fold_separation"] = pd.read_csv("out/fold_separation.csv").round(3).to_dict(orient="records")
S["dataset_audit"] = json.load(open("out/dataset_audit.json"))
S["flight"] = json.load(open("out/flight_geometry.json"))
# clusters per fold within cores
S["clusters_in_cores_per_fold"] = df[df.model == TAGS[0]][["fold", "clusters", "core_area_m2", "n_tiles", "tile_instances"]].to_dict(orient="records")
json.dump(S, open("out/summary.json", "w"), indent=1)
print(json.dumps({k: S[k] for k in ["primary_mean_sd", "primary_pooled", "primary_blocks", "thresholds", "fp_per_m2", "clusters_per_m2_cores", "core_area_m2", "clusters_in_cores_per_fold"]}, indent=1))
