"""Input and output locations used by the revision analysis.

All paths are relative to the repository root by default and can be overridden with environment variables,
e.g. SESAME_ORTHO=/data/odm_orthophoto.tif python wall_to_wall.py ...
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SESAME_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = os.environ.get("SESAME_DATA", str(ROOT / "dataset" / "sesame_dataset_v1")) + "/"     # 216 tiles + labels
GIS_DIR = os.environ.get("SESAME_GIS", str(ROOT / "gis")) + "/"                                    # sesame.shp, Field_border.shp
RUNS_DIR = os.environ.get("SESAME_RUNS", str(ROOT)) + "/"                     # contains yolo26_results/ and Rt_DETER/
ORTHO_TIF = os.environ.get("SESAME_ORTHO", str(ROOT / "orthomosaic" / "odm_orthophoto.tif"))       # not in the repository
MRK_FILE = os.environ.get("SESAME_MRK", str(ROOT / "flight" / "DJI_202607051136_008_Create-Area-Route7_Timestamp.MRK"))
WEIGHTS_BEST = Path(os.environ.get("SESAME_WEIGHTS_BEST", ROOT / "weights" / "best"))   # weights_<model>_fold<k>.pt, *_final.pt
WEIGHTS_LAST = Path(os.environ.get("SESAME_WEIGHTS_LAST", ROOT / "weights" / "last"))   # weights_<model>_fold<k>.pt (last.pt)
PREVIEW_DIR = os.environ.get("SESAME_PREVIEWS", str(ROOT / "revision_analysis" / "previews")) + "/"  # from make_previews.py
FIGS_DIR = os.environ.get("SESAME_FIGS", str(ROOT / "figures")) + "/"
os.makedirs(FIGS_DIR, exist_ok=True)
# the analysis scripts write to out/ relative to the working directory (run them from revision_analysis/)
os.makedirs(os.path.join("out", "final"), exist_ok=True)
