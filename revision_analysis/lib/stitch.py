"""Tiling grid of the mapping pipeline and seam-aware stitching of overlapping tiles.

Tiles of 512 x 512 px are placed with a stride of 448 px (64-px overlap); the last tile of a row or column is shifted
so that it ends at the image edge. A detection is kept only if its centre lies nearer to the centre of the tile that
produced it than to the centre of any other tile, i.e. between the midpoints of the overlaps with the neighbouring grid
positions. This is the overlapping-tile equivalent of the centre-in-core rule of the cluster-level evaluation; a global
NMS (IoU 0.5) is applied afterwards.
"""
import numpy as np

T, OVER = 512, 64
# georeferencing of the orthomosaic (EPSG:32652), used to convert pixel to map coordinates without opening the raster
W, H = 25000, 20999
X0, Y0 = 480919.25018164766, 4044525.1466317894
RX, RY = 0.0019999573967093604, 0.0019999711682572143


def positions(n, t=T, over=OVER):
    s = t - over
    p = list(range(0, max(1, n - t + 1), s))
    if p[-1] + t < n:
        p.append(n - t)
    return np.array(p)


def grid(w=W, h=H):
    return [(r, c) for r in positions(h) for c in positions(w)]


def _bounds(pos, n, t=T):
    lo, hi = np.empty(len(pos)), np.empty(len(pos))
    for i, p in enumerate(pos):
        lo[i] = 0 if i == 0 else (pos[i - 1] + t + p) / 2      # midpoint of the overlap with the previous tile
        hi[i] = n if i == len(pos) - 1 else (p + t + pos[i + 1]) / 2
    return dict(zip(pos.tolist(), lo)), dict(zip(pos.tolist(), hi))


def keep_mask(D, w=W, h=H):
    """D: detections with tile origin (row, col) and box (px0, py0, px1, py1) in full-orthomosaic pixels."""
    rlo, rhi = _bounds(positions(h), h)
    clo, chi = _bounds(positions(w), w)
    cx = (D.px0.values + D.px1.values) / 2
    cy = (D.py0.values + D.py1.values) / 2
    r, c = D.row.values, D.col.values
    return ((cy >= np.array([rlo[v] for v in r])) & (cy < np.array([rhi[v] for v in r])) &
            (cx >= np.array([clo[v] for v in c])) & (cx < np.array([chi[v] for v in c])))


def global_nms(D, iou=0.5):
    import torch
    from torchvision.ops import nms
    B = torch.tensor(D[["px0", "py0", "px1", "py1"]].values, dtype=torch.float64)
    S = torch.tensor(D.score.values, dtype=torch.float64)
    return D.iloc[nms(B, S, iou).numpy()].copy()


def to_map(K):
    """Add map coordinates of box centres (x, y) and box sizes in cm."""
    cx = (K.px0 + K.px1) / 2; cy = (K.py0 + K.py1) / 2
    K["x"] = X0 + cx * RX; K["y"] = Y0 - cy * RY
    K["width_cm"] = (K.px1 - K.px0) * RX * 100; K["height_cm"] = (K.py1 - K.py0) * RY * 100
    return K
