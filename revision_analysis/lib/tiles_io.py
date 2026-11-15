"""Reading 512-px tiles from the orthomosaic with the inference protocol of the revised manuscript."""
import cv2
import numpy as np
from rasterio.windows import Window

PAD, NET = 16, 640


def read_tile(ds, r, c, t=512):
    """Returns the network input (BGR, 672 x 672 px) for the tile with top-left pixel (r, c), or None if the tile
    contains no valid pixel. No-data pixels are set to black, the tile is resized to 640 px, and a 16-px grey border
    is added, as in the cross-validation."""
    win = Window(c, r, t, t)
    a = ds.read(4, window=win)
    if (a > 0).sum() == 0:
        return None
    rgb = ds.read([1, 2, 3], window=win)
    rgb[:, a == 0] = 0
    im = np.ascontiguousarray(np.transpose(rgb, (1, 2, 0))[..., ::-1])
    im = cv2.resize(im, (NET, NET), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def predict(model, batch, offs, extra, t=512):
    """Runs the model on a batch of tiles and returns rows (row, col, px0, py0, px1, py1, score) in orthomosaic pixels."""
    rows = []
    for (rr, cc), o in zip(offs, model.predict(source=batch, imgsz=NET + 2 * PAD, conf=0.01, device="cpu", verbose=False,
                                                max_det=300, **extra)):
        b = (o.boxes.xyxy.cpu().numpy() - PAD) * t / NET
        for bb, ss in zip(b, o.boxes.conf.cpu().numpy()):
            rows.append((rr, cc, bb[0] + cc, bb[1] + rr, bb[2] + cc, bb[3] + rr, float(ss)))
    return rows
