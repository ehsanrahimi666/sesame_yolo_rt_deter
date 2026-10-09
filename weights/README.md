# Trained weights

The weights are distributed as assets of the GitHub release **v1.0-weights** of this repository
(https://github.com/ehsanrahimi666/sesame_yolo_rt_deter/releases). All models were trained with Ultralytics 8.4.105.

| Asset | Description | Save as |
|---|---|---|
| `weights_yolo26n_fold<k>_best.pt` | YOLO26n, fold k: checkpoint with the highest held-out AP@0.5:0.95 during training | `weights/best/weights_yolo26n_fold<k>.pt` |
| `weights_yolo26n_fold<k>_last.pt` | YOLO26n, fold k: weights at the end of training (primary analysis) | `weights/last/weights_yolo26n_fold<k>.pt` |
| `weights_yolo26n_final.pt` | YOLO26n trained on all 216 tiles (wall-to-wall mapping) | `weights/best/weights_yolo26n_final.pt` |
| `weights_rtdetr-l_fold<k>_best.pt` | RT-DETR-l, fold k: best checkpoint | `weights/best/weights_rtdetr-l_fold<k>.pt` |
| `weights_rtdetr-l_fold<k>_last.pt` | RT-DETR-l, fold k: last checkpoint (primary analysis) | `weights/last/weights_rtdetr-l_fold<k>.pt` |
| `weights_rtdetr-l_final.pt` | RT-DETR-l trained on all 216 tiles | `weights/best/weights_rtdetr-l_final.pt` |

`k` = 0-4 is the held-out spatial fold of each model. Example (Linux/macOS):

```bash
mkdir -p weights/best weights/last
for m in yolo26n rtdetr-l; do
  for k in 0 1 2 3 4; do
    curl -L -o weights/best/weights_${m}_fold${k}.pt https://github.com/ehsanrahimi666/sesame_yolo_rt_deter/releases/download/v1.0-weights/weights_${m}_fold${k}_best.pt
    curl -L -o weights/last/weights_${m}_fold${k}.pt https://github.com/ehsanrahimi666/sesame_yolo_rt_deter/releases/download/v1.0-weights/weights_${m}_fold${k}_last.pt
  done
  curl -L -o weights/best/weights_${m}_final.pt https://github.com/ehsanrahimi666/sesame_yolo_rt_deter/releases/download/v1.0-weights/weights_${m}_final.pt
done
```

Usage in Python:

```python
from ultralytics import YOLO, RTDETR
yolo = YOLO("weights/best/weights_yolo26n_final.pt")
rtdetr = RTDETR("weights/best/weights_rtdetr-l_final.pt")
```
