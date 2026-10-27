# Test-set metrics on the patch split

Patch split: 1462 / 260 / 260 patches, split by source image (no leakage). Patches cover the whole image
area above the SEM info bar (5x4 grid per image, 64 px row overlap). PE patches with particles are
repeated 3x in train only. Every model is
scored on the same 260 test patches, including patches without particles. AP values are on a 0–100 scale.

| Model | Tool | Task | Classes | Test imgs | Box AP | AP50 | AP75 | Mask AP | AP50 | AP75 | P | R | F1 | Train time | Weights |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RT-DETR rtdetr-l_pe-x3_p30 | ultralytics | detection | 4 | 260 | 33.1 | 56.6 | 31.8 | – | – | – | 58.3 | 65.1 | 61.5 | 1:49:49 | `RT-DETR/runs/rtdetr-l/rtdetr-l_pe-x3_p30/weights/best.pt` |
| MaskRCNN R101-FPN | detectron2 | segmentation | 1 | 260 | 45.1 | 77.5 | 47.4 | 45.6 | 78.3 | 48.1 | 57.8 | 88.5 | 69.9 | 1:18:36 | `MaskRCNN/zoo/current/R101-FPN/model_final.pth` |
| MaskRCNN X101-FPN | detectron2 | segmentation | 1 | 260 | 45.3 | 77.4 | 48.0 | 45.9 | 78.4 | 48.6 | 0.0 | 0.0 | 0.0 | 1:46:54 | `MaskRCNN/zoo/current/X101-FPN/model_final.pth` |
| YOLOv8-seg yolov8n-seg_patches | ultralytics | segmentation | 1 | 260 | 49.1 | 80.8 | 53.2 | 37.4 | 74.0 | 35.1 | 71.9 | 70.4 | 71.1 | 0:17:21 | `YOLOv8-seg/runs/segment/yolov8n-seg_patches/weights/best.pt` |
| YOLOv26-seg yolo26n-seg_patches | ultralytics | segmentation | 1 | 260 | 49.4 | 80.4 | 52.9 | 38.1 | 75.0 | 34.1 | 74.5 | 69.9 | 72.2 | 0:42:30 | `YOLOv26-seg/runs/segment/yolo26n-seg_patches/weights/best.pt` |

## How to read this table
- **AP**: detectron2 reports COCO AP (pycocotools, 101-point interpolation, score threshold 0.05).
  Ultralytics reports its own mAP (its own interpolation, conf 0.001). The two are close but not
  identical. For a strict comparison, re-score the ultralytics predictions with pycocotools.
- **P / R / F1**: detectron2 rows use a fixed score threshold of 0.25, with greedy matching at
  IoU 0.5 (`evaluate.py` / `evaluate_improved_model.py`). Ultralytics rows use its mean
  precision/recall at the max-F1 confidence. The mask values are used for seg models.
- **Classes**: detection models predict 4 polymers (PP, PE, PS, PET). Segmentation models predict one
  class (Microplastic).
- **YOLO detection labels**: ultralytics drops 11 of 14,425 boxes that are exact duplicates of another
  box in the same patch (different polygons, same bbox).
