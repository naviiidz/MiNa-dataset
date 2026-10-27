Polymer-agnostic (1-class) models: box AP (%) on the 260 test patches, mean ± SD over training seeds

| Task | Model | Backbone | Box AP50 | Box AP75 | Box AP | Runs |
|---|---|---|---|---|---|---|
| Detection | Faster R-CNN | ResNet 101 | 77.6 ± 0.6 | 47.2 ± 0.3 | 44.9 ± 0.2 | 3 |
|  |  | ResNeXt 101 | 77.5 ± 0.2 | 47.6 ± 0.4 | 45.2 ± 0.0 | 3 |
|  | RT-DETR | RT-DETR-L | 82.0 ± 0.7 | 55.4 ± 0.5 | 51.4 ± 0.2 | 3 |
|  | YOLO | yolo26s | 81.3 ± 2.9 | 55.0 ± 4.6 | 50.7 ± 3.0 | 3 |
| Segmentation | Mask R-CNN | ResNet 101 | 77.0 ± 0.5 | 46.7 ± 0.7 | 44.7 ± 0.6 | 3 |
|  |  | ResNeXt 101 | 76.7 ± 0.8 | 47.0 ± 1.1 | 44.7 ± 0.7 | 3 |
|  | YOLO Segmentation | yolov8n-seg | 80.3 ± 0.5 | 53.1 ± 0.9 | 49.0 ± 0.7 | 3 |
|  |  | yolov26n-seg | 79.0 ± 1.6 | 50.7 ± 2.4 | 47.6 ± 1.8 | 3 |
