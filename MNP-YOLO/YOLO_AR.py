import os
import json
import numpy as np
from ultralytics import YOLOv10

def load_data():
    # Load the ground truth data
    label_dir = os.path.expanduser('~/datasets/MPDS2/labels/test')

    # Directory containing images
    image_dir = os.path.expanduser('~/datasets/MPDS2/images/test')

    # List of image paths
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.jpeg', '.png'))]

    # List of label paths
    label_paths = [os.path.join(label_dir, fname) for fname in os.listdir(label_dir) if fname.endswith('.txt')]

    # Load the ground truth data
    ground_truth_data = {}
    for label_path in label_paths:
        image_name = os.path.splitext(os.path.basename(label_path))[0]  # Remove the extension
        with open(label_path, 'r') as f:
            labels = f.readlines()
        image_annotations = []
        for label in labels:
            parts = label.strip().split()
            category_id = int(parts[0]) + 1  # COCO category IDs are 1-indexed
            x_center, y_center, width, height = map(float, parts[1:])
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            image_annotations.append({
                "bbox": [x_min, y_min, width, height],
                "category_id": category_id
            })
        ground_truth_data[image_name] = image_annotations

    # Load the predictions
    model = YOLOv10("/home/navid/yolov10/runs/detect/train112/weights/best.pt")
    results = model(image_paths)

    predictions = {}
    for i, result in enumerate(results):
        image_name = os.path.splitext(os.path.basename(image_paths[i]))[0]  # Remove the extension
        image_predictions = []
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].clone()
            width = x_max - x_min
            height = y_max - y_min
            score = box.conf[0].item()
            category_id = int(box.cls[0].item()) + 1  # COCO category IDs are 1-indexed

            # Normalize the predicted bounding box
            image_width, image_height = result.orig_shape[1], result.orig_shape[0]
            x_min /= image_width
            y_min /= image_height
            width /= image_width
            height /= image_height

            image_predictions.append({
                "bbox": [x_min.item(), y_min.item(), width.item(), height.item()],
                "score": score,
                "category_id": category_id
            })
        predictions[image_name] = image_predictions

    return ground_truth_data, predictions, image_paths

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min = x1
    y1_min = y1
    x1_max = x1 + w1
    y1_max = y1 + h1

    x2_min = x2
    y2_min = y2
    x2_max = x2 + w2
    y2_max = y2 + h2

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        inter_area = 0

    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def match_detections(gt_bboxes, pred_bboxes, iou_threshold):
    matches = []
    for pred in pred_bboxes:
        best_iou = 0
        best_gt_index = -1
        for i, gt_bbox in enumerate(gt_bboxes):
            iou = compute_iou(gt_bbox['bbox'], pred['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = i

        if best_iou >= iou_threshold:
            matches.append((best_gt_index, pred['score']))

    return matches

def calculate_recall(gt_dict, predictions, iou_threshold, score_threshold=0.3):
    true_positives = 0
    total_ground_truths = 0

    for image_name in gt_dict.keys():
        if image_name in predictions:
            gt_bboxes = gt_dict[image_name]
            pred_bboxes = [p for p in predictions[image_name] if p['score'] >= score_threshold]
            matches = match_detections(gt_bboxes, pred_bboxes, iou_threshold)

            matched_gt = set()
            for gt_index, _ in matches:
                if gt_index not in matched_gt:
                    true_positives += 1
                    matched_gt.add(gt_index)

            total_ground_truths += len(gt_bboxes)

    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
    return recall

def calculate_ar(gt_dict, predictions, score_threshold=0.3):
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    recalls = []

    for iou_threshold in iou_thresholds:
        recall = calculate_recall(gt_dict, predictions, iou_threshold, score_threshold)
        recalls.append(recall)

    ar = np.mean(recalls)
    return ar

# Load data
ground_truth_data, predictions, image_paths = load_data()

# Calculate AR0.5-0.95
ar = calculate_ar(ground_truth_data, predictions, score_threshold=0.3)
print(f"AR0.5-0.95: {ar:.4f}")
