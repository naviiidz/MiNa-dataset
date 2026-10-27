from ultralytics import YOLO, SAM
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
import pandas as pd
import json
from shapely.geometry import Polygon

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================
# Configuration
# =========================
CONFIG = {
    # Dataset
    'image_dir': '../../data/fragment_1/image',
    'gt_mask_dir': '../../data/fragment_1/label',  # <-- change this
    'gt_mask_extensions': ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'],

    # Ground-truth mask interpretation:
    # - If None: any non-zero pixel is foreground
    # - If list (e.g. [1, 255]): only those values are foreground
    'foreground_values': None,

    # Models
    'yolo_model': './rt-detr.pt',
    'sam_model': 'sam2.1_l.pt',

    # Inference
    'yolo_conf_thresholds': [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    'resize_dim': (260, 260),
    'prompt_type': 'box',  # 'box' or 'point'

    # Post-process + metrics
    'eval_iou_threshold': 0.5,
    'min_contour_area': 5,

    # Visualization + outputs
    'visualization_alpha': 0.5,
    'random_seed': 42,
    'output_dir': './outputs_semantic',
}


def get_device():
    if torch.cuda.is_available():
        return '0'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_models(yolo_path, sam_path, device):
    print(f'Loading models on device: {device}')
    yolo_model = YOLO(yolo_path)
    sam_model = SAM(sam_path)
    return yolo_model, sam_model


def find_gt_mask_path(image_name):
    stem = Path(image_name).stem
    mask_dir = Path(CONFIG['gt_mask_dir'])
    for ext in CONFIG['gt_mask_extensions']:
        p = mask_dir / f'{stem}{ext}'
        if p.exists():
            return str(p)
    return None


def load_gt_semantic_mask(image_name, image_shape):
    mask_path = find_gt_mask_path(image_name)
    if mask_path is None:
        return np.zeros(image_shape[:2], dtype=np.uint8), []

    gt = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        return np.zeros(image_shape[:2], dtype=np.uint8), []

    # Convert to single-channel semantic map if needed
    if gt.ndim == 3:
        # Treat any non-zero in any channel as foreground by default
        gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    else:
        gt_gray = gt

    if gt_gray.shape[:2] != image_shape[:2]:
        gt_gray = cv2.resize(gt_gray, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

    fg_vals = CONFIG['foreground_values']
    if fg_vals is None:
        gt_bin = (gt_gray > 0).astype(np.uint8)
    else:
        gt_bin = np.isin(gt_gray, fg_vals).astype(np.uint8)

    gt_polygons = mask_to_polygons(gt_bin, min_area=1)
    return gt_bin, gt_polygons


def mask_to_polygons(binary_mask, min_area=5):
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            polygons.append(cnt.reshape(-1, 2).tolist())
    return polygons


def create_mask_from_polygons(polygons, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for poly in polygons:
        if len(poly) >= 3:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
    return mask


def merge_predicted_polygons(polygons, image_shape, min_area=5):
    """Merge all predicted polygons by union on a raster mask, then re-extract components."""
    if not polygons:
        return []

    merged_mask = create_mask_from_polygons(polygons, image_shape)
    return mask_to_polygons(merged_mask, min_area=min_area)


def calculate_mask_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(intersection / union) if union > 0 else 0.0


def calculate_dice_coefficient(pred_mask, gt_mask):
    """Calculate Dice Similarity Coefficient (DSC) between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    denom = pred_sum + gt_sum
    return float(2.0 * intersection / denom) if denom > 0 else 0.0


def calculate_polygon_metrics(pred_polygons, gt_polygons, iou_threshold=0.5):
    if not gt_polygons and not pred_polygons:
        return 1.0, 1.0, 1.0, 0, 0, 0, [], [], []
    if not gt_polygons:
        return 0.0, 1.0, 0.0, 0, len(pred_polygons), 0, [], [], []
    if not pred_polygons:
        return 1.0, 0.0, 0.0, 0, 0, len(gt_polygons), [], [], []

    pred_shapes = []
    for poly in pred_polygons:
        if len(poly) >= 3:
            try:
                p = Polygon(poly)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_valid and p.area > 0:
                    pred_shapes.append(p)
            except Exception:
                pass

    gt_shapes = []
    for poly in gt_polygons:
        if len(poly) >= 3:
            try:
                g = Polygon(poly)
                if not g.is_valid:
                    g = g.buffer(0)
                if g.is_valid and g.area > 0:
                    gt_shapes.append(g)
            except Exception:
                pass

    if not pred_shapes and not gt_shapes:
        return 1.0, 1.0, 1.0, 0, 0, 0, [], [], []
    if not pred_shapes:
        return 1.0, 0.0, 0.0, 0, 0, len(gt_shapes), [], [], []
    if not gt_shapes:
        return 0.0, 1.0, 0.0, 0, len(pred_shapes), 0, [], [], []

    iou_matrix = np.zeros((len(pred_shapes), len(gt_shapes)), dtype=np.float64)
    for i, p in enumerate(pred_shapes):
        for j, g in enumerate(gt_shapes):
            try:
                inter = p.intersection(g).area
                uni = p.union(g).area
                iou_matrix[i, j] = (inter / uni) if uni > 0 else 0.0
            except Exception:
                iou_matrix[i, j] = 0.0

    matched_pred, matched_gt = set(), set()
    matched_pairs = []
    total_iou, match_count = 0.0, 0

    if SCIPY_AVAILABLE:
        cost = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        for pi, gi in zip(row_ind, col_ind):
            iou = iou_matrix[pi, gi]
            if iou >= iou_threshold:
                matched_pred.add(pi)
                matched_gt.add(gi)
                matched_pairs.append((pi, gi))
                total_iou += iou
                match_count += 1
    else:
        for i in range(len(pred_shapes)):
            best_j, best_iou = -1, 0.0
            for j in range(len(gt_shapes)):
                if j in matched_gt:
                    continue
                if iou_matrix[i, j] > best_iou and iou_matrix[i, j] >= iou_threshold:
                    best_iou = iou_matrix[i, j]
                    best_j = j
            if best_j != -1:
                matched_pred.add(i)
                matched_gt.add(best_j)
                matched_pairs.append((i, best_j))
                total_iou += best_iou
                match_count += 1

    tp = len(matched_pred)
    fp = len(pred_shapes) - tp
    fn = len(gt_shapes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    mean_iou = total_iou / match_count if match_count > 0 else 0.0

    return precision, recall, mean_iou, tp, len(pred_shapes), len(gt_shapes), matched_pairs, pred_shapes, gt_shapes


def compute_ap50_from_threshold_rows(threshold_rows):
    if not threshold_rows:
        return 0.0

    rows = sorted(threshold_rows, key=lambda r: r['confidence_threshold'], reverse=True)
    recalls = np.array([float(r['recall']) for r in rows], dtype=np.float64)
    precisions = np.array([float(r['precision']) for r in rows], dtype=np.float64)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def result_sort_key(result):
    s = result['image_summary']
    return (s['f1_score'], s['mean_iou'], s['mask_iou'], -s['area_mape'], s['precision'], s['recall'])


def build_polygon_data(image_name, confidence_threshold, polygons):
    rows = []
    for idx, poly in enumerate(polygons):
        if len(poly) < 3:
            continue
        try:
            shp = Polygon(poly)
            area = shp.area
            pts = np.array(poly, np.int32)
            x, y, w, h = cv2.boundingRect(pts.reshape((-1, 1, 2)))
            c = shp.centroid
            rows.append({
                'image_name': image_name,
                'polygon_id': idx,
                'confidence_threshold': confidence_threshold,
                'area_pixels': round(area, 2),
                'bbox_x': x,
                'bbox_y': y,
                'bbox_width': w,
                'bbox_height': h,
                'centroid_x': round(c.x, 2),
                'centroid_y': round(c.y, 2),
                'num_vertices': len(poly),
                'polygon_coords': json.dumps(poly),
            })
        except Exception:
            pass
    return rows


def evaluate_image_at_confidence(image_name, original_image, original_image_rgb, resized_image, gt_bin, gt_polygons,
                                 yolo_model, sam_model, device, confidence_threshold):
    print(f'Running YOLO detection (conf={confidence_threshold:.2f})...')
    try:
        yolo_results = yolo_model(resized_image, device=device, conf=confidence_threshold)
        bboxes = yolo_results[0].boxes.xyxy.tolist() if yolo_results[0].boxes is not None else []
    except Exception as e:
        print(f'Error during YOLO detection at conf={confidence_threshold:.2f}: {e}')
        bboxes = []

    original_h, original_w = original_image.shape[:2]
    resized_h, resized_w = resized_image.shape[:2]
    sx, sy = original_w / resized_w, original_h / resized_h

    scaled_bboxes = []
    scaled_points = []
    for x1, y1, x2, y2 in bboxes:
        bx = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        scaled_bboxes.append(bx)
        scaled_points.append([(bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0])

    pred_polygons = []
    sam_mask_count = 0

    if len(scaled_bboxes) == 0:
        print(f'YOLO found no detections at conf={confidence_threshold:.2f}.')
    else:
        try:
            print('Running SAM segmentation...')
            if CONFIG['prompt_type'] == 'point':
                labels = [1] * len(scaled_points)
                sam_results = sam_model(original_image_rgb, points=scaled_points, labels=labels, device=device, retina_masks=True)
            else:
                sam_results = sam_model(original_image_rgb, bboxes=scaled_bboxes, device=device, retina_masks=True)

            if sam_results[0].masks is None:
                print('SAM did not generate any masks.')
            else:
                masks = sam_results[0].masks.data
                sam_mask_count = len(masks)
                print(f'SAM generated {sam_mask_count} masks')

                for i in range(len(masks)):
                    try:
                        mask_np = masks[i].cpu().numpy().astype(np.uint8)
                        polys = mask_to_polygons(mask_np, min_area=CONFIG['min_contour_area'])
                        pred_polygons.extend(polys)
                    except Exception as e:
                        print(f'Error processing mask {i}: {e}')
        except Exception as e:
            print(f'Error during SAM segmentation at conf={confidence_threshold:.2f}: {e}')

    print(f'Pred polygon count (raw): {len(pred_polygons)}')

    # Merge all prediction polygons into connected components before evaluation
    merged_pred_polygons = merge_predicted_polygons(
        pred_polygons,
        original_image.shape,
        min_area=CONFIG['min_contour_area']
    )
    print(f'Pred polygon count (merged): {len(merged_pred_polygons)}')

    pred_mask = create_mask_from_polygons(merged_pred_polygons, original_image.shape)
    mask_iou = calculate_mask_iou(pred_mask, gt_bin)
    dice = calculate_dice_coefficient(pred_mask, gt_bin)

    precision, recall, mean_iou, tp, pred_count, gt_count, matched_pairs, pred_shapes, gt_shapes = calculate_polygon_metrics(
        merged_pred_polygons,
        gt_polygons,
        iou_threshold=CONFIG['eval_iou_threshold']
    )

    pred_area = float(pred_mask.sum())
    gt_area = float(gt_bin.sum())
    
    # Compute per-instance MAPE and DSC from matched polygon pairs only
    instance_mape_list = []
    instance_dsc_list = []
    for pred_idx, gt_idx in matched_pairs:
        if pred_idx < len(pred_shapes) and gt_idx < len(gt_shapes):
            pred_poly_area = pred_shapes[pred_idx].area
            gt_poly_area = gt_shapes[gt_idx].area
            if gt_poly_area > 0:
                mape_instance = abs(pred_poly_area - gt_poly_area) / gt_poly_area * 100.0
                instance_mape_list.append(mape_instance)
            
            # Calculate DSC for this matched pair (instance-level)
            pred_poly_mask = create_mask_from_polygons([merged_pred_polygons[pred_idx]], original_image.shape)
            gt_poly_mask = create_mask_from_polygons([gt_polygons[gt_idx]], original_image.shape)
            dsc_instance = calculate_dice_coefficient(pred_poly_mask, gt_poly_mask)
            instance_dsc_list.append(dsc_instance)
    
    mape = np.mean(instance_mape_list) if instance_mape_list else 0.0
    mean_dsc = np.mean(instance_dsc_list) if instance_dsc_list else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fp = pred_count - tp
    fn = gt_count - tp

    summary = {
        'image_name': image_name,
        'confidence_threshold': round(confidence_threshold, 2),
        'prompt_type': CONFIG['prompt_type'],
        'yolo_detections': len(scaled_bboxes),
        'sam_masks': sam_mask_count,
        'raw_pred_polygons': len(pred_polygons),
        'pred_polygons': pred_count,
        'gt_polygons': gt_count,
        'total_detections': pred_count,
        'total_area_pixels': round(pred_area, 2),
        'gt_total_area': round(gt_area, 2),
        'pred_total_area': round(pred_area, 2),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'mean_iou': round(mean_iou, 4),
        'mask_iou': round(mask_iou, 4),
        'mean_dsc': round(mean_dsc, 4),
        'dice_coefficient': round(dice, 4),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'area_mape': round(mape, 2),
        'is_best_threshold': False,
    }

    print(
        f"Metrics @ conf={confidence_threshold:.2f} - Precision: {precision:.3f}, Recall: {recall:.3f}, "
        f"F1: {f1:.3f}, Mean IoU: {mean_iou:.3f}, Mean DSC: {mean_dsc:.3f}, Mask IoU: {mask_iou:.3f}, DSC: {dice:.3f}, Area MAPE: {mape:.2f}%"
    )

    return {
        'image_summary': summary,
        'polygon_data': build_polygon_data(image_name, confidence_threshold, merged_pred_polygons),
        'pred_polygons': merged_pred_polygons,
    }


def save_overlay(image_path, image_name, best_confidence, pred_polygons):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert background to grayscale RGB to match requested visualization style.
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Fill masks with random colors on a copy, then alpha-blend for translucent interior.
        random.seed(CONFIG['random_seed'])
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                  for _ in range(len(pred_polygons))]

        overlay_fill = base.copy()
        for i, poly in enumerate(pred_polygons):
            if len(poly) < 3:
                continue
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay_fill, [pts], color=colors[i])

        final_img = cv2.addWeighted(overlay_fill, 0.40, base, 0.60, 0)

        # Draw strong red boundaries on top.
        for poly in pred_polygons:
            if len(poly) < 3:
                continue
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(final_img, [pts], True, (255, 0, 0), thickness=5)

        out_dir = Path(CONFIG['output_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(image_name).stem
        for old_overlay in out_dir.glob(f'output_{base_name}_bestconf*.png'):
            try:
                old_overlay.unlink()
            except Exception:
                pass

        out_path = out_dir / f'output_{base_name}_bestconf{best_confidence:.2f}.png'

        plt.figure(figsize=(8, 8))
        plt.imshow(final_img)
        plt.title(f"{image_name} - Predictions ({len(pred_polygons)} objects)")
        plt.axis('off')
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close()

        print(f'Visualization saved to: {out_path}')
    except Exception as e:
        print(f'Error during visualization for {image_name}: {e}')


if __name__ == '__main__':
    device = get_device()
    yolo_model, sam_model = load_models(CONFIG['yolo_model'], CONFIG['sam_model'], device)

    image_dir = Path(CONFIG['image_dir'])
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg')))

    if not image_files:
        print(f'No images found in {image_dir}')
        raise SystemExit(1)

    print(f'Found {len(image_files)} images to process')

    out_dir = Path(CONFIG['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    all_best_polygon_rows = []
    all_threshold_rows = []
    best_image_rows = []

    for image_file in image_files:
        image_name = image_file.name
        image_path = str(image_file)

        print(f'\n=== Processing: {image_name} ===')

        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f'Could not load image from {image_path}, skipping...')
            continue

        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(original_image, CONFIG['resize_dim'], interpolation=cv2.INTER_AREA)

        gt_bin, gt_polygons = load_gt_semantic_mask(image_name, original_image.shape)

        threshold_results = []
        best_result = None

        for conf in CONFIG['yolo_conf_thresholds']:
            result = evaluate_image_at_confidence(
                image_name,
                original_image,
                original_image_rgb,
                resized_image,
                gt_bin,
                gt_polygons,
                yolo_model,
                sam_model,
                device,
                conf,
            )
            threshold_results.append(result['image_summary'])
            if best_result is None or result_sort_key(result) > result_sort_key(best_result):
                best_result = result

        if best_result is None:
            print(f'No valid result for {image_name}, skipping...')
            continue

        best_conf = best_result['image_summary']['confidence_threshold']
        for r in threshold_results:
            r['is_best_threshold'] = (r['confidence_threshold'] == best_conf)

        ap50 = compute_ap50_from_threshold_rows(threshold_results)
        for r in threshold_results:
            r['ap50'] = round(ap50, 4)

        best_result['image_summary']['ap50'] = round(ap50, 4)

        all_threshold_rows.extend(threshold_results)
        best_image_rows.append(best_result['image_summary'])
        all_best_polygon_rows.extend(best_result['polygon_data'])

        print(f'Selected best confidence for {image_name}: {best_conf:.2f}')
        print(f'AP50 for {image_name}: {ap50:.4f}')

        save_overlay(image_path, image_name, best_conf, best_result['pred_polygons'])

    print('\n=== All images processed! ===')

    if all_best_polygon_rows:
        pd.DataFrame(all_best_polygon_rows).to_csv(out_dir / 'detailed_predictions_best_threshold_semantic.csv', index=False)
        print(f'Detailed predictions saved to: {out_dir / "detailed_predictions_best_threshold_semantic.csv"}')

    if all_threshold_rows:
        pd.DataFrame(all_threshold_rows).to_csv(out_dir / 'confidence_sweep_metrics_semantic.csv', index=False)
        print(f'Confidence sweep metrics saved to: {out_dir / "confidence_sweep_metrics_semantic.csv"}')

    if best_image_rows:
        df_best = pd.DataFrame(best_image_rows)
        df_best.to_csv(out_dir / 'image_best_threshold_metrics_semantic.csv', index=False)
        print(f'Image summaries saved to: {out_dir / "image_best_threshold_metrics_semantic.csv"}')

        overall = {
            'mean_precision': float(np.mean(df_best['precision'])),
            'mean_recall': float(np.mean(df_best['recall'])),
            'mean_f1_score': float(np.mean(df_best['f1_score'])),
            'mean_iou': float(np.mean(df_best['mean_iou'])),
            'mean_dsc': float(np.mean(df_best['mean_dsc'])),
            'mean_mask_iou': float(np.mean(df_best['mask_iou'])),
            'mean_mask_dice': float(np.mean(df_best['dice_coefficient'])),
            'map50': float(np.mean(df_best['ap50'])),
            'mean_area_mape': float(np.mean(df_best['area_mape'])),
            'total_true_positives': int(np.sum(df_best['true_positives'])),
            'total_false_positives': int(np.sum(df_best['false_positives'])),
            'total_false_negatives': int(np.sum(df_best['false_negatives'])),
            'total_images': int(len(df_best)),
            'tested_confidence_thresholds': ','.join([f'{x:.2f}' for x in CONFIG['yolo_conf_thresholds']]),
            'prompt_type': CONFIG['prompt_type'],
            'selection_metric': 'f1_score',
            'scipy_hungarian': SCIPY_AVAILABLE,
        }

        pd.DataFrame([overall]).to_csv(out_dir / 'overall_metrics_best_threshold_semantic.csv', index=False)
        print(f'Overall metrics saved to: {out_dir / "overall_metrics_best_threshold_semantic.csv"}')

        print('\n=== OVERALL EVALUATION METRICS (Semantic GT) ===')
        print(f"Mean Precision: {overall['mean_precision']:.4f}")
        print(f"Mean Recall: {overall['mean_recall']:.4f}")
        print(f"Mean F1-Score: {overall['mean_f1_score']:.4f}")
        print(f"Mean IoU (instance-level): {overall['mean_iou']:.4f}")
        print(f"Mean DSC (instance-level): {overall['mean_dsc']:.4f}")
        print(f"Mean Mask IoU (pixel-level): {overall['mean_mask_iou']:.4f}")
        print(f"Mean Mask DSC (pixel-level): {overall['mean_mask_dice']:.4f}")
        print(f"mAP50: {overall['map50']:.4f}")
        print(f"Mean Area MAPE: {overall['mean_area_mape']:.2f}%")
        print(
            f"Total TP: {overall['total_true_positives']}, "
            f"FP: {overall['total_false_positives']}, "
            f"FN: {overall['total_false_negatives']}"
        )

    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()
