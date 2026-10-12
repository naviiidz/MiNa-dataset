"""Detector + SAM: boxes from RT-DETR/YOLO prompt SAM 2.1, one mask per detected particle.

    python run_detector_sam.py --dataset MiNa_dataset             # fixed confidence 0.6
    python run_detector_sam.py --dataset dataset2_low_mag --sweep     # best of 0.5..0.9 per image
    python run_detector_sam.py --dataset fragment_1 --no-eval         # predictions only
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, SAM

import evaluation


SWEEP_CONFIDENCES = (0.5, 0.6, 0.7, 0.8, 0.9)
DETECTOR_INPUT_SIZE = (260, 260)  # images are resized to this before detection (the detector was trained on 256x256 patches)


def get_device():
    if torch.cuda.is_available():
        return '0'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def check_weights(path):
    # Ultralytics silently downloads stock weights when a path doesn't exist; fail instead
    if not Path(path).exists():
        raise SystemExit(f'Weights not found: {path}')


def remove_duplicate_masks(masks, iou_threshold):
    """Drop masks that overlap an already kept mask with IoU above the threshold.

    The detector often reports the same particle several times (once per class), which gives
    near-identical SAM masks. Larger masks are kept first.
    """
    areas = masks.flatten(1).sum(1)
    kept = []
    for i in torch.argsort(areas, descending=True).tolist():
        if areas[i] == 0:
            continue
        is_duplicate = False
        for j in kept:
            inter = (masks[i] & masks[j]).sum()
            if inter / (areas[i] + areas[j] - inter) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(i)
    return masks[kept]


def load_background_predictor(sam3_path, conf):
    """SAM 3 text-prompted predictor that segments the filter membrane behind the particles."""
    if not Path(sam3_path).exists():
        raise SystemExit(f'SAM 3 weights not found at {sam3_path}. Request access at '
                         f'https://huggingface.co/facebook/sam3 and download sam3.pt, or drop --exclude-background.')
    from ultralytics.models.sam import SAM3SemanticPredictor
    overrides = dict(conf=conf, task='segment', mode='predict', model=sam3_path,
                     half=torch.cuda.is_available(), verbose=False)
    return SAM3SemanticPredictor(overrides=overrides)


def make_predictor(args):
    check_weights(args.detector)
    check_weights(args.sam)
    device = get_device()
    print(f'Device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device == '0' else ''))
    detector = YOLO(args.detector)
    sam = SAM(args.sam)
    background = load_background_predictor(args.sam3, args.background_conf) if args.exclude_background else None

    def predict(image_bgr, conf):
        resized = cv2.resize(image_bgr, DETECTOR_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        boxes = detector(resized, device=device, conf=conf, verbose=False)[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        h, w = image_bgr.shape[:2]
        scale = np.array([w / DETECTOR_INPUT_SIZE[0], h / DETECTOR_INPUT_SIZE[1]] * 2)
        bboxes = boxes.xyxy.cpu().numpy() * scale
        if args.prompt == 'box':
            result = sam(image_bgr, bboxes=bboxes.tolist(), device=device, retina_masks=True, verbose=False)[0]
        else:
            centers = np.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2], 1)
            result = sam(image_bgr, points=centers.tolist(), labels=[1] * len(centers),
                         device=device, retina_masks=True, verbose=False)[0]
        if result.masks is None:
            return []
        masks = result.masks.data.bool()

        if background is not None:
            bg = background(image_bgr, text=args.background_text)[0]
            if bg.masks is not None:
                bg_mask = bg.masks.data.bool().any(0).to(masks.device)
                areas = masks.flatten(1).sum(1).clamp(min=1)
                on_background = (masks & bg_mask).flatten(1).sum(1) / areas
                masks = masks[on_background <= args.max_background_overlap] & ~bg_mask

        masks = remove_duplicate_masks(masks, args.dedupe_iou)
        polygons = [evaluation.instance_polygon(m) for m in masks.cpu().numpy()]
        return [p for p in polygons if p is not None]

    return predict


def main():
    parser = evaluation.base_argument_parser('Detector boxes prompt SAM 2.1; evaluate on a dataset.')
    parser.add_argument('--conf', type=float, default=0.6, help='Detector confidence threshold (default 0.6)')
    parser.add_argument('--sweep', action='store_true',
                        help=f'Try {SWEEP_CONFIDENCES} and keep each image\'s best by F1 (optimistic: '
                             f'chosen with the labels). Also reports a sweep AP50.')
    parser.add_argument('--detector', default='./model_zoo/rt-detr.pt', help='Detector weights (RT-DETR or YOLO)')
    parser.add_argument('--sam', default='./model_zoo/sam2.1_l.pt', help='SAM weights')
    parser.add_argument('--prompt', choices=['box', 'point'], default='box', help='Prompt SAM with boxes or box centres')
    parser.add_argument('--dedupe-iou', type=float, default=0.5,
                        help='Masks overlapping a kept mask above this IoU are dropped as duplicates')
    parser.add_argument('--exclude-background', action='store_true',
                        help='Use SAM 3 with a text prompt to remove the filter membrane from the masks')
    parser.add_argument('--sam3', default='./model_zoo/sam3.pt', help='SAM 3 weights (for --exclude-background)')
    parser.add_argument('--background-text', nargs='+', default=['porous filter membrane'],
                        help='Text prompt(s) describing the background')
    parser.add_argument('--background-conf', type=float, default=0.25, help='SAM 3 confidence for the background')
    parser.add_argument('--max-background-overlap', type=float, default=0.5,
                        help='Drop masks with more than this fraction of pixels on the background')
    args = parser.parse_args()

    if args.sweep and args.no_eval:
        parser.error('--sweep needs labels; drop --no-eval')
    confidences = SWEEP_CONFIDENCES if args.sweep else (args.conf,)
    settings = {k: v for k, v in vars(args).items() if k not in ('dataset', 'no_eval')}
    evaluation.run('detector_sam', make_predictor(args), args.dataset, confidences,
                   evaluate=not args.no_eval, settings=settings)


if __name__ == '__main__':
    main()
