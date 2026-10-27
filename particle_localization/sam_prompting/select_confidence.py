"""Choose the detector confidence threshold for detector + SAM on the validation patches.

SAM is run once per image on every box above the lowest threshold; each threshold of the grid then
keeps the masks of boxes at or above it. The threshold with the best F1 over the whole validation
set (not per image) is written to outputs/<name>/MiNa_val/confidence.json and is used unchanged on
the test patches and the external datasets.

    python select_confidence.py --detector ../RT-DETR/runs/rtdetr-l/rtdetr-l_agnostic_seed0/weights/best.pt \
        --prompt box --name detector_sam_box_seed0
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd

import evaluation
from run_detector_sam import make_predictor


GRID = np.round(np.arange(0.30, 0.71, 0.10), 2)


def main():
    parser = argparse.ArgumentParser(description='Choose the detector confidence on the validation patches.')
    parser.add_argument('--detector', required=True, help='Detector weights')
    parser.add_argument('--sam', default='./model_zoo/sam2.1_l.pt', help='SAM weights')
    parser.add_argument('--prompt', choices=['box', 'point'], default='box')
    parser.add_argument('--dedupe-iou', type=float, default=0.5)
    parser.add_argument('--name', required=True, help='Run name; output goes to outputs/<name>/MiNa_val/')
    parser.add_argument('--dataset', default='MiNa_val')
    args = parser.parse_args()
    args.exclude_background = False

    predict = make_predictor(args)
    cfg = evaluation.resolve_dataset(args.dataset)
    totals = {c: {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0} for c in GRID}
    images = evaluation.list_images(cfg['image_dir'])
    for n, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        _, gt_polygons = evaluation.load_ground_truth(cfg, path.name, image.shape)
        if gt_polygons is None:
            continue
        gt_foreground = evaluation.polygons_to_mask(gt_polygons, image.shape)
        scores, masks = predict.segment(image, float(GRID.min()))
        for conf in GRID:
            row = evaluation.score_image(predict.to_polygons(image, scores, masks, conf), gt_foreground, gt_polygons,
                                         image.shape)
            for key in totals[conf]:
                totals[conf][key] += row[key]
        if n % 50 == 0:
            print(f'{n}/{len(images)} images', flush=True)

    rows = []
    for conf, t in totals.items():
        tp, fp, fn = t['true_positives'], t['false_positives'], t['false_negatives']
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({'confidence': float(conf), **t, 'precision': round(p, 4), 'recall': round(r, 4),
                     'f1_score': round(2 * p * r / (p + r), 4) if p + r else 0.0})
    table = pd.DataFrame(rows)
    best = table.loc[table['f1_score'].idxmax()]
    out_dir = evaluation.OUTPUT_ROOT / args.name / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / 'confidence_grid.csv', index=False)
    (out_dir / 'confidence.json').write_text(json.dumps(
        {'confidence': float(best['confidence']), 'validation_f1': float(best['f1_score']),
         'detector': args.detector, 'prompt': args.prompt, 'dataset': args.dataset}, indent=2))
    print(table.to_string(index=False))
    print(f"\nChosen confidence {best['confidence']:.2f} (validation F1 {best['f1_score']:.3f}) -> {out_dir / 'confidence.json'}")


if __name__ == '__main__':
    main()
