"""Classical baseline without SAM: thresholding, morphology, distance transform and watershed.

    python run_watershed.py --dataset MiNa_dataset
    python run_watershed.py --dataset fragment_1 --threshold-method adaptive
"""
import cv2
import numpy as np

import evaluation


def watershed_segmentation(image_bgr, threshold_method='otsu', threshold_value=0,
                           morph_kernel_size=5, min_size=15):
    """Instance label map (0 = background) of the particles in one image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Threshold (particles become foreground)
    if threshold_method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif threshold_method == 'adaptive':
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    else:
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # 2. Opening removes small noise, closing fills small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. Distance transform; its Otsu-thresholded peaks are the sure-foreground seeds
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, markers = cv2.connectedComponents(sure_fg)

    # 4. Watershed from the seeds
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)

    # 5. Drop background (1), borders (-1) and particles smaller than min_size
    segmentation = np.zeros(markers.shape, np.int32)
    new_label = 1
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = markers == label
        if mask.sum() >= min_size:
            segmentation[mask] = new_label
            new_label += 1
    return segmentation


def main():
    parser = evaluation.base_argument_parser('Watershed baseline (no SAM); evaluate on a dataset.')
    parser.add_argument('--threshold-method', choices=['otsu', 'adaptive', 'fixed'], default='otsu')
    parser.add_argument('--threshold-value', type=int, default=0, help='Threshold for --threshold-method fixed')
    parser.add_argument('--morph-kernel-size', type=int, default=5, help='Kernel size for opening/closing')
    parser.add_argument('--min-size', type=int, default=15, help='Smallest particle kept, in pixels')
    args = parser.parse_args()

    def predict(image_bgr, conf):
        segmentation = watershed_segmentation(image_bgr, args.threshold_method, args.threshold_value,
                                              args.morph_kernel_size, args.min_size)
        return evaluation.label_map_to_polygons(segmentation)

    settings = {k: v for k, v in vars(args).items() if k not in ('dataset', 'no_eval')}
    evaluation.run('watershed', predict, args.dataset, evaluate=not args.no_eval, settings=settings)


if __name__ == '__main__':
    main()
