"""micro-sam automatic segmentation baseline (no detector): AMG, AIS or APG.

    python run_micro_sam.py --dataset MiNa_dataset                       # vit_l, AMG
    python run_micro_sam.py --dataset MiNa_dataset --model vit_b_lm --mode ais

AIS and APG need a micro-sam model (e.g. vit_b_lm, vit_l_lm); AMG works with any SAM model.
Results go to outputs/micro_sam_<mode>/<dataset>/.
"""
import cv2
import torch
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

import evaluation


# Post-processing settings per mode, tuned for microplastics
GENERATE_KWARGS = {
    'amg': {
        'pred_iou_thresh': 0.5,            # lowered from 0.88 for better recall on small objects
        'stability_score_thresh': 0.85,    # lowered from 0.95 to accept more candidates
        'box_nms_thresh': 0.65,            # tightened from 0.7 for less overlapping masks
        'crop_nms_thresh': 0.65,           # tightened from 0.7 to avoid redundancy
        'min_mask_region_area': 20,        # remove tiny noise (< 20 pixels)
    },
    'ais': {
        'center_distance_threshold': 0.4,  # tightened from 0.5 to prevent merging
        'boundary_distance_threshold': 0.4,
        'foreground_threshold': 0.7,       # increased from 0.5 to reduce false positives
        'foreground_smoothing': 1.0,
        'distance_smoothing': 1.4,
        'min_size': 15,                    # remove tiny noise (< 15 pixels)
    },
    'apg': {
        'center_distance_threshold': 0.5,
        'boundary_distance_threshold': 0.5,
        'foreground_threshold': 0.5,
        'nms_threshold': 0.9,
    },
}


def main():
    parser = evaluation.base_argument_parser('micro-sam automatic segmentation; evaluate on a dataset.')
    parser.add_argument('--model', default='vit_l',
                        help='micro-sam model type, e.g. vit_l (SAM), vit_b_lm / vit_l_lm (light microscopy), '
                             'vit_b_em_organelles (default vit_l)')
    parser.add_argument('--mode', choices=sorted(GENERATE_KWARGS), default='amg',
                        help='amg: automatic mask generation, ais: automatic instance segmentation, '
                             'apg: automatic prompt generation')
    parser.add_argument('--checkpoint', default=None, help='Optional fine-tuned checkpoint for --model')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''))
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=args.model, checkpoint=args.checkpoint, segmentation_mode=args.mode,
        is_tiled=False, device=device,
    )

    def predict(image_bgr, conf):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            segmentation = automatic_instance_segmentation(
                predictor=predictor, segmenter=segmenter, input_path=image_rgb, ndim=2,
                tile_shape=None, halo=None, verbose=False, **GENERATE_KWARGS[args.mode],
            )
        return evaluation.label_map_to_polygons(segmentation)

    settings = {**{k: v for k, v in vars(args).items() if k not in ('dataset', 'no_eval')},
                'generate_kwargs': GENERATE_KWARGS[args.mode]}
    evaluation.run(f'micro_sam_{args.mode}', predict, args.dataset, evaluate=not args.no_eval, settings=settings)


if __name__ == '__main__':
    main()
