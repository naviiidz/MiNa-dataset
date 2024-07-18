import torch, detectron2
import cv2
import os, json, random
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import shutil

# Setup detectron2 logger
setup_logger()

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)  # Recreate the directory after clearing

def get_MP_dicts(img_dir):
    json_file = os.path.join(img_dir, f"{MP_type.upper()}/{MP_type.lower()}.json")
    with open(json_file, encoding='utf-8-sig') as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for img_data in imgs_anns['images']:
        record = {}

        filename = os.path.join(img_dir, img_data["file_name"])
        height, width = img_data["height"], img_data["width"]

        record["file_name"] = filename
        record["image_id"] = img_data["id"]
        record["height"] = height
        record["width"] = width

        annos = [anno for anno in imgs_anns["annotations"] if anno["image_id"] == img_data["id"]]

        objs = []
        for anno in annos:
            if len(anno["segmentation"]) == 0:
                print(f"Warning: Empty segmentation for image_id {img_data['id']}")
                continue

            segmentation = anno["segmentation"][0]
            if len(segmentation) % 2 != 0:
                print(f"Warning: Odd number of segmentation points for image_id {img_data['id']}")
                continue

            px = segmentation[::2]
            py = segmentation[1::2]
            poly = [(x, y) for x, y in zip(px, py)]

            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [segmentation],  # Use the flat list directly
                "category_id": code_dict[MP_type]
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_gt(mp_type):
    # Load ground truth data
    with open(f'/home/navid/research/MPDetectron2/MPDS/test/{mp_type.upper()}/{mp_type.lower()}.json') as f:
        ground_truth_data = json.load(f)

    # Create the ground truth dictionary
    ground_truth = {}
    images = ground_truth_data['images']
    annotations = ground_truth_data['annotations']
    gt_id_to_name = {image['id']: os.path.splitext(os.path.basename(image['file_name']))[0] for image in images}
    for annotation in annotations:
        image_name = gt_id_to_name[annotation['image_id']]
        bbox_info = {'bbox': annotation['bbox'], 'category_id': code_dict2[annotation['category_id']]}
        if image_name not in ground_truth:
            ground_truth[image_name] = []
        ground_truth[image_name].append(bbox_info)
    return ground_truth

def convert_bbox_format(bbox, from_type="xywh", to_type="xyxy"):
    if from_type == "xywh" and to_type == "xyxy":
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    elif from_type == "xyxy" and to_type == "xywh":
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]
    return bbox

def iou(box1, box2):
    """ Calculate the Intersection over Union (IoU) of two bounding boxes. """
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def calculate_metrics_for_class(ground_truth, predictions, target_class, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0

    for image_name in predictions:
        gt_bboxes = [gt for gt in ground_truth.get(image_name, []) if gt['category_id'] == target_class]
        pred_bboxes = [pred for pred in predictions[image_name] if pred['category_id'] == target_class]

        matched_gt = [False] * len(gt_bboxes)
        matched_pred = [False] * len(pred_bboxes)

        for pred_idx, pred in enumerate(pred_bboxes):
            pred_box = convert_bbox_format(pred['bbox'], from_type="xyxy", to_type="xyxy")
            pred_iou = [iou(pred_box, convert_bbox_format(gt['bbox'], from_type="xywh", to_type="xyxy")) for gt in gt_bboxes]
            max_iou = max(pred_iou) if pred_iou else 0
            max_iou_idx = pred_iou.index(max_iou) if max_iou > 0 else -1

            if max_iou >= iou_threshold and max_iou_idx != -1 and not matched_gt[max_iou_idx]:
                tp += 1
                matched_gt[max_iou_idx] = True
                matched_pred[pred_idx] = True
            else:
                fp += 1

        fn += len([gt for gt in matched_gt if not gt])

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

if __name__=="__main__":
    MP_type = 'PP'
    clear_directory("output2")
    code_dict={'PET': 0, 'PE': 1, 'PP': 2, 'PS': 3}
    code_dict2={1928714511:0, 4258593460:1, 2416516703:2, 151015397:3}

    ground_truth = get_gt(MP_type)

    # Register dataset
    for d in ["test"]:
        json_dir = f"MPDS/{d}"
        DatasetCatalog.register("MP_" + d, lambda d=d: get_MP_dicts(json_dir))
        #MetadataCatalog.get("MP_" + d).set(thing_classes=['MNP'])
        MetadataCatalog.get("MP_" + d).set(thing_classes=['PET', 'PE', 'PP', 'PS'])

    MP_metadata = MetadataCatalog.get("MP_test")

    # Configure model for inference
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Path to the trained model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Number of classes (PET, PE, PP, PS)
    cfg.DATASETS.TEST = ("MP_test",)
    predictor = DefaultPredictor(cfg)

    pred_dict = {}

    # Run inference and visualize results
    for image_name in ground_truth.keys():
        test_images = os.path.join(f"MPDS/test/{MP_type.upper()}", image_name + '.png')
        if not os.path.exists(test_images):
            test_images = os.path.join(f"MPDS/test/{MP_type.upper()}", image_name + '.tif')

        im = cv2.imread(test_images)
        outputs = predictor(im)
        instances = outputs["instances"]

        pred_dict[image_name] = []

        if instances.has("pred_boxes"):
            for i in range(len(instances.pred_boxes)):
                bbox = instances.pred_boxes[i].tensor.tolist()[0]  # Convert tensor to list
                category_id = instances.pred_classes[i].item()  # Get the predicted class
                pred_dict[image_name].append({
                    'bbox': bbox,
                    'category_id': category_id  # Set the predicted class
                })

        v = Visualizer(im[:, :, ::-1],
                    metadata=MP_metadata,
                    scale=2,
                    instance_mode=ColorMode.IMAGE_BW  # Remove the colors of unsegmented pixels. This option is only available for segmentation models
                    )
        out = v.draw_instance_predictions(instances.to("cpu"))
        cv2.imshow("Sample Image", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Evaluate the model
    evaluator = COCOEvaluator("MP_test", output_dir="./output2")
    test_loader = build_detection_test_loader(cfg, "MP_test")
    inference_on_dataset(predictor.model, test_loader, evaluator)

    # Calculate precision, recall, and F1 score for a specific class
    target_class = code_dict[MP_type]
    metrics = calculate_metrics_for_class(ground_truth, pred_dict, target_class)

    print(f"Metrics for class {MP_type} (class ID: {target_class}):")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
