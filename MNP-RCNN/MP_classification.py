import patch_dataset_creation_non_random as patch_gen
import separate_train_test as sep_train_test

import torch, detectron2
import cv2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# Import common libraries
import numpy as np
import os, json, cv2, random

# Import common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import shutil

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)  # Recreate the directory after clearing

def get_MP_dicts(img_dir):
    dataset_dicts = []
    for MP_class in ['PET', 'PE', 'PP', 'PS']:
        json_file = os.path.join(img_dir, f"{MP_class}/{MP_class.lower()}.json")
        with open(json_file, encoding='utf-8-sig') as f:
            imgs_anns = json.load(f)

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
                    "category_id": code_dict[anno["category_id"]],  # Adjust the category_id to start from 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

clear_directory("output")
max_overlap=10
min_particle_size=10
patch_size=256
#code_dict={'PET': 1928714511, 'PE': 4258593460, 'PP': 2416516703, 'PS': 151015397}
code_dict={1928714511:0, 4258593460:1, 2416516703:2, 151015397:3}
sep_train_test.sep_train_val('all')

for d in ["train", "val", "test"]:
    json_dir=f"MPDS/{d}"
    DatasetCatalog.register("MP_" + d, lambda d=d: get_MP_dicts(json_dir))
    MetadataCatalog.get("MP_" + d).set(thing_classes=['PET', 'PE', 'PP', 'PS'])
MP_metadata = MetadataCatalog.get("MP_train")

print(MP_metadata)

# Calculate TOTAL_NUM_IMAGES dynamically
train_dataset_dicts = get_MP_dicts(f"MPDS/train")
TOTAL_NUM_IMAGES = len(train_dataset_dicts)

dataset_dicts = get_MP_dicts(json_dir)
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    print(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=MP_metadata, scale=2)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("Sample Image", out.get_image()[:, :, ::-1])  # Added window name here
    cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()  # Close all OpenCV windows

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("MP_train",)
cfg.DATASETS.TEST = ("MP_test",)
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Load pre-trained model

# Set anchor scales and ratios
# cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]  # The anchor sizes

cfg.SOLVER.IMS_PER_BATCH = 16  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.0015  # pick a good LR
single_iteration = 1 * cfg.SOLVER.IMS_PER_BATCH
iterations_for_one_epoch = TOTAL_NUM_IMAGES / single_iteration

cfg.SOLVER.MAX_ITER = int(iterations_for_one_epoch * 50)  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
print(cfg.SOLVER.MAX_ITER, "*Iterations")
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # four classes (PET, PE, PP, PS)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Inference should use the config with parameters that are used in training
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_MP_dicts("MPDS/test")
for d in random.sample(dataset_dicts, 8):
    im = cv2.imread(d["file_name"])
    print(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=MP_metadata,
                   scale=2,
                   instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Sample Image", out.get_image()[:, :, ::-1])  # Added window name here
    cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()  # Close all OpenCV windows

evaluator = COCOEvaluator("MP_test", output_dir="./output")
test_loader = build_detection_test_loader(cfg, "MP_test")
print(inference_on_dataset(predictor.model, test_loader, evaluator))
