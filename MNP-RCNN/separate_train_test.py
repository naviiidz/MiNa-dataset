import json
import os
import random
import shutil
import cv2
import numpy as np

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)  # Recreate the directory after clearing

def split_dataset(MP_type, new_json, new_images_dir, train_json, val_json, test_json, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15):
    with open(new_json, 'r') as f:
        coco = json.load(f)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if not os.path.exists(os.path.join(train_dir, MP_type)):
        os.makedirs(os.path.join(train_dir, MP_type))
    if not os.path.exists(os.path.exists(val_dir)):
        os.makedirs(os.path.join(val_dir, MP_type))
    if not os.path.exists(os.path.exists(test_dir)):
        os.makedirs(os.path.join(test_dir, MP_type))

    images = coco['images']
    annotations = coco['annotations']

    random.shuffle(images)
    train_size = int(len(images) * train_ratio)
    val_size = int(len(images) * val_ratio)

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    train_annotations = []
    val_annotations = []
    test_annotations = []

    train_image_ids = set(img['id'] for img in train_images)
    val_image_ids = set(img['id'] for img in val_images)
    test_image_ids = set(img['id'] for img in test_images)

    for ann in annotations:
        if ann['image_id'] in train_image_ids:
            train_annotations.append(ann)
        elif ann['image_id'] in val_image_ids:
            val_annotations.append(ann)
        elif ann['image_id'] in test_image_ids:
            test_annotations.append(ann)

    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco['categories']
    }

    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco['categories']
    }

    test_coco = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": coco['categories']
    }

    with open(train_json, 'w') as f:
        json.dump(train_coco, f)

    with open(val_json, 'w') as f:
        json.dump(val_coco, f)

    with open(test_json, 'w') as f:
        json.dump(test_coco, f)

    for img in train_images:
        src_path = os.path.join(new_images_dir, img['file_name'])
        dst_path = os.path.join(train_dir, img['file_name'])
        shutil.copyfile(src_path, dst_path)

    for img in val_images:
        src_path = os.path.join(new_images_dir, img['file_name'])
        dst_path = os.path.join(val_dir, img['file_name'])
        shutil.copyfile(src_path, dst_path)

    for img in test_images:
        src_path = os.path.join(new_images_dir, img['file_name'])
        dst_path = os.path.join(test_dir, img['file_name'])
        shutil.copyfile(src_path, dst_path)

def sep_train_val(MP_type):
    # Usage
    new_json = f'MPDS/Image_patches/{MP_type}/{MP_type.lower()}.json'
    new_images_dir = 'MPDS/Image_patches/'

    train_json = f'MPDS/train/{MP_type}/{MP_type.lower()}.json'
    val_json = f'MPDS/val/{MP_type}/{MP_type.lower()}.json'
    test_json = f'MPDS/test/{MP_type}/{MP_type.lower()}.json'
    train_dir = 'MPDS/train'
    val_dir = 'MPDS/val'
    test_dir = 'MPDS/test'
    clear_directory(os.path.join(train_dir, MP_type))
    clear_directory(os.path.join(val_dir, MP_type))
    clear_directory(os.path.join(test_dir, MP_type))
    split_dataset(MP_type, new_json, new_images_dir, train_json, val_json, test_json, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15)

