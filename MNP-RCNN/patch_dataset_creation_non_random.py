import json
import os
import cv2
import numpy as np
import shutil

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)  # Recreate the directory after clearing

def get_bbox_from_mask(mask):
    y_indices, x_indices = np.where(mask)
    if y_indices.size > 0 and x_indices.size > 0:
        x_min = int(np.min(x_indices))
        x_max = int(np.max(x_indices))
        y_min = int(np.min(y_indices))
        y_max = int(np.max(y_indices))
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    else:
        return None

def get_mask_from_polygon(segmentation, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=1)
    return mask

def create_new_coco_dataset(source_json, source_images_dir, new_json, new_images_dir, patch_size):
    with open(source_json, 'r') as f:
        coco = json.load(f)

    if not os.path.exists(new_images_dir):
        os.makedirs(new_images_dir)

    new_coco = {
        "images": [],
        "annotations": [],
        "categories": coco['categories']
    }

    annotation_id = 1
    image_id = 1

    for img in coco['images']:
        image_path = os.path.join(source_images_dir, img['file_name'])
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        height = 800  # Adjust this value based on your actual image height if needed

        for patch_y in range(0, height, patch_size):
            for patch_x in range(0, width, patch_size):
                if patch_x + patch_size <= width and patch_y + patch_size <= height:
                    patch = image[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]

                    contains_annotation = False
                    new_annotations = []

                    for ann in coco['annotations']:
                        if ann['image_id'] == img['id']:
                            segmentation = ann['segmentation']
                            mask = get_mask_from_polygon(segmentation, width, height)
                            patch_mask = mask[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]

                            if patch_mask.sum() > 0:
                                original_area = mask.sum()
                                patch_area = patch_mask.sum()

                                if patch_area / original_area >= 0.7:
                                    new_segmentation = []
                                    for polygon in segmentation:
                                        new_polygon = []
                                        for i in range(0, len(polygon), 2):
                                            x = polygon[i] - patch_x
                                            y = polygon[i + 1] - patch_y
                                            if 0 <= x < patch_size and 0 <= y < patch_size:
                                                new_polygon.append(x)
                                                new_polygon.append(y)
                                        if len(new_polygon) >= 6:  # at least three points to form a polygon
                                            new_segmentation.append(new_polygon)

                                    if new_segmentation:
                                        new_ann = ann.copy()
                                        new_ann['image_id'] = image_id
                                        new_ann['id'] = annotation_id
                                        new_ann['segmentation'] = new_segmentation
                                        new_ann['bbox'] = get_bbox_from_mask(patch_mask)
                                        new_ann["area"] = int(np.sum(patch_mask))
                                        new_ann["is_crowd"] = 0
                                        #code_dict={'PET': 1928714511, 'PE': 4258593460, 'PP': 2416516703, 'PS': 151015397}
                                        #code_dict={1928714511:0, 4258593460:1, 151015397:2, 2416516703:3}
                                        new_ann["category_id"] = ann["category_id"]
                                        new_annotations.append(new_ann)
                                        contains_annotation = True
                                        annotation_id += 1

                    if contains_annotation:
                        new_image_path = os.path.join(new_images_dir, img['file_name'].replace(".png", f"_{image_id}.png"))
                        cv2.imwrite(new_image_path, patch)

                        new_image_info = {
                            "id": image_id,
                            "file_name": img['file_name'].replace(".png", f"_{image_id}.png"),
                            "width": patch_size,
                            "height": patch_size
                        }
                        new_coco['images'].append(new_image_info)
                        new_coco['annotations'].extend(new_annotations)
                        image_id += 1

    with open(new_json, 'w') as f:
        json.dump(new_coco, f)

def gen_rand_patches(MP_type, patch_size, min_size):
    source_json = f'MPDS/Original_Images/{MP_type}/{MP_type.lower()}_fixed.json'
    source_images_dir = f'MPDS/Original_Images/'
    new_json = f'MPDS/Image_patches/{MP_type}/{MP_type.lower()}.json'
    new_images_dir = 'MPDS/Image_patches/'

    clear_directory(f'MPDS/Image_patches/{MP_type}')
    print("Train Test cleared")
    create_new_coco_dataset(source_json, source_images_dir, new_json, new_images_dir, patch_size)

#gen_rand_patches("PET", 256, 10)
