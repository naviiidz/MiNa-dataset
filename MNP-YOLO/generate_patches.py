import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

def read_bounding_boxes(file_path):
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                bboxes.append(list(map(float, parts)))
    return bboxes

def write_bounding_boxes(file_path, bboxes):
    with open(file_path, 'w') as f:
        for bbox in bboxes:
            f.write(' '.join(map(str, bbox)) + '\n')

def is_normalized(bbox):
    _, x_center, y_center, width, height = bbox
    return 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1

def remove_duplicates(bboxes):
    return [list(x) for x in set(tuple(x) for x in bboxes)]

def adjust_bboxes(bboxes, x_offset, y_offset, patch_size, original_width, original_height, category_id):
    adjusted_bboxes = []
    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox
        # Convert normalized coordinates to absolute coordinates
        x_center_abs = x_center * original_width
        y_center_abs = y_center * original_height
        width_abs = width * original_width
        height_abs = height * original_height
        
        # Adjust coordinates relative to the patch
        x_center_new = x_center_abs - x_offset
        y_center_new = y_center_abs - y_offset

        # Check if the bounding box center is within the patch
        if 0 <= x_center_new <= patch_size and 0 <= y_center_new <= patch_size:
            new_bbox = [
                category_id,
                x_center_new / patch_size,
                y_center_new / patch_size,
                width_abs / patch_size,
                height_abs / patch_size
            ]
            if is_normalized(new_bbox):
                adjusted_bboxes.append(new_bbox)
    return remove_duplicates(adjusted_bboxes)

def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def generate_patches(image_dir, bbox_dir, output_image_dir, category_id, clear_previous_data=True, patch_size=256, max_processing_height=600):
    if clear_previous_data:
        clear_directory(output_image_dir)
    
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
        
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif') or f.endswith('.png')]
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        bbox_path = os.path.join(bbox_dir, os.path.splitext(image_file)[0] + '.txt')

        if not os.path.exists(bbox_path):
            continue

        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Ensure we don't process patches from the bottom part with the scale bar
        effective_height = min(height, max_processing_height)
        
        bboxes = read_bounding_boxes(bbox_path)

        for y in range(0, effective_height, patch_size):
            for x in range(0, width, patch_size):
                patch = image[y:y+patch_size, x:x+patch_size]
                patch_bboxes = adjust_bboxes(bboxes, x, y, patch_size, width, height, category_id)
                
                # Skip patches with no annotations
                if not patch_bboxes:
                    continue
                
                patch_file_name = f"{os.path.splitext(image_file)[0]}_{x}_{y}.jpg"
                patch_file_path = os.path.join(output_image_dir, patch_file_name)
                patch_bbox_file_path = os.path.splitext(patch_file_path)[0] + '.txt'

                cv2.imwrite(patch_file_path, patch)
                write_bounding_boxes(patch_bbox_file_path, patch_bboxes)
                print(f"Saved patch {patch_file_name} with {len(patch_bboxes)} bounding boxes.")

def save_file_lists(train_files, val_files, test_files, main_dir):
    def write_list(file_path, files, subset):
        with open(file_path, 'w') as f:
            for item in files:
                relative_image_path = os.path.join(f"./images/{subset}", item)
                f.write(f"{relative_image_path}\n")
    
    write_list(os.path.join(main_dir, 'train.txt'), train_files, 'train')
    write_list(os.path.join(main_dir, 'val.txt'), val_files, 'val')
    write_list(os.path.join(main_dir, 'test.txt'), test_files, 'test')

def split_data(output_image_dir, output_lbl_dir, train_ratio=0.7, val_ratio=0.2, clear_previous_data=True):
    if clear_previous_data:
        clear_directory(os.path.join(output_image_dir, 'train'))
        clear_directory(os.path.join(output_image_dir, 'val'))
        clear_directory(os.path.join(output_image_dir, 'test'))
        clear_directory(os.path.join(output_lbl_dir, 'train'))
        clear_directory(os.path.join(output_lbl_dir, 'val'))
        clear_directory(os.path.join(output_lbl_dir, 'test'))    
    image_files = [f for f in os.listdir(output_image_dir) if f.endswith('.jpg')]
    
    train_files, test_files, _, _ = train_test_split(image_files, image_files, test_size=1-train_ratio, random_state=42)
    val_files, test_files, _, _ = train_test_split(test_files, test_files, test_size=(1 - train_ratio - val_ratio) / (1 - train_ratio), random_state=42)
    
    def move_files(files, subset_image_dir, subset_lbl_dir):
        if not os.path.exists(subset_image_dir):
            os.makedirs(subset_image_dir)
        for f in files:
            shutil.move(os.path.join(output_image_dir, f), os.path.join(subset_image_dir, f))
            shutil.move(os.path.splitext(os.path.join(output_image_dir, f))[0] + '.txt', os.path.join(subset_lbl_dir, os.path.splitext(f)[0] + '.txt'))
    
    move_files(train_files, os.path.join(output_image_dir, 'train'), os.path.join(output_lbl_dir, 'train'))
    move_files(val_files, os.path.join(output_image_dir, 'val'), os.path.join(output_lbl_dir, 'val'))
    move_files(test_files, os.path.join(output_image_dir, 'test'), os.path.join(output_lbl_dir, 'test'))
    
    save_file_lists(train_files, val_files, test_files, os.path.dirname(output_image_dir))

if __name__ == "__main__":
    image_directory = "yolo_images/PE"
    bbox_directory = "yolo_images/PE"
    output_image_directory = "images"
    output_lbl_dir = "labels"
    category_id = 0  # Set the desired category ID
    clear_previous_data = True  # Set to False to keep existing data and add new data on top
    
    # Generate patches
    generate_patches(image_directory, bbox_directory, output_image_directory, category_id, clear_previous_data)
    
    # Split data into Train, Test, Validation subsets
    split_data(output_image_directory, output_lbl_dir, clear_previous_data=clear_previous_data)
