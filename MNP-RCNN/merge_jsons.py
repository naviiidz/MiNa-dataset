import json
import os

def merge_coco_jsons(json_files, output_json):
    merged_coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_offset = 0
    annotation_id_offset = 0

    category_mapping = {}
    next_category_id = 0

    for json_file in json_files:
        with open(json_file, 'r') as f:
            coco = json.load(f)

        # Update category mapping
        for category in coco['categories']:
            if category['id'] not in category_mapping:
                category_mapping[category['id']] = next_category_id
                merged_coco['categories'].append({
                    "id": next_category_id,
                    "name": category['name'],
                    "supercategory": category.get('supercategory', '')
                })
                next_category_id += 1

        # Add images with updated ids
        for image in coco['images']:
            new_image = image.copy()
            new_image['id'] = image['id'] + image_id_offset
            merged_coco['images'].append(new_image)

        # Add annotations with updated ids and category ids
        for annotation in coco['annotations']:
            new_annotation = annotation.copy()
            new_annotation['id'] = annotation['id'] + annotation_id_offset
            new_annotation['image_id'] = annotation['image_id'] + image_id_offset
            new_annotation['category_id'] = category_mapping[annotation['category_id']]
            merged_coco['annotations'].append(new_annotation)

        # Update offsets
        image_id_offset += len(coco['images'])
        annotation_id_offset += len(coco['annotations'])

    # Save merged JSON
    with open(output_json, 'w') as f:
        json.dump(merged_coco, f)

# Example usage:
json_files = ['MPDS/Image_patches/all/ps.json', 'MPDS/Image_patches/all/pe.json', 'MPDS/Image_patches/all/pp.json', 'MPDS/Image_patches/all/pet.json']
output_json = 'MPDS/Image_patches/all/all.json'
merge_coco_jsons(json_files, output_json)
