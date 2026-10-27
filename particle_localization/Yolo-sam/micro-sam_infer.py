import os
import numpy as np
import cv2
import json
import time
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import random

import imageio.v3 as imageio

from micro_sam.util import get_cache_directory
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from micro_sam.sample_data import fetch_hela_2d_example_data, fetch_livecell_example_data, fetch_wholeslide_example_data
from shapely.geometry import Polygon


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def get_device():
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def setup_gpu_acceleration():
    """Setup GPU acceleration optimizations."""
    if torch.cuda.is_available():
        print(f"\n{'='*70}")
        print(f"GPU ACCELERATION SETUP")
        print(f"{'='*70}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.set_float32_matmul_precision('medium')
        
        print("✓ Optimizations Enabled:")
        print(f"  - CuDNN benchmarking: ON")
        print(f"  - Float32 matmul precision: MEDIUM")
        print(f"{'='*70}\n")
        return True
    else:
        print("\n⚠ GPU not available. Running on CPU.")
        return False
def count_parameters(model):
    """Count the number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def extract_polygons_from_mask(segmentation_mask, min_contour_area=5):
    """Extract polygon coordinates from a segmentation mask."""
    polygons = []
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(segmentation_mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    for label in unique_labels:
        # Create binary mask for this instance
        binary_mask = (segmentation_mask == label).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by minimum area
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
            
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to list of [x, y] coordinates
            poly_coords = simplified.reshape(-1, 2).tolist()
            
            if len(poly_coords) >= 3:  # Valid polygon needs at least 3 points
                polygons.append({
                    'coords': poly_coords,
                    'area': area,
                    'contour': contour
                })
    
    return polygons


def process_test_set(model_type, segmentation_mode, generate_kwargs, checkpoint=None):
    """Process all images in the test set and save prediction overlays in YOLO-SAM format with GPU acceleration."""
    test_dir = Path("../../data/patches/yolo_seg/images/test")
    
    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} does not exist.")
        return
    
    # Setup GPU acceleration
    has_gpu = setup_gpu_acceleration()
    device = get_device()
    
    # Create output directory based on segmentation mode
    output_dir = f"{segmentation_mode}_usam_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictor and segmenter
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=checkpoint,
        segmentation_mode=segmentation_mode,
        is_tiled=False,
    )
    
    # Move models to GPU if available
    if has_gpu and hasattr(predictor, 'model'):
        predictor.model = predictor.model.to(device)
        predictor.model.eval()
        print(f"✓ μSAM model moved to {device}")
    
    # Count and display model parameters
    total_params, trainable_params = count_parameters(predictor.model)
    print(f"\n{'='*60}")
    print(f"Model: {model_type}")
    print(f"Segmentation Mode: {segmentation_mode}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Get all image files
    image_files = sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return
    
    print(f"Processing {len(image_files)} images from test set...\n")
    
    # Lists to store results for CSV export
    all_polygon_results = []
    image_summaries = []
    inference_times = []
    
    overall_start_time = time.time()
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"[{idx}/{len(image_files)}] Processing {image_path.name}...")
            image_name = image_path.stem
            
            # Read image
            image = imageio.imread(image_path)
            
            # Generate segmentation with timing and GPU acceleration
            segmentation_start_time = time.time()
            
            # Use GPU acceleration context
            with torch.no_grad():  # Disable gradient computation for faster inference
                if has_gpu:
                    # Use Automatic Mixed Precision (AMP) for faster GPU inference
                    with torch.cuda.amp.autocast():
                        segmentation = automatic_instance_segmentation(
                            predictor=predictor,
                            segmenter=segmenter,
                            input_path=image,
                            ndim=2,
                            tile_shape=None,
                            halo=None,
                            **generate_kwargs
                        )
                else:
                    segmentation = automatic_instance_segmentation(
                        predictor=predictor,
                        segmenter=segmenter,
                        input_path=image,
                        ndim=2,
                        tile_shape=None,
                        halo=None,
                        **generate_kwargs
                    )
            
            segmentation_time = time.time() - segmentation_start_time
            inference_times.append(segmentation_time)
            
            # Extract polygons from segmentation mask
            polygons = extract_polygons_from_mask(segmentation, min_contour_area=5)
            
            # Prepare polygon data for CSV
            for poly_idx, poly_info in enumerate(polygons):
                poly_coords = poly_info['coords']
                area = poly_info['area']
                
                # Calculate bounding box
                pts = np.array(poly_coords, np.int32)
                x, y, w, h = cv2.boundingRect(pts.reshape((-1, 1, 2)))
                
                # Calculate centroid using shapely
                try:
                    shapely_poly = Polygon(poly_coords)
                    centroid = shapely_poly.centroid
                    cx, cy = centroid.x, centroid.y
                except:
                    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
                
                polygon_result = {
                    'image_name': image_path.name,
                    'polygon_id': poly_idx,
                    'area_pixels': round(area, 2),
                    'bbox_x': int(x),
                    'bbox_y': int(y),
                    'bbox_width': int(w),
                    'bbox_height': int(h),
                    'centroid_x': round(cx, 2),
                    'centroid_y': round(cy, 2),
                    'num_vertices': len(poly_coords),
                    'polygon_coords': json.dumps(poly_coords)
                }
                all_polygon_results.append(polygon_result)
            
            # Create image summary
            image_summary = {
                'image_name': image_path.name,
                'total_detections': len(polygons),
                'total_area_pixels': sum([p['area'] for p in polygons]),
                'avg_particle_area': round(np.mean([p['area'] for p in polygons]), 2) if polygons else 0,
            }
            image_summaries.append(image_summary)
            
            # Create visualization (similar to YOLO-SAM)
            if image.dtype != np.uint8:
                if image.max() <= 1:
                    image_uint8 = (image * 255).astype(np.uint8)
                else:
                    image_uint8 = image.astype(np.uint8)
            else:
                image_uint8 = image
            
            # Convert to RGB if grayscale
            if len(image_uint8.shape) == 2:
                image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image_uint8
            
            # Create overlay visualization
            overlay = image_rgb.copy()
            random.seed(42)
            colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                     for _ in range(len(polygons))]
            
            # Draw filled polygons
            for i, (poly_info, color) in enumerate(zip(polygons, colors)):
                pts = np.array(poly_info['coords'], np.int32)
                cv2.fillPoly(overlay, [pts], color)
            
            # Draw polygon outlines
            for i, (poly_info, color) in enumerate(zip(polygons, colors)):
                pts = np.array(poly_info['coords'], np.int32)
                cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
            
            # Create final composite image
            final_img = cv2.addWeighted(overlay, 0.5, image_rgb, 0.5, 0)
            
            # Save visualization
            plt.figure(figsize=(12, 12))
            plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
            plt.title(f"μSAM Predictions: {len(polygons)} particles detected (Image: {image_path.name})")
            plt.axis('off')
            
            output_path = os.path.join(output_dir, f"output_{image_name}_{segmentation_mode}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Visualization saved: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {str(e)}\n")
            continue
    
    # Export results to CSV (matching YOLO-SAM format)
    if all_polygon_results:
        df_detailed = pd.DataFrame(all_polygon_results)
        detailed_csv_path = os.path.join(output_dir, 'detailed_predictions.csv')
        df_detailed.to_csv(detailed_csv_path, index=False)
        print(f"\n✓ Detailed predictions saved: {detailed_csv_path}")
    
    if image_summaries:
        df_summary = pd.DataFrame(image_summaries)
        summary_csv_path = os.path.join(output_dir, f'image_summaries_{segmentation_mode}.csv')
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"✓ Image summaries saved: {summary_csv_path}")
    
    # Calculate and report inference time
    overall_time = time.time() - overall_start_time
    avg_inference_time = np.mean(inference_times) if inference_times else 0.0
    total_inference_time = sum(inference_times)
    
    print(f"\n{'='*60}")
    print(f"INFERENCE TIME SUMMARY")
    print(f"{'='*60}")
    print(f"Total Processing Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print(f"Total Inference Time: {total_inference_time:.2f} seconds ({total_inference_time/60:.2f} minutes)")
    print(f"Average Time per Image: {avg_inference_time:.3f} seconds")
    print(f"Images Processed: {len(inference_times)}")
    if len(inference_times) > 1:
        print(f"Inference Time - Min: {np.min(inference_times):.3f}s, Max: {np.max(inference_times):.3f}s, Std: {np.std(inference_times):.3f}s")
    print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete! Results saved to: {output_dir}")
    print(f"Total detections: {len(all_polygon_results)} across {len(image_summaries)} images")
    print(f"{'='*60}")
    
    # Clean up GPU memory if using CUDA
    if has_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU memory cleaned up\n")



def livecell_automatic_segmentation(model_type, segmentation_mode, generate_kwargs, checkpoint=None):
    """Run the automatic segmentation for an example image from the LIVECell dataset.

    See https://doi.org/10.1038/s41592-021-01249-6 for details on the data.
    """
    example_data = fetch_livecell_example_data(DATA_CACHE)
    
    #image = imageio.imread(example_data)
    # Read PS microplastics image instead
    image = imageio.imread("../../data/patches/yolo_seg/images/test/pet_PET-2000X_patch_000_000.png")  # Replace with the path to your microplastics image

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=checkpoint,  # Use the custom fine-tuned checkpoint
        segmentation_mode=segmentation_mode,
        is_tiled=False,  # Switch to 'True' in case you would like to perform tiling-window based prediction.
    )

    # Count and display model parameters
    total_params, trainable_params = count_parameters(predictor.model)
    print(f"\n{'='*60}")
    print(f"Model: {model_type}")
    print(f"Segmentation Mode: {segmentation_mode}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*60}\n")

    segmentation = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        tile_shape=None,  # If you set 'is_tiled' in 'get_predictor_and_segmeter' to True, set a tile shape
        halo=None,  # If you set 'is_tiled' in 'get_predictor_and_segmeter' to True, set a halo shape.
        **generate_kwargs
    )

    import napari
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation)
    napari.run()


def hela_automatic_segmentation(model_type, segmentation_mode, generate_kwargs):
    """Run the automatic segmentation for an example image from the Cell Tracking Challenge (HeLa 2d) dataset.
    """
    example_data = fetch_hela_2d_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=None,  # Replace this with your custom checkpoint.
        segmentation_mode=segmentation_mode,
        is_tiled=False,  # Switch to 'True' in case you would like to perform tiling-window based prediction.
    )

    segmentation = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        tile_shape=None,  # If you set 'is_tiled' in 'get_predictor_and_segmeter' to True, set a tile shape
        halo=None,  # If you set 'is_tiled' in 'get_predictor_and_segmeter' to True, set a halo shape.
        **generate_kwargs
    )

    print(segmentation)
    # import napari
    # v = napari.Viewer()
    # v.add_image(image)
    # v.add_labels(segmentation)
    # napari.run()


def wholeslide_automatic_segmentation(model_type, segmentation_mode, generate_kwargs):
    """Run the automatic segmentation with tiling for an example whole-slide image from the
    NeurIPS Cell Segmentation challenge.
    """
    example_data = fetch_wholeslide_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=None,  # Replace this with your custom checkpoint.
        segmentation_mode=segmentation_mode,
        is_tiled=True,
    )

    segmentation = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        tile_shape=(1024, 1024),
        halo=(256, 256),
        **generate_kwargs
    )

    import napari
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation)
    napari.run()


def main():
    # The choice of Segment Anything model.
    # SEM models, these are not working well
    model_type = "vit_l_em_organelles" #Model for mitochodria and nuclei in electron microscopy data with ViT Large image encoder
    model_type = "vit_b_em_organelles" #Model for mitochodria and nuclei in electron microscopy data with ViT Base
    model_type = "vit_t_em_organelles" #Model for mitochodria and nuclei in electron microscopy data with ViT Tiny

    # The light microscopy generalist models from Segment Anything for Microscopy
    # Base model is the best model here
    model_type = "vit_l" 
    #model_type = "vit_b_lm" 
    #model_type = "vit_t_lm" 

    # The medical imaging generalist models from MedicoSAM 
    # 
    # model_type = "vit_b_medical_imaging" 

    # Use the fine-tuned microplastics model
    # model_type = "vit_t"
    # # Use the checkpoint directly (includes decoder)
    # checkpoint_path = "/home/navid/research/uSAM/micro-sam/examples/finetuning/checkpoints/sam_microplastics/best.pt"


    # The histopathology generalist models from PathoSAM 
    # model_type = "vit_h_histopathology"
    # model_type = "vit_l_histopathology"
    # model_type = "vit_b_histopathology"
    



    
    # the automatic mask generation (AMG): supported by all our models.
    # the automatic instance segmentation (AIS): supported by 'micro-sam' models.
    # the automatic prompt generation (APG): supported by 'micro-sam' models.
    segmentation_mode = "amg"  # available choices for automatic segmentation modes are 'amg' / 'ais' / 'apg'.

    # Post-processing parameters for automatic segmentation.
    if segmentation_mode == "amg":  # AMG parameters - optimized for microplastics
        generate_kwargs = {
            "pred_iou_thresh": 0.5,               # Lowered from 0.88 for better recall on small objects
            "stability_score_thresh": 0.85,        # Lowered from 0.95 to accept more candidates
            "box_nms_thresh": 0.65,                # Tightened from 0.7 for less overlapping masks
            "crop_nms_thresh": 0.65,               # Tightened from 0.7 to avoid redundancy
            "min_mask_region_area": 20,            # Added filtering to remove tiny noise (>20 pixels)
        }
    elif segmentation_mode == "ais":  # AIS parameters - optimized for microplastics
        generate_kwargs = {
            "center_distance_threshold": 0.4,      # Tightened from 0.5 to prevent merging
            "boundary_distance_threshold": 0.4,    # Tightened from 0.5 for better separation
            "foreground_threshold": 0.7,           # Increased from 0.5 to reduce false positives
            "foreground_smoothing": 1.0,           # Keep smooth for robustness
            "distance_smoothing": 1.4,             # Slightly reduced from 1.6 for clarity
            "min_size": 15,                        # Added filtering to remove tiny noise (>15 pixels)
        }
    elif segmentation_mode == "apg":  # APG parameters
        generate_kwargs = {
            "center_distance_threshold": 0.5,
            "boundary_distance_threshold": 0.5,
            "foreground_threshold": 0.5,
            "nms_threshold": 0.9,
        }
    else:
        raise ValueError("The selected 'segmentation_mode' is not a supported segmentation method.")

    # Process all test set images and save prediction overlays
    process_test_set(model_type, segmentation_mode, generate_kwargs)#, checkpoint=checkpoint_path)

    # Automatic segmentation for livecell data.
    # livecell_automatic_segmentation(model_type, segmentation_mode, generate_kwargs)#, checkpoint=checkpoint_path)

    # Automatic segmentation for cell tracking challenge hela data.
    # hela_automatic_segmentation(model_type, segmentation_mode, generate_kwargs)

    # Automatic segmentation for a whole slide image.
    # wholeslide_automatic_segmentation(model_type, segmentation_mode, generate_kwargs)


# The corresponding CLI call for hela_automatic_segmentation:
# (replace with cache directory on your machine)
# $ micro_sam.automatic_segmentation -i /home/pape/.cache/micro_sam/sample_data/hela-2d-image.png -o hela-2d-image_segmentation.tif  # noqa
if __name__ == "__main__":
    main()