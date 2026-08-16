import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from utils.config import load_config
from models.task_heads import Phase2MultiTaskModel

def load_model(config_path, model_path, device):
    """Loads a specific model based on its configuration file and Kaggle weights path."""
    config = load_config(config_path)
    model = Phase2MultiTaskModel(config)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights file {model_path} not found. Please check Kaggle paths.")
        
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def post_process_and_find_polygons(mask, kernel_size=3, min_area=250):
    """
    Applies OpenCV Morphological cleaning and extracts polygons larger than min_area.
    """
    # 1. Morphological Cleaning
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # 2. Extract Polygons
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Size Filtering
    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            valid_contours.append(contour)
            
    return valid_contours

def draw_polygons(image, contours, color_rgb, fill_alpha=0.3):
    """
    Draws semi-transparent filled polygons with solid borders on the image.
    Color should be RGB e.g., (255, 0, 0) for Red.
    """
    overlay = image.copy()
    
    # Draw filled polygons
    cv2.fillPoly(overlay, contours, color=color_rgb)
    
    # Blend with original image to make it transparent
    cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0, image)
    
    # Draw solid borders
    cv2.drawContours(image, contours, -1, color=color_rgb, thickness=2)
    
    return image

def main():
    parser = argparse.ArgumentParser(description="Kaggle Demo: Polygon Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input drone image")
    parser.add_argument("--building_model", type=str, required=True, help="Path to building phase2_best.pt")
    parser.add_argument("--road_model", type=str, required=True, help="Path to road phase2_best.pt")
    parser.add_argument("--output", type=str, default="demo_output.png", help="Path to save visual output")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Models
    print("Initializing specialized experts...")
    building_model = load_model("src/configs/phase4_finetune.yaml", args.building_model, device)
    road_model = load_model("src/configs/phase4_finetune.yaml", args.road_model, device)
    
    # 2. Prepare Image
    img_pil = Image.open(args.image).convert('RGB')
    
    # Resize to 512x512 as required by the model
    img_tensor = TF.resize(img_pil, [512, 512], interpolation=TF.InterpolationMode.BILINEAR)
    img_tensor = TF.to_tensor(img_tensor).unsqueeze(0).to(device) # [1, 3, 512, 512]
    
    # 3. Run AI Inference
    print("Extracting AI Feature Maps...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
            b_preds = building_model.semantic_inference(img_tensor, ['building'])
            r_preds = road_model.semantic_inference(img_tensor, ['road'])
            
    # Extract binary masks (threshold at 0.5)
    raw_b_mask = (b_preds['building'][0, 0] > 0.5).cpu().numpy().astype(np.uint8)
    raw_r_mask = (r_preds['road'][0, 0] > 0.5).cpu().numpy().astype(np.uint8)
    
    # 4. Post-Processing & Polygon Extraction
    print("Running OpenCV Post-Processing and Size Thresholding...")
    building_polygons = post_process_and_find_polygons(raw_b_mask, kernel_size=3, min_area=250)
    road_polygons = post_process_and_find_polygons(raw_r_mask, kernel_size=3, min_area=250)
    
    print(f"Detected {len(building_polygons)} valid buildings.")
    print(f"Detected {len(road_polygons)} valid roads.")
    
    # 5. Visualization
    print("Rendering High-Fidelity Polygons...")
    # Convert original resized image to numpy array (RGB)
    original_resized = TF.resize(img_pil, [512, 512], interpolation=TF.InterpolationMode.BILINEAR)
    viz_image = np.array(original_resized)
    
    # Draw Buildings in Red (RGB: 255, 0, 0)
    viz_image = draw_polygons(viz_image, building_polygons, color_rgb=(255, 0, 0), fill_alpha=0.3)
    
    # Draw Roads in Green (RGB: 0, 255, 0)
    viz_image = draw_polygons(viz_image, road_polygons, color_rgb=(0, 255, 0), fill_alpha=0.3)
    
    # 6. Plot and Save
    plt.figure(figsize=(18, 9))
    
    # Subplot 1: Original
    plt.subplot(1, 2, 1)
    plt.title("Original Drone Orthophoto (512x512)", fontsize=16)
    plt.imshow(original_resized)
    plt.axis('off')
    
    # Subplot 2: AI Predictions
    plt.subplot(1, 2, 2)
    plt.title("AI Pipeline Outputs (Mask2Former + Post-Processing)", fontsize=16)
    plt.imshow(viz_image)
    
    # Add a legend manually since we drew raw pixels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(1, 0, 0, 0.5), edgecolor='red', label=f'Buildings ({len(building_polygons)})'),
        Patch(facecolor=(0, 1, 0, 0.5), edgecolor='green', label=f'Roads ({len(road_polygons)})')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n✅ Presentation slide successfully saved to: {args.output}")

if __name__ == "__main__":
    main()
