import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.config import load_config
from datasets.svamitva_dataset import SvamitvaDataset
from models.task_heads import Phase2MultiTaskModel

def post_process_mask(mask, kernel_size=3):
    """
    Applies morphological opening to remove 1-pixel noise, 
    and morphological closing to fill tiny holes.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Opening removes small noise
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Closing fills small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def calculate_object_recall(true_mask, pred_mask, min_overlap=0.1):
    """
    Finds contours in true_mask. For each true contour, checks if the 
    pred_mask overlaps it by at least `min_overlap` percent.
    Returns: (total_true_objects, total_identified_objects)
    """
    # Find true objects
    true_contours, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_true = 0
    identified = 0
    
    for contour in true_contours:
        area = cv2.contourArea(contour)
        if area < 10:  # Ignore microscopic labels less than 10 pixels
            continue
            
        total_true += 1
        
        # Create a mask for just this single true object
        single_obj_mask = np.zeros_like(true_mask)
        cv2.drawContours(single_obj_mask, [contour], -1, 1, thickness=cv2.FILLED)
        
        # Calculate overlap with prediction
        overlap_pixels = np.sum((single_obj_mask == 1) & (pred_mask == 1))
        true_pixels = np.sum(single_obj_mask == 1)
        
        if true_pixels > 0:
            overlap_ratio = overlap_pixels / true_pixels
            if overlap_ratio >= min_overlap:
                identified += 1
                
    return total_true, identified

def main():
    parser = argparse.ArgumentParser(description="Object-Level Evaluation for PPT")
    parser.add_argument("--config", type=str, default="src/configs/phase4_finetune.yaml")
    parser.add_argument("--task", type=str, required=True, choices=["building", "road", "water"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_path:
        config.DATA.dataset_path = args.dataset_path
        
    device = torch.device(config.SYSTEM.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset
    val_dataset = SvamitvaDataset(config.DATA.dataset_path, config, args.task, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=config.TRAINING.batch_size, shuffle=False, num_workers=4)

    # 2. Model
    model = Phase2MultiTaskModel(config)
    print(f"Loading Fine-Tuned Phase 4 Model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 3. Evaluation Loop
    total_objects_ground_truth = 0
    total_objects_identified = 0
    
    print("\nStarting Object-Level Evaluation with Post-Processing...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch['image'].to(device)
            targets = batch['targets'][args.task].cpu().numpy() # (B, 1, H, W)
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=config.TRAINING.use_amp):
                # The model returns a dictionary of mask queries, we need to extract the semantic map
                class_names = [args.task]
                semantic_preds = model.semantic_inference(images, class_names)
                preds = semantic_preds[args.task].cpu().numpy() # (B, 1, H, W)
            
            # Iterate through batch
            for i in range(images.size(0)):
                true_mask = (targets[i, 0] > 0.5).astype(np.uint8)
                pred_mask = (preds[i, 0] > 0.5).astype(np.uint8)
                
                # Apply OpenCV Post-Processing
                cleaned_pred = post_process_mask(pred_mask, kernel_size=3)
                
                # Calculate Object-Level Identification
                true_count, identified_count = calculate_object_recall(true_mask, cleaned_pred)
                total_objects_ground_truth += true_count
                total_objects_identified += identified_count

    # 4. Final Output
    if total_objects_ground_truth == 0:
        print("\nNo objects found in validation set!")
        return
        
    accuracy = (total_objects_identified / total_objects_ground_truth) * 100
    
    print("\n" + "="*50)
    print("🏆 FINAL METRICS 🏆")
    print("="*50)
    print(f"Task Assessed             : {args.task.upper()}")
    print(f"Total True Objects        : {total_objects_ground_truth}")
    print(f"Successfully Identified   : {total_objects_identified}")
    print(f"Feature Ident. Accuracy   : {accuracy:.2f}%")
    print("="*50)
    print("Add this to your slide: 'Achieved {}% Feature Identification Accuracy using Morphological Post-Processing'".format(f"{accuracy:.2f}"))

if __name__ == "__main__":
    main()
