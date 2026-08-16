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

def calculate_object_metrics(true_mask, pred_mask, min_overlap=0.1):
    """
    Calculates Object-Level True Positives (Recall) and False Positives (Precision)
    """
    # Find contours
    true_contours, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    true_positives = 0
    total_true = 0
    
    # Calculate Recall (True Positives)
    for contour in true_contours:
        if cv2.contourArea(contour) < 50:  # Ground truth might have small valid fragments
            continue
        total_true += 1
        
        single_obj_mask = np.zeros_like(true_mask)
        cv2.drawContours(single_obj_mask, [contour], -1, 1, thickness=cv2.FILLED)
        
        overlap_pixels = np.sum((single_obj_mask == 1) & (pred_mask == 1))
        true_pixels = np.sum(single_obj_mask == 1)
        
        if true_pixels > 0 and (overlap_pixels / true_pixels) >= min_overlap:
            true_positives += 1
            
    # Calculate False Positives (Requires strict size filtering)
    false_positives = 0
    total_pred = 0
    
    for contour in pred_contours:
        # Strict Size Filtering: Delete any predicted "object" smaller than 250 pixels (a ~15x15 smudge)
        if cv2.contourArea(contour) < 250:
            continue
        total_pred += 1
        
        single_pred_mask = np.zeros_like(pred_mask)
        cv2.drawContours(single_pred_mask, [contour], -1, 1, thickness=cv2.FILLED)
        
        overlap_pixels = np.sum((single_pred_mask == 1) & (true_mask == 1))
        pred_pixels = np.sum(single_pred_mask == 1)
        
        if pred_pixels > 0 and (overlap_pixels / pred_pixels) < min_overlap:
            false_positives += 1
            
    return total_true, true_positives, total_pred, false_positives

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

    val_dataset = SvamitvaDataset(config.DATA.dataset_path, config, args.task, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=config.TRAINING.batch_size, shuffle=False, num_workers=4)

    model = Phase2MultiTaskModel(config)
    print(f"Loading Fine-Tuned Phase 4 Model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    total_true_objs = 0
    total_tp_objs = 0
    total_pred_objs = 0
    total_fp_objs = 0
    
    total_pixel_correct = 0
    total_pixels = 0
    total_intersection = 0
    total_union = 0
    
    print("\nStarting Object-Level Evaluation with Post-Processing...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch['image'].to(device)
            targets = batch['targets'][args.task].cpu().numpy()
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=config.TRAINING.use_amp):
                semantic_preds = model.semantic_inference(images, [args.task])
                preds = semantic_preds[args.task].cpu().numpy()
            
            for i in range(images.size(0)):
                true_mask = (targets[i, 0] > 0.5).astype(np.uint8)
                pred_mask = (preds[i, 0] > 0.5).astype(np.uint8)
                
                cleaned_pred = post_process_mask(pred_mask, kernel_size=3)
                
                # Object Metrics
                t_true, t_tp, t_pred, t_fp = calculate_object_metrics(true_mask, cleaned_pred)
                total_true_objs += t_true
                total_tp_objs += t_tp
                total_pred_objs += t_pred
                total_fp_objs += t_fp
                
                # Pixel Metrics
                total_pixel_correct += np.sum(true_mask == cleaned_pred)
                total_pixels += true_mask.size
                total_intersection += np.sum((true_mask == 1) & (cleaned_pred == 1))
                total_union += np.sum((true_mask == 1) | (cleaned_pred == 1))

    if total_true_objs == 0:
        print("\nNo objects found in validation set!")
        return
        
    # Calculate Final Object Metrics
    obj_recall = (total_tp_objs / total_true_objs) * 100 if total_true_objs > 0 else 0
    obj_precision = (total_tp_objs / (total_tp_objs + total_fp_objs)) * 100 if (total_tp_objs + total_fp_objs) > 0 else 0
    obj_f1 = 2 * (obj_precision * obj_recall) / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0
    
    # Calculate Final Pixel Metrics
    pixel_acc = (total_pixel_correct / total_pixels) * 100
    pixel_iou = (total_intersection / total_union) * 100 if total_union > 0 else 0
    
    print("\n" + "="*50)
    print("🏆 FINAL METRICS 🏆")
    print("="*50)
    print(f"Task Assessed             : {args.task.upper()}")
    print("-" * 50)
    print("OBJECT-LEVEL METRICS (Feature Identification)")
    print(f"Total True Objects        : {total_true_objs}")
    print(f"Successfully Identified   : {total_tp_objs}")
    print(f"False Positives (Noise)   : {total_fp_objs}")
    print(f"Object Recall (Accuracy)  : {obj_recall:.2f}%")
    print(f"Object Precision          : {obj_precision:.2f}%")
    print(f"Object F1-Score           : {obj_f1:.2f}%")
    print("-" * 50)
    print("PIXEL-LEVEL METRICS (Post-Processed)")
    print(f"Pixel Accuracy            : {pixel_acc:.2f}%")
    print(f"Pixel IoU                 : {pixel_iou:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
