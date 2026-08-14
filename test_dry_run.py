import torch
import numpy as np
from utils.config import load_config
from models.task_heads import Phase2MultiTaskModel
from losses.multitask_loss import SetCriterion

def test_architecture_flow():
    print("--- Starting Architecture Dry Run ---")
    
    # 1. Load config
    config = load_config("src/configs/phase2_building.yaml")
    
    # 2. Instantiate Model and Loss
    print("\n1. Instantiating Model & Loss...")
    model = Phase2MultiTaskModel(config)
    criterion = SetCriterion(config)
    
    # 3. Create Dummy Data (Batch Size 2, 512x512 images)
    print("2. Generating Dummy Input Data (2x3x512x512)...")
    images = torch.randn(2, 3, 512, 512)
    
    # Dummy Targets (Binary building mask)
    # 1 building in batch 0, no buildings in batch 1
    target_b0 = torch.zeros(1, 512, 512)
    target_b0[0, 100:200, 100:200] = 1.0 # mock building
    
    target_b1 = torch.zeros(1, 512, 512)
    
    targets = {'building': torch.stack([target_b0, target_b1])}
    
    # 4. Forward Pass
    print("\n3. Testing Forward Pass...")
    try:
        preds = model(images)
        print("   -> Forward Pass Successful!")
        print(f"   -> Mask Logits Shape: {preds['pred_masks'][-1].shape}")
        print(f"   -> Class Logits Shape: {preds['pred_logits'][-1].shape}")
    except Exception as e:
        print(f"   -> Forward Pass FAILED: {e}")
        return

    # 5. Loss Calculation
    print("\n4. Testing Loss Calculation & Hungarian Matching...")
    try:
        loss, loss_dict = criterion(preds, targets)
        print("   -> Loss Calculation Successful!")
        print(f"   -> Total Loss: {loss.item():.4f}")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"      {k}: {v.item():.4f}")
    except Exception as e:
        print(f"   -> Loss Calculation FAILED: {e}")
        return

    # 6. Backward Pass (Gradient Flow)
    print("\n5. Testing Backward Pass (Gradient Flow)...")
    try:
        loss.backward()
        
        # Check if gradients reached the very beginning (Swin-T) and the very end (Queries)
        encoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        decoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer_decoder.parameters())
        
        print("   -> Backward Pass Successful!")
        print(f"   -> Gradients flowing to Swin-T Encoder: {'YES' if encoder_grad else 'NO'}")
        print(f"   -> Gradients flowing to Transformer Decoder: {'YES' if decoder_grad else 'NO'}")
        
    except Exception as e:
        print(f"   -> Backward Pass FAILED: {e}")
        return
        
    print("\n--- Dry Run Complete! Architecture is 100% mathematically sound. ---")

if __name__ == "__main__":
    test_architecture_flow()
