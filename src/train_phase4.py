import os
import random
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from datasets.svamitva_dataset import SvamitvaDataset
from models.task_heads import Phase2MultiTaskModel
from training.trainer_phase2 import Phase2Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Svamitva Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/phase4_finetune.yaml", help="Path to fine-tune config")
    parser.add_argument("--task", type=str, required=True, choices=["building", "road", "water"], help="Which task to fine-tune")
    parser.add_argument("--base_model", type=str, required=True, help="Path to Phase 2 best model")
    
    # Kaggle Overrides
    parser.add_argument("--dataset_path", type=str, default=None, help="Override path to Svamitva FilteredData")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Apply Overrides
    if args.dataset_path:
        config.DATA.dataset_path = args.dataset_path
    if args.batch_size:
        config.TRAINING.batch_size = args.batch_size
    if args.epochs:
        config.TRAINING.epochs = args.epochs
    
    # Configure the model to focus ONLY on the specific task being fine-tuned
    config.LOSS.weights = {args.task: 1.0}
    config.SYSTEM.output_dir = f"outputs_phase4_{args.task}"
        
    set_seed(config.SYSTEM.seed)
    
    device = torch.device(config.SYSTEM.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset & Splits (Using the new Svamitva loader)
    print(f"Setting up Svamitva dataset for task: {args.task}...")
    
    train_dataset = SvamitvaDataset(config.DATA.dataset_path, config, args.task, is_train=True)
    val_dataset = SvamitvaDataset(config.DATA.dataset_path, config, args.task, is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.TRAINING.batch_size, 
        shuffle=True, 
        num_workers=config.TRAINING.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.TRAINING.batch_size, 
        shuffle=False, 
        num_workers=config.TRAINING.num_workers,
        pin_memory=True
    )
    
    # 2. Model setup
    print("\nInitializing Phase 4 Fine-Tuning Model...")
    model = Phase2MultiTaskModel(config)
    
    # LOAD THE PHASE 2 WEIGHTS INTO THE FULL ARCHITECTURE
    print(f"Loading Base Phase 2 Model from {args.base_model}...")
    checkpoint = torch.load(args.base_model, map_location='cpu', weights_only=False)
    # The checkpoint contains 'model_state_dict', 'optimizer_state_dict', 'epoch', 'best_metric'
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Base Phase 2 weights successfully loaded!")
    
    # FREEZE THE ENCODER to prevent catastrophic forgetting on tiny dataset
    print("Freezing the Swin Encoder to prevent overfitting on 660 images...")
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
            
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("PHASE 4 FINE-TUNING SUMMARY")
    print("=" * 50)
    print(f"Task         : {args.task.upper()}")
    print(f"Base Model   : {args.base_model}")
    print(f"Train Images : {len(train_dataset)}")
    print(f"Val Images   : {len(val_dataset)}")
    print(f"Epochs       : {config.TRAINING.epochs}")
    print(f"Encoder LR   : {config.OPTIMIZER.encoder_lr}")
    print("=" * 50 + "\n")
    
    # 3. Trainer
    # We pass resume_path=None because we just loaded the model weights manually above.
    # We DO NOT want to resume the optimizer states or the epoch number from Phase 2!
    # We are starting a fresh Phase 4 optimizer.
    trainer = Phase2Trainer(model, config, train_loader, val_loader, device, resume_path=None)
    
    # Add the CosineAnnealingLR scheduler to gracefully lower the LR during fine-tuning
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, 
        T_max=config.TRAINING.epochs
    )
    
    # Hook the scheduler step into the trainer fit loop (quick monkey patch)
    original_validate = trainer.validate
    def validate_and_step(epoch):
        metrics = original_validate(epoch)
        trainer.scheduler.step()
        return metrics
    trainer.validate = validate_and_step
    
    # 4. Fit
    trainer.fit()
    
if __name__ == "__main__":
    main()
