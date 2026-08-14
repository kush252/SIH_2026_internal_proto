import os
import random
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from datasets.generic_binary_dataset import GenericBinaryDataset
from models.task_heads import Phase2MultiTaskModel
from training.trainer_phase2 import Phase2Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Supervised Segmentation Training")
    parser.add_argument("--config", type=str, default="configs/phase2_building.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config.SYSTEM.seed)
    
    device = torch.device(config.SYSTEM.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset & Splits
    print(f"Setting up dataset from {config.DATA.dataset_path}...")
    
    train_dataset = GenericBinaryDataset(config.DATA.dataset_path, config, is_train=True)
    val_dataset = GenericBinaryDataset(config.DATA.dataset_path, config, is_train=False)
    
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
    print("Initializing Phase 2 Model...")
    model = Phase2MultiTaskModel(config)
    
    # 3. Trainer
    trainer = Phase2Trainer(model, config, train_loader, val_loader, device)
    
    # 4. Fit
    trainer.fit()
    
if __name__ == "__main__":
    main()
