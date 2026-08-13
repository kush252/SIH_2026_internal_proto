import torch
from torch.utils.data import DataLoader
from utils.config import load_config
from datasets.transforms import SimMIMTransform
from datasets.unified_ssl_dataset import UnifiedSSLDataset
from models.simmim import SimMIM
from training.trainer import Trainer
import os
import random
import numpy as np
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="SimMIM Phase 1 Training")
    parser.add_argument('--config', type=str, default=r"d:\Kush\2nd Year\Hackathons\SIH\src\configs\simmim_swin_t.yaml")
    parser.add_argument('--kaggle_dir', type=str, help="Path to Kaggle dataset (overrides config)")
    parser.add_argument('--landcover_dir', type=str, help="Path to Landcover dataset (overrides config)")
    parser.add_argument('--output_dir', type=str, help="Path to save outputs (overrides config)")
    parser.add_argument('--resume', type=str, help="Path to checkpoint .pt file to resume training")
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.kaggle_dir:
        config.DATA.kaggle_path = args.kaggle_dir
    if args.landcover_dir:
        config.DATA.landcover_path = args.landcover_dir
    if args.output_dir:
        config.SYSTEM.output_dir = args.output_dir
    
    set_seed(config.SYSTEM.seed)
    
    print("Setting up datasets...")
    transform = SimMIMTransform(config)
    dataset = UnifiedSSLDataset(config, transform=transform)
    
    # Use the full dataset for production (num_samples=None uses all patches)
    sampler = dataset.get_sampler(num_samples=None)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.TRAINING.batch_size,
        sampler=sampler,
        num_workers=config.TRAINING.num_workers,
        pin_memory=True
    )
    
    print("Initializing Model...")
    model = SimMIM(config)
    
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        config=config,
        output_dir=config.SYSTEM.output_dir,
        resume_path=args.resume
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
