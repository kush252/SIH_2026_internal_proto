import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from phase3_roof_classifier.datasets.roofnet import RoofNetDataset
from phase3_roof_classifier.models.classifier import RoofClassifier
from phase3_roof_classifier.training.trainer import Trainer
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', type=str, default=r'd:\Kush\2nd Year\Hackathons\SIH\phase3_roof_classifier\datasets\roofnet_phase3_split.csv')
    parser.add_argument('--images_dir', type=str, default=r'd:\Kush\2nd Year\Hackathons\SIH\data\roofnet\xBD_cropped_roofs\xBD_cropped_roofs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_name', type=str, default='convnext_tiny')
    parser.add_argument('--save_dir', type=str, default='phase3_roof_classifier/outputs')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Datasets & DataLoaders
    print("Loading datasets...")
    train_dataset = RoofNetDataset(args.metadata_csv, args.images_dir, split='train')
    val_dataset = RoofNetDataset(args.metadata_csv, args.images_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Model
    print(f"Initializing model {args.model_name}...")
    model = RoofClassifier(model_name=args.model_name, num_classes=4, pretrained=True)
    model = model.to(device)
    
    # 3. Loss & Optimizer
    # Class weights from split.py calculation
    # {'RCC': 11.47, 'TILED': 1.53, 'TIN': 0.55, 'OTHER': 0.68}
    weights = torch.tensor([11.47, 1.53, 0.55, 0.68], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 4. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir
    )
    
    # 5. Train
    print("Starting training...")
    trainer.fit(num_epochs=args.epochs)

if __name__ == '__main__':
    main()
