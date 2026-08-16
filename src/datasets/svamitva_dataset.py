import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SvamitvaDataset(Dataset):
    def __init__(self, data_dir, config, task, is_train=True):
        """
        data_dir: Path to FilteredData (must contain 'images' and 'masks' folders)
        config: Loaded config (phase4_finetune.yaml)
        task: 'building', 'road', or 'water'
        is_train: True for training, False for validation
        """
        self.data_dir = data_dir
        self.config = config
        self.task = task
        self.is_train = is_train
        
        self.images_dir = os.path.join(data_dir, 'Images')
        self.masks_dir = os.path.join(data_dir, 'Masks')
        
        # Load filenames
        all_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg'))]
        # Basic 80/20 split on the fly using a seeded RNG
        rng = np.random.RandomState(42)
        rng.shuffle(all_files)
        
        split_idx = int(len(all_files) * 0.8)
        if self.is_train:
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]
            
        print(f"Loaded {len(self.files)} Svamitva files for {'Train' if is_train else 'Val'}.")
        
        # Get target RGB color from config based on task
        if task not in config.DATA.classes:
            raise ValueError(f"Task '{task}' not found in config.DATA.classes")
        self.target_rgb = config.DATA.classes[task] # e.g., [255, 0, 0]
        
        # Image augmentations
        self.img_size = config.DATA.img_size
        
        self.train_transforms = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ColorJitter(brightness=config.AUGMENTATION.color_jitter,
                          contrast=config.AUGMENTATION.color_jitter,
                          saturation=config.AUGMENTATION.color_jitter),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.images_dir, filename)
        
        # In Svamitva, masks might have different extensions. We try exact match or replace extension with .png
        mask_filename = filename
        if not os.path.exists(os.path.join(self.masks_dir, mask_filename)):
            mask_filename = os.path.splitext(filename)[0] + '.png'
            
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        image = Image.open(img_path).convert('RGB')
        
        # Try loading mask; if corrupt or missing, return an empty mask
        try:
            mask_rgb = Image.open(mask_path).convert('RGB')
            # Resize mask nearest neighbor
            mask_rgb = mask_rgb.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
            mask_arr = np.array(mask_rgb) # (H, W, 3)
            
            # Extract binary mask based on target RGB color
            target = np.array(self.target_rgb)
            # Find pixels where all 3 channels match the target color
            binary_mask = np.all(mask_arr == target, axis=-1).astype(np.float32)
        except Exception as e:
            binary_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            
        # Apply transforms
        if self.is_train:
            # Spatial augmentations must apply to both image and mask equally
            # We skip rotation/flip for simplicity right now to preserve the mask mappings,
            # but you can add custom geometric transforms here.
            image = self.train_transforms(image)
        else:
            image = self.val_transforms(image)
            
        # (1, H, W)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)
        
        targets = {self.task: mask_tensor}
        
        return {
            'image': image,
            'targets': targets,
            'filename': filename
        }
