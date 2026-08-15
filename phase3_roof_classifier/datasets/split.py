import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

def create_geographic_split(metadata_path, output_dir, train_pct=0.7, val_pct=0.15, random_seed=42):
    """
    Reads roofnet_metadata.csv, drops missing images, groups by disaster/region, 
    and creates a train/val/test split without geographic leakage.
    """
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    # 0. DROP MISSING GOOGLE MAPS IMAGES
    initial_len = len(df)
    df = df.dropna(subset=['filename'])
    print(f"Dropped {initial_len - len(df)} missing images. Remaining: {len(df)}")
    
    # 1. Map classes
    class_mapping = {
        'AmorphousConcrete': 'RCC',
        'ClayTiles': 'TILED',
        'ConcreteTiles': 'TILED',
        'MetalSheetMaterials': 'TIN',
    }
    
    def map_class(x):
        return class_mapping.get(x, 'OTHER')
        
    df['phase3_class'] = df['material_class'].apply(map_class)
    
    # 2. Extract Geographic Proxy from xBD Filename (e.g. 'Haiti-hurricane-matthew')
    def get_disaster_name(fname):
        if pd.isna(fname):
            return 'unknown'
        parts = str(fname).split('_')
        return parts[0] if len(parts) > 0 else 'unknown'
        
    df['geographic_group'] = df['filename'].apply(get_disaster_name)
    
    print("\nGeographic Groups (Disasters):")
    print(df['geographic_group'].value_counts())
    
    # 3. Perform group split
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_pct, random_state=random_seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df['geographic_group']))
    
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()
    
    test_pct = 1.0 - train_pct - val_pct
    val_relative_pct = val_pct / (val_pct + test_pct)
    
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_relative_pct, random_state=random_seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['geographic_group']))
    
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()
    
    # Assign new split labels
    train_df['new_split'] = 'train'
    val_df['new_split'] = 'val'
    test_df['new_split'] = 'test'
    
    final_df = pd.concat([train_df, val_df, test_df])
    
    print("\nNew Geographic Split Distribution:")
    print(final_df['new_split'].value_counts())
    
    print("\nClass Distribution in Train:")
    print(train_df['phase3_class'].value_counts())
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'roofnet_phase3_split.csv')
    final_df.to_csv(out_path, index=False)
    print(f"\nSaved new split metadata to {out_path}")

if __name__ == "__main__":
    metadata_path = r"d:\Kush\2nd Year\Hackathons\SIH\data\roofnet\roofnet_metadata.csv"
    output_dir = r"d:\Kush\2nd Year\Hackathons\SIH\phase3_roof_classifier\datasets"
    create_geographic_split(metadata_path, output_dir)
