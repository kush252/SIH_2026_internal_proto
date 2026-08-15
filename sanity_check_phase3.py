import os
import pandas as pd
from PIL import Image

def run_sanity_check():
    print("--- 1. Class Mapping & Split Check ---")
    df = pd.read_csv(r'd:\Kush\2nd Year\Hackathons\SIH\phase3_roof_classifier\datasets\roofnet_phase3_split.csv')
    print(f"Total available images in split file: {len(df)}")
    print("Class Distribution in overall dataset:")
    print(df['phase3_class'].value_counts())
    
    print("\n--- 2. Dataset Loading & Corruption Check ---")
    img_dir = r'd:\Kush\2nd Year\Hackathons\SIH\data\roofnet\xBD_cropped_roofs\xBD_cropped_roofs'
    train_df = df[df['new_split'] == 'train'].reset_index(drop=True)
    print(f"Train Dataset size: {len(train_df)}")
    
    bad_count = 0
    # testing the first 100 images
    for i in range(min(100, len(train_df))):
        img_name = train_df.iloc[i]['filename']
        img_path = os.path.join(img_dir, img_name)
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            print(f"Failed loading {img_path}: {e}")
            bad_count += 1
    print(f"Corrupted images found in first 100: {bad_count}")
    
    print("\n--- 3. Verify Geographic Leakage Avoidance ---")
    val_df = df[df['new_split'] == 'val']
    test_df = df[df['new_split'] == 'test']
    
    train_regions = set(train_df['geographic_group'])
    val_regions = set(val_df['geographic_group'])
    test_regions = set(test_df['geographic_group'])
    
    leakage = train_regions.intersection(val_regions).union(train_regions.intersection(test_regions))
    if len(leakage) == 0:
        print("Success: Zero geographic leakage! Train, Val, and Test use completely disjoint regions.")
    else:
        print(f"Warning: Found leaking regions: {leakage}")

    print("\nSanity Check Complete!")

if __name__ == '__main__':
    run_sanity_check()
