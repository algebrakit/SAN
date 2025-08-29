#!/usr/bin/env python3
"""
Prepare CROHME 2016 data for SAN model training and testing.
This script:
1. Converts images to pickle format
2. Generates hybrid tree labels from LaTeX
3. Creates train/test splits
"""

import os
import glob
import cv2
import pickle as pkl
from tqdm import tqdm
import subprocess
import sys

def convert_images_to_pkl(image_dir, output_file, verbose=True):
    """Convert BMP images to pickle format."""
    images = glob.glob(os.path.join(image_dir, '*.bmp'))
    image_dict = {}
    
    if verbose:
        print(f"Converting {len(images)} images from {image_dir}...")
    
    for item in tqdm(images, disable=not verbose):
        img = cv2.imread(item)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Remove _0.bmp suffix to match label format
            key = os.path.basename(item).replace('_0.bmp', '')
            image_dict[key] = img
    
    with open(output_file, 'wb') as f:
        pkl.dump(image_dict, f)
    
    if verbose:
        print(f"Saved {len(image_dict)} images to {output_file}")
    return len(image_dict)

def prepare_label_file(input_caption, output_latex):
    """Prepare label file in the format expected by gen_hybrid_data.py"""
    with open(input_caption, 'r') as f:
        lines = f.readlines()
    
    with open(output_latex, 'w') as f:
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                # Add .jpg extension to match expected format
                f.write(f"{parts[0]}.jpg {parts[1]}\n")
    
    print(f"Prepared {len(lines)} labels in {output_latex}")

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    crohme_dir = os.path.join(base_dir, 'CHROME-2016')
    
    # Training data
    train_image_dir = os.path.join(crohme_dir, 'TrainingSet', 'off_image_train')
    train_caption = os.path.join(crohme_dir, 'TrainingSet', 'train_caption.txt')
    
    # Test data  
    test_image_dir = os.path.join(crohme_dir, 'TestSet', 'off_image_test')
    test_caption = os.path.join(crohme_dir, 'TestSet', 'test_caption.txt')
    
    # Output files
    train_image_pkl = os.path.join(base_dir, 'train_image.pkl')
    test_image_pkl = os.path.join(base_dir, 'test_image.pkl')
    train_latex = os.path.join(base_dir, 'train_latex.txt')
    test_latex = os.path.join(base_dir, 'test_latex.txt')
    
    print("=== CROHME 2016 Data Preparation ===\n")
    
    # Step 1: Convert images to pickle format
    print("Step 1: Converting images to pickle format...")
    train_count = convert_images_to_pkl(train_image_dir, train_image_pkl)
    test_count = convert_images_to_pkl(test_image_dir, test_image_pkl)
    
    # Step 2: Prepare label files
    print("\nStep 2: Preparing label files...")
    prepare_label_file(train_caption, train_latex)
    if os.path.exists(test_caption):
        prepare_label_file(test_caption, test_latex)
    
    # Step 3: Generate hybrid tree labels
    print("\nStep 3: Generating hybrid tree labels...")
    print("You need to run gen_hybrid_data.py to convert LaTeX to hybrid format:")
    print(f"  python3 gen_hybrid_data.py")
    print(f"  (Make sure to update the paths in gen_hybrid_data.py)")
    
    print("\n=== Summary ===")
    print(f"Training images: {train_count}")
    print(f"Test images: {test_count}")
    print(f"\nGenerated files:")
    print(f"  - {train_image_pkl}")
    print(f"  - {test_image_pkl}")
    print(f"  - {train_latex}")
    if os.path.exists(test_caption):
        print(f"  - {test_latex}")
    
    print("\n=== Next Steps ===")
    print("1. Update gen_hybrid_data.py with correct paths")
    print("2. Run: python3 gen_hybrid_data.py")
    print("3. Update config.yaml with data paths")
    print("4. Run: python3 train.py --config config.yaml")

if __name__ == '__main__':
    main()