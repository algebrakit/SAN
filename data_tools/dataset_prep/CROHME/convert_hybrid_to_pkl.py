#!/usr/bin/env python3
"""Convert hybrid tree files to pickle format for training."""

import os
import glob
import pickle as pkl
import sys
from tqdm import tqdm

def convert_hybrid_to_pkl(hybrid_dir, output_file):
    """Convert hybrid tree text files to pickle format."""
    hybrid_files = glob.glob(os.path.join(hybrid_dir, '*.txt'))
    label_dict = {}
    
    print(f"Converting {len(hybrid_files)} hybrid tree files...")
    
    for filepath in tqdm(hybrid_files):
        # Get filename without extension
        filename = os.path.basename(filepath).replace('.txt', '')
        
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        label_dict[filename] = lines
    
    with open(output_file, 'wb') as f:
        pkl.dump(label_dict, f)
    
    print(f"Saved {len(label_dict)} labels to {output_file}")
    return len(label_dict)

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_hybrid_to_pkl.py <hyb folder> <output_file>")
        print("Example: python convert_hybrid_to_pkl.py train_hyb train_label.pkl")
        sys.exit(1)

    hyb_folder = sys.argv[1]
    output_file = sys.argv[2]

    # Convert training labels
    train_count = convert_hybrid_to_pkl(hyb_folder, output_file)
    
    print(f"\nTotal: {train_count} training labels")


if __name__ == '__main__':
    main()