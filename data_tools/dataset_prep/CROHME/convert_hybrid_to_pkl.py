#!/usr/bin/env python3
"""Convert hybrid tree files to pickle format for training."""

import os
import glob
import pickle as pkl
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
    # Convert training labels
    train_count = convert_hybrid_to_pkl('train_hyb', 'train_label.pkl')
    
    # If test hybrid data exists, convert it too
    if os.path.exists('test_hyb'):
        test_count = convert_hybrid_to_pkl('test_hyb', 'test_label.pkl')
        print(f"\nTotal: {train_count} training labels, {test_count} test labels")
    else:
        print(f"\nTotal: {train_count} training labels")
        print("Note: Test labels not found. You'll need to generate test_hyb first.")

if __name__ == '__main__':
    main()