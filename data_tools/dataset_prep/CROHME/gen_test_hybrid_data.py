#!/usr/bin/env python3
"""Generate hybrid tree data for test set."""

import os
import glob
import pickle as pkl
from tqdm import tqdm

def prepare_test_labels():
    """Create test_label.pkl using existing test_caption.txt format."""
    # Read the existing test caption file
    with open('test_caption.txt', 'r') as f:
        lines = f.readlines()
    
    # Convert to the format expected by the model
    label_dict = {}
    print(f"Processing {len(lines)} test labels...")
    
    for line in tqdm(lines):
        parts = line.strip().split('\t')
        if len(parts) == 2:
            filename = parts[0]  # e.g., "18_em_0"
            latex_expr = parts[1]  # e.g., "x _ { k } x x _ { k } + y _ { k } y x _ { k }"
            
            # For now, create simple sequential labels (not full hybrid tree)
            # This is a simplified approach - the model expects hybrid format
            # but we can start with basic sequential representation
            tokens = latex_expr.split()
            simple_labels = []
            
            # Create simple sequential representation
            for i, token in enumerate(tokens):
                parent_id = max(0, i-1)  # Previous token as parent
                parent_token = tokens[parent_id] if i > 0 else '<sos>'
                simple_labels.append(f"{i+1}\t{token}\t{parent_id}\t{parent_token}\tNone\tNone\tNone\tNone\tNone\tNone\tNone")
            
            # Add end token
            simple_labels.append(f"{len(tokens)+1}\t<eos>\t{len(tokens)}\t{tokens[-1] if tokens else '<sos>'}\tNone\tNone\tNone\tNone\tNone\tNone\tNone")
            
            label_dict[filename] = simple_labels
    
    # Save to pickle
    with open('test_label.pkl', 'wb') as f:
        pkl.dump(label_dict, f)
    
    print(f"Saved {len(label_dict)} test labels to test_label.pkl")
    return len(label_dict)

if __name__ == '__main__':
    prepare_test_labels()