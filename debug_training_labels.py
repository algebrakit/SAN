#!/usr/bin/env python3
"""
Debug what labels are actually being passed to the model during training
"""
import torch
import os
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words, InkMLDataset

def debug_training_labels():
    print("=== Debugging Training Labels ===")
    
    config_path = "config_tiny.yaml"
    params = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    
    # Load vocabulary and create dataset
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7
    
    # Create dataset
    dataset = InkMLDataset(params, params['inkml_folder'], words, is_train=True)
    print(f"Dataset has {len(dataset)} samples")
    
    # Get the sample
    sample = dataset.data_samples[0]
    print(f"\nFile: {os.path.basename(sample['file_path'])}")
    print(f"Ground truth: '{sample['label']}'")
    
    # Get processed data 
    stroke_tensor, stroke_masks, stroke_positions, target_labels = dataset[0]
    
    print(f"\nTarget labels shape: {target_labels.shape}")
    print(f"Target labels content:")
    
    for pos in range(min(10, target_labels.shape[0])):
        label_row = target_labels[pos]
        word_token = label_row[0].item()
        struct_token = label_row[1].item() 
        parent_id = label_row[2].item()
        relation_token = label_row[3].item()
        
        word_symbol = words.words_index_dict.get(word_token, f"UNK_{word_token}")
        struct_symbol = words.words_index_dict.get(struct_token, f"UNK_{struct_token}")
        relation_symbol = words.words_index_dict.get(relation_token, f"UNK_{relation_token}")
        
        print(f"  {pos}: word={word_token}('{word_symbol}') struct={struct_token}('{struct_symbol}') parent={parent_id} relation={relation_token}('{relation_symbol}')")
        
        if word_symbol == '<eos>':
            break

if __name__ == '__main__':
    debug_training_labels()