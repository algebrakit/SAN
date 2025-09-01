#!/usr/bin/env python3
"""
Debug script to check label conversion for the tiny dataset
"""

import sys
import os
import torch
from dataset_stroke import StrokeNormalizer, InkMLDataset
from latex_parser import LaTeXToSANConverter
from utils import load_config

def debug_label_conversion():
    print("=== Debug Label Conversion ===")
    
    # Load config
    config_path = "config_tiny.yaml"
    params = load_config(config_path)
    
    # Load vocabulary  
    from dataset import Words
    words = Words(params['word_path'])
    
    print(f"Vocabulary size: {len(words)}")
    print(f"'L' index in vocabulary: {words.words_dict.get('L', 'NOT FOUND')}")
    print(f"Token at index 40: '{words.words_index_dict.get(40, 'NOT FOUND')}'")
    print(f"Token at index 69: '{words.words_index_dict.get(69, 'NOT FOUND')}'")
    
    # Initialize normalizer
    normalizer = StrokeNormalizer(params)
    
    # Test the specific L sample
    inkml_file = "tiny_dataset/00a11cdf8d773ca3.inkml"
    print(f"\n=== Debugging {inkml_file} ===")
    
    # Parse the InkML
    parsed_data = normalizer.parse_inkml(inkml_file)
    if parsed_data is None:
        print("Failed to parse InkML!")
        return
        
    print(f"Original label: '{parsed_data['label']}'")
    print(f"Number of strokes: {len(parsed_data['strokes'])}")
    
    # Test LaTeX converter directly
    print(f"\n=== Testing LaTeX Converter ===")
    latex_converter = LaTeXToSANConverter(params['word_path'], params.get('max_sequence_length', 20))
    
    # Convert the label
    label_text = parsed_data['label']
    converted_labels = latex_converter.convert(label_text)
    
    print(f"Input label: '{label_text}'")
    print(f"Converted labels shape: {converted_labels.shape}")
    print(f"Converted labels dtype: {converted_labels.dtype}")
    
    # Check the first few positions in the converted labels
    print(f"\nDecoding converted labels:")
    for i in range(min(10, converted_labels.shape[0])):
        word_token = converted_labels[i, 0].item()  # Word token (first column)
        if word_token in words.words_index_dict:
            symbol = words.words_index_dict[word_token]
            print(f"  Position {i}: token {word_token} = '{symbol}'")
            if symbol == '</s>':  # End of sequence
                break
        else:
            print(f"  Position {i}: token {word_token} = [UNKNOWN]")
    
    # Test the dataset loading
    print(f"\n=== Testing Dataset Loading ===")
    try:
        dataset = InkMLDataset(params, params['inkml_folder'], words, is_train=True)
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        
        # Get the specific sample
        sample_found = False
        for i in range(len(dataset)):
            sample = dataset.data_samples[i]
            if sample['file_path'].endswith('00a11cdf8d773ca3.inkml'):
                print(f"Found sample at index {i}")
                print(f"Sample label: '{sample['label']}'")
                
                # Get the processed data from dataset
                stroke_tensor, stroke_masks, stroke_positions, real_labels = dataset[i]
                print(f"Dataset returned labels shape: {real_labels.shape}")
                
                # Decode the dataset labels
                print(f"Dataset labels (first 10 positions):")
                for j in range(min(10, real_labels.shape[0])):
                    word_token = real_labels[j, 0].item()
                    if word_token in words.words_index_dict:
                        symbol = words.words_index_dict[word_token]
                        print(f"  Position {j}: token {word_token} = '{symbol}'")
                        if symbol == '</s>':
                            break
                    else:
                        print(f"  Position {j}: token {word_token} = [UNKNOWN]")
                
                sample_found = True
                break
        
        if not sample_found:
            print("Sample not found in dataset!")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_label_conversion()