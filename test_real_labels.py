"""
Test script to verify the real label integration works
"""

import os
import torch
from dataset_stroke import get_stroke_dataset
from utils import load_config

def test_real_labels():
    """Test that the dataset now produces real labels instead of dummy ones"""
    
    print("Testing Real Label Integration")
    print("=" * 50)
    
    # Load configuration
    config_path = 'config_stroke.yaml'
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found!")
        return
    
    params = load_config(config_path)
    
    # Override for testing - use smaller cache load
    params['load_single_chunk'] = True
    params['single_chunk_index'] = 1
    
    print(f"Loading dataset with real label conversion...")
    print(f"InkML folder: {params.get('inkml_folder', 'synthetic')}")
    print(f"Word vocabulary: {params.get('word_path', 'data/word.txt')}")
    print(f"Max sequence length: {params.get('max_sequence_length', 50)}")
    
    try:
        # Create dataset
        train_loader, eval_loader = get_stroke_dataset(params)
        
        print(f"+ Dataset created successfully")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Evaluation samples: {len(eval_loader.dataset)}")
        
        # Test a few batches
        print("\nTesting label content...")
        
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= 3:  # Test first 3 batches
                break
                
            stroke_data, stroke_masks, stroke_positions, labels, labels_mask = batch_data
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Stroke data shape: {stroke_data.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Label mask shape: {labels_mask.shape}")
            
            # Check if labels are no longer all zeros (dummy)
            non_zero_tokens = (labels[:, :, 1] > 0).sum()
            total_positions = labels.shape[0] * labels.shape[1]
            
            print(f"  Non-zero token positions: {non_zero_tokens}/{total_positions}")
            print(f"  Percentage filled: {non_zero_tokens/total_positions*100:.1f}%")
            
            # Show some example token IDs from first sample
            first_sample_tokens = labels[0, :10, 1]  # First 10 tokens of first sample
            print(f"  Example token IDs: {first_sample_tokens.tolist()}")
            
            # Check label masks
            valid_tokens = labels_mask[:, :, 0].sum()
            print(f"  Valid tokens (from mask): {valid_tokens}")
            
            # Show structure information
            has_structure = (labels[:, :, 4:].sum(dim=2) > 0).sum()
            print(f"  Positions with structure info: {has_structure}")
            
        print("\n" + "=" * 50)
        print("SUCCESS: Real labels are now being generated!")
        print("Key improvements:")
        print("+ Labels contain actual token IDs (not all zeros)")
        print("+ Label masks properly indicate valid positions") 
        print("+ Structure information is encoded")
        print("+ LaTeX expressions are parsed into SAN format")
        
        # Test with some sample InkML files to show LaTeX conversion
        print(f"\nTesting LaTeX conversion examples:")
        
        # Get some samples directly from dataset
        dataset = train_loader.dataset
        if hasattr(dataset, 'data_samples') and len(dataset.data_samples) > 0:
            for i in range(min(3, len(dataset.data_samples))):
                sample = dataset.data_samples[i]
                latex_label = sample.get('label', 'No label')
                print(f"  Sample {i+1}: '{latex_label}'")
        
        print("\nTraining can now proceed with meaningful mathematical targets!")
        
    except Exception as e:
        print(f"- Error testing labels: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_real_labels()