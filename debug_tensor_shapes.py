#!/usr/bin/env python3

import torch
import sys

def test_tensor_operations():
    """Test the exact tensor operations that are failing"""
    
    print("Testing tensor operations...")
    
    # Simulate the exact tensors from the error
    batch_size = 8
    
    # Features tensor (what stroke encoder should output)
    features = torch.randn(batch_size, 684, 20, 100)
    print(f"Features shape: {features.shape}")
    
    # Original image mask (what I create)
    images_mask_orig = torch.ones(batch_size, 1, 20, 100)
    print(f"Original mask shape: {images_mask_orig.shape}")
    
    # Test different ratio values to see which gives dimension 7
    for ratio in [14, 15, 16, 17]:
        downsampled_mask = images_mask_orig[:, :, ::ratio, ::ratio]
        print(f"Ratio {ratio}: downsampled mask shape = {downsampled_mask.shape}")
        
        width = downsampled_mask.shape[-1]
        if width == 7:
            print(f"  *** FOUND IT! Ratio {ratio} gives width {width}")
        
        # Test the multiplication that's failing
        try:
            result = (features * downsampled_mask).sum(-1).sum(-1)
            print(f"  Multiplication successful: result shape {result.shape}")
        except Exception as e:
            print(f"  ERROR with ratio {ratio}: {e}")
    
    # Test with exact dimension 7
    print("\nTesting with exact width 7:")
    mask_with_7 = torch.ones(batch_size, 1, 20, 7)
    print(f"Mask shape: {mask_with_7.shape}")
    
    try:
        result = (features * mask_with_7).sum(-1).sum(-1)
        print(f"Multiplication successful: result shape {result.shape}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == '__main__':
    test_tensor_operations()