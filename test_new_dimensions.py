#!/usr/bin/env python3

import torch

def test_new_dimensions():
    """Test new dimensions with ratio 16"""
    
    # Test with new dimensions
    height, width = 16, 96
    tensor = torch.ones(2, 1, height, width)
    print(f"Original tensor shape: {tensor.shape}")
    
    ratio = 16
    sliced = tensor[:, :, ::ratio, ::ratio]
    print(f"Ratio {ratio}: sliced shape = {sliced.shape}")
    
    # Show indices
    height_indices = list(range(0, height, ratio))
    width_indices = list(range(0, width, ratio))
    print(f"Height indices: {height_indices} -> length {len(height_indices)}")
    print(f"Width indices: {width_indices} -> length {len(width_indices)}")
    
    # Test if this would work for the decoder
    features = torch.randn(2, 684, height, width)
    feature_mask = sliced
    
    print(f"\nTesting multiplication:")
    print(f"Features shape: {features.shape}")
    print(f"Feature mask shape: {feature_mask.shape}")
    
    try:
        result = (features * feature_mask).sum(-1).sum(-1)
        print(f"SUCCESS: Result shape {result.shape}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == '__main__':
    test_new_dimensions()