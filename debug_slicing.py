#!/usr/bin/env python3

import torch

def test_slicing():
    """Test tensor slicing to understand the issue"""
    
    # Create test tensor with exact dimensions
    tensor = torch.ones(2, 1, 20, 100)
    print(f"Original tensor shape: {tensor.shape}")
    
    # Test different slice operations
    for ratio in [15, 16, 17]:
        sliced = tensor[:, :, ::ratio, ::ratio]
        print(f"Ratio {ratio}: sliced shape = {sliced.shape}")
        
        # Show what the slicing actually does
        height_indices = list(range(0, 20, ratio))
        width_indices = list(range(0, 100, ratio))
        print(f"  Height indices: {height_indices} -> length {len(height_indices)}")
        print(f"  Width indices: {width_indices} -> length {len(width_indices)}")
        print(f"  Expected shape: [2, 1, {len(height_indices)}, {len(width_indices)}]")
        print()

if __name__ == '__main__':
    test_slicing()