#!/usr/bin/env python3

import torch
import torch.nn as nn
from utils import load_config
from models.Backbone_stroke import StrokeBackbone

def quick_test():
    """Quick test with synthetic data to isolate the issue"""
    
    print("Quick test with synthetic data...")
    
    # Load config
    params = load_config('config_stroke.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    
    # Add missing parameters
    params['word_num'] = 1000  # Placeholder
    params['struct_num'] = 7   # Standard
    
    print(f"Using device: {device}")
    
    # Create model
    model = StrokeBackbone(params).to(device)
    print("Model created")
    
    # Create synthetic test data (same dimensions as expected)
    batch_size = 1  # Single sample for testing
    max_strokes = 20
    max_points = 100
    max_length = 10
    
    stroke_data = torch.randn(batch_size, max_strokes, max_points, 3).to(device)
    stroke_masks = torch.ones(batch_size, max_strokes, max_points).to(device)  # All valid
    stroke_positions = torch.randn(batch_size, max_strokes, 4).to(device)
    labels = torch.randint(0, 100, (batch_size, max_length, 11)).long().to(device)
    labels_mask = torch.ones(batch_size, max_length, 2).float().to(device)
    
    print(f"Test data shapes:")
    print(f"  stroke_data: {stroke_data.shape}")
    print(f"  stroke_masks: {stroke_masks.shape}")
    print(f"  stroke_positions: {stroke_positions.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  labels_mask: {labels_mask.shape}")
    
    try:
        with torch.no_grad():
            predictions, losses = model(stroke_data, stroke_masks, stroke_positions, labels, labels_mask, is_train=False)
        print("SUCCESS: Forward pass completed!")
        print(f"Predictions shapes: {[p.shape for p in predictions]}")
        print(f"Losses: {losses}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    quick_test()