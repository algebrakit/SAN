#!/usr/bin/env python3

import torch
from utils import load_config
from dataset_stroke import get_stroke_dataset
from models.stroke_encoder import StrokeAwareEncoder

def test_stroke_encoder():
    """Test just the stroke encoder without the decoder"""
    
    # Load config
    params = load_config('config_stroke.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    
    print("Testing stroke encoder only...")
    
    # Create data loader
    train_loader, _ = get_stroke_dataset(params)
    
    # Create encoder
    encoder = StrokeAwareEncoder(params).to(device)
    
    print(f"Encoder created with {sum(p.numel() for p in encoder.parameters()):,} parameters")
    
    # Test one batch
    for batch_data in train_loader:
        stroke_data, stroke_masks, stroke_positions, labels, labels_mask = batch_data
        
        stroke_data = stroke_data.to(device)
        stroke_masks = stroke_masks.to(device)
        stroke_positions = stroke_positions.to(device)
        
        print(f"Input shapes:")
        print(f"  stroke_data: {stroke_data.shape}")
        print(f"  stroke_masks: {stroke_masks.shape}")
        print(f"  stroke_positions: {stroke_positions.shape}")
        
        with torch.no_grad():
            output = encoder(stroke_data, stroke_masks, stroke_positions)
            print(f"Output shape: {output.shape}")
            print(f"Expected: [batch, 684, 20, 100]")
            print(f"Match: {output.shape == (stroke_data.shape[0], 684, 20, 100)}")
        
        break  # Only test one batch
    
    print("Encoder test completed successfully!")

if __name__ == '__main__':
    test_stroke_encoder()