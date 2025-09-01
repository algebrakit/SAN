#!/usr/bin/env python3
"""
Quick inference test on tiny dataset to verify training worked
"""
import os
import torch
import numpy as np
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words
from models.Backbone_stroke import StrokeBackbone

def test_inference():
    print("=== Testing Tiny Dataset Inference ===")
    
    # Load config
    config_path = "config_tiny.yaml"
    params = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    
    print(f"Using device: {device}")
    
    # Load vocabulary
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7  # Same as original SAN
    
    # Initialize stroke normalizer
    normalizer = StrokeNormalizer(params)
    
    # Create model
    print("Creating model...")
    model = StrokeBackbone(params)
    model = model.to(device)
    
    # Use the known working single L checkpoint
    import glob
    checkpoint_dirs = glob.glob("checkpoints_single_L/*/")
    if not checkpoint_dirs:
        print("No checkpoint directories found!")
        return
    
    # Get the most recent checkpoint directory with the correct training (2025-09-01-11-41)
    target_dir = None
    for cdir in checkpoint_dirs:
        if "2025-09-01-11-41" in cdir:
            target_dir = cdir
            break
    
    if not target_dir:
        print("No matching checkpoint directory found!")
        return
    
    dir_name = os.path.basename(target_dir.rstrip('/\\'))
    checkpoint_dir = os.path.join(target_dir, dir_name)
    
    # Look for any checkpoint (they may all have 0.0000 word rate due to accuracy calculation bug)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
        
    # Get the latest checkpoint (highest epoch number)
    best_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if best_checkpoint is None:
        print("No checkpoint found!")
        return
        
    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
    print(f"Loading checkpoint: {best_checkpoint}")
    print(f"From directory: {checkpoint_dir}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Test on the sample with vocabulary-compatible label
    inkml_file = "tiny_dataset/00a11cdf8d773ca3.inkml"
    print(f"\nTesting inference on: {inkml_file}")
    
    # Parse InkML
    parsed_data = normalizer.parse_inkml(inkml_file)
    if parsed_data is None:
        print("Failed to parse InkML!")
        return
    
    print(f"Ground truth label: {parsed_data['label']}")
    print(f"Number of strokes: {len(parsed_data['strokes'])}")
    
    # Normalize strokes - pass just the strokes list
    normalized_data = normalizer.normalize_expression(parsed_data['strokes'])
    
    # Convert strokes to tensor format (same as dataset __getitem__)
    stroke_tensor = torch.zeros(normalizer.max_strokes, normalizer.max_points_per_stroke, 3)
    
    for stroke_idx, stroke in enumerate(normalized_data['normalized_strokes']):
        for point_idx, (x, y, t) in enumerate(stroke['points']):
            stroke_tensor[stroke_idx, point_idx, 0] = x
            stroke_tensor[stroke_idx, point_idx, 1] = y  
            stroke_tensor[stroke_idx, point_idx, 2] = t
    
    stroke_masks = torch.tensor(normalized_data['stroke_masks'], dtype=torch.float32)
    stroke_positions = torch.tensor(normalized_data['stroke_positions'], dtype=torch.float32)
    
    print(f"Normalized stroke data shape: {stroke_tensor.shape}")
    
    # Add batch dimension and move to device
    stroke_data = stroke_tensor.unsqueeze(0).to(device)
    stroke_masks = stroke_masks.unsqueeze(0).to(device)
    stroke_positions = stroke_positions.unsqueeze(0).to(device)
    
    # Create dummy labels for inference (we don't actually use them for prediction)
    batch_size = 1
    max_length = 20
    labels = torch.zeros(batch_size, max_length, 11, dtype=torch.long).to(device)
    labels_mask = torch.ones(batch_size, max_length, 2, dtype=torch.float32).to(device)
    
    print(f"Input shapes:")
    print(f"  Stroke data: {stroke_data.shape}")
    print(f"  Stroke masks: {stroke_masks.shape}")
    print(f"  Stroke positions: {stroke_positions.shape}")
    
    # Run inference
    with torch.no_grad():
        try:
            predictions, losses = model(stroke_data, stroke_masks, stroke_positions, labels, labels_mask, is_train=False)
            word_probs, struct_probs = predictions
            
            print(f"\nInference successful!")
            print(f"Word predictions shape: {word_probs.shape}")
            print(f"Struct predictions shape: {struct_probs.shape}")
            
            # Get predicted sequence
            predicted_tokens = torch.argmax(word_probs[0], dim=-1)  # [seq_len]
            
            print(f"\nPredicted token sequence:")
            predicted_symbols = []
            for i, token_id in enumerate(predicted_tokens):
                token_id = token_id.item()
                if token_id in words.words_index_dict:
                    symbol = words.words_index_dict[token_id]
                    predicted_symbols.append(symbol)
                    print(f"  Step {i+1}: {token_id} -> '{symbol}'")
                    if symbol == '<eos>':  # End of sequence - corrected token name
                        break
                else:
                    print(f"  Step {i+1}: {token_id} -> [UNKNOWN]")
            
            # Filter out start/end tokens for clean display
            content_symbols = [s for s in predicted_symbols if s not in ['<sos>', '<eos>']]
            predicted_latex = ' '.join(content_symbols)
            
            print(f"\nPredicted LaTeX: {predicted_latex}")
            print(f"Ground truth: {parsed_data['label']}")
            
            # Calculate simple match
            if predicted_latex.strip() == parsed_data['label'].strip():
                print("EXACT MATCH!")
            else:
                print("Different from ground truth")
                
        except Exception as e:
            print(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_inference()