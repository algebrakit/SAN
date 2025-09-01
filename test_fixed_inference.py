#!/usr/bin/env python3
"""
Test inference with the newly trained model that has fixed labels
"""
import os
import torch
import numpy as np
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words
from models.Backbone_stroke import StrokeBackbone
from simple_latex_converter import SimpleLaTeXConverter

def test_fixed_inference():
    print("=== Testing Fixed Label Inference ===")
    
    # Load config
    config_path = "config_tiny.yaml"
    params = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    
    print(f"Using device: {device}")
    
    # Load vocabulary
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7
    
    # Initialize stroke normalizer
    normalizer = StrokeNormalizer(params)
    
    # Create model
    print("Creating model...")
    model = StrokeBackbone(params)
    model = model.to(device)
    
    # Find the best checkpoint from the new training
    checkpoint_dir = "checkpoints_tiny/SAN_Stroke_Tiny_2025-09-01-09-50_StrokeEncoder_Decoder-SAN_decoder_max_strokes-5_max_points-100/SAN_Stroke_Tiny_2025-09-01-09-50_StrokeEncoder_Decoder-SAN_decoder_max_strokes-5_max_points-100"
    
    # Look for checkpoint with highest expression accuracy
    best_checkpoint = None
    best_expr_rate = -1
    best_word_rate = -1
    
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth') and 'WordRate' in file:
            try:
                # Extract metrics from filename
                parts = file.split('_')
                for i, part in enumerate(parts):
                    if part.startswith('ExpRate-'):
                        expr_rate = float(part.split('-')[1])
                    elif part.startswith('WordRate-'):
                        word_rate = float(part.split('-')[1])
                
                # Prefer higher expression rate, then higher word rate
                if expr_rate > best_expr_rate or (expr_rate == best_expr_rate and word_rate > best_word_rate):
                    best_expr_rate = expr_rate
                    best_word_rate = word_rate
                    best_checkpoint = file
            except:
                continue
    
    if best_checkpoint is None:
        print("No checkpoint found!")
        return
        
    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
    print(f"Loading best checkpoint: {best_checkpoint}")
    print(f"Word accuracy: {best_word_rate:.4f}, Expression accuracy: {best_expr_rate:.4f}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Test on the simple 'L' sample
    inkml_file = "tiny_dataset/00a11cdf8d773ca3.inkml"
    print(f"\n=== Testing 'L' Character ===")
    print(f"File: {inkml_file}")
    
    # Parse InkML
    parsed_data = normalizer.parse_inkml(inkml_file)
    if parsed_data is None:
        print("Failed to parse InkML!")
        return
    
    print(f"Ground truth label: '{parsed_data['label']}'")
    print(f"Expected token sequence: <sos> (1) -> L (69) -> <eos> (0)")
    
    # Normalize strokes
    normalized_data = normalizer.normalize_expression(parsed_data['strokes'])
    
    # Convert to tensor format
    stroke_tensor = torch.zeros(normalizer.max_strokes, normalizer.max_points_per_stroke, 3)
    
    for stroke_idx, stroke in enumerate(normalized_data['normalized_strokes']):
        for point_idx, (x, y, t) in enumerate(stroke['points']):
            stroke_tensor[stroke_idx, point_idx, 0] = x
            stroke_tensor[stroke_idx, point_idx, 1] = y  
            stroke_tensor[stroke_idx, point_idx, 2] = t
    
    stroke_masks = torch.tensor(normalized_data['stroke_masks'], dtype=torch.float32)
    stroke_positions = torch.tensor(normalized_data['stroke_positions'], dtype=torch.float32)
    
    # Add batch dimension and move to device
    stroke_data = stroke_tensor.unsqueeze(0).to(device)
    stroke_masks = stroke_masks.unsqueeze(0).to(device)
    stroke_positions = stroke_positions.unsqueeze(0).to(device)
    
    # Create dummy labels for inference
    batch_size = 1
    max_length = 20
    labels = torch.zeros(batch_size, max_length, 11, dtype=torch.long).to(device)
    labels_mask = torch.ones(batch_size, max_length, 2, dtype=torch.float32).to(device)
    
    print(f"\nInput tensor shapes:")
    print(f"  Stroke data: {stroke_data.shape}")
    print(f"  Stroke masks: {stroke_masks.shape}")
    print(f"  Stroke positions: {stroke_positions.shape}")
    
    # Run inference
    with torch.no_grad():
        try:
            predictions, losses = model(stroke_data, stroke_masks, stroke_positions, labels, labels_mask, is_train=False)
            word_probs, struct_probs = predictions
            
            print(f"\n🎉 Inference successful!")
            print(f"Word predictions shape: {word_probs.shape}")
            print(f"Struct predictions shape: {struct_probs.shape}")
            
            # Get predicted sequence
            predicted_tokens = torch.argmax(word_probs[0], dim=-1)  # [seq_len]
            
            print(f"\n📊 Predicted Token Sequence:")
            predicted_symbols = []
            for i, token_id in enumerate(predicted_tokens):
                token_id = token_id.item()
                if token_id in words.words_index_dict:
                    symbol = words.words_index_dict[token_id]
                    predicted_symbols.append(symbol)
                    print(f"  Step {i+1}: token {token_id} → '{symbol}'")
                    if symbol == '<eos>':  # End of sequence
                        break
                else:
                    print(f"  Step {i+1}: token {token_id} → [UNKNOWN]")
                
                if i >= 9:  # Limit output
                    break
            
            predicted_latex = ' '.join([s for s in predicted_symbols if s not in ['<sos>', '<eos>']])
            print(f"\n🎯 Results:")
            print(f"Ground truth:  '{parsed_data['label']}'")
            print(f"Predicted:     '{predicted_latex}'")
            
            # Check if it matches
            if predicted_latex.strip() == parsed_data['label'].strip():
                print("✅ PERFECT MATCH! The model correctly predicted 'L'!")
            else:
                print("❌ Different prediction")
                
            # Show confidence scores for key tokens
            print(f"\n📈 Confidence scores for key positions:")
            for pos in range(min(3, word_probs.shape[1])):
                probs = torch.softmax(word_probs[0, pos], dim=0)
                top_k = torch.topk(probs, 3)
                print(f"  Position {pos}:")
                for idx, (prob, token_id) in enumerate(zip(top_k.values, top_k.indices)):
                    token_id = token_id.item()
                    prob = prob.item()
                    symbol = words.words_index_dict.get(token_id, f"UNK_{token_id}")
                    print(f"    {idx+1}. '{symbol}' (token {token_id}): {prob:.4f}")
                    
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_fixed_inference()