#!/usr/bin/env python3
"""
Simple test inference with the newly trained model
"""
import os
import torch
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words
from models.Backbone_stroke import StrokeBackbone

def test_simple():
    config_path = "config_tiny.yaml"
    params = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7
    
    normalizer = StrokeNormalizer(params)
    model = StrokeBackbone(params)
    model = model.to(device)
    
    # Load the perfectly trained single L checkpoint
    checkpoint_dir = "checkpoints_single_L/SAN_Stroke_Tiny_2025-09-01-10-01_StrokeEncoder_Decoder-SAN_decoder_max_strokes-5_max_points-100/SAN_Stroke_Tiny_2025-09-01-10-01_StrokeEncoder_Decoder-SAN_decoder_max_strokes-5_max_points-100"
    
    # Get the perfect checkpoint (should be epoch 2)
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*ExpRate-1.0000*.pth"))
    if not checkpoints:
        print("No perfect checkpoints found!")
        return
    
    # Get the latest perfect checkpoint
    best_checkpoint = max(checkpoints, key=os.path.getmtime)
    best_checkpoint = os.path.basename(best_checkpoint)
    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
    
    print(f"Loading checkpoint: {best_checkpoint}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Test on 'L'
    inkml_file = "single_L_dataset/00a11cdf8d773ca3.inkml"
    parsed_data = normalizer.parse_inkml(inkml_file)
    print(f"Ground truth: {parsed_data['label']}")
    
    normalized_data = normalizer.normalize_expression(parsed_data['strokes'])
    
    stroke_tensor = torch.zeros(normalizer.max_strokes, normalizer.max_points_per_stroke, 3)
    for stroke_idx, stroke in enumerate(normalized_data['normalized_strokes']):
        for point_idx, (x, y, t) in enumerate(stroke['points']):
            stroke_tensor[stroke_idx, point_idx, 0] = x
            stroke_tensor[stroke_idx, point_idx, 1] = y  
            stroke_tensor[stroke_idx, point_idx, 2] = t
    
    stroke_masks = torch.tensor(normalized_data['stroke_masks'], dtype=torch.float32)
    stroke_positions = torch.tensor(normalized_data['stroke_positions'], dtype=torch.float32)
    
    stroke_data = stroke_tensor.unsqueeze(0).to(device)
    stroke_masks = stroke_masks.unsqueeze(0).to(device)
    stroke_positions = stroke_positions.unsqueeze(0).to(device)
    
    labels = torch.zeros(1, 20, 11, dtype=torch.long).to(device)
    labels_mask = torch.ones(1, 20, 2, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions, losses = model(stroke_data, stroke_masks, stroke_positions, labels, labels_mask, is_train=False)
        word_probs, struct_probs = predictions
        
        predicted_tokens = torch.argmax(word_probs[0], dim=-1)
        
        print("\nPredicted sequence:")
        predicted_symbols = []
        for i, token_id in enumerate(predicted_tokens[:5]):
            token_id = token_id.item()
            if token_id in words.words_index_dict:
                symbol = words.words_index_dict[token_id]
                predicted_symbols.append(symbol)
                print(f"  {i}: token {token_id} = '{symbol}'")
                if symbol == '<eos>':
                    break
            else:
                print(f"  {i}: token {token_id} = [UNKNOWN]")
        
        predicted_latex = ' '.join([s for s in predicted_symbols if s not in ['<sos>', '<eos>']])
        print(f"\nResult:")
        print(f"Ground truth: '{parsed_data['label']}'")
        print(f"Predicted:    '{predicted_latex}'")
        
        if predicted_latex.strip() == parsed_data['label'].strip():
            print("SUCCESS - Perfect match!")
        else:
            print("Different prediction")

if __name__ == '__main__':
    test_simple()