#!/usr/bin/env python3
"""
Debug the new model - compare training vs inference
"""
import torch
import os
import glob
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words, InkMLDataset
from models.Backbone_stroke import StrokeBackbone

def debug_new_model():
    print("=== Debugging New Model - Training vs Inference ===")
    
    config_path = "config_tiny.yaml"
    params = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    
    # Load vocabulary and create dataset
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7
    
    # Create dataset
    dataset = InkMLDataset(params, params['inkml_folder'], words, is_train=True)
    
    # Create model
    model = StrokeBackbone(params)
    model = model.to(device)
    
    # Load latest checkpoint
    checkpoint_dir = "checkpoints_single_L"
    latest_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("SAN_Stroke_Tiny_2025-09-01-10")]
    latest_dir = max(latest_dirs)
    full_checkpoint_dir = os.path.join(checkpoint_dir, latest_dir, latest_dir)
    
    checkpoints = glob.glob(os.path.join(full_checkpoint_dir, "*.pth"))
    best_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Loading: {os.path.basename(best_checkpoint)}")
    
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Get data
    stroke_tensor, stroke_masks, stroke_positions, target_labels = dataset[0]
    
    # Add batch dimension
    stroke_data = stroke_tensor.unsqueeze(0).to(device)
    stroke_masks = stroke_masks.unsqueeze(0).to(device)
    stroke_positions = stroke_positions.unsqueeze(0).to(device)
    target_labels = target_labels.unsqueeze(0).to(device)
    
    # Create labels mask
    labels_mask = torch.ones(1, target_labels.shape[1], 2, dtype=torch.float32).to(device)
    
    print(f"Target labels:")
    for pos in range(3):
        word_token = target_labels[0, pos, 0].item()
        word_symbol = words.words_index_dict.get(word_token, f"UNK_{word_token}")
        print(f"  {pos}: {word_token} = '{word_symbol}'")
    
    with torch.no_grad():
        print(f"\n=== TRAINING MODE (teacher forcing) ===")
        predictions_train, _ = model(stroke_data, stroke_masks, stroke_positions, 
                                   target_labels, labels_mask, is_train=True)
        word_probs_train, struct_probs_train = predictions_train
        
        # Check predictions for each position
        for pos in range(3):
            predicted_token = torch.argmax(word_probs_train[0, pos], dim=-1).item()
            target_token = target_labels[0, pos, 0].item()
            
            pred_symbol = words.words_index_dict.get(predicted_token, f"UNK_{predicted_token}")
            target_symbol = words.words_index_dict.get(target_token, f"UNK_{target_token}")
            
            # Get confidence
            probs = torch.softmax(word_probs_train[0, pos], dim=0)
            confidence = probs[predicted_token].item()
            target_confidence = probs[target_token].item()
            
            match = "MATCH" if predicted_token == target_token else "DIFF"
            print(f"  Pos {pos} {match}: pred={predicted_token}('{pred_symbol}',{confidence:.4f}) target={target_token}('{target_symbol}',{target_confidence:.4f})")
        
        print(f"\n=== INFERENCE MODE (autoregressive) ===")
        dummy_labels = torch.zeros_like(target_labels)
        predictions_inf, _ = model(stroke_data, stroke_masks, stroke_positions, 
                                 dummy_labels, labels_mask, is_train=False)
        word_probs_inf, struct_probs_inf = predictions_inf
        
        # Check first few predictions
        for pos in range(3):
            predicted_token = torch.argmax(word_probs_inf[0, pos], dim=-1).item()
            pred_symbol = words.words_index_dict.get(predicted_token, f"UNK_{predicted_token}")
            
            probs = torch.softmax(word_probs_inf[0, pos], dim=0)
            confidence = probs[predicted_token].item()
            
            # What should we predict at this position?
            expected_token = target_labels[0, pos, 0].item()
            expected_symbol = words.words_index_dict.get(expected_token, f"UNK_{expected_token}")
            expected_confidence = probs[expected_token].item()
            
            match = "MATCH" if predicted_token == expected_token else "DIFF"
            print(f"  Pos {pos} {match}: pred={predicted_token}('{pred_symbol}',{confidence:.4f}) expected={expected_token}('{expected_symbol}',{expected_confidence:.4f})")

if __name__ == '__main__':
    debug_new_model()