#!/usr/bin/env python3
"""
Debug the model forward pass during training vs inference
"""
import torch
import os
import glob
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words, InkMLDataset
from models.Backbone_stroke import StrokeBackbone

def debug_model_forward():
    print("=== Debugging Model Forward Pass ===")
    
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
    
    # Load checkpoint
    checkpoint_dir = "checkpoints_single_L/SAN_Stroke_Tiny_2025-09-01-10-01_StrokeEncoder_Decoder-SAN_decoder_max_strokes-5_max_points-100/SAN_Stroke_Tiny_2025-09-01-10-01_StrokeEncoder_Decoder-SAN_decoder_max_strokes-5_max_points-100"
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*ExpRate-1.0000*.pth"))
    
    if not checkpoints:
        print("No perfect checkpoints found!")
        return
    
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
    
    print(f"\nInput shapes:")
    print(f"  stroke_data: {stroke_data.shape}")
    print(f"  target_labels: {target_labels.shape}")
    print(f"  labels_mask: {labels_mask.shape}")
    
    with torch.no_grad():
        print(f"\n=== TRAINING MODE (teacher forcing) ===")
        predictions_train, losses_train = model(stroke_data, stroke_masks, stroke_positions, 
                                               target_labels, labels_mask, is_train=True)
        word_probs_train, struct_probs_train = predictions_train
        
        print(f"Word predictions shape: {word_probs_train.shape}")
        
        # Check predictions for each position
        for pos in range(3):  # Just first 3 positions
            predicted_token = torch.argmax(word_probs_train[0, pos], dim=-1).item()
            target_token = target_labels[0, pos, 0].item()
            
            pred_symbol = words.words_index_dict.get(predicted_token, f"UNK_{predicted_token}")
            target_symbol = words.words_index_dict.get(target_token, f"UNK_{target_token}")
            
            # Get confidence
            probs = torch.softmax(word_probs_train[0, pos], dim=0)
            confidence = probs[predicted_token].item()
            
            print(f"  Pos {pos}: predicted={predicted_token}('{pred_symbol}') target={target_token}('{target_symbol}') conf={confidence:.4f}")
        
        print(f"\n=== INFERENCE MODE (autoregressive) ===")
        dummy_labels = torch.zeros_like(target_labels)
        predictions_inf, losses_inf = model(stroke_data, stroke_masks, stroke_positions, 
                                          dummy_labels, labels_mask, is_train=False)
        word_probs_inf, struct_probs_inf = predictions_inf
        
        print(f"Word predictions shape: {word_probs_inf.shape}")
        
        # Check first few predictions
        for pos in range(3):
            predicted_token = torch.argmax(word_probs_inf[0, pos], dim=-1).item()
            pred_symbol = words.words_index_dict.get(predicted_token, f"UNK_{predicted_token}")
            
            probs = torch.softmax(word_probs_inf[0, pos], dim=0)
            confidence = probs[predicted_token].item()
            
            print(f"  Pos {pos}: predicted={predicted_token}('{pred_symbol}') conf={confidence:.4f}")

if __name__ == '__main__':
    debug_model_forward()