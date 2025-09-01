#!/usr/bin/env python3
"""
Debug what the model learned in training vs what it does in inference
"""
import torch
import os
import glob
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words, InkMLDataset
from models.Backbone_stroke import StrokeBackbone

def debug_training_vs_inference():
    print("=== Debug Training vs Inference Behavior ===")
    
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
    latest_dirs = [d for d in os.listdir(checkpoint_dir) if "2025-09-01-10-37" in d]
    latest_dir = max(latest_dirs)
    full_checkpoint_dir = os.path.join(checkpoint_dir, latest_dir, latest_dir)
    
    checkpoints = glob.glob(os.path.join(full_checkpoint_dir, "*ExpRate-1.0000*.pth"))
    if not checkpoints:
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
    
    labels_mask = torch.ones(1, target_labels.shape[1], 2, dtype=torch.float32).to(device)
    
    print(f"Target sequence:")
    for pos in range(3):
        word_token = target_labels[0, pos, 0].item()
        word_symbol = words.words_index_dict.get(word_token, f"UNK_{word_token}")
        print(f"  {pos}: {word_token} = '{word_symbol}'")
    
    with torch.no_grad():
        print(f"\n=== TRAINING MODE (teacher forcing) ===")
        predictions_train, _ = model(stroke_data, stroke_masks, stroke_positions, 
                                   target_labels, labels_mask, is_train=True)
        word_probs_train, _ = predictions_train
        
        print(f"What the model learned to predict for each input:")
        for pos in range(3):
            input_token = target_labels[0, pos, 0].item() if pos > 0 else 1  # First input is always SOS
            predicted_token = torch.argmax(word_probs_train[0, pos], dim=-1).item()
            target_token = target_labels[0, pos, 0].item()
            
            input_symbol = words.words_index_dict.get(input_token, f"UNK_{input_token}")
            pred_symbol = words.words_index_dict.get(predicted_token, f"UNK_{predicted_token}")
            target_symbol = words.words_index_dict.get(target_token, f"UNK_{target_token}")
            
            # Get confidence
            probs = torch.softmax(word_probs_train[0, pos], dim=0)
            confidence = probs[predicted_token].item()
            target_confidence = probs[target_token].item()
            
            match = "MATCH" if predicted_token == target_token else "DIFF"
            print(f"  Pos {pos} {match}: input='{input_symbol}' -> pred='{pred_symbol}'({confidence:.4f}) target='{target_symbol}'({target_confidence:.4f})")
        
        print(f"\n=== INFERENCE MODE (autoregressive) ===")
        dummy_labels = torch.zeros_like(target_labels)
        predictions_inf, _ = model(stroke_data, stroke_masks, stroke_positions, 
                                 dummy_labels, labels_mask, is_train=False)
        word_probs_inf, _ = predictions_inf
        
        print(f"Autoregressive predictions:")
        for pos in range(5):
            predicted_token = torch.argmax(word_probs_inf[0, pos], dim=-1).item()
            pred_symbol = words.words_index_dict.get(predicted_token, f"UNK_{predicted_token}")
            
            probs = torch.softmax(word_probs_inf[0, pos], dim=0)
            confidence = probs[predicted_token].item()
            
            print(f"  Step {pos}: pred='{pred_symbol}' (token {predicted_token}) conf={confidence:.4f}")
            
            if predicted_token == 0:  # <eos>
                print("    -> Should stop here")
                break
            elif predicted_token == 1:  # <sos>
                print("    -> This creates an infinite loop!")

if __name__ == '__main__':
    debug_training_vs_inference()