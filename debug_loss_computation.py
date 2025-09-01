#!/usr/bin/env python3
"""
Debug the loss computation to see why the model isn't learning
"""
import torch
import os
import glob
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words, InkMLDataset
from models.Backbone_stroke import StrokeBackbone

def debug_loss_computation():
    print("=== Debugging Loss Computation ===")
    
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
    
    # Don't load existing checkpoint - start fresh to see what happens
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
        struct_token = target_labels[0, pos, 1].item()
        parent_id = target_labels[0, pos, 2].item()
        relation_token = target_labels[0, pos, 3].item()
        
        word_symbol = words.words_index_dict.get(word_token, f"UNK_{word_token}")
        struct_symbol = words.words_index_dict.get(struct_token, f"UNK_{struct_token}")
        relation_symbol = words.words_index_dict.get(relation_token, f"UNK_{relation_token}")
        
        print(f"  {pos}: word={word_token}('{word_symbol}') struct={struct_token}('{struct_symbol}') parent={parent_id} relation={relation_token}('{relation_symbol}')")
    
    with torch.no_grad():
        print(f"\n=== FORWARD PASS (training mode) ===")
        predictions, losses = model(stroke_data, stroke_masks, stroke_positions, 
                                  target_labels, labels_mask, is_train=True)
        word_probs, struct_probs = predictions
        
        if len(losses) == 4:
            word_loss, struct_loss, parent_loss, kl_loss = losses
            print(f"Losses: word={word_loss.item():.6f}, struct={struct_loss.item():.6f}, parent={parent_loss.item():.6f}, kl={kl_loss.item():.6f}")
        else:
            word_loss, struct_loss = losses
            print(f"Losses: word={word_loss.item():.6f}, struct={struct_loss.item():.6f}")
        
        print(f"\nPredictions at each position:")
        for pos in range(3):
            predicted_token = torch.argmax(word_probs[0, pos], dim=-1).item()
            target_token = target_labels[0, pos, 0].item()
            
            pred_symbol = words.words_index_dict.get(predicted_token, f"UNK_{predicted_token}")
            target_symbol = words.words_index_dict.get(target_token, f"UNK_{target_token}")
            
            # Get raw logits and probabilities
            logits = word_probs[0, pos]
            probs = torch.softmax(logits, dim=0)
            
            target_logit = logits[target_token].item()
            target_prob = probs[target_token].item()
            pred_logit = logits[predicted_token].item()
            pred_prob = probs[predicted_token].item()
            
            print(f"  Pos {pos}:")
            print(f"    Target: {target_token}('{target_symbol}') logit={target_logit:.4f} prob={target_prob:.4f}")
            print(f"    Predicted: {predicted_token}('{pred_symbol}') logit={pred_logit:.4f} prob={pred_prob:.4f}")
            
            # Check if gradients would flow correctly
            word_loss_single = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([target_token], device=device))
            print(f"    Single-position word loss: {word_loss_single.item():.6f}")

if __name__ == '__main__':
    debug_loss_computation()