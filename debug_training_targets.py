#!/usr/bin/env python3
"""
Debug what the model is actually being trained to predict
"""
import torch
import os
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words, InkMLDataset
from models.Backbone_stroke import StrokeBackbone

def debug_training_targets():
    print("=== Debugging Training Targets ===")
    
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
    print(f"Dataset has {len(dataset)} samples")
    
    # Test the simple 'L' sample (index 3)
    sample = dataset.data_samples[3]  # This should be the L sample
    print(f"\nTesting sample: {os.path.basename(sample['file_path'])}")
    print(f"Ground truth: '{sample['label']}'")
    
    # Get processed data 
    stroke_tensor, stroke_masks, stroke_positions, target_labels = dataset[3]
    
    print(f"\nTarget labels shape: {target_labels.shape}")
    print(f"Target labels content (what model should predict):")
    
    for pos in range(min(10, target_labels.shape[0])):
        word_token = target_labels[pos, 0].item()
        word_symbol = words.words_index_dict.get(word_token, f"UNK_{word_token}")
        print(f"  Position {pos}: token {word_token} = '{word_symbol}'")
        if word_symbol == '<eos>':
            break
    
    print(f"\nNow let's trace what happens during teacher forcing:")
    print(f"The decoder should learn these input -> output mappings:")
    
    # Remember: with correct teacher forcing, input is previous token
    for pos in range(min(3, target_labels.shape[0])):
        if pos == 0:
            input_token = 1  # <sos> is always the first input
            input_symbol = '<sos>'
        else:
            input_token = target_labels[pos-1, 0].item()
            input_symbol = words.words_index_dict.get(input_token, f"UNK_{input_token}")
        
        output_token = target_labels[pos, 0].item()
        output_symbol = words.words_index_dict.get(output_token, f"UNK_{output_token}")
        
        print(f"  Step {pos}: Input='{input_symbol}' (token {input_token}) -> Target='{output_symbol}' (token {output_token})")
        
        if output_symbol == '<eos>':
            break
    
    # Create a simple model to test training step
    model = StrokeBackbone(params)
    model = model.to(device)
    model.train()
    
    # Add batch dimension
    stroke_data = stroke_tensor.unsqueeze(0).to(device)
    stroke_masks = stroke_masks.unsqueeze(0).to(device)
    stroke_positions = stroke_positions.unsqueeze(0).to(device)
    target_labels_batch = target_labels.unsqueeze(0).to(device)
    labels_mask = torch.ones(1, target_labels.shape[0], 2, dtype=torch.float32).to(device)
    
    print(f"\n=== Testing Training Forward Pass ===")
    with torch.no_grad():
        predictions, losses = model(stroke_data, stroke_masks, stroke_positions, 
                                  target_labels_batch, labels_mask, is_train=True)
        word_probs, struct_probs = predictions
        
        print(f"Model predictions vs targets:")
        for pos in range(min(3, word_probs.shape[1])):
            predicted_token = torch.argmax(word_probs[0, pos], dim=-1).item()
            target_token = target_labels[pos, 0].item()
            
            pred_symbol = words.words_index_dict.get(predicted_token, f"UNK_{predicted_token}")
            target_symbol = words.words_index_dict.get(target_token, f"UNK_{target_token}")
            
            # Get top 3 predictions
            probs = torch.softmax(word_probs[0, pos], dim=0)
            top_3 = torch.topk(probs, 3)
            
            match = "MATCH" if predicted_token == target_token else "WRONG"
            print(f"  Position {pos} [{match}]: Target='{target_symbol}' Predicted='{pred_symbol}'")
            print(f"    Top 3 predictions:")
            for i, (prob, token_id) in enumerate(zip(top_3.values, top_3.indices)):
                token_id = token_id.item()
                prob = prob.item()
                symbol = words.words_index_dict.get(token_id, f"UNK_{token_id}")
                print(f"      {i+1}. '{symbol}' (token {token_id}): {prob:.4f}")
            
            if target_symbol == '<eos>':
                break

if __name__ == '__main__':
    debug_training_targets()