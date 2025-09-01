#!/usr/bin/env python3
"""
Test the final model with all fixes applied
"""
import torch
import os
import glob
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words, InkMLDataset
from models.Backbone_stroke import StrokeBackbone

def test_final_model():
    print("=== Testing Final Model with All Fixes ===")
    
    config_path = "config_tiny.yaml"
    params = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    
    # Load vocabulary and create dataset
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7
    
    # Create dataset
    dataset = InkMLDataset(params, params['inkml_folder'], words, is_train=False)
    print(f"Dataset has {len(dataset)} samples")
    
    # Create model
    model = StrokeBackbone(params)
    model = model.to(device)
    
    # Load the latest checkpoint from fixed training (2025-09-01-11-41)
    checkpoint_dir = "checkpoints_single_L"
    latest_dirs = [d for d in os.listdir(checkpoint_dir) if "2025-09-01-11-41" in d]
    if not latest_dirs:
        print("No latest checkpoints found!")
        return False
    
    latest_dir = max(latest_dirs)
    full_checkpoint_dir = os.path.join(checkpoint_dir, latest_dir, latest_dir)
    print(f"Looking in: {full_checkpoint_dir}")
    
    # Get perfect checkpoints from new training
    checkpoints = glob.glob(os.path.join(full_checkpoint_dir, "*ExpRate-1.0000*.pth"))
    if not checkpoints:
        # Get any checkpoint if no perfect ones
        checkpoints = glob.glob(os.path.join(full_checkpoint_dir, "*.pth"))
    
    if not checkpoints:
        print("No checkpoints found in latest training!")
        return False
    
    best_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Loading: {os.path.basename(best_checkpoint)}")
    
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Test the single L sample
    sample = dataset.data_samples[0]
    print(f"\nFile: {os.path.basename(sample['file_path'])}")
    print(f"Ground truth: '{sample['label']}'")
    
    # Get processed data
    stroke_tensor, stroke_masks, stroke_positions, target_labels = dataset[0]
    
    # Add batch dimension
    stroke_data = stroke_tensor.unsqueeze(0).to(device)
    stroke_masks = stroke_masks.unsqueeze(0).to(device)
    stroke_positions = stroke_positions.unsqueeze(0).to(device)
    
    print(f"\nExpected sequence:")
    for pos in range(min(5, target_labels.shape[0])):
        word_token = target_labels[pos, 0].item()
        if word_token in words.words_index_dict:
            symbol = words.words_index_dict[word_token]
            print(f"  {pos}: token {word_token} = '{symbol}'")
            if symbol == '<eos>':
                break
    
    with torch.no_grad():
        print(f"\n=== AUTOREGRESSIVE INFERENCE ===")
        # Inference mode - let decoder generate autoregressively
        dummy_labels = torch.zeros(1, 20, 11, dtype=torch.long).to(device)
        dummy_mask = torch.ones(1, 20, 2, dtype=torch.float32).to(device)
        
        predictions, losses = model(stroke_data, stroke_masks, stroke_positions, 
                                  dummy_labels, dummy_mask, is_train=False)
        word_probs, struct_probs = predictions
        
        print(f"Word predictions shape: {word_probs.shape}")
        
        # Decode the predictions
        predicted_tokens = torch.argmax(word_probs[0], dim=-1)
        predicted_symbols = []
        
        print(f"\nDecoded sequence:")
        for pos in range(min(10, predicted_tokens.shape[0])):
            token_id = predicted_tokens[pos].item()
            if token_id in words.words_index_dict:
                symbol = words.words_index_dict[token_id]
                predicted_symbols.append(symbol)
                print(f"  {pos}: token {token_id} = '{symbol}'")
                if symbol == '<eos>':
                    break
            else:
                print(f"  {pos}: token {token_id} = [UNKNOWN]")
        
        # Extract content (remove start/end tokens)
        predicted_text = ' '.join([s for s in predicted_symbols if s not in ['<sos>', '<eos>']])
        
        print(f"\n*** FINAL RESULT ***")
        print(f"Ground truth: '{sample['label']}'")
        print(f"Predicted:    '{predicted_text}'")
        
        if predicted_text.strip() == sample['label'].strip():
            print("SUCCESS - PERFECT MATCH!")
            print("The stroke-aware SAN model with all fixes is working perfectly!")
            print("Autoregressive inference is generating the correct sequence!")
            return True
        else:
            print("Different result - analyzing...")
            
            # Show confidence scores
            print(f"\nConfidence analysis:")
            for pos in range(min(5, word_probs.shape[1])):
                if pos < len(predicted_symbols):
                    probs = torch.softmax(word_probs[0, pos], dim=0)
                    top_3 = torch.topk(probs, 3)
                    print(f"Position {pos}:")
                    for i, (prob, token_id) in enumerate(zip(top_3.values, top_3.indices)):
                        token_id = token_id.item()
                        prob = prob.item()
                        symbol = words.words_index_dict.get(token_id, f"UNK_{token_id}")
                        print(f"  {i+1}. '{symbol}' (token {token_id}): {prob:.4f}")
            
            return False

if __name__ == '__main__':
    success = test_final_model()
    if success:
        print("\nAll debugging complete! The model is ready for real training.")
    else:
        print("\nStill needs debugging...")
    exit(0 if success else 1)