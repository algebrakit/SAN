#!/usr/bin/env python3
"""
Test the model trained on the full tiny dataset
"""
import torch
import os
import glob
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words, InkMLDataset
from models.Backbone_stroke import StrokeBackbone

def test_tiny_dataset():
    print("=== Testing Model Trained on Full Tiny Dataset ===")
    
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
    
    # Load the latest checkpoint from expanded vocabulary training (2025-09-01-11-02)
    checkpoint_dir = "checkpoints_single_L"
    latest_dirs = [d for d in os.listdir(checkpoint_dir) if "2025-09-01-11-02" in d]
    if not latest_dirs:
        print("No latest checkpoints found!")
        return False
    
    latest_dir = max(latest_dirs)
    full_checkpoint_dir = os.path.join(checkpoint_dir, latest_dir, latest_dir)
    print(f"Looking in: {full_checkpoint_dir}")
    
    # Get best checkpoint
    checkpoints = glob.glob(os.path.join(full_checkpoint_dir, "*.pth"))
    if not checkpoints:
        print("No checkpoints found!")
        return False
    
    best_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Loading: {os.path.basename(best_checkpoint)}")
    
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Test all samples in the dataset
    total_correct = 0
    total_samples = len(dataset)
    
    for idx in range(total_samples):
        sample = dataset.data_samples[idx]
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{total_samples}: {os.path.basename(sample['file_path'])}")
        print(f"Ground truth: '{sample['label']}'")
        
        # Get processed data
        stroke_tensor, stroke_masks, stroke_positions, target_labels = dataset[idx]
        
        # Add batch dimension
        stroke_data = stroke_tensor.unsqueeze(0).to(device)
        stroke_masks = stroke_masks.unsqueeze(0).to(device)
        stroke_positions = stroke_positions.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Autoregressive inference
            dummy_labels = torch.zeros(1, 20, 11, dtype=torch.long).to(device)
            dummy_mask = torch.ones(1, 20, 2, dtype=torch.float32).to(device)
            
            predictions, losses = model(stroke_data, stroke_masks, stroke_positions, 
                                      dummy_labels, dummy_mask, is_train=False)
            word_probs, struct_probs = predictions
            
            # Decode the predictions
            predicted_tokens = torch.argmax(word_probs[0], dim=-1)
            predicted_symbols = []
            
            print(f"Decoded sequence:")
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
            
            print(f"\nResult:")
            print(f"Ground truth: '{sample['label']}'")
            print(f"Predicted:    '{predicted_text}'")
            
            if predicted_text.strip() == sample['label'].strip():
                print("SUCCESS - PERFECT MATCH!")
                total_correct += 1
            else:
                print("Different prediction")
                
                # Show top predictions for first few positions
                print(f"\nTop predictions:")
                for pos in range(min(3, word_probs.shape[1])):
                    probs = torch.softmax(word_probs[0, pos], dim=0)
                    top_3 = torch.topk(probs, 3)
                    print(f"Position {pos}:")
                    for i, (prob, token_id) in enumerate(zip(top_3.values, top_3.indices)):
                        token_id = token_id.item()
                        prob = prob.item()
                        symbol = words.words_index_dict.get(token_id, f"UNK_{token_id}")
                        print(f"  {i+1}. '{symbol}' (token {token_id}): {prob:.4f}")
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"Correct predictions: {total_correct}/{total_samples}")
    print(f"Success rate: {total_correct/total_samples*100:.1f}%")
    
    if total_correct == total_samples:
        print("PERFECT! All samples predicted correctly!")
        print("The stroke-aware SAN model is working perfectly!")
        return True
    elif total_correct > 0:
        print(f"Good progress! {total_correct} out of {total_samples} samples correct.")
        return False
    else:
        print("Still needs more training...")
        return False

if __name__ == '__main__':
    success = test_tiny_dataset()
    exit(0 if success else 1)