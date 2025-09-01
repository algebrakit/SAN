#!/usr/bin/env python3
"""
Test what the model actually learned during training
"""
import torch
from utils import load_config
from dataset_stroke import StrokeNormalizer, Words, InkMLDataset
from models.Backbone_stroke import StrokeBackbone

def test_training_data():
    print("=== Testing What Model Learned ===")
    
    config_path = "config_tiny.yaml"
    params = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    
    # Load vocabulary and create dataset exactly like training
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7
    
    # Create the dataset (same as training)
    dataset = InkMLDataset(params, params['inkml_folder'], words, is_train=True)
    print(f"Dataset has {len(dataset)} samples")
    
    # Create model
    model = StrokeBackbone(params)
    model = model.to(device)
    
    # Load latest checkpoint
    import glob
    import os
    checkpoint_dir = "checkpoints_tiny/SAN_Stroke_Tiny_2025-09-01-09-56_StrokeEncoder_Decoder-SAN_decoder_max_strokes-5_max_points-100/SAN_Stroke_Tiny_2025-09-01-09-56_StrokeEncoder_Decoder-SAN_decoder_max_strokes-5_max_points-100"
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    
    print(f"Loading: {os.path.basename(latest_checkpoint)}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Test each sample in the dataset
    for i in range(len(dataset)):
        print(f"\n--- Sample {i} ---")
        sample = dataset.data_samples[i]
        print(f"File: {os.path.basename(sample['file_path'])}")
        print(f"Label: '{sample['label']}'")
        
        # Get the processed data from dataset (this is what training sees)
        stroke_tensor, stroke_masks, stroke_positions, target_labels = dataset[i]
        
        print(f"Target label shape: {target_labels.shape}")
        print("Target sequence:")
        for pos in range(min(5, target_labels.shape[0])):
            word_token = target_labels[pos, 0].item()
            if word_token in words.words_index_dict:
                symbol = words.words_index_dict[word_token]
                print(f"  {pos}: token {word_token} = '{symbol}'")
                if symbol == '<eos>':
                    break
            else:
                print(f"  {pos}: token {word_token} = [UNKNOWN]")
        
        # Run inference with the same data format as training
        stroke_data = stroke_tensor.unsqueeze(0).to(device)  # Add batch dim
        stroke_masks = stroke_masks.unsqueeze(0).to(device)
        stroke_positions = stroke_positions.unsqueeze(0).to(device)
        
        # Create label mask (same as training)
        labels_mask = torch.ones(1, target_labels.shape[0], 2, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            # Add batch dimension to target labels for the forward pass
            target_batch = target_labels.unsqueeze(0).to(device)
            
            predictions, losses = model(stroke_data, stroke_masks, stroke_positions, target_batch, labels_mask, is_train=False)
            word_probs, struct_probs = predictions
            
            print("Predicted sequence:")
            predicted_tokens = torch.argmax(word_probs[0], dim=-1)
            for pos in range(min(5, predicted_tokens.shape[0])):
                token_id = predicted_tokens[pos].item()
                if token_id in words.words_index_dict:
                    symbol = words.words_index_dict[token_id]
                    print(f"  {pos}: token {token_id} = '{symbol}'")
                    if symbol == '<eos>':
                        break
                else:
                    print(f"  {pos}: token {token_id} = [UNKNOWN]")
        
        # Check if this specific sample matches our 'L' file
        if sample['file_path'].endswith('00a11cdf8d773ca3.inkml'):
            print(f"\n🎯 This is the 'L' sample!")
            predicted_symbols = []
            for pos in range(min(5, predicted_tokens.shape[0])):
                token_id = predicted_tokens[pos].item()
                if token_id in words.words_index_dict:
                    symbol = words.words_index_dict[token_id]
                    predicted_symbols.append(symbol)
                    if symbol == '<eos>':
                        break
            
            predicted_text = ' '.join([s for s in predicted_symbols if s not in ['<sos>', '<eos>']])
            print(f"Ground truth: '{sample['label']}'")
            print(f"Predicted:    '{predicted_text}'")
            
            if predicted_text.strip() == sample['label'].strip():
                print("✅ SUCCESS - Perfect match!")
                return True
            else:
                print("❌ Still not matching")
                
        if i >= 4:  # Limit output for readability
            break
    
    return False

if __name__ == '__main__':
    test_training_data()