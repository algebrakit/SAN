#!/usr/bin/env python3
"""
Debugging utility for stroke recognition data pipeline.
Use this to inspect data quality and identify issues.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_stroke import StrokeNormalizer, InkMLDataset, Words
from utils import load_config

def debug_single_inkml(inkml_path, config_path='config_stroke.yaml'):
    """Debug a single InkML file through the entire pipeline"""
    print(f"=== Debugging InkML: {os.path.basename(inkml_path)} ===")
    
    # Load config
    params = load_config(config_path)
    normalizer = StrokeNormalizer(params)
    
    # Parse InkML
    parsed_data = normalizer.parse_inkml(inkml_path)
    if parsed_data is None:
        print("❌ Failed to parse InkML file")
        return None
    
    print(f"[OK] Original label: {parsed_data['label']}")
    print(f"[OK] Number of strokes: {len(parsed_data['strokes'])}")
    
    # Print original coordinates
    print("\n--- Original Stroke Coordinates ---")
    for i, stroke in enumerate(parsed_data['strokes'][:3]):  # First 3 strokes
        if stroke['points']:
            xs = [p[0] for p in stroke['points'][:5]]  # First 5 points
            ys = [p[1] for p in stroke['points'][:5]]
            print(f"Stroke {i}: X=[{', '.join(f'{x:.1f}' for x in xs)}...], "
                  f"Y=[{', '.join(f'{y:.1f}' for y in ys)}...]")
    
    # Normalize
    normalized_data = normalizer.normalize_expression(parsed_data['strokes'])
    
    print(f"\n--- Normalization Results ---")
    print(f"✓ Valid strokes: {normalized_data['num_valid_strokes']}")
    metadata = normalized_data['expression_metadata']
    print(f"✓ Scale factor: {metadata['scale_factor']:.4f}")
    print(f"✓ Offset: ({metadata['offset'][0]:.2f}, {metadata['offset'][1]:.2f})")
    print(f"✓ Aspect ratio: {metadata['aspect_ratio']:.2f}")
    print(f"✓ Target dimensions: {metadata['target_width']} × {metadata['target_height']}")
    
    # Print normalized coordinates
    print("\n--- Normalized Stroke Coordinates ---")
    for i, stroke in enumerate(normalized_data['normalized_strokes'][:3]):
        if stroke['points'] and any(p != (0,0,0) for p in stroke['points'][:5]):
            xs = [p[0] for p in stroke['points'][:5] if p != (0,0,0)]
            ys = [p[1] for p in stroke['points'][:5] if p != (0,0,0)]
            if xs and ys:
                print(f"Stroke {i}: X=[{', '.join(f'{x:.2f}' for x in xs)}], "
                      f"Y=[{', '.join(f'{y:.2f}' for y in ys)}]")
    
    # Check stroke positions
    print(f"\n--- Stroke Positions ---")
    positions = normalized_data['stroke_positions'][:5]  # First 5 strokes
    for i, (cx, cy, w, h) in enumerate(positions):
        if cx != 0 or cy != 0 or w != 0 or h != 0:
            print(f"Stroke {i}: center=({cx:.2f}, {cy:.2f}), size=({w:.2f} × {h:.2f})")
    
    return normalized_data

def debug_dataset_sample(config_path='config_stroke.yaml', num_samples=3):
    """Debug dataset loading and processing"""
    print(f"=== Debugging Dataset Loading ===")
    
    params = load_config(config_path)
    
    # Load vocabulary
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7
    
    print(f"✓ Vocabulary size: {len(words)}")
    print(f"✓ Sample words: {[words.words_index_dict[i] for i in range(min(10, len(words)))]}")
    
    # Create dataset
    inkml_folder = params.get('inkml_folder', 'synthetic')
    if not os.path.exists(inkml_folder):
        print(f"❌ InkML folder not found: {inkml_folder}")
        return None
    
    dataset = InkMLDataset(params, inkml_folder, words, is_train=True)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Check first few samples
    for i in range(min(num_samples, len(dataset))):
        print(f"\n--- Sample {i} ---")
        try:
            stroke_tensor, stroke_masks, stroke_positions, labels = dataset[i]
            
            print(f"✓ Stroke tensor shape: {stroke_tensor.shape}")
            print(f"✓ Stroke masks shape: {stroke_masks.shape}")
            print(f"✓ Positions shape: {stroke_positions.shape}")
            print(f"✓ Labels shape: {labels.shape}")
            
            # Check for valid data
            valid_strokes = stroke_masks.sum(dim=1) > 0
            num_valid = valid_strokes.sum().item()
            print(f"✓ Valid strokes: {num_valid}")
            
            # Check label content
            valid_labels = labels[:, 1] > 0  # Token IDs > 0
            label_tokens = labels[valid_labels, 1][:10]  # First 10 tokens
            if len(label_tokens) > 0:
                decoded_tokens = [words.words_index_dict.get(int(t.item()), '<unk>') for t in label_tokens]
                print(f"✓ Label tokens: {' '.join(decoded_tokens)}")
            else:
                print("❌ No valid label tokens found!")
                
        except Exception as e:
            print(f"❌ Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()

def debug_coordinate_ranges(config_path='config_stroke.yaml', num_files=10):
    """Debug coordinate ranges across multiple files"""
    print(f"=== Debugging Coordinate Ranges Across {num_files} Files ===")
    
    params = load_config(config_path)
    normalizer = StrokeNormalizer(params)
    inkml_folder = params.get('inkml_folder', 'synthetic')
    
    if not os.path.exists(inkml_folder):
        print(f"❌ InkML folder not found: {inkml_folder}")
        return
    
    # Get sample files
    import glob
    files = glob.glob(os.path.join(inkml_folder, "*.inkml"))[:num_files]
    
    all_original_coords = []
    all_normalized_coords = []
    all_positions = []
    
    for file_path in files:
        try:
            parsed_data = normalizer.parse_inkml(file_path)
            if parsed_data is None:
                continue
                
            # Original coordinates
            for stroke in parsed_data['strokes']:
                for x, y, t in stroke['points']:
                    all_original_coords.append((x, y))
            
            # Normalized data
            normalized_data = normalizer.normalize_expression(parsed_data['strokes'])
            for stroke in normalized_data['normalized_strokes']:
                for x, y, t in stroke['points']:
                    if x != 0 or y != 0:  # Skip padding
                        all_normalized_coords.append((x, y))
            
            # Stroke positions
            positions = normalized_data['stroke_positions']
            for cx, cy, w, h in positions:
                if cx != 0 or cy != 0:  # Skip padding
                    all_positions.append((cx, cy, w, h))
                    
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            continue
    
    if all_original_coords:
        orig_xs, orig_ys = zip(*all_original_coords)
        print(f"[OK] Original coordinates:")
        print(f"  X range: [{min(orig_xs):.1f}, {max(orig_xs):.1f}]")
        print(f"  Y range: [{min(orig_ys):.1f}, {max(orig_ys):.1f}]")
    
    if all_normalized_coords:
        norm_xs, norm_ys = zip(*all_normalized_coords)
        print(f"✓ Normalized coordinates:")
        print(f"  X range: [{min(norm_xs):.2f}, {max(norm_xs):.2f}]")
        print(f"  Y range: [{min(norm_ys):.2f}, {max(norm_ys):.2f}]")
    
    if all_positions:
        cxs, cys, ws, hs = zip(*all_positions)
        print(f"✓ Stroke positions:")
        print(f"  Center X range: [{min(cxs):.2f}, {max(cxs):.2f}]")
        print(f"  Center Y range: [{min(cys):.2f}, {max(cys):.2f}]")
        print(f"  Width range: [{min(ws):.2f}, {max(ws):.2f}]")
        print(f"  Height range: [{min(hs):.2f}, {max(hs):.2f}]")

def main():
    """Main debugging function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug stroke recognition data pipeline')
    parser.add_argument('--config', default='config_stroke.yaml', help='Config file path')
    parser.add_argument('--inkml', help='Debug specific InkML file')
    parser.add_argument('--dataset', action='store_true', help='Debug dataset loading')
    parser.add_argument('--coords', action='store_true', help='Debug coordinate ranges')
    parser.add_argument('--all', action='store_true', help='Run all debug tests')
    parser.add_argument('--num-files', type=int, default=10, help='Number of files to test')
    parser.add_argument('--num-samples', type=int, default=3, help='Number of dataset samples to check')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    
    if args.inkml:
        if os.path.exists(args.inkml):
            debug_single_inkml(args.inkml, args.config)
        else:
            print(f"❌ InkML file not found: {args.inkml}")
    
    if args.dataset or args.all:
        debug_dataset_sample(args.config, args.num_samples)
    
    if args.coords or args.all:
        debug_coordinate_ranges(args.config, args.num_files)
    
    if not any([args.inkml, args.dataset, args.coords, args.all]):
        print("No debug option specified. Use --help to see available options.")
        print("\nQuick start:")
        print(f"  python {__file__} --all  # Run all debug tests")
        print(f"  python {__file__} --inkml your_file.inkml  # Debug specific file")

if __name__ == '__main__':
    main()