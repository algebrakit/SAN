#!/usr/bin/env python3
"""
Create a smaller chunk file for fast testing from existing cache data.
"""

import pickle
import os

def create_train_val_chunks():
    # Paths
    cache_dir = 'cache'
    input_chunk = 'inkml_dataset_3fb8ffc968db6720903651d8f1033846.pkl.chunk0'
    train_output = 'inkml_dataset_3fb8ffc968db6720903651d8f1033846_train_10k.pkl'
    val_output = 'inkml_dataset_3fb8ffc968db6720903651d8f1033846_val_10k.pkl'
    
    input_path = os.path.join(cache_dir, input_chunk)
    train_path = os.path.join(cache_dir, train_output)
    val_path = os.path.join(cache_dir, val_output)
    
    # Number of samples for each dataset
    N_TRAIN_SAMPLES = 10000
    N_VAL_SAMPLES = 10000
    
    print(f"Loading data from: {input_path}")
    
    try:
        with open(input_path, 'rb') as f:
            full_data = pickle.load(f)
        
        print(f"Original chunk contains: {len(full_data)} samples")
        
        # Create train and validation splits
        train_data = full_data[:N_TRAIN_SAMPLES]
        val_data = full_data[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES + N_VAL_SAMPLES]
        
        print(f"Creating training chunk with: {len(train_data)} samples")
        print(f"Creating validation chunk with: {len(val_data)} samples")
        
        # Save training chunk
        with open(train_path, 'wb') as f:
            pickle.dump(train_data, f)
        
        # Save validation chunk
        with open(val_path, 'wb') as f:
            pickle.dump(val_data, f)
        
        # Get file sizes
        original_size = os.path.getsize(input_path) / (1024*1024)  # MB
        train_size = os.path.getsize(train_path) / (1024*1024)     # MB
        val_size = os.path.getsize(val_path) / (1024*1024)         # MB
        
        print(f"Datasets created successfully!")
        print(f"  Original: {original_size:.1f} MB ({len(full_data)} samples)")
        print(f"  Training: {train_size:.1f} MB ({len(train_data)} samples)")
        print(f"  Validation: {val_size:.1f} MB ({len(val_data)} samples)")
        print(f"  Train saved to: {train_path}")
        print(f"  Val saved to: {val_path}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        print("Available cache files:")
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                if f.endswith('.pkl') or 'chunk' in f:
                    print(f"  {f}")
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Creating training and validation chunks...")
    success = create_train_val_chunks()
    
    if success:
        print("\nTo use these chunks:")
        print("1. Modify dataset_stroke.py to load 'train_10k' for training")
        print("2. Modify dataset_stroke.py to load 'val_10k' for validation")
        print("3. Or temporarily rename files to replace existing chunks")
    else:
        print("\nFailed to create chunks")