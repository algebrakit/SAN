#!/usr/bin/env python3
"""
Dataset splitting script for handwritten mathematical expression recognition.
Splits the dataset into train/validation/test sets with 80/10/10 ratio.
"""

import shutil
import random
from pathlib import Path

def split_dataset(data_dir=".", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset into train/validation/test sets.

    Args:
        data_dir: Directory containing labels.txt and images/ folder
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.1)
        test_ratio: Ratio for test set (default: 0.1)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    data_path = Path(data_dir)
    labels_file = data_path / "labels.txt"
    images_dir = data_path / "images"

    # Check if required files exist
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Read all labels
    print("Reading labels...")
    with open(labels_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse labels and extract image filenames
    data_entries = []
    for line in lines:
        line = line.strip()
        if line:
            # Format: image_filename.bmp	LaTeX_expression
            parts = line.split('\t')
            if len(parts) >= 2:
                image_filename = parts[0].strip()
                latex_expression = parts[1].strip()
                data_entries.append((line, image_filename))

    print(f"Found {len(data_entries)} data entries")

    # Shuffle the data
    random.shuffle(data_entries)

    # Calculate split indices
    total_samples = len(data_entries)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    # Split the data
    train_data = data_entries[:train_end]
    val_data = data_entries[train_end:val_end]
    test_data = data_entries[val_end:]

    print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create output directories
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split_name, split_data in splits.items():
        print(f"\nCreating {split_name} split...")

        # Create directories
        split_dir = data_path / split_name
        split_images_dir = split_dir / "images"
        split_dir.mkdir(exist_ok=True)
        split_images_dir.mkdir(exist_ok=True)

        # Write labels file
        labels_path = split_dir / "labels.txt"
        with open(labels_path, 'w', encoding='utf-8') as f:
            for label_line, image_filename in split_data:
                f.write(f"{label_line}\n")

        # Copy image files
        print(f"Copying {len(split_data)} images for {split_name}...")
        for _, image_filename in split_data:
            src_image = images_dir / image_filename
            dst_image = split_images_dir / image_filename

            if src_image.exists():
                shutil.copy2(src_image, dst_image)
            else:
                print(f"Warning: Image not found: {src_image}")

        print(f"Created {split_name} split with {len(split_data)} samples")

    print(f"\nDataset splitting completed!")
    print(f"Original dataset: {total_samples} samples")
    if total_samples > 0:
        print(f"Train: {len(train_data)} samples ({len(train_data)/total_samples*100:.1f}%)")
        print(f"Validation: {len(val_data)} samples ({len(val_data)/total_samples*100:.1f}%)")
        print(f"Test: {len(test_data)} samples ({len(test_data)/total_samples*100:.1f}%)")
    else:
        print("No data entries found to split!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset into train/val/test sets")
    parser.add_argument("--data_dir", default=".", help="Directory containing labels.txt and images/")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")

    args = parser.parse_args()

    split_dataset(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )