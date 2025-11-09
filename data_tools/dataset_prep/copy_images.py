#!/usr/bin/env python3
"""
Copy image files listed in images.txt from source directory to destination directory.
"""

import os
import shutil
from pathlib import Path

# Configuration
IMAGES_LIST = "data_tools/dataset_prep/images.txt"
SOURCE_DIR = "/Users/martijnslob/handwriting-data/mathwriting-2024/images"
DEST_DIR = "data/newsymbols"

def main():
    # Get the script's parent directory (repository root)
    repo_root = Path(__file__).parent.parent.parent

    # Resolve paths
    images_list_path = repo_root / IMAGES_LIST
    source_dir = Path(SOURCE_DIR)
    dest_dir = repo_root / DEST_DIR

    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Read the list of images
    with open(images_list_path, 'r') as f:
        image_files = [line.strip() for line in f if line.strip()]

    print(f"Found {len(image_files)} images to copy")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print()

    # Copy each image
    copied = 0
    skipped = 0
    errors = 0

    for image_file in image_files:
        source_path = source_dir / image_file
        dest_path = dest_dir / image_file

        if not source_path.exists():
            print(f"⚠ Source file not found: {image_file}")
            skipped += 1
            continue

        try:
            shutil.copy2(source_path, dest_path)
            copied += 1
            if copied % 10 == 0:
                print(f"Copied {copied}/{len(image_files)} images...")
        except Exception as e:
            print(f"✗ Error copying {image_file}: {e}")
            errors += 1

    print()
    print("=== Summary ===")
    print(f"Successfully copied: {copied}")
    print(f"Skipped (not found): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total: {len(image_files)}")

if __name__ == "__main__":
    main()
