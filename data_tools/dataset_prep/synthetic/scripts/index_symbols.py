#!/usr/bin/env python3
"""
Symbol Library Indexer

This script indexes all handwritten symbol InkML files from the symbols directory,
grouping them by their LaTeX label to enable random selection during synthesis.

Usage:
    python3 index_symbols.py --symbols-dir symbols --output symbol_index.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm


class SymbolIndexer:
    """Indexes symbol InkML files by their LaTeX labels."""

    def __init__(self, symbols_dir: Path):
        """
        Initialize the symbol indexer.

        Args:
            symbols_dir: Path to directory containing symbol InkML files
        """
        self.symbols_dir = Path(symbols_dir)
        self.symbol_index: Dict[str, List[str]] = defaultdict(list)
        self.symbol_stats: Dict[str, Dict] = {}

    def parse_inkml_label(self, file_path: Path) -> Optional[str]:
        """
        Parse an InkML file and extract the symbol label.

        Args:
            file_path: Path to the InkML file

        Returns:
            str: The symbol label (LaTeX format) or None if not found
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Find annotation with type="label"
            for annotation in root.findall('.//{http://www.w3.org/2003/InkML}annotation'):
                if annotation.get('type') == 'label':
                    return annotation.text

            return None
        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")
            return None

    def get_stroke_bounds(self, file_path: Path) -> Optional[Tuple[float, float, float, float]]:
        """
        Get bounding box of strokes in an InkML file.

        Args:
            file_path: Path to the InkML file

        Returns:
            Tuple of (xmin, ymin, xmax, ymax) or None if error
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            all_x = []
            all_y = []

            # Extract all points from all traces
            for trace in root.findall('.//{http://www.w3.org/2003/InkML}trace'):
                points = trace.text.split(',')
                for point in points:
                    coords = point.strip().split()
                    if len(coords) >= 2:
                        all_x.append(float(coords[0]))
                        all_y.append(float(coords[1]))

            if not all_x or not all_y:
                return None

            return (min(all_x), min(all_y), max(all_x), max(all_y))

        except Exception as e:
            print(f"Error getting bounds for {file_path.name}: {e}")
            return None

    def index_symbols(self) -> None:
        """
        Index all InkML files in the symbols directory.
        Groups files by their label and computes statistics.
        """
        # Find all InkML files
        inkml_files = list(self.symbols_dir.glob("*.inkml"))

        if not inkml_files:
            print(f"Warning: No InkML files found in {self.symbols_dir}")
            return

        print(f"Indexing {len(inkml_files)} InkML files...")

        # Process each file
        failed_count = 0
        for file_path in tqdm(inkml_files, desc="Processing symbols"):
            label = self.parse_inkml_label(file_path)

            if label is None:
                failed_count += 1
                continue

            # Add to index (store just the filename, not full path)
            self.symbol_index[label].append(file_path.name)

        print(f"\nIndexing complete:")
        print(f"  Successfully indexed: {len(inkml_files) - failed_count}")
        print(f"  Failed to parse: {failed_count}")
        print(f"  Unique labels: {len(self.symbol_index)}")

    def compute_statistics(self) -> Dict:
        """
        Compute statistics about the symbol library.

        Returns:
            Dictionary containing statistics
        """
        total_files = sum(len(files) for files in self.symbol_index.values())
        variants_per_label = [len(files) for files in self.symbol_index.values()]

        stats = {
            "total_files": total_files,
            "unique_labels": len(self.symbol_index),
            "min_variants": min(variants_per_label) if variants_per_label else 0,
            "max_variants": max(variants_per_label) if variants_per_label else 0,
            "avg_variants": sum(variants_per_label) / len(variants_per_label) if variants_per_label else 0,
            "labels_with_single_variant": sum(1 for v in variants_per_label if v == 1),
            "labels_with_10plus_variants": sum(1 for v in variants_per_label if v >= 10),
        }

        return stats

    def get_label_distribution(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Get the most common labels by number of variants.

        Args:
            top_n: Number of top labels to return

        Returns:
            List of (label, count) tuples, sorted by count descending
        """
        label_counts = [(label, len(files)) for label, files in self.symbol_index.items()]
        label_counts.sort(key=lambda x: x[1], reverse=True)
        return label_counts[:top_n]

    def save_index(self, output_path: Path) -> None:
        """
        Save the symbol index to a JSON file.

        Args:
            output_path: Path where to save the index
        """
        output_data = {
            "symbols": dict(self.symbol_index),  # Convert defaultdict to regular dict
            "statistics": self.compute_statistics(),
            "metadata": {
                "symbols_directory": str(self.symbols_dir),
                "total_inkml_files": sum(len(files) for files in self.symbol_index.values()),
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nIndex saved to: {output_path}")

    def print_statistics(self) -> None:
        """Print detailed statistics about the symbol library."""
        stats = self.compute_statistics()

        print("\n" + "="*60)
        print("SYMBOL LIBRARY STATISTICS")
        print("="*60)
        print(f"Total InkML files:              {stats['total_files']}")
        print(f"Unique LaTeX labels:            {stats['unique_labels']}")
        print(f"Average variants per label:     {stats['avg_variants']:.1f}")
        print(f"Min variants for any label:     {stats['min_variants']}")
        print(f"Max variants for any label:     {stats['max_variants']}")
        print(f"Labels with 1 variant only:     {stats['labels_with_single_variant']}")
        print(f"Labels with 10+ variants:       {stats['labels_with_10plus_variants']}")

        print("\n" + "="*60)
        print("TOP 20 LABELS BY VARIANT COUNT")
        print("="*60)

        distribution = self.get_label_distribution(20)
        for i, (label, count) in enumerate(distribution, 1):
            # Escape label for display
            display_label = label if len(label) <= 20 else label[:17] + "..."
            print(f"{i:2d}. {display_label:20s} : {count:3d} variants")

        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index symbol InkML files by their LaTeX labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--symbols-dir',
        type=str,
        default='symbols',
        help="Directory containing symbol InkML files"
    )

    parser.add_argument(
        '--output',
        type=str,
        default='symbol_index.json',
        help="Output JSON file path"
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate paths
    symbols_dir = Path(args.symbols_dir)
    output_path = Path(args.output)

    if not symbols_dir.exists():
        print(f"Error: Symbols directory '{symbols_dir}' does not exist")
        return 1

    if not symbols_dir.is_dir():
        print(f"Error: '{symbols_dir}' is not a directory")
        return 1

    # Create indexer and process files
    indexer = SymbolIndexer(symbols_dir)
    indexer.index_symbols()

    # Print statistics
    indexer.print_statistics()

    # Save index
    indexer.save_index(output_path)

    print("\n✓ Symbol indexing complete!")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
