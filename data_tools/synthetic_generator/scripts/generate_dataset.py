#!/usr/bin/env python3
"""
End-to-End Dataset Generator

This script processes LaTeX expressions and generates a complete dataset:
1. Generates bounding boxes from LaTeX expressions
2. Synthesizes handwritten InkML from bounding boxes
3. Saves InkML files
4. Creates labels.txt file mapping InkML files to LaTeX expressions

Usage:
    python3 generate_dataset.py --input latex.txt --output-dir output
"""

import sys
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional

# Add stroke processing to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'stroke_processing'))

# Import our modules
from generate_boxes import LaTeXToDVIBoxes
from synthesize_expression import ExpressionSynthesizer
from stroke_transformer import StrokeSet


class DatasetGenerator:
    """End-to-end dataset generator from LaTeX to InkML and labels."""

    def __init__(
        self,
        symbol_index_path: Path,
        symbols_dir: Path,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the dataset generator.

        Args:
            symbol_index_path: Path to symbol_index.json
            symbols_dir: Directory containing symbol InkML files
            random_seed: Random seed for reproducible synthesis
        """
        self.symbol_index_path = Path(symbol_index_path)
        self.symbols_dir = Path(symbols_dir)

        # Initialize components
        self.bbox_generator = LaTeXToDVIBoxes(dpi=72, keep_temp=False)
        self.synthesizer = ExpressionSynthesizer(
            symbol_index_path,
            symbols_dir,
            random_seed=random_seed
        )

        self.stats = {
            "total_expressions": 0,
            "successful": 0,
            "failed_bbox": 0,
            "failed_synthesis": 0,
            "failed_saving": 0,
        }

    def read_latex_file(self, latex_file: Path) -> List[str]:
        """
        Read LaTeX expressions from file.

        Args:
            latex_file: Path to file with LaTeX expressions (one per line)

        Returns:
            List of LaTeX expressions
        """
        try:
            with open(latex_file, 'r', encoding='utf-8') as f:
                expressions = [line.strip() for line in f if line.strip() and line.strip()[0]!='#']
            return expressions
        except Exception as e:
            print(f"Error reading LaTeX file: {e}")
            return []

    def generate_bboxes_batch(
        self,
        expressions: List[str],
        work_dir: Path
    ) -> List[Optional[Dict]]:
        """
        Generate bounding boxes for a batch of expressions.

        Args:
            expressions: List of LaTeX expressions
            work_dir: Working directory for temporary files

        Returns:
            List of bbox dictionaries (None for failed expressions)
        """
        results = []

        for i, expression in enumerate(expressions, 1):
            print(f"[{i}/{len(expressions)}] Generating bboxes for: {expression}")

            bbox_data = self.bbox_generator.process_expression(expression, work_dir)

            if bbox_data:
                results.append(bbox_data)
                # print(f"  ✓ Generated {len(bbox_data['bboxes'])} bounding boxes")
            else:
                results.append(None)
                self.stats["failed_bbox"] += 1
                print(f"  ✗ Failed to generate bounding boxes")

        return results

    def synthesize_batch(
        self,
        bbox_data_list: List[Optional[Dict]]
    ) -> List[Optional[StrokeSet]]:
        """
        Synthesize handwritten InkML for a batch of bounding boxes.

        Args:
            bbox_data_list: List of bbox dictionaries

        Returns:
            List of StrokeSets (None for failed expressions)
        """
        results = []

        for i, bbox_data in enumerate(bbox_data_list, 1):
            if bbox_data is None:
                results.append(None)
                continue

            expression = bbox_data['label']
            print(f"[{i}/{len(bbox_data_list)}] Synthesizing: {expression}")

            stroke_set, metadata = self.synthesizer.synthesize_from_bboxes(
                bbox_data,
                preserve_aspect=True,
                skip_missing=True
            )

            if stroke_set and metadata['symbols_skipped']==0:
                results.append(stroke_set)
            else:
                results.append(None)
                self.stats["failed_synthesis"] += 1
                print(f"  ✗ Failed to synthesize (unknown symbols {metadata['missing_symbols']})")

        return results

    def save_inkml_batch(
        self,
        stroke_sets: List[Optional[StrokeSet]],
        expressions: List[str],
        output_dir: Path
    ) -> List[Optional[Path]]:
        """
        Save StrokeSets as InkML files.

        Args:
            stroke_sets: List of StrokeSets
            expressions: List of LaTeX expressions (for filenames and labels)
            output_dir: Output directory for InkML files

        Returns:
            List of InkML file paths (None for failed saves)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for i, (stroke_set, expression) in enumerate(zip(stroke_sets, expressions), 1):
            if stroke_set is None:
                results.append(None)
                continue

            # Generate filename
            safe_expr = expression.replace('\\', '').replace('{', '').replace('}', '')
            safe_expr = safe_expr.replace('/', '_').replace(' ', '_').replace('.', '_')
            if len(safe_expr) > 40:
                safe_expr = safe_expr[:40]

            filename = f"expr_{i:04d}_{safe_expr}.inkml"
            output_path = output_dir / filename

            print(f"[{i}/{len(stroke_sets)}] Saving: {filename}")

            try:
                # Save using synthesizer's save_inkml method
                self.synthesizer.save_inkml(
                    stroke_set,
                    output_path,
                    label=expression,
                    normalized_label=expression  # Could use normalizer here
                )

                results.append(output_path)
                print(f"  ✓ Saved to {filename}")

            except Exception as e:
                results.append(None)
                self.stats["failed_saving"] += 1
                print(f"  ✗ Saving error: {e}")

        return results

    def create_labels_file(
        self,
        inkml_paths: List[Optional[Path]],
        expressions: List[str],
        output_path: Path
    ) -> None:
        """
        Create labels.txt file mapping InkML files to LaTeX expressions.

        Args:
            inkml_paths: List of InkML file paths
            expressions: List of LaTeX expressions
            output_path: Path to labels.txt
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for inkml_path, expression in zip(inkml_paths, expressions):
                    if inkml_path is not None:
                        # Write: filename.inkml\texpression
                        f.write(f"{inkml_path.name}\t{expression}\n")

            # Count successful entries
            successful_count = sum(1 for p in inkml_paths if p is not None)
            print(f"\n✓ Created labels.txt with {successful_count} entries")

        except Exception as e:
            print(f"Error creating labels file: {e}")

    def generate_dataset(
        self,
        latex_file: Path,
        output_dir: Path
    ) -> None:
        """
        Generate complete dataset from LaTeX expressions.

        Args:
            latex_file: Path to LaTeX expressions file
            output_dir: Output directory for InkML files and labels
        """
        # Read LaTeX expressions
        print("="*70)
        print("STEP 1: Reading LaTeX expressions")
        print("="*70)
        expressions = self.read_latex_file(latex_file)
        self.stats["total_expressions"] = len(expressions)

        if not expressions:
            print("No expressions to process")
            return

        print(f"Found {len(expressions)} expressions")

        # Create working directory for bounding boxes
        work_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Generate bounding boxes
            print("\n" + "="*70)
            print("STEP 2: Generating bounding boxes from LaTeX")
            print("="*70)
            bbox_data_list = self.generate_bboxes_batch(expressions, work_dir)

            # Step 2: Synthesize handwritten InkML
            print("\n" + "="*70)
            print("STEP 3: Synthesizing handwritten expressions")
            print("="*70)
            stroke_sets = self.synthesize_batch(bbox_data_list)

            # Step 3: Save InkML files
            print("\n" + "="*70)
            print("STEP 4: Saving InkML files")
            print("="*70)
            inkml_dir = output_dir / "inkml"
            inkml_paths = self.save_inkml_batch(stroke_sets, expressions, inkml_dir)

            # Step 4: Create labels file
            print("\n" + "="*70)
            print("STEP 5: Creating labels.txt")
            print("="*70)
            labels_path = output_dir / "labels.txt"
            self.create_labels_file(inkml_paths, expressions, labels_path)

            # Calculate success count
            self.stats["successful"] = sum(1 for p in inkml_paths if p is not None)

        finally:
            # Cleanup working directory
            shutil.rmtree(work_dir, ignore_errors=True)

    def print_stats(self) -> None:
        """Print generation statistics."""
        print("\n" + "="*70)
        print("DATASET GENERATION STATISTICS")
        print("="*70)
        print(f"Total expressions:          {self.stats['total_expressions']}")
        print(f"Successfully generated:     {self.stats['successful']}")
        print(f"Failed (bbox generation):   {self.stats['failed_bbox']}")
        print(f"Failed (synthesis):         {self.stats['failed_synthesis']}")
        print(f"Failed (saving):            {self.stats['failed_saving']}")

        if self.stats['total_expressions'] > 0:
            success_rate = 100 * self.stats['successful'] / self.stats['total_expressions']
            print(f"Success rate:               {success_rate:.1f}%")

        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate complete dataset from LaTeX expressions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        default='latex.txt',
        help="Input file with LaTeX expressions (one per line)"
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help="Output directory for InkML files and labels"
    )

    parser.add_argument(
        '--symbol-index',
        type=str,
        default='output/symbol_index.json',
        help="Path to symbol index JSON"
    )

    parser.add_argument(
        '--symbols-dir',
        type=str,
        default='symbols',
        help="Directory containing symbol InkML files"
    )

    parser.add_argument(
        '--seed',
        type=int,
        help="Random seed for reproducible generation"
    )

    args = parser.parse_args()

    # Validate paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        return 1

    symbol_index_path = Path(args.symbol_index)
    if not symbol_index_path.exists():
        print(f"Error: Symbol index '{symbol_index_path}' not found")
        print("Run: python3 scripts/index_symbols.py first")
        return 1

    symbols_dir = Path(args.symbols_dir)
    if not symbols_dir.exists():
        print(f"Error: Symbols directory '{symbols_dir}' not found")
        return 1

    output_dir = Path(args.output_dir)

    # Create generator
    generator = DatasetGenerator(
        symbol_index_path,
        symbols_dir,
        random_seed=args.seed
    )

    # Generate dataset
    generator.generate_dataset(input_path, output_dir)

    # Print statistics
    generator.print_stats()

    print(f"\n✓ Dataset generation complete!")
    print(f"   InkML files: {output_dir}/inkml/")
    print(f"   Labels: {output_dir}/labels.txt")

    return 0 if generator.stats['successful'] > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
