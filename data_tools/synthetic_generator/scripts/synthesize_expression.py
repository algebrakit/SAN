#!/usr/bin/env python3
"""
Handwritten Expression Synthesizer

This script generates synthetic handwritten mathematical expression InkML files
by placing real handwritten symbols from the InkML library into bounding boxes
from LaTeX/DVI compilation.

Usage:
    python3 synthesize_expression.py --index symbol_index.json --boxes boxes.jsonl --output output/
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from stroke_transformer import StrokeSet, combine_stroke_sets, fit_to_bbox
from models import BoundingBox


class ExpressionSynthesizer:
    """Synthesizes handwritten expressions from bounding boxes and symbol library."""

    def __init__(
        self,
        symbol_index_path: Path,
        symbols_dir: Path,
        random_seed: Optional[int] = None,
        debug: bool = False
    ):
        """
        Initialize the synthesizer.

        Args:
            symbol_index_path: Path to symbol_index.json
            symbols_dir: Directory containing symbol InkML files
            random_seed: Random seed for reproducible variant selection
            debug: Enable debug logging for bounding boxes
        """
        self.symbol_index_path = Path(symbol_index_path)
        self.symbols_dir = Path(symbols_dir)
        self.debug = debug
        self.symbol_index: Dict[str, List[str]] = {}
        self.stats = {
            "expressions_processed": 0,
            "symbols_placed": 0,
            "symbols_skipped": 0,
            "missing_symbols": set(),
        }

        if random_seed is not None:
            random.seed(random_seed)

        self.load_symbol_index()

    def load_symbol_index(self) -> None:
        """Load the symbol index from JSON file."""
        try:
            with open(self.symbol_index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.symbol_index = data['symbols']

            print(f"Loaded symbol index with {len(self.symbol_index)} unique symbols")

        except Exception as e:
            print(f"Error loading symbol index: {e}")
            raise

    def get_random_variant(self, symbol: str) -> Optional[Path]:
        """
        Get a random InkML file for the given symbol.

        Args:
            symbol: LaTeX symbol string

        Returns:
            Path to InkML file, or None if symbol not in library
        """
        if symbol not in self.symbol_index:
            return None

        variants = self.symbol_index[symbol]
        if not variants:
            return None

        # Choose random variant
        variant_filename = random.choice(variants)
        return self.symbols_dir / variant_filename

    def synthesize_from_bboxes(
        self,
        bboxes_data: Dict,
        preserve_aspect: bool = True,
        skip_missing: bool = True
    ) -> Tuple[Optional[StrokeSet], Dict]:
        """
        Synthesize a handwritten expression from bounding boxes.

        Args:
            bboxes_data: Dictionary with 'label', 'normalizedLabel', and 'bboxes' list
            preserve_aspect: If True, preserve aspect ratio of symbols
            skip_missing: If True, skip missing symbols; if False, raise error

        Returns:
            Tuple of (StrokeSet or None, metadata dict)
        """
        label = bboxes_data.get('label', '')
        normalized_label = bboxes_data.get('normalizedLabel', label)
        bboxes = bboxes_data.get('bboxes', [])

        if not bboxes:
            return None, {"error": "No bounding boxes provided"}

        stroke_sets = []
        trace_id_counter = 0  # Track stroke IDs for debug output
        metadata = {
            "label": label,
            "normalized_label": normalized_label,
            "symbols_placed": 0,
            "symbols_skipped": 0,
            "missing_symbols": [],
        }

        for bbox_dict in bboxes:
            # Convert dict to BoundingBox object for type safety
            bbox = BoundingBox.from_dict(bbox_dict)

            # Skip empty tokens or whitespace
            if not bbox.token or bbox.token.isspace():
                continue

            # Get random variant for this symbol
            inkml_path = self.get_random_variant(bbox.token)

            if inkml_path is None or not inkml_path.exists():
                # Symbol not in library
                metadata["missing_symbols"].append(bbox.token)
                metadata["symbols_skipped"] += 1
                self.stats["symbols_skipped"] += 1
                self.stats["missing_symbols"].add(bbox.token)

                if not skip_missing:
                    raise ValueError(f"Symbol '{bbox.token}' not found in library")

                continue

            try:
                # Load symbol strokes
                symbol_strokes = StrokeSet.from_inkml(inkml_path)

                # Fit to bounding box with padding to prevent overlap
                # Use 5% padding for content symbols, 0% for structural symbols
                structural_symbols = ['\\sqrt', '\\frac', '\\overline', '\\underline']
                padding = 0.0 if bbox.token in structural_symbols else 0.05

                # Disable aspect preservation for structural symbols (frac bars, overlines, underlines)
                # to allow them to stretch to correct dimensions
                preserve_symbol_aspect = preserve_aspect and bbox.token not in ['\\frac', '\\overline', '\\underline']

                fitted_strokes = fit_to_bbox(
                    symbol_strokes,
                    bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max,
                    preserve_aspect=preserve_symbol_aspect,
                    padding=padding,
                    glyph_height=bbox.glyph_height,
                    glyph_depth=bbox.glyph_depth
                )

                # Debug logging for bounding box info
                num_strokes = len(fitted_strokes.strokes)
                if self.debug:
                    stroke_ids = list(range(trace_id_counter, trace_id_counter + num_strokes))
                    print(f"\n[BOX] Token: '{bbox.token}'")
                    print(f"  Position: ({bbox.x_min:.2f}, {bbox.y_min:.2f}) → ({bbox.x_max:.2f}, {bbox.y_max:.2f})")
                    print(f"  Dimensions: {bbox.width:.2f} × {bbox.height:.2f}")
                    if bbox.glyph_height is not None:
                        print(f"  Glyph: height={bbox.glyph_height:.2f}, depth={bbox.glyph_depth:.2f}")
                    print(f"  Symbol file: {inkml_path.name}")
                    print(f"  Strokes: {stroke_ids}")
                    # Detailed stroke info
                    for i, stroke in enumerate(fitted_strokes.strokes):
                        stroke_id = trace_id_counter + i
                        s_x_min, s_y_min = float(stroke.x.min()), float(stroke.y.min())
                        s_x_max, s_y_max = float(stroke.x.max()), float(stroke.y.max())
                        print(f"    Stroke {stroke_id}: ({s_x_min:.2f}, {s_y_min:.2f}) → ({s_x_max:.2f}, {s_y_max:.2f})")
                trace_id_counter += num_strokes

                stroke_sets.append(fitted_strokes)
                metadata["symbols_placed"] += 1
                self.stats["symbols_placed"] += 1

            except Exception as e:
                print(f"Warning: Error processing symbol '{bbox.token}' from {inkml_path.name}: {e}")
                metadata["symbols_skipped"] += 1
                self.stats["symbols_skipped"] += 1
                continue

        if not stroke_sets:
            return None, metadata

        # Combine all strokes
        combined = combine_stroke_sets(stroke_sets)
        return combined, metadata

    def save_inkml(
        self,
        stroke_set: StrokeSet,
        output_path: Path,
        label: str,
        normalized_label: Optional[str] = None
    ) -> None:
        """
        Save stroke set as InkML file.

        Args:
            stroke_set: StrokeSet to save
            output_path: Path where to save InkML file
            label: Original LaTeX label
            normalized_label: Normalized LaTeX label (optional)
        """
        # Create root element
        root = ET.Element('ink')
        root.set('xmlns', 'http://www.w3.org/2003/InkML')

        # Add annotations
        label_annotation = ET.SubElement(root, 'annotation')
        label_annotation.set('type', 'label')
        label_annotation.text = label

        if normalized_label:
            norm_annotation = ET.SubElement(root, 'annotation')
            norm_annotation.set('type', 'normalizedLabel')
            norm_annotation.text = normalized_label

        # Add annotation for synthetic data
        synthetic_annotation = ET.SubElement(root, 'annotation')
        synthetic_annotation.set('type', 'inkCreationMethod')
        synthetic_annotation.text = 'synthetic'

        # Add trace format
        trace_format = ET.SubElement(root, 'traceFormat')
        for channel_name, channel_type in [('X', 'decimal'), ('Y', 'decimal'), ('T', 'decimal')]:
            channel = ET.SubElement(trace_format, 'channel')
            channel.set('name', channel_name)
            channel.set('type', channel_type)
            if channel_name == 'T':
                channel.set('units', 'ms')

        # Add traces
        traces = stroke_set.to_inkml_traces()
        for trace in traces:
            root.append(trace)

        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')  # Pretty print (Python 3.9+)

        with open(output_path, 'wb') as f:
            tree.write(f, encoding='utf-8', xml_declaration=True)

    def process_boxes_file(
        self,
        boxes_file: Path,
        output_dir: Path,
        preserve_aspect: bool = True,
        skip_missing: bool = True,
        max_expressions: Optional[int] = None
    ) -> None:
        """
        Process a JSONL file of bounding boxes and generate InkML files.

        Args:
            boxes_file: Path to boxes.jsonl file
            output_dir: Directory to save generated InkML files
            preserve_aspect: If True, preserve aspect ratio of symbols
            skip_missing: If True, skip missing symbols
            max_expressions: Maximum number of expressions to process (None = all)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(boxes_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        except Exception as e:
            print(f"Error reading boxes file: {e}")
            return

        print(f"Processing {len(lines)} expressions from {boxes_file.name}...")

        processed = 0
        for i, line in enumerate(lines):
            if max_expressions and processed >= max_expressions:
                break

            if not line.strip():
                continue

            try:
                bboxes_data = json.loads(line)

                # Synthesize expression
                stroke_set, metadata = self.synthesize_from_bboxes(
                    bboxes_data,
                    preserve_aspect=preserve_aspect,
                    skip_missing=skip_missing
                )

                if stroke_set is None:
                    print(f"  [{i+1}] Skipped (no strokes generated)")
                    continue

                # Generate output filename
                label = metadata['label'].replace(' ', '_').replace('/', '_')
                if len(label) > 50:
                    label = label[:50]
                output_filename = f"expr_{i+1:04d}_{label}.inkml"
                output_path = output_dir / output_filename

                # Save InkML
                self.save_inkml(
                    stroke_set,
                    output_path,
                    metadata['label'],
                    metadata.get('normalized_label')
                )

                processed += 1
                self.stats["expressions_processed"] += 1

                print(f"  [{i+1}] Generated: {output_filename}")
                print(f"       Placed {metadata['symbols_placed']} symbols, "
                      f"skipped {metadata['symbols_skipped']}")

            except Exception as e:
                print(f"  [{i+1}] Error: {e}")
                continue

        print(f"\nProcessed {processed} expressions")

    def print_stats(self) -> None:
        """Print synthesis statistics."""
        print("\n" + "="*60)
        print("SYNTHESIS STATISTICS")
        print("="*60)
        print(f"Expressions processed:    {self.stats['expressions_processed']}")
        print(f"Symbols placed:           {self.stats['symbols_placed']}")
        print(f"Symbols skipped:          {self.stats['symbols_skipped']}")
        print(f"Unique missing symbols:   {len(self.stats['missing_symbols'])}")

        if self.stats['missing_symbols']:
            print("\nMissing symbols:")
            for symbol in sorted(self.stats['missing_symbols']):
                display = symbol if len(symbol) <= 30 else symbol[:27] + "..."
                print(f"  - {display}")

        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synthesize handwritten expressions from bounding boxes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--index',
        type=str,
        default='output/symbol_index.json',
        help="Path to symbol index JSON file"
    )

    parser.add_argument(
        '--symbols-dir',
        type=str,
        default='symbols',
        help="Directory containing symbol InkML files"
    )

    parser.add_argument(
        '--boxes',
        type=str,
        required=True,
        help="Path to bounding boxes JSONL file"
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/synthesized',
        help="Output directory for generated InkML files"
    )

    parser.add_argument(
        '--max-expressions',
        type=int,
        help="Maximum number of expressions to process"
    )

    parser.add_argument(
        '--no-preserve-aspect',
        action='store_true',
        help="Don't preserve aspect ratio (stretch symbols to fill boxes)"
    )

    parser.add_argument(
        '--fail-on-missing',
        action='store_true',
        help="Fail if a symbol is missing (default: skip missing symbols)"
    )

    parser.add_argument(
        '--seed',
        type=int,
        help="Random seed for reproducible variant selection"
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug logging for bounding boxes"
    )

    args = parser.parse_args()

    # Validate paths
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Error: Symbol index '{index_path}' not found")
        return 1

    symbols_dir = Path(args.symbols_dir)
    if not symbols_dir.exists():
        print(f"Error: Symbols directory '{symbols_dir}' not found")
        return 1

    boxes_path = Path(args.boxes)
    if not boxes_path.exists():
        print(f"Error: Boxes file '{boxes_path}' not found")
        return 1

    output_dir = Path(args.output)

    # Initialize synthesizer
    synthesizer = ExpressionSynthesizer(
        index_path,
        symbols_dir,
        random_seed=args.seed,
        debug=args.debug
    )

    # Process boxes file
    synthesizer.process_boxes_file(
        boxes_path,
        output_dir,
        preserve_aspect=not args.no_preserve_aspect,
        skip_missing=not args.fail_on_missing,
        max_expressions=args.max_expressions
    )

    # Print statistics
    synthesizer.print_stats()

    print("\n✓ Expression synthesis complete!")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
