#!/usr/bin/env python3
"""
LaTeX to Bounding Box Generator

This script generates bounding box JSONL files from LaTeX expressions by:
1. Compiling LaTeX to DVI
2. Parsing DVI to extract glyph positions
3. Mapping glyphs back to LaTeX tokens
4. Outputting in boxes.jsonl format

Usage:
    python3 generate_boxes.py --input latex.txt --output boxes.jsonl
"""

import json
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from utils import tokenize_latex
from models import Glyph, DVIBox, BoundingBox
from glyph_handlers import (
    GlyphMappingContext,
    CasesHandler,
    SqrtHandler,
    LognlHandler,
    LargeOperatorHandler,
    ExtendedDelimiterHandler,
    create_standard_bbox
)
from latex_compiler import create_latex_document, compile_to_dvi
from dvi_parser import parse_dvi
from token_expander import expand_latex_tokens
from box_classifier import classify_box


class LaTeXToDVIBoxes:
    """Converts LaTeX expressions to bounding box data via DVI compilation."""

    def __init__(self, dpi: int = 72, keep_temp: bool = False):
        """
        Initialize the converter.

        Args:
            dpi: DPI for DVI rendering (affects coordinate scale)
            keep_temp: If True, keep temporary files for debugging
        """
        self.dpi = dpi
        self.keep_temp = keep_temp
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }

    def handle_environment_token(
        self,
        token: str,
        tokens: List[str],
        glyphs: List[Glyph],
        glyph_idx: int
    ) -> Tuple[bool, Optional[str], Optional[BoundingBox]]:
        r"""
        Handle \begin{...} and \end{...} environment tokens.

        This function centralizes all logic for environment constructs like
        cases, pmatrix, bmatrix, etc.

        Args:
            token: The current token being processed
            tokens: Full list of LaTeX tokens
            glyphs: List of Glyph objects from DVI
            glyph_idx: Current index in glyphs list

        Returns:
            Tuple of (should_skip, mapped_token, bbox):
            - should_skip: True if token should be skipped without consuming a glyph
            - mapped_token: Mapped token string (e.g., '(' for pmatrix), or None to skip
            - bbox: Precomputed BoundingBox if special handling is needed, None otherwise
        """
        # Matrix environments: pmatrix, bmatrix, Bmatrix, matrix
        # These produce large delimiter glyphs from cmex10 font
        matrix_delimiters = {
            '\\begin{pmatrix}': '(',
            '\\end{pmatrix}': ')',
            '\\begin{bmatrix}': '[',
            '\\end{bmatrix}': ']',
            '\\begin{Bmatrix}': '\\{',
            '\\end{Bmatrix}': '\\}',
        }

        if token in matrix_delimiters:
            # Matrix delimiters produce large extended glyphs from cmex10 font
            # Check if current glyph is a matrix delimiter
            if glyph_idx < len(glyphs):
                glyph = glyphs[glyph_idx]

                # Matrix delimiters have:
                # - Large depth (> 20.0) to span multiple rows
                # - Elevated y-position (vertically centered)
                # - Minimal height
                is_matrix_delimiter = (glyph.depth > 15.0 and glyph.y > 5.0)

                if is_matrix_delimiter:
                    # Matrix delimiters need to be centered on the actual content
                    # Determine if this is opening or closing delimiter based on token
                    is_opening = token.startswith('\\begin{')

                    # Scan glyphs to find content vertical extent in DVI coords
                    content_top_dvi = float('inf')  # Smallest y = highest visual position
                    content_bottom_dvi = float('-inf')  # Largest y = lowest visual position

                    if is_opening:
                        # Opening delimiter: scan forward for content
                        for next_glyph in glyphs[glyph_idx + 1:]:
                            # Stop if we hit another matrix delimiter (closing one)
                            if next_glyph.depth > 15.0 and next_glyph.y > 5.0:
                                break

                            # Note: Extended delimiters from \left( and \right) don't appear in DVI glyphs
                            # They're created during bbox mapping, so no need to filter by depth here

                            # Track vertical extent of content
                            # In DVI: y is reference point, extends up by height and down by depth
                            glyph_top = next_glyph.y - next_glyph.height
                            glyph_bottom = next_glyph.y + next_glyph.depth
                            content_top_dvi = min(content_top_dvi, glyph_top)
                            content_bottom_dvi = max(content_bottom_dvi, glyph_bottom)
                    else:
                        # Closing delimiter: scan backward for content
                        for prev_glyph in reversed(glyphs[:glyph_idx]):
                            # Stop if we hit another matrix delimiter (opening one)
                            if prev_glyph.depth > 15.0 and prev_glyph.y > 5.0:
                                break

                            # Note: Extended delimiters from \left( and \right) don't appear in DVI glyphs
                            # They're created during bbox mapping, so no need to filter by depth here

                            # Track vertical extent of content
                            glyph_top = prev_glyph.y - prev_glyph.height
                            glyph_bottom = prev_glyph.y + prev_glyph.depth
                            content_top_dvi = min(content_top_dvi, glyph_top)
                            content_bottom_dvi = max(content_bottom_dvi, glyph_bottom)

                    # Convert DVI coords to negated coords (our output system)
                    # In DVI: smaller y = higher visual position
                    # In output: larger y = higher visual position (negated)
                    content_top_negated = -content_top_dvi     # Highest visual position
                    content_bottom_negated = -content_bottom_dvi  # Lowest visual position

                    # Calculate content height and add 10% padding for visual breathing room
                    content_height = content_top_negated - content_bottom_negated
                    padding = content_height * 0.10

                    # Position delimiter to span content with padding
                    # This aligns delimiter extent with actual content, not an abstract center point
                    mapped_token = matrix_delimiters[token]
                    bbox = BoundingBox(
                        token=mapped_token,
                        x_min=float(glyph.x),
                        y_min=float(content_bottom_negated - padding),  # Bottom edge with padding
                        x_max=float(glyph.x + glyph.width),
                        y_max=float(content_top_negated + padding)      # Top edge with padding
                    )
                    return (False, mapped_token, bbox)

            # If glyph doesn't match matrix delimiter characteristics, skip token
            return (True, None, None)

        # Handle \begin{matrix} and \end{matrix} (no delimiters)
        if token in ['\\begin{matrix}', '\\end{matrix}']:
            return (True, None, None)

        # Handle cases environment
        if token == '\\begin{cases}':
            # Map to \{ token for symbol library matching
            return (False, '\\{', None)  # Will use standard bbox calculation with special extent

        # Handle \end{cases} and \\ (line breaks) - no glyphs
        if token == '\\end{cases}' or token == '\\\\':
            return (True, None, None)

        # Not an environment token
        return (False, None, None)

    def map_glyphs_to_tokens(
        self,
        tokens: List[str],
        glyphs: List[Glyph],
        boxes: List[DVIBox]
    ) -> List[BoundingBox]:
        """
        Map DVI glyphs and boxes to LaTeX tokens and create bounding boxes.

        This uses a simple sequential mapping strategy for glyphs,
        and classification for boxes. Special constructs (sqrt, cases, lognl,
        large operators) are handled by dedicated handler classes.

        Args:
            tokens: List of LaTeX tokens
            glyphs: List of Glyph objects from DVI
            boxes: List of DVIBox objects from DVI

        Returns:
            List of BoundingBox objects, sorted by x-position
        """
        # Expand multi-character tokens (like \cos -> c, o, s) to match DVI glyphs
        expanded_tokens = expand_latex_tokens(tokens)

        # Find sqrt overline boxes for proper sqrt sizing
        sqrt_boxes: Dict[int, DVIBox] = {}
        if '\\sqrt' in expanded_tokens:
            for box in boxes:
                if box.y > 5.0 and box.width > 2.0:
                    for i, glyph in enumerate(glyphs):
                        if abs(glyph.x - (box.x - glyph.width)) < 2.0 and abs(glyph.y - box.y) < 1.0:
                            sqrt_boxes[i] = box
                            break

        # Create mapping context
        ctx = GlyphMappingContext(
            tokens=expanded_tokens,
            glyphs=glyphs,
            sqrt_boxes=sqrt_boxes
        )

        # Process tokens
        while ctx.has_tokens():
            token = ctx.tokens[ctx.token_idx]
            ctx.advance_token()

            if not ctx.has_glyphs():
                break

            # Handle environment tokens (cases, pmatrix, bmatrix, etc.)
            should_skip, mapped_token, precomputed_bbox = self.handle_environment_token(
                token, ctx.tokens, ctx.glyphs, ctx.glyph_idx
            )

            if should_skip:
                continue

            if precomputed_bbox:
                ctx.append_bbox(precomputed_bbox)
                ctx.advance_glyph()
                continue

            if mapped_token:
                token = mapped_token

            # Skip structural tokens and spacing commands
            if self._is_structural_token(token):
                continue

            glyph = ctx.current_glyph

            # Try specialized handlers in order
            if CasesHandler.can_handle(token, ctx.tokens):
                CasesHandler.handle(ctx, token)
                continue

            if SqrtHandler.can_handle(token):
                SqrtHandler.handle(ctx, token)
                continue

            if LognlHandler.can_handle(token):
                LognlHandler.handle(ctx, token)
                continue

            if LargeOperatorHandler.can_handle(token, glyph):
                LargeOperatorHandler.handle(ctx, token)
                continue

            if ExtendedDelimiterHandler.can_handle(glyph):
                bbox = ExtendedDelimiterHandler.create_bbox(token, glyph)
                ctx.append_bbox(bbox)
                ctx.advance_glyph()
                continue

            # Standard bbox for normal glyphs
            bbox = create_standard_bbox(token, glyph)
            ctx.append_bbox(bbox)
            ctx.advance_glyph()

        # Process boxes (fraction bars, overlines, etc.)
        for box in boxes:
            box_token = classify_box(box, glyphs, ctx.tokens)
            if box_token:
                bbox = BoundingBox(
                    token=box_token,
                    x_min=float(box.x),
                    y_min=float(-box.y - box.height),
                    x_max=float(box.x + box.width),
                    y_max=float(-box.y)
                )
                ctx.append_bbox(bbox)

        # Sort by x-position to maintain left-to-right reading order
        ctx.bboxes.sort(key=lambda b: b.x_min)

        return ctx.bboxes

    def _is_structural_token(self, token: str) -> bool:
        """Check if token is a structural/spacing token that doesn't render as a glyph."""
        structural_tokens = [
            '^', '_', '{', '}', '\\ ', '\\,', '\\;', '\\:', '\\!', '\\quad', '\\qquad',
            '\\mathrm', '\\mathbf', '\\mathit', '\\mathcal', '\\mathsf', '\\mathtt',
            '\\mathfrak', '\\mathnormal'
        ]
        return token in structural_tokens or token.startswith('\\frac')

    def process_expression(
        self,
        expression: str,
        work_dir: Optional[Path] = None
    ) -> Optional[Dict]:
        """
        Process a single LaTeX expression to generate bounding boxes.

        Args:
            expression: LaTeX math expression
            work_dir: Optional working directory (creates temp dir if None)

        Returns:
            Dictionary with label, normalizedLabel, and bboxes, or None if failed
        """
        # Create working directory
        if work_dir is None:
            temp_dir = tempfile.mkdtemp()
            work_dir = Path(temp_dir)
            cleanup = True
        else:
            cleanup = False

        try:
            # Tokenize
            tokens = tokenize_latex(expression)

            # Create LaTeX document
            latex_doc = create_latex_document(expression)

            # Compile to DVI
            dvi_path = compile_to_dvi(latex_doc, work_dir)
            if dvi_path is None:
                return None

            # Parse DVI
            glyphs, boxes = parse_dvi(dvi_path, dpi=self.dpi)
            if not glyphs:
                return None

            # Map glyphs and boxes to tokens
            bboxes = self.map_glyphs_to_tokens(tokens, glyphs, boxes)

            if not bboxes:
                return None

            # Create result - convert BoundingBox objects to dicts for JSON serialization
            result = {
                "label": expression,
                "normalizedLabel": expression,  # Could use latex_normalizer here
                "bboxes": [bbox.to_dict() for bbox in bboxes]
            }

            return result

        finally:
            # Cleanup temp directory
            if cleanup and not self.keep_temp:
                shutil.rmtree(work_dir, ignore_errors=True)

    def process_file(self, input_file: Path, output_file: Path) -> None:
        """
        Process a file of LaTeX expressions.

        Args:
            input_file: Path to input file (one expression per line)
            output_file: Path to output JSONL file
        """
        # Read input
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                expressions = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading input file: {e}")
            return

        print(f"Processing {len(expressions)} expressions from {input_file.name}...")

        # Process each expression
        results = []
        for i, expression in enumerate(expressions, 1):
            self.stats["total"] += 1

            print(f"[{i}/{len(expressions)}] {expression}")

            result = self.process_expression(expression)

            if result:
                results.append(result)
                self.stats["successful"] += 1
                print(f"  ✓ Generated {len(result['bboxes'])} bounding boxes")
            else:
                self.stats["failed"] += 1
                self.stats["errors"].append((i, expression))
                print(f"  ✗ Failed")

        # Write output
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

            print(f"\n✓ Wrote {len(results)} expressions to {output_file}")

        except Exception as e:
            print(f"Error writing output file: {e}")

    def print_stats(self) -> None:
        """Print processing statistics."""
        print("\n" + "="*60)
        print("BOUNDING BOX GENERATION STATISTICS")
        print("="*60)
        print(f"Total expressions:     {self.stats['total']}")
        print(f"Successful:            {self.stats['successful']}")
        print(f"Failed:                {self.stats['failed']}")

        if self.stats['failed'] > 0:
            success_rate = 100 * self.stats['successful'] / self.stats['total']
            print(f"Success rate:          {success_rate:.1f}%")

        if self.stats['errors']:
            print(f"\nFailed expressions:")
            for idx, expr in self.stats['errors'][:10]:
                print(f"  [{idx}] {expr}")
            if len(self.stats['errors']) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more")

        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate bounding boxes from LaTeX expressions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        default='latex.txt',
        help="Input file with LaTeX expressions (one per line)"
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/boxes_generated.jsonl',
        help="Output JSONL file path"
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=72,
        help="DPI for DVI rendering (affects coordinate scale)"
    )

    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help="Keep temporary LaTeX/DVI files for debugging"
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create converter
    converter = LaTeXToDVIBoxes(dpi=args.dpi, keep_temp=args.keep_temp)

    # Process file
    converter.process_file(input_path, output_path)

    # Print statistics
    converter.print_stats()

    return 0 if converter.stats['failed'] == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
