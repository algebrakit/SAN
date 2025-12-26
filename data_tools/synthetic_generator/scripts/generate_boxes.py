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
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re

from utils import tokenize_latex as _tokenize_latex
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

try:
    import matplotlib.dviread as dviread
except ImportError:
    print("Error: matplotlib is required for DVI parsing")
    print("Install with: pip install matplotlib")
    exit(1)


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

    def tokenize_latex(self, latex: str) -> List[str]:
        """
        Tokenize a LaTeX string into individual symbols/commands.

        Args:
            latex: LaTeX expression string

        Returns:
            List of tokens
        """
        return _tokenize_latex(latex)

    def create_latex_document(self, expression: str) -> str:
        """
        Create a minimal LaTeX document containing the expression.

        Args:
            expression: LaTeX math expression

        Returns:
            Complete LaTeX document as string
        """
        return f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{textcomp}}
\\pagestyle{{empty}}
\\newcommand\\lognl[1][]{{\\mathop{{ {{}}^{{#1}}\\mathrm{{log}} }} }}
\\newcommand\\degree{{^\\circ}}
\\begin{{document}}
${expression}$
\\end{{document}}
"""

    def compile_to_dvi(self, latex_content: str, work_dir: Path) -> Optional[Path]:
        """
        Compile LaTeX content to DVI file.

        Args:
            latex_content: Complete LaTeX document
            work_dir: Working directory for compilation

        Returns:
            Path to DVI file, or None if compilation failed
        """
        # Write LaTeX file
        tex_file = work_dir / "document.tex"
        tex_file.write_text(latex_content, encoding='utf-8')

        # Find LaTeX binary
        latex_cmd = '/Library/TeX/texbin/latex'
        if not Path(latex_cmd).exists():
            # Fallback to PATH
            latex_cmd = 'latex'

        # Compile with latex
        try:
            result = subprocess.run(
                [latex_cmd, '-interaction=nonstopmode', 'document.tex'],
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )

            dvi_file = work_dir / "document.dvi"
            log_file = work_dir / "document.log"

            # Check for LaTeX compilation errors in log file
            if log_file.exists():
                log_content = log_file.read_text(encoding='utf-8', errors='ignore')

                # Check for critical errors
                error_patterns = [
                    '! Undefined control sequence',
                    '! LaTeX Error:',
                    '! Missing',
                    '! Emergency stop'
                ]

                for pattern in error_patterns:
                    if pattern in log_content:
                        # Extract context around the error for better diagnostics
                        lines = log_content.split('\n')
                        for i, line in enumerate(lines):
                            if pattern in line:
                                print(f"  LaTeX compilation error detected:")
                                print(f"  {pattern}")
                                if '! Undefined control sequence' in pattern and i + 1 < len(lines):
                                    # Try to extract the undefined command
                                    next_line = lines[i + 1]
                                    if next_line.strip():
                                        print(f"  {next_line.strip()}")
                                break
                        return None

                # Check for warnings about missing characters
                warning_patterns = [
                    'Missing character:',
                    'Some font shapes were not available',
                ]

                for pattern in warning_patterns:
                    if pattern in log_content:
                        print(f"  LaTeX compilation warning detected:")
                        lines = log_content.split('\n')
                        warning_lines = [line.strip() for line in lines if pattern in line]
                        # Show first few warnings for context
                        for warning in warning_lines[:3]:
                            print(f"  {warning}")
                        if len(warning_lines) > 3:
                            print(f"  ... and {len(warning_lines) - 3} more warnings")
                        return None

            if dvi_file.exists():
                return dvi_file
            else:
                return None

        except subprocess.TimeoutExpired:
            print(f"  Compilation timeout")
            return None
        except Exception as e:
            print(f"  Compilation error: {e}")
            return None

    def parse_dvi(self, dvi_path: Path) -> Tuple[List[Glyph], List[DVIBox]]:
        """
        Parse DVI file to extract glyph positions and boxes.

        Args:
            dvi_path: Path to DVI file

        Returns:
            Tuple of (glyphs, boxes):
            - glyphs: List of Glyph objects
            - boxes: List of DVIBox objects
        """
        glyphs: List[Glyph] = []
        boxes: List[DVIBox] = []

        # Ensure TeX binaries are in PATH so matplotlib can find TFM files via kpsewhich
        import os
        old_path = os.environ.get('PATH', '')
        if '/Library/TeX/texbin' not in old_path:
            os.environ['PATH'] = '/Library/TeX/texbin:' + old_path

        try:
            with dviread.Dvi(str(dvi_path), dpi=self.dpi) as dvi:
                for page in dvi:
                    # Extract glyphs
                    for x, y, font, glyph_code, width in page.text:
                        # Convert glyph code to character
                        # Note: This is a simplified mapping
                        try:
                            char = chr(glyph_code) if glyph_code < 128 else f"\\glyph{{{glyph_code}}}"
                        except:
                            char = f"\\glyph{{{glyph_code}}}"

                        # Extract actual height and depth from font metrics
                        # NOTE: page.text already contains scaled x,y,width values
                        # but height/depth from font._height_depth_of() are in DVI units
                        # We need to scale them to match the page coordinate system
                        height_depth_dvi = font._height_depth_of(glyph_code)

                        # Calculate scaling factor (same as matplotlib dviread uses)
                        scaling_factor = self.dpi / (72.27 * 2**16)

                        # Scale height and depth to page coordinates
                        height = height_depth_dvi[0] * scaling_factor
                        depth = height_depth_dvi[1] * scaling_factor

                        glyphs.append(Glyph(
                            x=x,
                            y=y,
                            char=char,
                            width=width,
                            height=height,
                            depth=depth
                        ))

                    # Extract boxes (fraction bars, overlines, etc.)
                    for box in page.boxes:
                        boxes.append(DVIBox(
                            x=box.x,
                            y=box.y,
                            width=box.width,
                            height=box.height
                        ))

        except Exception as e:
            print(f"  DVI parsing error: {e}")
            return [], []

        return glyphs, boxes

    def classify_box(
        self,
        box: DVIBox,
        glyphs: List[Glyph],
        tokens: List[str]
    ) -> Optional[str]:
        """
        Classify a DVI box to determine what LaTeX symbol it represents.

        Args:
            box: DVIBox object with position and dimensions
            glyphs: List of all Glyph objects in the expression
            tokens: List of LaTeX tokens (for context)

        Returns:
            Token string ('\\frac', '\\overline', '\\underline') or None to skip
        """
        # Check for fraction bar:
        # - Horizontal box (width >> height)
        # - Has glyphs above and below at similar x-position
        if box.width > box.height * 2:  # Horizontal line
            # Check if this is a sqrt overline (part of radical symbol)
            # Heuristic: if we have \\sqrt in tokens, check if this box is at high y-position
            # Sqrt overlines are typically at y > 5
            if '\\sqrt' in tokens and box.y > 5.0:
                # This is likely the sqrt overline, which is already part of the radical glyph
                return None

            # Look for glyphs above and below this box
            glyphs_above = [g for g in glyphs
                          if abs(g.x - box.x) < box.width and g.y > box.y + box.height]
            glyphs_below = [g for g in glyphs
                          if abs(g.x - box.x) < box.width and g.y < box.y]

            if glyphs_above and glyphs_below:
                # Likely a fraction bar with numerator and denominator
                return '\\frac'

            # Check for overline (line above glyphs)
            if glyphs_below and not glyphs_above and box.y > 4.0:
                return '\\overline'

            # Check for underline (line below glyphs)
            if glyphs_above and not glyphs_below and box.y < -1.0:
                return '\\underline'

        return None

    def expand_latex_tokens(self, tokens: List[str]) -> List[str]:
        """
        Expand multi-character LaTeX commands to match actual DVI glyphs.

        Examples:
            \\cos -> ['c', 'o', 's']
            \\left( -> ['(']
            \\sin -> ['s', 'i', 'n']
            x -> ['x']

        Args:
            tokens: List of LaTeX tokens

        Returns:
            Expanded list of tokens matching DVI glyphs
        """
        # Function names that render as individual letters
        # NOTE: Only include built-in LaTeX function names here, NOT custom macros
        # Custom macros like \lognl should NOT be expanded to letters
        function_names = {
            'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
            'sinh', 'cosh', 'tanh', 'coth',
            'arcsin', 'arccos', 'arctan',
            'log', 'ln', 'exp',
            'lim', 'sup', 'inf',
            'max', 'min',
            'det', 'dim', 'deg',
            'gcd', 'arg'
        }

        # Sizing commands that don't produce glyphs
        sizing_commands = {'left', 'right', 'big', 'Big', 'bigg', 'Bigg', 'bigl', 'bigr', 'Bigl', 'Bigr'}

        # Custom macro expansions (defined in LaTeX preamble)
        # These macros expand to other symbols that exist in the symbol library
        macro_expansions = {
            'degree': '\\circ',  # \degree is defined as ^\circ in preamble
        }

        expanded = []

        for token in tokens:
            # Handle LaTeX commands
            if token.startswith('\\'):
                cmd_name = token[1:]  # Remove backslash

                # Skip sizing commands
                if cmd_name in sizing_commands:
                    continue

                # Expand custom macros
                if cmd_name in macro_expansions:
                    expanded.append(macro_expansions[cmd_name])
                # Expand function names to individual letters
                elif cmd_name in function_names:
                    expanded.extend(list(cmd_name))
                else:
                    # Keep as-is (Greek letters, special symbols, etc.)
                    expanded.append(token)
            else:
                # Regular characters, keep as-is
                expanded.append(token)

        return expanded

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
        expanded_tokens = self.expand_latex_tokens(tokens)

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
            box_token = self.classify_box(box, glyphs, ctx.tokens)
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
            tokens = self.tokenize_latex(expression)

            # Create LaTeX document
            latex_doc = self.create_latex_document(expression)

            # Compile to DVI
            dvi_path = self.compile_to_dvi(latex_doc, work_dir)
            if dvi_path is None:
                return None

            # Parse DVI
            glyphs, boxes = self.parse_dvi(dvi_path)
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
