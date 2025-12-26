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
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
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
            Tuple of (should_skip, mapped_token, bbox_dict):
            - should_skip: True if token should be skipped without consuming a glyph
            - mapped_token: Mapped token string (e.g., '(' for pmatrix), or None to skip
            - bbox_dict: Precomputed bbox dictionary if special handling is needed, None otherwise
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
                    bbox = {
                        "token": mapped_token,
                        "xMin": float(glyph.x),
                        "yMin": float(content_bottom_negated - padding),  # Bottom edge with padding
                        "xMax": float(glyph.x + glyph.width),
                        "yMax": float(content_top_negated + padding)      # Top edge with padding
                    }
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
    ) -> List[Dict]:
        """
        Map DVI glyphs and boxes to LaTeX tokens and create bounding boxes.

        This uses a simple sequential mapping strategy for glyphs,
        and classification for boxes.

        Args:
            tokens: List of LaTeX tokens
            glyphs: List of Glyph objects from DVI
            boxes: List of DVIBox objects from DVI

        Returns:
            List of bounding box dictionaries, sorted by x-position
        """
        bboxes = []

        # Expand multi-character tokens (like \cos -> c, o, s) to match DVI glyphs
        tokens = self.expand_latex_tokens(tokens)

        # Find sqrt overline boxes for proper sqrt sizing
        sqrt_boxes: Dict[int, DVIBox] = {}  # Maps glyph index to sqrt overline box
        if '\\sqrt' in tokens:
            for box in boxes:
                # Sqrt overlines are at high y-position and wide
                if box.y > 5.0 and box.width > 2.0:
                    # Find the sqrt glyph this box corresponds to
                    for i, glyph in enumerate(glyphs):
                        # Sqrt radical glyph starts near the box start
                        if abs(glyph.x - (box.x - glyph.width)) < 2.0 and abs(glyph.y - box.y) < 1.0:
                            sqrt_boxes[i] = box
                            break

        # Simple strategy: match tokens to glyphs sequentially
        # Skip tokens that don't produce glyphs (like ^, _, etc.)
        glyph_idx = 0
        token_idx = 0

        while token_idx < len(tokens):
            token = tokens[token_idx]
            token_idx += 1
            if glyph_idx >= len(glyphs):
                break

            # Handle environment tokens (cases, pmatrix, bmatrix, etc.)
            should_skip, mapped_token, precomputed_bbox = self.handle_environment_token(
                token, tokens, glyphs, glyph_idx
            )

            if should_skip:
                # Token produces no glyph, skip it
                continue

            if precomputed_bbox:
                # Environment token with special bbox handling (e.g., matrix delimiters)
                bboxes.append(precomputed_bbox)
                glyph_idx += 1
                continue

            if mapped_token:
                # Environment token mapped to different token (e.g., \begin{cases} -> \{)
                token = mapped_token

            # Skip structural tokens and spacing commands that don't render as glyphs
            # Note: \frac is skipped here (only renders as box), but \sqrt produces a glyph
            # Font-changing commands (\mathrm, \mathbf, etc.) don't produce glyphs, only style content
            if token in ['^', '_', '{', '}', '\\ ', '\\,', '\\;', '\\:', '\\!', '\\quad', '\\qquad',
                        '\\mathrm', '\\mathbf', '\\mathit', '\\mathcal', '\\mathsf', '\\mathtt',
                        '\\mathfrak', '\\mathnormal'] or \
               token.startswith('\\frac'):
                continue

            # Get current glyph
            glyph = glyphs[glyph_idx]

            # Special handling for cases environment brace - extend to cover all content
            if token == '\\{' and '\\begin{cases}' in tokens:
                # Find the vertical extent of all content after the brace
                # In DVI coords: smaller y is higher, larger y is lower
                # Initialize to extreme values so content determines extent (not brace position)
                min_y = float('inf')  # Will be updated to smallest y (highest position)
                max_y = float('-inf')  # Will be updated to largest y (lowest position)

                prev_glyph_right = glyph.x + glyph.width  # Track previous glyph's right edge

                for next_glyph in glyphs[glyph_idx + 1:]:
                    gap = next_glyph.x - prev_glyph_right

                    # Stop if we hit another elevated glyph (another brace)
                    # Braces are at highly elevated y-positions (> 15.0 in DVI coords)
                    # Fraction numerators are at y ≈ 10-11, so use 15.0 threshold
                    if next_glyph.y > 15.0:
                        break

                    # Stop if there's a large horizontal gap (indicates next cases environment)
                    if gap > 10.0:
                        break

                    min_y = min(min_y, next_glyph.y)
                    max_y = max(max_y, next_glyph.y)
                    # Use actual glyph height (extends downward in DVI)
                    max_y = max(max_y, next_glyph.y + next_glyph.total_height)
                    prev_glyph_right = next_glyph.x + next_glyph.width

                # Create bbox that spans all content lines
                # Note: DVI y goes down, we negate for standard math convention
                bbox = {
                    "token": token,
                    "xMin": float(glyph.x),
                    "yMin": float(-max_y),  # Lowest point in standard coords
                    "xMax": float(glyph.x + glyph.width),
                    "yMax": float(-min_y + 2.0)  # Highest point in standard coords + padding
                }
                bboxes.append(bbox)
                glyph_idx += 1  # Skip the brace glyph
                continue  # Skip common bbox append
            # Special handling for sqrt radical
            elif token == '\\sqrt':
                # Check if this sqrt has an index: next token is '['
                has_index = token_idx < len(tokens) and tokens[token_idx] == '['

                # Track the rightmost x-position of index glyphs (where sqrt bbox should start)
                index_x_max = None

                # If there's an index, we need to process index glyphs first
                # Index glyphs appear at elevated y-position before the radical glyph
                if has_index:
                    # Skip opening '[' token
                    if token_idx < len(tokens) and tokens[token_idx] == '[':
                        token_idx += 1

                    # Process index glyphs until we find the radical
                    while glyph_idx < len(glyphs) and glyph_idx not in sqrt_boxes:
                        idx_glyph = glyphs[glyph_idx]

                        # Track the rightmost edge of index glyphs
                        # The sqrt bbox should start after the index to avoid overlap
                        if index_x_max is None:
                            index_x_max = idx_glyph.x + idx_glyph.width
                        else:
                            index_x_max = max(index_x_max, idx_glyph.x + idx_glyph.width)

                        if token_idx < len(tokens):
                            index_token = tokens[token_idx]
                            token_idx += 1

                            # Create bbox for index glyph
                            bbox = {
                                "token": index_token,
                                "xMin": float(idx_glyph.x),
                                "yMin": float(-idx_glyph.y - idx_glyph.total_height),
                                "xMax": float(idx_glyph.x + idx_glyph.width),
                                "yMax": float(-idx_glyph.y)
                            }
                            bboxes.append(bbox)

                        glyph_idx += 1

                    # Skip closing ']' token
                    if token_idx < len(tokens) and tokens[token_idx] == ']':
                        token_idx += 1

                # Now process the radical if we're at one
                if glyph_idx in sqrt_boxes:
                    radical_glyph = glyphs[glyph_idx]
                    # Use the sqrt overline box to determine proper extent
                    sqrt_box = sqrt_boxes[glyph_idx]

                    # Find vertical extent by scanning all glyphs under the overline
                    # In DVI coords: y increases downward, so max_y is the bottom
                    min_y = radical_glyph.y
                    max_y = radical_glyph.y
                    radicand_glyph_count = 0  # Count glyphs in radicand to skip them later

                    # The overline box width tells us the horizontal extent of the radicand
                    # box_x is where the overline starts, box_width is its length
                    radicand_x_max = sqrt_box.x + sqrt_box.width

                    prev_glyph_right = sqrt_box.x  # Track previous glyph's right edge

                    for next_glyph in glyphs[glyph_idx + 1:]:
                        # Stop if we've gone past the overline box or hit another sqrt
                        if next_glyph.x >= radicand_x_max or next_glyph.y > 5.0:
                            break

                        # Detect gaps that indicate non-continuous content
                        # - Backwards jump (gap < -1.0) indicates fraction denominator or other content
                        # - Large forward gap (> 5.0) indicates next expression part
                        # Stop processing entirely - vertical extent should only cover continuous content
                        gap = next_glyph.x - prev_glyph_right
                        if gap < -1.0 or gap > 5.0:
                            break

                        # Count continuous radicand glyphs and update extents
                        radicand_glyph_count += 1
                        prev_glyph_right = next_glyph.x + next_glyph.width

                        # Update vertical extent for continuous radicand glyphs only
                        min_y = min(min_y, next_glyph.y)
                        max_y = max(max_y, next_glyph.y + next_glyph.total_height)

                    # Create bbox that covers from top of radicand to bottom of content
                    # Use overline box width for horizontal extent (not actual glyph positions)
                    # This ensures the sqrt bbox covers the full radicand including fractions
                    # In standard coords (negated DVI): yMin=bottom, yMax=top
                    # Note: If index exists, start after the index to avoid overlap
                    # Otherwise start at radical glyph position
                    bbox = {
                        "token": token,
                        "xMin": float(index_x_max if index_x_max is not None else radical_glyph.x),
                        "yMin": float(-max_y - 2.0),  # Bottom of content (most negative)
                        "xMax": float(radicand_x_max),  # Use overline box width
                        "yMax": float(-min_y + 2.0)  # Top of radicand (least negative)
                    }
                    bboxes.append(bbox)
                    glyph_idx += 1  # Skip the radical glyph

                    # Create separate bboxes for radicand content
                    # The radicand glyphs need their own bboxes so they get synthesized
                    for _ in range(radicand_glyph_count):
                        if glyph_idx >= len(glyphs):
                            break

                        # Get radicand glyph info
                        rad_content_glyph = glyphs[glyph_idx]

                        # Get corresponding token (should be the radicand content)
                        radicand_token = tokens[token_idx] if token_idx < len(tokens) else rad_content_glyph.char

                        # Create bbox for this radicand glyph
                        radicand_bbox = {
                            "token": radicand_token,
                            "xMin": float(rad_content_glyph.x),
                            "yMin": float(-rad_content_glyph.y - rad_content_glyph.total_height),
                            "xMax": float(rad_content_glyph.x + rad_content_glyph.width),
                            "yMax": float(-rad_content_glyph.y)
                        }
                        bboxes.append(radicand_bbox)

                        glyph_idx += 1
                        token_idx += 1

                    # Skip any spacing tokens like '\ ' that don't correspond to glyphs
                    while token_idx < len(tokens) and tokens[token_idx] in ['\\ ', '\\,', '\\;', '\\:', '\\!', ' ']:
                        token_idx += 1

                    continue  # Skip the common bbox append at line 510
            # Special handling for \lognl custom macro
            # \lognl[n] expands to: superscript 'n' + 'l' + 'o' + 'g'
            elif token == '\\lognl':
                # \lognl produces N+3 glyphs: N superscript characters, then 'l', 'o', 'g'
                # We need to dynamically count how many superscript glyphs there are

                # Count superscript tokens by looking ahead for '[' ... ']' sequence
                superscript_token_count = 0
                if token_idx < len(tokens) and tokens[token_idx] == '[':
                    # Scan ahead to count tokens between brackets
                    scan_idx = token_idx + 1
                    while scan_idx < len(tokens) and tokens[scan_idx] != ']':
                        superscript_token_count += 1
                        scan_idx += 1

                # Process N superscript glyphs (all at elevated y-position)
                for i in range(superscript_token_count):
                    if glyph_idx >= len(glyphs):
                        break
                    sup_glyph = glyphs[glyph_idx]
                    bbox = {
                        "token": sup_glyph.char,  # Use the actual character
                        "xMin": float(sup_glyph.x),
                        "yMin": float(-sup_glyph.y - sup_glyph.total_height),
                        "xMax": float(sup_glyph.x + sup_glyph.width),
                        "yMax": float(-sup_glyph.y)
                    }
                    bboxes.append(bbox)
                    glyph_idx += 1

                # Next 3 glyphs: 'l', 'o', 'g'
                # Collect all three glyphs first to normalize their heights
                log_glyphs_list: List[Glyph] = []
                for i in range(3):
                    if glyph_idx + i >= len(glyphs):
                        break
                    log_glyphs_list.append(glyphs[glyph_idx + i])

                # Calculate max height among the 'log' letters for normalization
                max_height = 0.0
                for log_glyph in log_glyphs_list:
                    max_height = max(max_height, log_glyph.total_height)

                # Create bboxes with normalized heights
                for log_glyph in log_glyphs_list:
                    # Use max_height for all letters to make them uniform
                    bbox = {
                        "token": log_glyph.char,  # 'l', 'o', or 'g'
                        "xMin": float(log_glyph.x),
                        "yMin": float(-log_glyph.y - max_height),  # Normalized height
                        "xMax": float(log_glyph.x + log_glyph.width),
                        "yMax": float(-log_glyph.y)
                    }
                    bboxes.append(bbox)
                    glyph_idx += 1

                # Skip the bracket tokens: '[', ..., ']'  (e.g., [10] in \lognl[10]{x})
                # \lognl takes an optional argument in square brackets
                if token_idx < len(tokens) and tokens[token_idx] == '[':
                    token_idx += 1  # Skip '['
                    # Skip all superscript tokens
                    for i in range(superscript_token_count):
                        if token_idx < len(tokens):
                            token_idx += 1
                    # Skip ']'
                    if token_idx < len(tokens) and tokens[token_idx] == ']':
                        token_idx += 1
                # Continue to next token without appending another bbox
                continue
            else:
                # Standard bbox calculation
                # Use actual font metrics for height

                # Detect extended delimiters created by \left and \right commands
                # These are positioned at vertical center of content, not baseline
                # Characteristics: large depth, elevated y-position, small height
                is_extended_delimiter = (glyph.depth > 10.0 and glyph.y > 5.0 and glyph.height < 1.0)

                # Detect inline large operators (sum, prod, int, bigcup, etc.)
                # These have similar characteristics but need different positioning
                # Characteristics: large depth (> 9.0), elevated y (5.0 < y < 10.0), minimal height (< 1.0)
                is_large_operator = (glyph.depth > 9.0 and 5.0 < glyph.y < 10.0 and glyph.height < 1.0)

                if is_extended_delimiter:
                    # Extended delimiters need special positioning
                    # These delimiters should span from baseline upward to cover the content
                    # Force bottom edge (yMin) to be at baseline (0 in negated coords)
                    bbox = {
                        "token": token,
                        "xMin": float(glyph.x),
                        "yMin": float(-glyph.total_height),  # Bottom at baseline, extends up
                        "xMax": float(glyph.x + glyph.width),
                        "yMax": float(0.0)                    # Top aligned with baseline
                    }
                elif is_large_operator:
                    # Large operators in inline mode need baseline alignment
                    # Position so the operator top aligns with baseline, extending downward
                    # This ensures the operator aligns with following content on the same baseline
                    bbox = {
                        "token": token,
                        "xMin": float(glyph.x),
                        "yMin": float(-glyph.total_height),  # Bottom edge extends below baseline
                        "xMax": float(glyph.x + glyph.width),
                        "yMax": float(0.0)             # Top aligned with baseline
                    }

                    # Handle subscript/superscript for large operators
                    # In DVI: operator, superscript glyphs, subscript glyphs
                    # In tokens: operator, subscript tokens, superscript tokens
                    # Need to reorder glyph consumption to match token order

                    # Append the operator bbox first
                    bboxes.append(bbox)
                    glyph_idx += 1

                    # Look ahead to detect superscript and subscript glyphs
                    superscript_glyph_count = 0
                    subscript_glyph_count = 0

                    # Scan glyphs after the operator to count superscript (y > 2.0) and subscript (y < -1.0)
                    # Only consider glyphs that are horizontally near the operator (within width + 15 units)
                    operator_x_max = glyph.x + glyph.width + 15.0  # Generous range for sub/superscripts

                    for scan_glyph in glyphs[glyph_idx:]:
                        # Stop if we've moved too far horizontally (beyond operator's immediate vicinity)
                        if scan_glyph.x > operator_x_max:
                            break

                        # Superscript glyphs have elevated positive y-position (above baseline in DVI)
                        if scan_glyph.y > 2.0 and scan_glyph.y < 8.0:  # Superscript range
                            superscript_glyph_count += 1
                        # Subscript glyphs have negative y-position (below baseline in DVI)
                        elif scan_glyph.y < -1.0:
                            subscript_glyph_count += 1
                        else:
                            # No more sub/superscript glyphs
                            break

                    # Now process in token order: subscripts first, then superscripts
                    # Process subscript glyphs (which appear AFTER superscript glyphs in DVI)
                    subscript_start_glyph = glyph_idx + superscript_glyph_count
                    for i in range(subscript_glyph_count):
                        if subscript_start_glyph + i >= len(glyphs):
                            break
                        if token_idx >= len(tokens):
                            break

                        sub_glyph = glyphs[subscript_start_glyph + i]
                        sub_token = tokens[token_idx]
                        token_idx += 1

                        # Create bbox for subscript glyph
                        sub_bbox = {
                            "token": sub_token,
                            "xMin": float(sub_glyph.x),
                            "yMin": float(-sub_glyph.y - sub_glyph.total_height),
                            "xMax": float(sub_glyph.x + sub_glyph.width),
                            "yMax": float(-sub_glyph.y)
                        }
                        bboxes.append(sub_bbox)

                    # Process superscript glyphs (which appear BEFORE subscript glyphs in DVI)
                    for i in range(superscript_glyph_count):
                        if glyph_idx + i >= len(glyphs):
                            break
                        if token_idx >= len(tokens):
                            break

                        sup_glyph = glyphs[glyph_idx + i]
                        sup_token = tokens[token_idx]
                        token_idx += 1

                        # Create bbox for superscript glyph
                        sup_bbox = {
                            "token": sup_token,
                            "xMin": float(sup_glyph.x),
                            "yMin": float(-sup_glyph.y - sup_glyph.total_height),
                            "xMax": float(sup_glyph.x + sup_glyph.width),
                            "yMax": float(-sup_glyph.y)
                        }
                        bboxes.append(sup_bbox)

                    # Advance glyph_idx past all sub/superscript glyphs
                    glyph_idx += superscript_glyph_count + subscript_glyph_count

                    # Skip the common bbox append since we already appended
                    continue

                else:
                    # Standard bbox calculation for normal glyphs
                    # Note: DVI y-coordinates go down, we negate for standard math convention
                    bbox = {
                        "token": token,
                        "xMin": float(glyph.x),
                        "yMin": float(-glyph.y - glyph.total_height),
                        "xMax": float(glyph.x + glyph.width),
                        "yMax": float(-glyph.y)
                    }

            bboxes.append(bbox)
            glyph_idx += 1

        # Process boxes (fraction bars, overlines, etc.)
        for box in boxes:
            box_token = self.classify_box(box, glyphs, tokens)
            if box_token:
                bbox = {
                    "token": box_token,
                    "xMin": float(box.x),
                    "yMin": float(-box.y - box.height),
                    "xMax": float(box.x + box.width),
                    "yMax": float(-box.y)
                }
                bboxes.append(bbox)

        # Sort by x-position to maintain left-to-right reading order
        bboxes.sort(key=lambda b: b["xMin"])

        return bboxes

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

            # Create result
            result = {
                "label": expression,
                "normalizedLabel": expression,  # Could use latex_normalizer here
                "bboxes": bboxes
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
