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
        # Pattern matches LaTeX commands and single characters
        command_pattern = re.compile(
            r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)'
        )

        tokens = []
        s = latex

        while s:
            if s[0] == '\\':
                # Match LaTeX command
                match = command_pattern.match(s)
                if match:
                    tokens.append(match.group(0))
                    s = s[len(match.group(0)):]
                else:
                    tokens.append(s[0])
                    s = s[1:]
            elif s[0] in '{}^_':
                # Structural characters (not rendered as glyphs)
                s = s[1:]
            elif s[0].isspace():
                # Skip whitespace
                s = s[1:]
            else:
                # Regular character
                tokens.append(s[0])
                s = s[1:]

        return tokens

    def identify_letter_groups(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Identify consecutive letter sequences that should have normalized heights.
        Examples: 'cos', 'sin', 'km', 'minute', 'of'

        Args:
            tokens: List of tokens after expansion

        Returns:
            List of (start_index, end_index) tuples for letter groups
        """
        groups = []
        i = 0
        while i < len(tokens):
            # Check if current token is a single letter
            if len(tokens[i]) == 1 and tokens[i].isalpha():
                start = i
                # Count consecutive letters
                while i < len(tokens) and len(tokens[i]) == 1 and tokens[i].isalpha():
                    i += 1
                # Only group if 2+ consecutive letters
                if i - start >= 2:
                    groups.append((start, i))
            else:
                i += 1
        return groups

    def create_latex_document(self, expression: str) -> str:
        """
        Create a minimal LaTeX document containing the expression.

        Args:
            expression: LaTeX math expression

        Returns:
            Complete LaTeX document as string
        """
        return f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\pagestyle{{empty}}
\\newcommand\\lognl[1][]{{\\mathop{{ {{}}^{{#1}}\\mathrm{{log}} }} }}
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

    def parse_dvi(self, dvi_path: Path) -> Tuple[List[Tuple[float, float, str, float]], List[Tuple[float, float, float, float]]]:
        """
        Parse DVI file to extract glyph positions and boxes.

        Args:
            dvi_path: Path to DVI file

        Returns:
            Tuple of (glyphs, boxes):
            - glyphs: List of (x, y, char, width) tuples
            - boxes: List of (x, y, width, height) tuples
        """
        glyphs = []
        boxes = []

        # Ensure TeX binaries are in PATH so matplotlib can find TFM files via kpsewhich
        import os
        old_path = os.environ.get('PATH', '')
        if '/Library/TeX/texbin' not in old_path:
            os.environ['PATH'] = '/Library/TeX/texbin:' + old_path

        try:
            with dviread.Dvi(str(dvi_path), dpi=self.dpi) as dvi:
                for page in dvi:
                    # Extract glyphs
                    for x, y, font, glyph, width in page.text:
                        # Convert glyph code to character
                        # Note: This is a simplified mapping
                        try:
                            char = chr(glyph) if glyph < 128 else f"\\glyph{{{glyph}}}"
                        except:
                            char = f"\\glyph{{{glyph}}}"

                        glyphs.append((x, y, char, width))

                    # Extract boxes (fraction bars, overlines, etc.)
                    for box in page.boxes:
                        boxes.append((box.x, box.y, box.width, box.height))

        except Exception as e:
            print(f"  DVI parsing error: {e}")
            return [], []

        return glyphs, boxes

    def classify_box(
        self,
        box: Tuple[float, float, float, float],
        glyphs: List[Tuple[float, float, str, float]],
        tokens: List[str]
    ) -> Optional[str]:
        """
        Classify a DVI box to determine what LaTeX symbol it represents.

        Args:
            box: Tuple of (x, y, width, height)
            glyphs: List of all glyphs in the expression
            tokens: List of LaTeX tokens (for context)

        Returns:
            Token string ('\\frac', '\\overline', '\\underline') or None to skip
        """
        x, y, width, height = box

        # Check for fraction bar:
        # - Horizontal box (width >> height)
        # - Has glyphs above and below at similar x-position
        if width > height * 2:  # Horizontal line
            # Check if this is a sqrt overline (part of radical symbol)
            # Heuristic: if we have \\sqrt in tokens, check if this box is at high y-position
            # Sqrt overlines are typically at y > 5
            if '\\sqrt' in tokens and y > 5.0:
                # This is likely the sqrt overline, which is already part of the radical glyph
                return None

            # Look for glyphs above and below this box
            glyphs_above = [g for g in glyphs
                          if abs(g[0] - x) < width and g[1] > y + height]
            glyphs_below = [g for g in glyphs
                          if abs(g[0] - x) < width and g[1] < y]

            if glyphs_above and glyphs_below:
                # Likely a fraction bar with numerator and denominator
                return '\\frac'

            # Check for overline (line above glyphs)
            if glyphs_below and not glyphs_above and y > 4.0:
                return '\\overline'

            # Check for underline (line below glyphs)
            if glyphs_above and not glyphs_below and y < -1.0:
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

        expanded = []

        for token in tokens:
            # Handle LaTeX commands
            if token.startswith('\\'):
                cmd_name = token[1:]  # Remove backslash

                # Skip sizing commands
                if cmd_name in sizing_commands:
                    continue

                # Expand function names to individual letters
                if cmd_name in function_names:
                    expanded.extend(list(cmd_name))
                else:
                    # Keep as-is (Greek letters, special symbols, etc.)
                    expanded.append(token)
            else:
                # Regular characters, keep as-is
                expanded.append(token)

        return expanded

    def map_glyphs_to_tokens(
        self,
        tokens: List[str],
        glyphs: List[Tuple[float, float, str, float]],
        boxes: List[Tuple[float, float, float, float]]
    ) -> List[Dict]:
        """
        Map DVI glyphs and boxes to LaTeX tokens and create bounding boxes.

        This uses a simple sequential mapping strategy for glyphs,
        and classification for boxes.

        Args:
            tokens: List of LaTeX tokens
            glyphs: List of (x, y, char, width) from DVI
            boxes: List of (x, y, width, height) from DVI

        Returns:
            List of bounding box dictionaries, sorted by x-position
        """
        bboxes = []

        # Expand multi-character tokens (like \cos -> c, o, s) to match DVI glyphs
        tokens = self.expand_latex_tokens(tokens)

        # Identify consecutive letter groups for height normalization
        letter_groups = self.identify_letter_groups(tokens)

        # Create lookup for fast checking during token processing
        token_group_map = {}
        for start, end in letter_groups:
            for idx in range(start, end):
                token_group_map[idx] = (start, end)

        # Find sqrt overline boxes for proper sqrt sizing
        sqrt_boxes = {}  # Maps glyph index to sqrt overline box
        if '\\sqrt' in tokens:
            for box in boxes:
                box_x, box_y, box_width, box_height = box
                # Sqrt overlines are at high y-position and wide
                if box_y > 5.0 and box_width > 2.0:
                    # Find the sqrt glyph this box corresponds to
                    for i, (glyph_x, glyph_y, char, glyph_width) in enumerate(glyphs):
                        # Sqrt radical glyph starts near the box start
                        if abs(glyph_x - (box_x - glyph_width)) < 2.0 and abs(glyph_y - box_y) < 1.0:
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

            # Special handling for cases environment - it produces a left brace glyph
            if token == '\\begin{cases}':
                # Map to \{ token for symbol library matching
                token = '\\{'
            elif token == '\\end{cases}' or token == '\\\\':
                # \end{cases} and \\ (line break) produce no glyphs, skip them
                continue
            # Skip structural tokens and spacing commands that don't render as glyphs
            # Note: \frac is skipped here (only renders as box), but \sqrt produces a glyph
            # Font-changing commands (\mathrm, \mathbf, etc.) don't produce glyphs, only style content
            elif token in ['^', '_', '{', '}', '\\ ', '\\,', '\\;', '\\:', '\\!', '\\quad', '\\qquad',
                          '\\mathrm', '\\mathbf', '\\mathit', '\\mathcal', '\\mathsf', '\\mathtt',
                          '\\mathfrak', '\\mathnormal'] or \
               token.startswith('\\frac') or \
               token.startswith('\\begin') or token.startswith('\\end'):
                continue

            # Get glyph info
            x, y, char, width = glyphs[glyph_idx]

            # Special handling for cases environment brace - extend to cover all content
            if token == '\\{' and '\\begin{cases}' in tokens:
                # Find the vertical extent of all content after the brace
                # In DVI coords: smaller y is higher, larger y is lower
                min_y = y  # Highest point (most negative in standard coords)
                max_y = y  # Lowest point (most positive in standard coords)
                for next_glyph_x, next_glyph_y, _, next_width in glyphs[glyph_idx + 1:]:
                    min_y = min(min_y, next_glyph_y)  # Find highest DVI y (top)
                    max_y = max(max_y, next_glyph_y)  # Find lowest DVI y (bottom)
                    # Estimate glyph height (extends downward in DVI)
                    glyph_height = next_width * 1.2
                    max_y = max(max_y, next_glyph_y + glyph_height)

                # Create bbox that spans all content lines
                # Note: DVI y goes down, we negate for standard math convention
                bbox = {
                    "token": token,
                    "xMin": float(x),
                    "yMin": float(-max_y),  # Lowest point in standard coords
                    "xMax": float(x + width),
                    "yMax": float(-min_y + 2.0)  # Highest point in standard coords + padding
                }
            # Special handling for sqrt radical
            elif token == '\\sqrt' and glyph_idx in sqrt_boxes:
                # Use the sqrt overline box to determine proper extent
                box_x, box_y, box_width, box_height = sqrt_boxes[glyph_idx]

                # Sqrt should extend from the radical to the end of actual content
                # and from the top of the overline down to cover all content below
                sqrt_x_min = x
                sqrt_x_max = box_x  # Start with box position

                # Find both horizontal and vertical extent by scanning content glyphs
                # Content glyphs are at baseline (y ≈ 0), radical is elevated (y ≈ 8)
                # In DVI coords: y increases downward, so max_y is the bottom
                min_y = y
                max_y = y
                radicand_glyph_count = 0  # Count glyphs in radicand to skip them later

                # The overline box width tells us the horizontal extent of the radicand
                # box_x is where the overline starts, box_width is its length
                radicand_x_max = box_x + box_width

                for next_glyph_x, next_glyph_y, _, next_width in glyphs[glyph_idx + 1:]:
                    # Content glyphs are at baseline (y ≈ 0)
                    # Only include glyphs within the sqrt overline box extent
                    if abs(next_glyph_y) < 2.0 and next_glyph_x < radicand_x_max:
                        radicand_glyph_count += 1  # Count this radicand glyph

                        # Update horizontal extent based on actual glyph position
                        glyph_right = next_glyph_x + next_width
                        sqrt_x_max = max(sqrt_x_max, glyph_right)

                        # Update vertical extent
                        min_y = min(min_y, next_glyph_y)
                        glyph_height = next_width * 1.2
                        max_y = max(max_y, next_glyph_y + glyph_height)
                    # Stop if we've gone past the overline box or hit an elevated glyph
                    elif next_glyph_x >= radicand_x_max or next_glyph_y > 5.0:
                        break

                # Create bbox that covers from top of radicand to bottom of content
                # In standard coords (negated DVI): yMin=bottom, yMax=top
                bbox = {
                    "token": token,
                    "xMin": float(sqrt_x_min),
                    "yMin": float(-max_y - 2.0),  # Bottom of content (most negative)
                    "xMax": float(sqrt_x_max),
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
                    x, y, char, width = glyphs[glyph_idx]
                    height = width * 1.2

                    # Get corresponding token (should be the radicand content)
                    radicand_token = tokens[token_idx] if token_idx < len(tokens) else char

                    # Create bbox for this radicand glyph
                    radicand_bbox = {
                        "token": radicand_token,
                        "xMin": float(x),
                        "yMin": float(-y - height),
                        "xMax": float(x + width),
                        "yMax": float(-y)
                    }
                    bboxes.append(radicand_bbox)

                    glyph_idx += 1
                    token_idx += 1

                # Skip any spacing tokens like '\ ' that don't correspond to glyphs
                while token_idx < len(tokens) and tokens[token_idx] in ['\\ ', '\\,', '\\;', '\\:', '\\!', ' ']:
                    token_idx += 1

                continue  # Skip the common bbox append at line 510
            # Special handling for \lognl custom macro
            # \lognl{n} expands to: superscript 'n' + 'l' + 'o' + 'g'
            elif token == '\\lognl':
                # \lognl produces 4 glyphs: superscript digit, then 'l', 'o', 'g'
                # We need to create separate bboxes for each glyph

                # First glyph: superscript (current glyph)
                superscript_x, superscript_y, superscript_char, superscript_width = glyphs[glyph_idx]
                superscript_height = superscript_width * 1.2
                bbox = {
                    "token": superscript_char,  # Use the actual digit character
                    "xMin": float(superscript_x),
                    "yMin": float(-superscript_y - superscript_height),
                    "xMax": float(superscript_x + superscript_width),
                    "yMax": float(-superscript_y)
                }
                bboxes.append(bbox)
                glyph_idx += 1

                # Next 3 glyphs: 'l', 'o', 'g'
                # Collect all three glyphs first to normalize their heights
                log_glyphs = []
                for i in range(3):
                    if glyph_idx + i >= len(glyphs):
                        break
                    log_glyphs.append(glyphs[glyph_idx + i])

                # Calculate max height among the 'log' letters for normalization
                max_height = 0.0
                for log_x, log_y, log_char, log_width in log_glyphs:
                    estimated_height = log_width * 1.2
                    max_height = max(max_height, estimated_height)

                # Create bboxes with normalized heights
                for log_x, log_y, log_char, log_width in log_glyphs:
                    # Use max_height for all letters to make them uniform
                    bbox = {
                        "token": log_char,  # 'l', 'o', or 'g'
                        "xMin": float(log_x),
                        "yMin": float(-log_y - max_height),  # Normalized height
                        "xMax": float(log_x + log_width),
                        "yMax": float(-log_y)
                    }
                    bboxes.append(bbox)
                    glyph_idx += 1

                # Skip the next 3 tokens: '[', digit, ']'  (e.g., [3] in \lognl[3]{x})
                # \lognl takes an optional argument in square brackets
                if token_idx < len(tokens) and tokens[token_idx] == '[':
                    token_idx += 1  # Skip '['
                    if token_idx < len(tokens):
                        token_idx += 1  # Skip the digit
                    if token_idx < len(tokens) and tokens[token_idx] == ']':
                        token_idx += 1  # Skip ']'
                # Continue to next token without appending another bbox
                continue
            # Check if this token is part of a letter group (for height normalization)
            elif (token_idx - 1) in token_group_map:  # -1 because we incremented at line 389
                group_start, group_end = token_group_map[token_idx - 1]

                # Check if we're at the START of the group (process once for entire group)
                if token_idx - 1 == group_start:
                    group_size = group_end - group_start

                    # Collect all glyphs in this letter group
                    group_glyphs = []
                    for i in range(group_size):
                        if glyph_idx + i < len(glyphs):
                            group_glyphs.append(glyphs[glyph_idx + i])

                    # Calculate max height for normalization (same technique as \lognl)
                    max_height = 0.0
                    for g_x, g_y, g_char, g_width in group_glyphs:
                        estimated_height = g_width * 1.2
                        max_height = max(max_height, estimated_height)

                    # Create bboxes with normalized heights
                    for g_x, g_y, g_char, g_width in group_glyphs:
                        bbox = {
                            "token": g_char,  # Use actual character
                            "xMin": float(g_x),
                            "yMin": float(-g_y - max_height),  # Normalized height!
                            "xMax": float(g_x + g_width),
                            "yMax": float(-g_y)
                        }
                        bboxes.append(bbox)

                    # Advance indices to skip the entire group
                    glyph_idx += group_size
                    token_idx += group_size - 1  # -1 because main loop will increment
                    continue  # Skip standard bbox creation
            else:
                # Standard bbox calculation
                # Estimate height (simplified - would need font metrics for accuracy)
                height = width * 1.2  # Rough approximation

                # Create bounding box
                # Note: DVI y-coordinates go down, we negate for standard math convention
                bbox = {
                    "token": token,
                    "xMin": float(x),
                    "yMin": float(-y - height),
                    "xMax": float(x + width),
                    "yMax": float(-y)
                }

            bboxes.append(bbox)
            glyph_idx += 1

        # Process boxes (fraction bars, overlines, etc.)
        for box in boxes:
            box_token = self.classify_box(box, glyphs, tokens)
            if box_token:
                x, y, width, height = box
                bbox = {
                    "token": box_token,
                    "xMin": float(x),
                    "yMin": float(-y - height),
                    "xMax": float(x + width),
                    "yMax": float(-y)
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
