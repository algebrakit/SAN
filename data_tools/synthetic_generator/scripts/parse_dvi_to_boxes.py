#!/usr/bin/env python3
"""
DVI to Bounding Box Parser

This script parses an existing DVI file and generates bounding box data.
Useful when LaTeX compilation is handled separately or pre-existing DVI files are available.

Usage:
    python3 parse_dvi_to_boxes.py --dvi latex/latex.dvi --expression "2\sqrt{1+\frac{1}{x}}"
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re

try:
    import matplotlib.dviread as dviread
except ImportError:
    print("Error: matplotlib is required for DVI parsing")
    print("Install with: pip install matplotlib")
    exit(1)


def tokenize_latex(latex: str) -> List[str]:
    """
    Tokenize a LaTeX string into individual symbols/commands.

    Args:
        latex: LaTeX expression string

    Returns:
        List of tokens
    """
    command_pattern = re.compile(
        r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)'
    )

    tokens = []
    s = latex

    while s:
        if s[0] == '\\':
            match = command_pattern.match(s)
            if match:
                tokens.append(match.group(0))
                s = s[len(match.group(0)):]
            else:
                tokens.append(s[0])
                s = s[1:]
        elif s[0] in '{}^_':
            s = s[1:]
        elif s[0].isspace():
            s = s[1:]
        else:
            tokens.append(s[0])
            s = s[1:]

    return tokens


def parse_dvi(dvi_path: Path, dpi: int = 72) -> List[Tuple[float, float, str, float]]:
    """
    Parse DVI file to extract glyph positions.

    Args:
        dvi_path: Path to DVI file
        dpi: DPI for rendering

    Returns:
        List of (x, y, char, width) tuples
    """
    glyphs = []

    try:
        print(f"Parsing {dvi_path}...")
        with dviread.Dvi(str(dvi_path), dpi=dpi) as dvi:
            for page_num, page in enumerate(dvi):
                print(f"  Page {page_num + 1}: {len(list(page.text))} glyphs")

                for x, y, font, glyph, width in page.text:
                    # Get character representation
                    try:
                        char = chr(glyph) if glyph < 256 else f"glyph_{glyph}"
                    except:
                        char = f"glyph_{glyph}"

                    print(f"    Glyph: x={x:.2f}, y={y:.2f}, char={char!r}, width={width:.2f}")
                    glyphs.append((x, y, char, width))

    except Exception as e:
        print(f"Error parsing DVI: {e}")
        import traceback
        traceback.print_exc()
        return []

    return glyphs


def create_bounding_boxes(
    tokens: List[str],
    glyphs: List[Tuple[float, float, str, float]]
) -> List[Dict]:
    """
    Create bounding boxes by mapping glyphs to tokens.

    Args:
        tokens: List of LaTeX tokens
        glyphs: List of (x, y, char, width) from DVI

    Returns:
        List of bounding box dictionaries
    """
    bboxes = []
    glyph_idx = 0

    print(f"\nMapping {len(tokens)} tokens to {len(glyphs)} glyphs...")
    print(f"Tokens: {tokens}")

    for token in tokens:
        if glyph_idx >= len(glyphs):
            print(f"  Warning: Ran out of glyphs at token '{token}'")
            break

        # Skip structural tokens
        if token in ['^', '_', '{', '}'] or \
           token.startswith('\\frac') or \
           token.startswith('\\sqrt') or \
           token.startswith('\\begin') or \
           token.startswith('\\end'):
            print(f"  Skipping structural token: {token}")
            continue

        # Get glyph
        x, y, char, width = glyphs[glyph_idx]

        # Estimate height (using font metrics would be better)
        height = width * 1.5

        bbox = {
            "token": token,
            "xMin": float(x),
            "yMin": float(-y - height),
            "xMax": float(x + width),
            "yMax": float(-y)
        }

        print(f"  Token '{token}' → glyph '{char}' at ({x:.2f}, {y:.2f})")
        bboxes.append(bbox)
        glyph_idx += 1

    return bboxes


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parse DVI file to extract bounding boxes"
    )

    parser.add_argument(
        '--dvi',
        type=str,
        required=True,
        help="Path to DVI file"
    )

    parser.add_argument(
        '--expression',
        type=str,
        required=True,
        help="LaTeX expression (for token mapping)"
    )

    parser.add_argument(
        '--output',
        type=str,
        help="Output JSON file (default: print to stdout)"
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=72,
        help="DPI for DVI rendering"
    )

    args = parser.parse_args()

    # Validate DVI file
    dvi_path = Path(args.dvi)
    if not dvi_path.exists():
        print(f"Error: DVI file '{dvi_path}' not found")
        return 1

    # Tokenize expression
    tokens = tokenize_latex(args.expression)

    # Parse DVI
    glyphs = parse_dvi(dvi_path, args.dpi)

    if not glyphs:
        print("Error: No glyphs found in DVI file")
        return 1

    # Create bounding boxes
    bboxes = create_bounding_boxes(tokens, glyphs)

    # Create result
    result = {
        "label": args.expression,
        "normalizedLabel": args.expression,
        "bboxes": bboxes
    }

    # Output
    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_json + '\n', encoding='utf-8')
        print(f"\n✓ Wrote bounding boxes to {output_path}")
    else:
        print("\nGenerated bounding boxes:")
        print(output_json)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
