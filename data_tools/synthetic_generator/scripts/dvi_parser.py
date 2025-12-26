"""DVI file parsing utilities.

This module handles parsing DVI files to extract glyph positions
and box elements using matplotlib's dviread module.
"""

import os
from pathlib import Path
from typing import List, Tuple

try:
    import matplotlib.dviread as dviread
except ImportError:
    raise ImportError("matplotlib is required for DVI parsing. Install with: pip install matplotlib")

from models import Glyph, DVIBox


def parse_dvi(dvi_path: Path, dpi: int = 72) -> Tuple[List[Glyph], List[DVIBox]]:
    """
    Parse DVI file to extract glyph positions and boxes.

    This function extracts all text glyphs and box elements (fraction bars,
    overlines, etc.) from a DVI file. It handles coordinate scaling and
    font metric extraction.

    Args:
        dvi_path: Path to DVI file
        dpi: DPI for rendering (affects coordinate scale)

    Returns:
        Tuple of (glyphs, boxes):
        - glyphs: List of Glyph objects with positions and font metrics
        - boxes: List of DVIBox objects (fraction bars, overlines, etc.)
    """
    glyphs: List[Glyph] = []
    boxes: List[DVIBox] = []

    # Ensure TeX binaries are in PATH so matplotlib can find TFM files via kpsewhich
    old_path = os.environ.get('PATH', '')
    if '/Library/TeX/texbin' not in old_path:
        os.environ['PATH'] = '/Library/TeX/texbin:' + old_path

    try:
        with dviread.Dvi(str(dvi_path), dpi=dpi) as dvi:
            for page in dvi:
                # Extract glyphs
                for x, y, font, glyph_code, width in page.text:
                    # Convert glyph code to character
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
                    scaling_factor = dpi / (72.27 * 2**16)

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
