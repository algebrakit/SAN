"""Box classification utilities for DVI output.

This module classifies DVI box elements (horizontal rules) to determine
what LaTeX construct they represent (fraction bars, overlines, underlines).
"""

from typing import List, Optional

from models import Glyph, DVIBox


def classify_box(
    box: DVIBox,
    glyphs: List[Glyph],
    tokens: List[str]
) -> Optional[str]:
    """
    Classify a DVI box to determine what LaTeX symbol it represents.

    DVI boxes are horizontal rules that can represent:
    - Fraction bars (\\frac): horizontal line with glyphs above and below
    - Overlines (\\overline): line above glyphs only
    - Underlines (\\underline): line below glyphs only
    - Sqrt overlines: part of radical symbol (filtered out)

    Args:
        box: DVIBox object with position and dimensions
        glyphs: List of all Glyph objects in the expression
        tokens: List of LaTeX tokens (for context)

    Returns:
        Token string ('\\frac', '\\overline', '\\underline') or None to skip
    """
    # Check for horizontal line (width >> height)
    if box.width > box.height * 2:
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
