"""Glyph mapping handlers for special LaTeX constructs.

This module extracts the special case handling logic from map_glyphs_to_tokens
into reusable handler classes. Each handler deals with a specific LaTeX construct
that requires non-trivial glyph-to-token mapping.

Handlers:
- CasesHandler: \begin{cases} ... \end{cases} environment
- SqrtHandler: \sqrt with optional index
- LognlHandler: Custom \lognl macro
- LargeOperatorHandler: \sum, \prod, etc. with sub/superscripts
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from models import Glyph, DVIBox, BoundingBox


@dataclass
class GlyphMappingContext:
    """Context for tracking state during glyph-to-token mapping.

    This class encapsulates the mutable state that handlers need to
    read and modify during the mapping process.

    Attributes:
        tokens: List of expanded LaTeX tokens
        glyphs: List of Glyph objects from DVI parsing
        token_idx: Current position in tokens list
        glyph_idx: Current position in glyphs list
        bboxes: Accumulated bounding boxes
        sqrt_boxes: Pre-computed sqrt overline boxes (glyph_idx -> box)
    """
    tokens: List[str]
    glyphs: List[Glyph]
    token_idx: int = 0
    glyph_idx: int = 0
    bboxes: List[BoundingBox] = field(default_factory=list)
    sqrt_boxes: Dict[int, DVIBox] = field(default_factory=dict)

    @property
    def current_token(self) -> Optional[str]:
        """Get current token or None if exhausted."""
        if self.token_idx < len(self.tokens):
            return self.tokens[self.token_idx]
        return None

    @property
    def current_glyph(self) -> Optional[Glyph]:
        """Get current glyph or None if exhausted."""
        if self.glyph_idx < len(self.glyphs):
            return self.glyphs[self.glyph_idx]
        return None

    def advance_token(self) -> None:
        """Move to next token."""
        self.token_idx += 1

    def advance_glyph(self) -> None:
        """Move to next glyph."""
        self.glyph_idx += 1

    def peek_token(self, offset: int = 1) -> Optional[str]:
        """Look ahead at token without advancing."""
        idx = self.token_idx + offset
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def append_bbox(self, bbox: BoundingBox) -> None:
        """Add a bounding box to the result list."""
        self.bboxes.append(bbox)

    def has_glyphs(self) -> bool:
        """Check if there are more glyphs to process."""
        return self.glyph_idx < len(self.glyphs)

    def has_tokens(self) -> bool:
        """Check if there are more tokens to process."""
        return self.token_idx < len(self.tokens)


class CasesHandler:
    """Handler for \\begin{cases} environment brace.

    The cases environment produces a left brace that should span all content
    lines vertically. This handler scans subsequent glyphs to determine
    the vertical extent of the brace.
    """

    @staticmethod
    def can_handle(token: str, tokens: List[str]) -> bool:
        """Check if this handler should process the token."""
        return token == '\\{' and '\\begin{cases}' in tokens

    @staticmethod
    def handle(ctx: GlyphMappingContext, token: str) -> bool:
        """Process the cases brace and create its bounding box.

        Args:
            ctx: The mapping context
            token: The current token (should be '\\{')

        Returns:
            True if handled (caller should continue to next token)
        """
        glyph = ctx.current_glyph
        if glyph is None:
            return False

        # Find the vertical extent of all content after the brace
        # In DVI coords: smaller y is higher, larger y is lower
        # Initialize to extreme values so content determines extent
        min_y = float('inf')
        max_y = float('-inf')

        prev_glyph_right = glyph.x + glyph.width

        for next_glyph in ctx.glyphs[ctx.glyph_idx + 1:]:
            gap = next_glyph.x - prev_glyph_right

            # Stop if we hit another elevated glyph (another brace)
            # Braces are at highly elevated y-positions (> 15.0 in DVI coords)
            if next_glyph.y > 15.0:
                break

            # Stop if there's a large horizontal gap (indicates next cases environment)
            if gap > 10.0:
                break

            min_y = min(min_y, next_glyph.y)
            max_y = max(max_y, next_glyph.y)
            max_y = max(max_y, next_glyph.y + next_glyph.total_height)
            prev_glyph_right = next_glyph.x + next_glyph.width

        # Create bbox that spans all content lines
        bbox = BoundingBox(
            token=token,
            x_min=float(glyph.x),
            y_min=float(-max_y),
            x_max=float(glyph.x + glyph.width),
            y_max=float(-min_y + 2.0)
        )
        ctx.append_bbox(bbox)
        ctx.advance_glyph()

        return True


class SqrtHandler:
    """Handler for \\sqrt with optional index.

    Square roots are complex because:
    - They may have an optional index in brackets [n]
    - The radical glyph is separate from the radicand content
    - The overline box determines the horizontal extent
    """

    @staticmethod
    def can_handle(token: str) -> bool:
        """Check if this handler should process the token."""
        return token == '\\sqrt'

    @staticmethod
    def handle(ctx: GlyphMappingContext, token: str) -> bool:
        """Process the sqrt and create bounding boxes.

        Args:
            ctx: The mapping context
            token: The current token (should be '\\sqrt')

        Returns:
            True if handled
        """
        # Check if this sqrt has an index: next token is '['
        has_index = ctx.peek_token(0) == '['

        # Track the rightmost x-position of index glyphs
        index_x_max = None

        # Process index glyphs if present
        if has_index:
            # Skip opening '[' token
            if ctx.peek_token(0) == '[':
                ctx.advance_token()

            # Process index glyphs until we find the radical
            while ctx.has_glyphs() and ctx.glyph_idx not in ctx.sqrt_boxes:
                idx_glyph = ctx.glyphs[ctx.glyph_idx]

                # Track the rightmost edge of index glyphs
                if index_x_max is None:
                    index_x_max = idx_glyph.x + idx_glyph.width
                else:
                    index_x_max = max(index_x_max, idx_glyph.x + idx_glyph.width)

                if ctx.has_tokens():
                    index_token = ctx.tokens[ctx.token_idx]
                    ctx.advance_token()

                    bbox = BoundingBox.from_glyph(index_token, idx_glyph)
                    ctx.append_bbox(bbox)

                ctx.advance_glyph()

            # Skip closing ']' token
            if ctx.peek_token(0) == ']':
                ctx.advance_token()

        # Process the radical if we're at one
        if ctx.glyph_idx in ctx.sqrt_boxes:
            radical_glyph = ctx.glyphs[ctx.glyph_idx]
            sqrt_box = ctx.sqrt_boxes[ctx.glyph_idx]

            # Find vertical extent by scanning glyphs under the overline
            min_y = radical_glyph.y
            max_y = radical_glyph.y
            radicand_glyph_count = 0

            radicand_x_max = sqrt_box.x + sqrt_box.width
            prev_glyph_right = sqrt_box.x

            for next_glyph in ctx.glyphs[ctx.glyph_idx + 1:]:
                # Stop if we've gone past the overline box or hit another sqrt
                if next_glyph.x >= radicand_x_max or next_glyph.y > 5.0:
                    break

                # Detect gaps that indicate non-continuous content
                gap = next_glyph.x - prev_glyph_right
                if gap < -1.0 or gap > 5.0:
                    break

                radicand_glyph_count += 1
                prev_glyph_right = next_glyph.x + next_glyph.width
                min_y = min(min_y, next_glyph.y)
                max_y = max(max_y, next_glyph.y + next_glyph.total_height)

            # Create sqrt bbox
            bbox = BoundingBox(
                token=token,
                x_min=float(index_x_max if index_x_max is not None else radical_glyph.x),
                y_min=float(-max_y - 2.0),
                x_max=float(radicand_x_max),
                y_max=float(-min_y + 2.0)
            )
            ctx.append_bbox(bbox)
            ctx.advance_glyph()

            # Create bboxes for radicand content
            for _ in range(radicand_glyph_count):
                if not ctx.has_glyphs():
                    break

                rad_glyph = ctx.glyphs[ctx.glyph_idx]
                radicand_token = ctx.tokens[ctx.token_idx] if ctx.has_tokens() else rad_glyph.char

                radicand_bbox = BoundingBox.from_glyph(radicand_token, rad_glyph)
                ctx.append_bbox(radicand_bbox)

                ctx.advance_glyph()
                ctx.advance_token()

            # Skip spacing tokens
            while ctx.has_tokens() and ctx.tokens[ctx.token_idx] in ['\\ ', '\\,', '\\;', '\\:', '\\!', ' ']:
                ctx.advance_token()

        return True


class LognlHandler:
    """Handler for custom \\lognl macro.

    The \\lognl[n] macro expands to superscript characters followed by 'l', 'o', 'g'.
    This handler processes the superscript glyphs and normalizes the log letter heights.
    """

    @staticmethod
    def can_handle(token: str) -> bool:
        """Check if this handler should process the token."""
        return token == '\\lognl'

    @staticmethod
    def handle(ctx: GlyphMappingContext, token: str) -> bool:
        """Process the lognl macro and create bounding boxes.

        Args:
            ctx: The mapping context
            token: The current token (should be '\\lognl')

        Returns:
            True if handled
        """
        # Count superscript tokens by looking for '[' ... ']' sequence
        superscript_token_count = 0
        if ctx.peek_token(0) == '[':
            scan_idx = ctx.token_idx + 1
            while scan_idx < len(ctx.tokens) and ctx.tokens[scan_idx] != ']':
                superscript_token_count += 1
                scan_idx += 1

        # Process superscript glyphs
        for _ in range(superscript_token_count):
            if not ctx.has_glyphs():
                break
            sup_glyph = ctx.glyphs[ctx.glyph_idx]
            bbox = BoundingBox.from_glyph(sup_glyph.char, sup_glyph)
            ctx.append_bbox(bbox)
            ctx.advance_glyph()

        # Collect 'l', 'o', 'g' glyphs
        log_glyphs: List[Glyph] = []
        for i in range(3):
            if ctx.glyph_idx + i < len(ctx.glyphs):
                log_glyphs.append(ctx.glyphs[ctx.glyph_idx + i])

        # Create bboxes using centralized coordinate conversion
        # This preserves proper descender positioning for 'g'
        for log_glyph in log_glyphs:
            bbox = BoundingBox.from_glyph(log_glyph.char, log_glyph)
            ctx.append_bbox(bbox)
            ctx.advance_glyph()

        # Skip bracket tokens
        if ctx.peek_token(0) == '[':
            ctx.advance_token()  # Skip '['
            for _ in range(superscript_token_count):
                if ctx.has_tokens():
                    ctx.advance_token()
            if ctx.peek_token(0) == ']':
                ctx.advance_token()  # Skip ']'

        return True


class LargeOperatorHandler:
    """Handler for large operators (\\sum, \\prod, \\int, etc.) with subscripts/superscripts.

    Large operators have a complex glyph ordering: in DVI output, superscript glyphs
    appear before subscript glyphs, but in LaTeX tokens, subscript tokens appear first.
    This handler reorders the processing to match token order.
    """

    # Tokens that are typically large operators
    LARGE_OPERATORS = {'\\sum', '\\prod', '\\int', '\\bigcup', '\\bigcap', '\\bigoplus',
                       '\\bigotimes', '\\coprod', '\\oint'}

    @staticmethod
    def is_large_operator_glyph(glyph: Glyph) -> bool:
        """Check if glyph has characteristics of a large operator.

        Characteristics: large depth (> 9.0), elevated y (5.0 < y < 10.0), minimal height (< 1.0)
        """
        return glyph.depth > 9.0 and 5.0 < glyph.y < 10.0 and glyph.height < 1.0

    @staticmethod
    def can_handle(token: str, glyph: Optional[Glyph]) -> bool:
        """Check if this handler should process the token."""
        if glyph is None:
            return False
        return LargeOperatorHandler.is_large_operator_glyph(glyph)

    @staticmethod
    def handle(ctx: GlyphMappingContext, token: str) -> bool:
        """Process large operator with subscript/superscript reordering.

        Args:
            ctx: The mapping context
            token: The current token

        Returns:
            True if handled
        """
        glyph = ctx.current_glyph
        if glyph is None:
            return False

        # Create operator bbox
        bbox = BoundingBox(
            token=token,
            x_min=float(glyph.x),
            y_min=float(-glyph.total_height),
            x_max=float(glyph.x + glyph.width),
            y_max=float(0.0)
        )
        ctx.append_bbox(bbox)
        ctx.advance_glyph()

        # Count superscript and subscript glyphs
        superscript_glyph_count = 0
        subscript_glyph_count = 0
        operator_x_max = glyph.x + glyph.width + 15.0

        for scan_glyph in ctx.glyphs[ctx.glyph_idx:]:
            if scan_glyph.x > operator_x_max:
                break

            if scan_glyph.y > 2.0 and scan_glyph.y < 8.0:
                superscript_glyph_count += 1
            elif scan_glyph.y < -1.0:
                subscript_glyph_count += 1
            else:
                break

        # Process in token order: subscripts first, then superscripts
        # Subscript glyphs appear AFTER superscript glyphs in DVI
        subscript_start_glyph = ctx.glyph_idx + superscript_glyph_count

        for i in range(subscript_glyph_count):
            if subscript_start_glyph + i >= len(ctx.glyphs):
                break
            if not ctx.has_tokens():
                break

            sub_glyph = ctx.glyphs[subscript_start_glyph + i]
            sub_token = ctx.tokens[ctx.token_idx]
            ctx.advance_token()

            sub_bbox = BoundingBox.from_glyph(sub_token, sub_glyph)
            ctx.append_bbox(sub_bbox)

        # Superscript glyphs appear BEFORE subscript glyphs in DVI
        for i in range(superscript_glyph_count):
            if ctx.glyph_idx + i >= len(ctx.glyphs):
                break
            if not ctx.has_tokens():
                break

            sup_glyph = ctx.glyphs[ctx.glyph_idx + i]
            sup_token = ctx.tokens[ctx.token_idx]
            ctx.advance_token()

            sup_bbox = BoundingBox.from_glyph(sup_token, sup_glyph)
            ctx.append_bbox(sup_bbox)

        # Advance past all sub/superscript glyphs
        ctx.glyph_idx += superscript_glyph_count + subscript_glyph_count

        return True


class ExtendedDelimiterHandler:
    """Handler for extended delimiters created by \\left and \\right.

    Extended delimiters have large depth and elevated y-position, and need
    special positioning to span the content they enclose.
    """

    @staticmethod
    def is_extended_delimiter(glyph: Glyph) -> bool:
        """Check if glyph is an extended delimiter.

        Characteristics: large depth (> 10.0), elevated y (> 5.0), small height (< 1.0)
        """
        return glyph.depth > 10.0 and glyph.y > 5.0 and glyph.height < 1.0

    @staticmethod
    def can_handle(glyph: Optional[Glyph]) -> bool:
        """Check if this handler should process the glyph."""
        if glyph is None:
            return False
        return ExtendedDelimiterHandler.is_extended_delimiter(glyph)

    @staticmethod
    def create_bbox(token: str, glyph: Glyph) -> BoundingBox:
        """Create a bounding box for an extended delimiter.

        Extended delimiters span from baseline upward.
        """
        return BoundingBox(
            token=token,
            x_min=float(glyph.x),
            y_min=float(-glyph.total_height),
            x_max=float(glyph.x + glyph.width),
            y_max=float(0.0),
            glyph_height=float(glyph.height),
            glyph_depth=float(glyph.depth)
        )


def create_standard_bbox(token: str, glyph: Glyph) -> BoundingBox:
    """Create a standard bounding box for a normal glyph.

    Args:
        token: The LaTeX token
        glyph: The glyph from DVI

    Returns:
        BoundingBox with DVI-to-output coordinate conversion
    """
    return BoundingBox.from_glyph(token, glyph)
