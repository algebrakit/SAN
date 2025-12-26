"""Type definitions for the synthetic expression generator.

This module defines the core data structures used throughout the pipeline:
- Glyph: A character extracted from DVI output
- DVIBox: A box element (fraction bars, overlines) from DVI
- BoundingBox: A positioned token in output coordinates
- ProcessingStats: Statistics for tracking processing results
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass(frozen=True)
class Glyph:
    """A glyph extracted from DVI output.

    Coordinates are in DVI space where:
    - X increases to the right (normal)
    - Y increases downward (inverted from standard math coordinates)
    - The baseline is at y=0
    - Superscripts have positive y values
    - Subscripts have negative y values

    Attributes:
        x: X position (left edge of glyph)
        y: Y position (baseline reference point in DVI coords)
        char: Character or glyph code string
        width: Glyph width in DVI units
        height: Height above baseline
        depth: Depth below baseline (positive value)
    """
    x: float
    y: float
    char: str
    width: float
    height: float
    depth: float

    @property
    def total_height(self) -> float:
        """Total vertical extent (height + depth)."""
        return self.height + self.depth

    @property
    def right(self) -> float:
        """Right edge x-coordinate."""
        return self.x + self.width

    @property
    def top_dvi(self) -> float:
        """Top edge in DVI coords (y - height, since y increases down)."""
        return self.y - self.height

    @property
    def bottom_dvi(self) -> float:
        """Bottom edge in DVI coords (y + depth)."""
        return self.y + self.depth


@dataclass(frozen=True)
class DVIBox:
    """A box element from DVI output (fraction bars, overlines, underlines, etc).

    Coordinates are in DVI space (y increases downward).

    Attributes:
        x: X position (left edge)
        y: Y position
        width: Box width
        height: Box height
    """
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        """Right edge x-coordinate."""
        return self.x + self.width


@dataclass
class BoundingBox:
    """A bounding box for a LaTeX token in output space.

    Coordinates are in standard math space where:
    - X increases to the right
    - Y increases upward (normal math coordinates)

    The coordinate conversion from DVI to output space involves
    negating the y-axis.

    Attributes:
        token: The LaTeX token this box represents
        x_min: Left edge (minimum x)
        y_min: Bottom edge (minimum y, lowest visual position)
        x_max: Right edge (maximum x)
        y_max: Top edge (maximum y, highest visual position)
        glyph_height: Height above baseline (from original glyph, optional)
        glyph_depth: Depth below baseline (from original glyph, optional)
    """
    token: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    glyph_height: Optional[float] = None
    glyph_depth: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Uses camelCase keys to match existing output format.
        """
        result = {
            "token": self.token,
            "xMin": self.x_min,
            "yMin": self.y_min,
            "xMax": self.x_max,
            "yMax": self.y_max
        }
        if self.glyph_height is not None:
            result["glyphHeight"] = self.glyph_height
        if self.glyph_depth is not None:
            result["glyphDepth"] = self.glyph_depth
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BoundingBox':
        """Create from dictionary (e.g., loaded from JSON).

        Handles both camelCase and snake_case keys.
        """
        return cls(
            token=d.get("token", ""),
            x_min=d.get("xMin", d.get("x_min", 0.0)),
            y_min=d.get("yMin", d.get("y_min", 0.0)),
            x_max=d.get("xMax", d.get("x_max", 0.0)),
            y_max=d.get("yMax", d.get("y_max", 0.0)),
            glyph_height=d.get("glyphHeight"),
            glyph_depth=d.get("glyphDepth")
        )

    @classmethod
    def from_glyph(cls, token: str, glyph: Glyph) -> 'BoundingBox':
        """Create bbox from a glyph with DVI-to-output coordinate conversion.

        Coordinate System:
        - DVI: y increases downward, baseline at glyph.y
        - Output: negative y = up, positive y = down, baseline at y=0

        For a glyph at baseline (glyph.y = 0):
        - Height extends ABOVE baseline → negative y (up)
        - Depth extends BELOW baseline → positive y (down)

        Example: 'g' with height=4.29, depth=1.94
        - y_min = -4.29 (top of body, above baseline)
        - y_max = +1.94 (bottom of descender, below baseline)

        Args:
            token: The LaTeX token for this glyph
            glyph: The Glyph object from DVI parsing

        Returns:
            BoundingBox in output coordinates
        """
        return cls(
            token=token,
            x_min=glyph.x,
            y_min=-glyph.y - glyph.height,  # Top: height above baseline
            x_max=glyph.x + glyph.width,
            y_max=-glyph.y + glyph.depth,   # Bottom: depth below baseline
            glyph_height=glyph.height,
            glyph_depth=glyph.depth
        )

    @classmethod
    def from_dvi_coords(
        cls,
        token: str,
        x: float,
        y: float,
        width: float,
        height: float,
        depth: float = 0.0
    ) -> 'BoundingBox':
        """Create bbox from raw DVI coordinates.

        Uses the same coordinate conversion as from_glyph():
        - y_min = top of glyph (height above baseline)
        - y_max = bottom of glyph (depth below baseline)

        Args:
            token: The LaTeX token
            x: X position in DVI coords
            y: Y position in DVI coords (baseline)
            width: Width of the glyph/box
            height: Height above baseline
            depth: Depth below baseline (default 0)

        Returns:
            BoundingBox in output coordinates
        """
        return cls(
            token=token,
            x_min=x,
            y_min=-y - height,   # Top: height above baseline
            x_max=x + width,
            y_max=-y + depth,    # Bottom: depth below baseline
            glyph_height=height,
            glyph_depth=depth
        )

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y_max - self.y_min

    @property
    def center_x(self) -> float:
        """X coordinate of the center."""
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self) -> float:
        """Y coordinate of the center."""
        return (self.y_min + self.y_max) / 2


@dataclass
class ProcessingStats:
    """Statistics for tracking processing results.

    Used by various stages of the pipeline to track success/failure rates
    and collect error information for debugging.

    Attributes:
        total: Total number of items processed
        successful: Number of successfully processed items
        failed: Number of failed items
        errors: List of (index, item) tuples for failed items
    """
    total: int = 0
    successful: int = 0
    failed: int = 0
    errors: List[Tuple[int, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.successful / self.total

    def record_success(self) -> None:
        """Record a successful processing."""
        self.total += 1
        self.successful += 1

    def record_failure(self, index: int, item: str) -> None:
        """Record a failed processing.

        Args:
            index: Index of the failed item
            item: The item that failed (e.g., LaTeX expression)
        """
        self.total += 1
        self.failed += 1
        self.errors.append((index, item))

    def print_summary(self, title: str = "PROCESSING STATISTICS", width: int = 60) -> None:
        """Print a formatted summary of the statistics.

        Args:
            title: Title for the statistics section
            width: Width of the separator lines
        """
        print("\n" + "=" * width)
        print(title)
        print("=" * width)
        print(f"Total processed:       {self.total}")
        print(f"Successful:            {self.successful}")
        print(f"Failed:                {self.failed}")
        if self.total > 0:
            print(f"Success rate:          {self.success_rate:.1f}%")
        if self.errors:
            print(f"\nFailed items ({len(self.errors)}):")
            for idx, item in self.errors[:10]:  # Show first 10
                display = item[:50] + "..." if len(item) > 50 else item
                print(f"  [{idx}] {display}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")
        print("=" * width)
