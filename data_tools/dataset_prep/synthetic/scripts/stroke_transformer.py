#!/usr/bin/env python3
"""
InkML Stroke Transformer

This module provides utilities to transform InkML strokes:
- Normalize to unit bounding box
- Scale to target dimensions
- Translate to target position
- Compute bounding boxes
- Combine multiple stroke sets

These transformations are used to place handwritten symbols from the InkML
library into target bounding boxes for expression synthesis.
"""

import numpy as np
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET
from pathlib import Path


class Stroke:
    """Represents a single stroke with x, y, and optional time coordinates."""

    def __init__(self, points: np.ndarray):
        """
        Initialize a stroke.

        Args:
            points: Nx2 or Nx3 numpy array of (x, y) or (x, y, t) coordinates
        """
        self.points = np.array(points, dtype=np.float64)

        if self.points.shape[1] < 2:
            raise ValueError("Stroke must have at least x, y coordinates")

    @property
    def x(self) -> np.ndarray:
        """Get x coordinates."""
        return self.points[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Get y coordinates."""
        return self.points[:, 1]

    @property
    def t(self) -> Optional[np.ndarray]:
        """Get time coordinates if available."""
        if self.points.shape[1] >= 3:
            return self.points[:, 2]
        return None

    def copy(self) -> 'Stroke':
        """Create a deep copy of the stroke."""
        return Stroke(self.points.copy())


class StrokeSet:
    """Collection of strokes representing a symbol or expression."""

    def __init__(self, strokes: List[Stroke]):
        """
        Initialize a stroke set.

        Args:
            strokes: List of Stroke objects
        """
        self.strokes = strokes

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Compute the bounding box of all strokes.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        if not self.strokes:
            return (0.0, 0.0, 0.0, 0.0)

        all_x = np.concatenate([stroke.x for stroke in self.strokes])
        all_y = np.concatenate([stroke.y for stroke in self.strokes])

        return (float(np.min(all_x)), float(np.min(all_y)),
                float(np.max(all_x)), float(np.max(all_y)))

    def normalize(self) -> 'StrokeSet':
        """
        Normalize strokes to unit bounding box [0, 1] x [0, 1].

        Returns:
            New StrokeSet with normalized strokes
        """
        x_min, y_min, x_max, y_max = self.get_bounding_box()

        width = x_max - x_min
        height = y_max - y_min

        # Handle degenerate cases
        if width == 0:
            width = 1.0
        if height == 0:
            height = 1.0

        # Normalize to [0, 1] range
        normalized_strokes = []
        for stroke in self.strokes:
            normalized_points = stroke.points.copy()
            normalized_points[:, 0] = (stroke.x - x_min) / width
            normalized_points[:, 1] = (stroke.y - y_min) / height
            normalized_strokes.append(Stroke(normalized_points))

        return StrokeSet(normalized_strokes)

    def scale(self, width: float, height: float,
              preserve_aspect: bool = True) -> 'StrokeSet':
        """
        Scale strokes to target dimensions.

        Args:
            width: Target width
            height: Target height
            preserve_aspect: If True, preserve aspect ratio (fit within target box)

        Returns:
            New StrokeSet with scaled strokes
        """
        if preserve_aspect:
            # Get current dimensions
            x_min, y_min, x_max, y_max = self.get_bounding_box()
            curr_width = x_max - x_min
            curr_height = y_max - y_min

            if curr_width == 0:
                curr_width = 1.0
            if curr_height == 0:
                curr_height = 1.0

            # Compute scale to fit within target box
            scale_x = width / curr_width
            scale_y = height / curr_height
            scale = min(scale_x, scale_y)

            scale_x = scale
            scale_y = scale
        else:
            # Stretch to fill target box
            x_min, y_min, x_max, y_max = self.get_bounding_box()
            curr_width = x_max - x_min
            curr_height = y_max - y_min

            if curr_width == 0:
                curr_width = 1.0
            if curr_height == 0:
                curr_height = 1.0

            scale_x = width / curr_width
            scale_y = height / curr_height

        # Apply scaling
        scaled_strokes = []
        for stroke in self.strokes:
            scaled_points = stroke.points.copy()
            scaled_points[:, 0] = (stroke.x - x_min) * scale_x
            scaled_points[:, 1] = (stroke.y - y_min) * scale_y
            scaled_strokes.append(Stroke(scaled_points))

        return StrokeSet(scaled_strokes)

    def translate(self, dx: float, dy: float) -> 'StrokeSet':
        """
        Translate strokes by (dx, dy).

        Args:
            dx: Translation in x direction
            dy: Translation in y direction

        Returns:
            New StrokeSet with translated strokes
        """
        translated_strokes = []
        for stroke in self.strokes:
            translated_points = stroke.points.copy()
            translated_points[:, 0] += dx
            translated_points[:, 1] += dy
            translated_strokes.append(Stroke(translated_points))

        return StrokeSet(translated_strokes)

    def copy(self) -> 'StrokeSet':
        """Create a deep copy of the stroke set."""
        return StrokeSet([stroke.copy() for stroke in self.strokes])

    @staticmethod
    def from_inkml(inkml_path: Path) -> 'StrokeSet':
        """
        Load strokes from an InkML file.

        Args:
            inkml_path: Path to InkML file

        Returns:
            StrokeSet loaded from file
        """
        tree = ET.parse(inkml_path)
        root = tree.getroot()

        strokes = []
        for trace in root.findall('.//{http://www.w3.org/2003/InkML}trace'):
            points_text = trace.text
            if not points_text:
                continue

            # Parse points: "x1 y1 t1,x2 y2 t2,..."
            point_strs = points_text.split(',')
            points = []

            for point_str in point_strs:
                coords = point_str.strip().split()
                if len(coords) >= 2:
                    # Convert to float
                    point = [float(coords[0]), float(coords[1])]
                    if len(coords) >= 3:
                        point.append(float(coords[2]))
                    points.append(point)

            if points:
                strokes.append(Stroke(np.array(points)))

        return StrokeSet(strokes)

    def to_inkml_traces(self, trace_id_start: int = 0) -> List[ET.Element]:
        """
        Convert strokes to InkML trace elements.

        Args:
            trace_id_start: Starting trace ID number

        Returns:
            List of XML trace elements
        """
        traces = []

        for i, stroke in enumerate(self.strokes):
            trace = ET.Element('trace')
            trace.set('id', str(trace_id_start + i))

            # Format points
            points_strs = []
            for j in range(len(stroke.points)):
                if stroke.points.shape[1] == 2:
                    # x, y only
                    points_strs.append(f"{stroke.x[j]:.2f} {stroke.y[j]:.2f} 0.0")
                else:
                    # x, y, t
                    points_strs.append(
                        f"{stroke.x[j]:.2f} {stroke.y[j]:.2f} {stroke.t[j]:.1f}"
                    )

            trace.text = ','.join(points_strs)
            traces.append(trace)

        return traces


def combine_stroke_sets(stroke_sets: List[StrokeSet]) -> StrokeSet:
    """
    Combine multiple stroke sets into one.

    Args:
        stroke_sets: List of StrokeSet objects

    Returns:
        Single StrokeSet containing all strokes
    """
    all_strokes = []
    for stroke_set in stroke_sets:
        all_strokes.extend(stroke_set.strokes)

    return StrokeSet(all_strokes)


def fit_to_bbox(
    stroke_set: StrokeSet,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    preserve_aspect: bool = True,
    padding: float = 0.0
) -> StrokeSet:
    """
    Transform strokes to fit within a target bounding box.

    This is a convenience function that combines scale and translate operations.

    Args:
        stroke_set: Input stroke set
        x_min: Target bounding box minimum x
        y_min: Target bounding box minimum y
        x_max: Target bounding box maximum x
        y_max: Target bounding box maximum y
        preserve_aspect: If True, preserve aspect ratio
        padding: Padding ratio (0.0-1.0) to shrink target bbox and create gaps.
                 E.g., 0.05 = 5% padding on each side

    Returns:
        New StrokeSet fitted to target bounding box
    """
    # Apply padding to create gaps between symbols
    if padding > 0:
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        pad_x = bbox_width * padding
        pad_y = bbox_height * padding

        # Shrink target bbox by padding
        x_min += pad_x
        x_max -= pad_x
        y_min += pad_y
        y_max -= pad_y

    target_width = x_max - x_min
    target_height = y_max - y_min

    # Scale to target dimensions
    scaled = stroke_set.scale(target_width, target_height, preserve_aspect)

    # If aspect ratio was preserved, center the result
    if preserve_aspect:
        scaled_bbox = scaled.get_bounding_box()
        scaled_width = scaled_bbox[2] - scaled_bbox[0]
        scaled_height = scaled_bbox[3] - scaled_bbox[1]

        # Center horizontally and vertically
        dx = x_min + (target_width - scaled_width) / 2 - scaled_bbox[0]
        dy = y_min + (target_height - scaled_height) / 2 - scaled_bbox[1]
    else:
        # Just translate to target position
        dx = x_min
        dy = y_min

    return scaled.translate(dx, dy)


# Example usage and testing
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stroke_transformer.py <inkml_file>")
        print("\nThis will load an InkML file and demonstrate transformations.")
        sys.exit(1)

    inkml_path = Path(sys.argv[1])

    if not inkml_path.exists():
        print(f"Error: File '{inkml_path}' not found")
        sys.exit(1)

    # Load strokes
    print(f"Loading strokes from {inkml_path.name}...")
    stroke_set = StrokeSet.from_inkml(inkml_path)
    print(f"  Loaded {len(stroke_set.strokes)} strokes")

    # Get original bounding box
    bbox = stroke_set.get_bounding_box()
    print(f"  Original bbox: ({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})")
    print(f"  Size: {bbox[2]-bbox[0]:.1f} x {bbox[3]-bbox[1]:.1f}")

    # Normalize
    print("\nNormalizing to unit box...")
    normalized = stroke_set.normalize()
    norm_bbox = normalized.get_bounding_box()
    print(f"  Normalized bbox: ({norm_bbox[0]:.3f}, {norm_bbox[1]:.3f}) to ({norm_bbox[2]:.3f}, {norm_bbox[3]:.3f})")

    # Scale to 100x100
    print("\nScaling to 100x100 (preserve aspect)...")
    scaled = stroke_set.scale(100, 100, preserve_aspect=True)
    scaled_bbox = scaled.get_bounding_box()
    print(f"  Scaled bbox: ({scaled_bbox[0]:.1f}, {scaled_bbox[1]:.1f}) to ({scaled_bbox[2]:.1f}, {scaled_bbox[3]:.1f})")
    print(f"  Size: {scaled_bbox[2]-scaled_bbox[0]:.1f} x {scaled_bbox[3]-scaled_bbox[1]:.1f}")

    # Translate
    print("\nTranslating by (50, 50)...")
    translated = scaled.translate(50, 50)
    trans_bbox = translated.get_bounding_box()
    print(f"  Translated bbox: ({trans_bbox[0]:.1f}, {trans_bbox[1]:.1f}) to ({trans_bbox[2]:.1f}, {trans_bbox[3]:.1f})")

    # Fit to specific bounding box
    print("\nFitting to bbox (200, 300) to (500, 400)...")
    fitted = fit_to_bbox(stroke_set, 200, 300, 500, 400, preserve_aspect=True)
    fitted_bbox = fitted.get_bounding_box()
    print(f"  Fitted bbox: ({fitted_bbox[0]:.1f}, {fitted_bbox[1]:.1f}) to ({fitted_bbox[2]:.1f}, {fitted_bbox[3]:.1f})")

    print("\n✓ Transformations completed successfully!")
