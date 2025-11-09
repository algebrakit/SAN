#!/usr/bin/env python3
"""
Quick InkML Renderer

Simple utility to render InkML files to images for visual verification.

Usage:
    python3 render_inkml.py <inkml_file> <output_image>
"""

import sys
import argparse
from pathlib import Path

# Add stroke processing to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'stroke_processing'))

from stroke2img import strokes_to_image, save_as_bmp
from stroke_transformer import StrokeSet


def render_inkml(inkml_path: Path, output_path: Path, image_size=(800, 400)):
    """
    Render an InkML file to a BMP image.

    Args:
        inkml_path: Path to InkML file
        output_path: Path for output BMP
        image_size: Tuple of (width, height)
    """
    # Load strokes
    print(f"Loading {inkml_path.name}...")
    stroke_set = StrokeSet.from_inkml(inkml_path)
    print(f"  Loaded {len(stroke_set.strokes)} strokes")

    # Get bounding box
    bbox = stroke_set.get_bounding_box()
    print(f"  Bounding box: ({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})")
    print(f"  Size: {bbox[2]-bbox[0]:.1f} x {bbox[3]-bbox[1]:.1f}")

    # Convert to list format for rendering
    strokes_list = []
    for stroke in stroke_set.strokes:
        stroke_points = [[float(x), float(y)] for x, y in zip(stroke.x, stroke.y)]
        strokes_list.append(stroke_points)

    # Render to image
    print(f"\nRendering to {image_size[0]}x{image_size[1]} image...")
    img = strokes_to_image(
        strokes_list,
        image_size=image_size,
        line_thickness=3,
        padding=20,
        background_color=255,  # White background
        stroke_color=0         # Black strokes
    )

    # Save image
    print(f"Saving to {output_path}...")
    if save_as_bmp(img, str(output_path)):
        print(f"✓ Successfully rendered {inkml_path.name}")
        return True
    else:
        print(f"✗ Failed to save image")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Render InkML file to BMP image"
    )

    parser.add_argument(
        'inkml',
        type=str,
        help="Input InkML file"
    )

    parser.add_argument(
        'output',
        type=str,
        nargs='?',
        help="Output BMP file (default: same name with .bmp extension)"
    )

    parser.add_argument(
        '--size',
        type=str,
        default='800x400',
        help="Image size as WIDTHxHEIGHT"
    )

    args = parser.parse_args()

    # Parse paths
    inkml_path = Path(args.inkml)
    if not inkml_path.exists():
        print(f"Error: File '{inkml_path}' not found")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = inkml_path.with_suffix('.bmp')

    # Parse image size
    try:
        width, height = map(int, args.size.split('x'))
        image_size = (width, height)
    except:
        print(f"Error: Invalid size format '{args.size}'. Use WIDTHxHEIGHT")
        return 1

    # Render
    success = render_inkml(inkml_path, output_path, image_size)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
