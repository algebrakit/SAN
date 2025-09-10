"""
Stroke to Image Converter Module

This module provides functionality to convert stroke data from InkML files
into BMP image format.
"""

import numpy as np
from typing import List, Tuple, Optional
import cv2


def strokes_to_image(
    strokes: List[List[List[float]]], 
    image_size: Tuple[int, int] = (400, 400),
    line_thickness: int = 2,
    padding: int = 20,
    background_color: int = 0,
    stroke_color: int = 255
) -> np.ndarray:
    """
    Convert strokes to a BMP image.
    
    Args:
        strokes: List of strokes, where each stroke is a list of [x, y] points
        image_size: Tuple of (width, height) for the output image
        line_thickness: Thickness of the stroke lines
        padding: Padding around the strokes
        background_color: Background color (0-255, default white)
        stroke_color: Stroke color (0-255, default black)
        
    Returns:
        np.ndarray: Image array in BMP format (grayscale)
    """
    if not strokes:
        # Return blank image if no strokes
        return np.full(image_size[::-1], background_color, dtype=np.uint8)
    
    # Find bounding box of all strokes
    all_points = []
    for stroke in strokes:
        for point in stroke:
            if len(point) >= 2:
                all_points.append(point[:2])
    
    if not all_points:
        return np.full(image_size[::-1], background_color, dtype=np.uint8)
    
    all_points = np.array(all_points)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    
    # Calculate scale to fit strokes in image with padding
    stroke_width = max_x - min_x
    stroke_height = max_y - min_y
    
    if stroke_width == 0:
        stroke_width = 1
    if stroke_height == 0:
        stroke_height = 1
    
    # Calculate scale factors
    scale_x = (image_size[0] - 2 * padding) / stroke_width
    scale_y = (image_size[1] - 2 * padding) / stroke_height
    scale = min(scale_x, scale_y)
    
    # Create blank image
    image = np.full((image_size[1], image_size[0]), background_color, dtype=np.uint8)
    
    # Draw each stroke
    for stroke in strokes:
        if len(stroke) < 2:
            continue
            
        # Transform and scale points
        transformed_points = []
        for point in stroke:
            if len(point) >= 2:
                x = int((point[0] - min_x) * scale + padding)
                y = int((point[1] - min_y) * scale + padding)
                transformed_points.append([x, y])
        
        # Draw lines between consecutive points
        if len(transformed_points) >= 2:
            points = np.array(transformed_points, dtype=np.int32)
            for i in range(len(points) - 1):
                cv2.line(
                    image,
                    tuple(points[i]),
                    tuple(points[i + 1]),
                    stroke_color,
                    line_thickness,
                    cv2.LINE_AA
                )
    
    return image



def save_as_bmp(image: np.ndarray, file_path: str) -> bool:
    """
    Save image array as BMP file.
    
    Args:
        image: Image array
        file_path: Path where to save the BMP file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        return cv2.imwrite(file_path, image)
    except Exception as e:
        print(f"Error saving BMP: {e}")
        return False