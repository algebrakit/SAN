"""
InkML Parser Module

This module provides functionality to parse InkML files and extract symbol labels
for mathematical handwriting recognition analysis.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np


def parse_inkml_label(file_path: Path) -> Optional[str]:
    """
    Parse an InkML file and extract the symbol label.
    
    Args:
        file_path (Path): Path to the InkML file
        
    Returns:
        str: The symbol label (LaTeX format) or None if not found
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Find annotation with type="label"
        for annotation in root.findall('.//{http://www.w3.org/2003/InkML}annotation'):
            if annotation.get('type') == 'label':
                return annotation.text
        
        return None
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def get_inkml_files(directory: Path) -> List[Path]:
    """
    Get all InkML files from a directory.
    
    Args:
        directory (Path): Directory containing InkML files
        
    Returns:
        List[Path]: List of paths to InkML files
    """
    return list(directory.glob("*.inkml"))


class InkMLParser:
    """InkML Parser class for compatibility with notebook inference code."""
    
    def __init__(self):
        pass
    
    def parse_file(self, file_path):
        """
        Parse an InkML file and extract strokes and label.
        
        Args:
            file_path (str or Path): Path to the InkML file
            
        Returns:
            tuple: (strokes_list, label) where strokes_list contains stroke points
        """
        from stroke_preprocessor import StrokePreprocessor
        
        # Convert to Path object
        file_path = Path(file_path)
        
        # Use StrokePreprocessor to parse the file
        preprocessor = StrokePreprocessor()
        label, strokes_array = preprocessor.parse_inkml_strokes(file_path)
        
        if strokes_array is None:
            return [], label
        
        # Convert numpy array back to list of strokes
        # Group by stroke_id
        strokes_list = []
        unique_stroke_ids = np.unique(strokes_array[:, 2])
        
        for stroke_id in unique_stroke_ids:
            mask = strokes_array[:, 2] == stroke_id
            stroke_points = strokes_array[mask][:, :2]  # Get x, y coordinates
            strokes_list.append(stroke_points.tolist())
        
        return strokes_list, label


def extract_sample_labels(file_path: Path, num_samples: int = 3) -> List[Tuple[str, Optional[str]]]:
    """
    Extract labels from a sample of InkML files for testing.
    
    Args:
        file_path (Path): Directory containing InkML files
        num_samples (int): Number of sample files to process
        
    Returns:
        List[Tuple[str, Optional[str]]]: List of (filename, label) pairs
    """
    directory = Path(file_path)
    sample_files = get_inkml_files(directory)[:num_samples]
    
    results = []
    for file_path in sample_files:
        label = parse_inkml_label(file_path)
        results.append((file_path.name, label))
    
    return results


def validate_inkml_structure(file_path: Path) -> bool:
    """
    Validate that an InkML file has the expected structure.
    
    Args:
        file_path (Path): Path to the InkML file
        
    Returns:
        bool: True if file has valid structure, False otherwise
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Check for required namespace
        if 'http://www.w3.org/2003/InkML' not in root.tag:
            return False
            
        # Check for label annotation
        has_label = False
        for annotation in root.findall('.//{http://www.w3.org/2003/InkML}annotation'):
            if annotation.get('type') == 'label':
                has_label = True
                break
        
        # Check for trace elements (stroke data)
        traces = root.findall('.//{http://www.w3.org/2003/InkML}trace')
        has_traces = len(traces) > 0
        
        return has_label and has_traces
    except Exception:
        return False