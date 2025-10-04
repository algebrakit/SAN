#!/usr/bin/env python3
"""
InkML to Image Batch Converter

This tool converts inkML files containing handwritten mathematical expressions
to BMP images with black background and white foreground, following the same
approach used in the SAN server.

Usage:
    python3 tools/inkml_to_images.py --input /path/to/inkml/files --output /path/to/images
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Add the stroke processing directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'stroke_processing'))

from stroke_preprocessor import StrokePreprocessor
from scale_strokes import rescale_strokes
from stroke2img import strokes_to_image, save_as_bmp


@dataclass
class ConversionStats:
    """Statistics for the conversion process."""
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = 0
    end_time: float = 0
    
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0


class InkMLBatchConverter:
    """Batch converter for inkML files to BMP images."""
    
    def __init__(self, stroke_length: int = 40, image_size: Tuple[int, int] = (400, 400), 
                 line_thickness: int = 2, padding: int = 2):
        """
        Initialize the batch converter.
        
        Args:
            stroke_length: Target stroke length for rescaling (pixels)
            image_size: Output image size as (width, height)
            line_thickness: Thickness of stroke lines
            padding: Padding around strokes in pixels
        """
        self.stroke_length = stroke_length
        self.image_size = image_size
        self.line_thickness = line_thickness
        self.padding = padding
        self.preprocessor = StrokePreprocessor()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def find_inkml_files(self, input_dir: Path) -> List[Path]:
        """
        Find all inkML files in the input directory recursively.
        
        Args:
            input_dir: Input directory path
            
        Returns:
            List of inkML file paths
        """
        inkml_files = []
        for pattern in ['*.inkml', '*.InkML']:
            inkml_files.extend(input_dir.rglob(pattern))
        return sorted(inkml_files)
    
    def convert_single_file(self, inkml_path: Path, output_dir: Path, 
                          preserve_structure: bool = True) -> Tuple[bool, str]:
        """
        Convert a single inkML file to BMP image.
        
        Args:
            inkml_path: Path to inkML file
            output_dir: Output directory path
            preserve_structure: Whether to preserve directory structure
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Create fresh preprocessor for each file to avoid any state issues
            preprocessor = StrokePreprocessor()
            
            # Parse inkML file to get strokes and label
            label, strokes_array = preprocessor.parse_inkml_strokes(inkml_path)
            
            if strokes_array is None or len(strokes_array) == 0:
                return False, f"No stroke data found in {inkml_path.name}"
            
            # Convert numpy array back to list of strokes for compatibility
            strokes_list = []
            unique_stroke_ids = sorted(set(strokes_array[:, 2]))
            
            for stroke_id in unique_stroke_ids:
                mask = strokes_array[:, 2] == stroke_id
                stroke_points = strokes_array[mask][:, :2]  # Get x, y coordinates
                strokes_list.append(stroke_points.tolist())
            
            if not strokes_list:
                return False, f"No valid strokes found in {inkml_path.name}"
            
            # Rescale strokes using the same approach as the server
            strokes_norm, size = rescale_strokes(strokes_list, self.stroke_length)
            
            # Calculate image size based on rescaled strokes
            calculated_size = (int(size[0]) + 4, int(size[1]) + 4)
            
            # Use either calculated size or configured size
            final_image_size = calculated_size if calculated_size[0] > 0 and calculated_size[1] > 0 else self.image_size
            
            # Convert strokes to image with black background (0) and white foreground (255)
            img = strokes_to_image(
                strokes_norm, 
                image_size=final_image_size, 
                line_thickness=self.line_thickness, 
                padding=self.padding,
                background_color=0,    # Black background
                stroke_color=255       # White strokes
            )
            
            # Determine output path
            if preserve_structure:
                # Preserve relative directory structure
                relative_path = inkml_path.relative_to(inkml_path.parents[0] if len(inkml_path.parts) == 1 else inkml_path.parents[-1])
                output_path = output_dir / relative_path.with_suffix('.bmp')
            else:
                # Flat structure
                output_path = output_dir / f"{inkml_path.stem}.bmp"
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            if save_as_bmp(img, str(output_path)):
                return True, f"Successfully converted {inkml_path.name} -> {output_path.name}"
            else:
                return False, f"Failed to save image for {inkml_path.name}"
                
        except Exception as e:
            return False, f"Error converting {inkml_path.name}: {str(e)}"
    
    def convert_batch(self, input_dir: Path, output_dir: Path, 
                     max_workers: int = 1, preserve_structure: bool = True) -> ConversionStats:
        """
        Convert all inkML files in a directory to BMP images.
        
        Args:
            input_dir: Input directory containing inkML files
            output_dir: Output directory for BMP images
            max_workers: Number of parallel workers (1 for sequential processing)
            preserve_structure: Whether to preserve directory structure
            
        Returns:
            ConversionStats object with processing statistics
        """
        stats = ConversionStats()
        stats.start_time = time.time()
        
        # Find all inkML files
        self.logger.info(f"Scanning for inkML files in {input_dir}")
        inkml_files = self.find_inkml_files(input_dir)
        stats.total_files = len(inkml_files)
        
        if stats.total_files == 0:
            self.logger.warning(f"No inkML files found in {input_dir}")
            stats.end_time = time.time()
            return stats
        
        self.logger.info(f"Found {stats.total_files} inkML files")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files
        if max_workers == 1:
            # Sequential processing
            for i, inkml_path in enumerate(inkml_files, 1):
                success, message = self.convert_single_file(inkml_path, output_dir, preserve_structure)
                
                if success:
                    stats.successful += 1
                    self.logger.debug(message)
                else:
                    stats.failed += 1
                    self.logger.error(message)
                
                # Progress update
                if i % 100 == 0 or i == stats.total_files:
                    self.logger.info(f"Progress: {i}/{stats.total_files} files processed "
                                   f"({stats.successful} successful, {stats.failed} failed)")
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.convert_single_file, inkml_path, output_dir, preserve_structure): inkml_path
                    for inkml_path in inkml_files
                }
                
                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_file):
                    inkml_path = future_to_file[future]
                    completed += 1
                    
                    try:
                        success, message = future.result()
                        if success:
                            stats.successful += 1
                            self.logger.debug(message)
                        else:
                            stats.failed += 1
                            self.logger.error(message)
                    except Exception as e:
                        stats.failed += 1
                        self.logger.error(f"Error processing {inkml_path.name}: {str(e)}")
                    
                    # Progress update
                    if completed % 100 == 0 or completed == stats.total_files:
                        self.logger.info(f"Progress: {completed}/{stats.total_files} files processed "
                                       f"({stats.successful} successful, {stats.failed} failed)")
        
        stats.end_time = time.time()
        return stats
    
    def print_summary(self, stats: ConversionStats):
        """Print conversion summary."""
        duration = stats.duration()
        rate = stats.total_files / duration if duration > 0 else 0
        
        print("\n" + "="*50)
        print("CONVERSION SUMMARY")
        print("="*50)
        print(f"Total files:     {stats.total_files}")
        print(f"Successful:      {stats.successful}")
        print(f"Failed:          {stats.failed}")
        print(f"Success rate:    {stats.successful/stats.total_files*100:.1f}%" if stats.total_files > 0 else "Success rate:    0%")
        print(f"Duration:        {duration:.2f} seconds")
        print(f"Processing rate: {rate:.1f} files/second")
        print("="*50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert inkML files to BMP images for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        required=True,
        help="Input directory containing inkML files"
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        required=True,
        help="Output directory for BMP images"
    )
    
    parser.add_argument(
        '--stroke-length', 
        type=int, 
        default=50,
        help="Target stroke length for rescaling (pixels)"
    )
    
    parser.add_argument(
        '--image-size', 
        type=str, 
        default="400x400",
        help="Default image size as WIDTHxHEIGHT (e.g., 400x400)"
    )
    
    parser.add_argument(
        '--line-thickness', 
        type=int, 
        default=2,
        help="Thickness of stroke lines in pixels"
    )
    
    parser.add_argument(
        '--padding', 
        type=int, 
        default=2,
        help="Padding around strokes in pixels"
    )
    
    parser.add_argument(
        '--parallel', '-p', 
        type=int, 
        default=1,
        help="Number of parallel workers (1 for sequential processing)"
    )
    
    parser.add_argument(
        '--preserve-structure', 
        action='store_true',
        help="Preserve directory structure in output"
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse image size
    try:
        width, height = map(int, args.image_size.split('x'))
        image_size = (width, height)
    except ValueError:
        print(f"Error: Invalid image size format '{args.image_size}'. Use WIDTHxHEIGHT (e.g., 400x400)")
        return 1
    
    # Validate paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 1
    
    if not input_dir.is_dir():
        print(f"Error: Input path '{input_dir}' is not a directory")
        return 1
    
    # Initialize converter
    converter = InkMLBatchConverter(
        stroke_length=args.stroke_length,
        image_size=image_size,
        line_thickness=args.line_thickness,
        padding=args.padding
    )
    
    print(f"Starting batch conversion...")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size:       {image_size[0]}x{image_size[1]} (default)")
    print(f"Stroke length:    {args.stroke_length} pixels")
    print(f"Line thickness:   {args.line_thickness} pixels")
    print(f"Padding:          {args.padding} pixels")
    print(f"Parallel workers: {args.parallel}")
    print(f"Preserve structure: {args.preserve_structure}")
    
    # Run conversion
    stats = converter.convert_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=args.parallel,
        preserve_structure=args.preserve_structure
    )
    
    # Print summary
    converter.print_summary(stats)
    
    # Return appropriate exit code
    return 0 if stats.failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())