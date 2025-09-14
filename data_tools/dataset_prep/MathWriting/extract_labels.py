#!/usr/bin/env python3
"""
InkML Label Extractor for MathWriting Dataset

This tool extracts LaTeX labels from InkML files in the MathWriting dataset
and outputs them in a format compatible with the SAN training pipeline.

Usage:
    python3 extract_labels.py --input /path/to/inkml/files --output labels.txt
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Add the stroke processing directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'stroke_processing'))

from stroke_preprocessor import StrokePreprocessor


@dataclass
class ExtractionStats:
    """Statistics for the label extraction process."""
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    no_label: int = 0
    start_time: float = 0
    end_time: float = 0

    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0


class MathWritingLabelExtractor:
    """Extract LaTeX labels from MathWriting dataset InkML files."""

    def __init__(self, output_extension: str = ".bmp"):
        """
        Initialize the label extractor.

        Args:
            output_extension: File extension to use in output (e.g., .bmp, .jpg)
        """
        self.output_extension = output_extension
        self.preprocessor = StrokePreprocessor()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def find_inkml_files(self, input_dir: Path) -> List[Path]:
        """
        Find all InkML files in the input directory.

        Args:
            input_dir: Input directory path

        Returns:
            List of InkML file paths
        """
        inkml_files = []
        for pattern in ['*.inkml', '*.InkML']:
            inkml_files.extend(input_dir.glob(pattern))
        return sorted(inkml_files)

    def extract_single_label(self, inkml_path: Path) -> Tuple[bool, str, str]:
        """
        Extract label from a single InkML file.

        Args:
            inkml_path: Path to InkML file

        Returns:
            Tuple of (success, filename_with_extension, label)
        """
        try:
            # Use existing preprocessor to parse the file
            label, strokes_array = self.preprocessor.parse_inkml_strokes(inkml_path)

            if label is None:
                return False, "", f"No label found in {inkml_path.name}"

            # Create output filename with desired extension
            output_filename = inkml_path.stem + self.output_extension

            # Clean up label (remove extra whitespace)
            clean_label = label.strip()

            return True, output_filename, clean_label

        except Exception as e:
            return False, "", f"Error extracting label from {inkml_path.name}: {str(e)}"

    def extract_batch(self, input_dir: Path, output_file: Path,
                     max_workers: int = 1) -> ExtractionStats:
        """
        Extract labels from all InkML files in a directory.

        Args:
            input_dir: Input directory containing InkML files
            output_file: Output file path for labels
            max_workers: Number of parallel workers (1 for sequential processing)

        Returns:
            ExtractionStats object with processing statistics
        """
        stats = ExtractionStats()
        stats.start_time = time.time()

        # Find all InkML files
        self.logger.info(f"Scanning for InkML files in {input_dir}")
        inkml_files = self.find_inkml_files(input_dir)
        stats.total_files = len(inkml_files)

        if stats.total_files == 0:
            self.logger.warning(f"No InkML files found in {input_dir}")
            stats.end_time = time.time()
            return stats

        self.logger.info(f"Found {stats.total_files} InkML files")

        # Store results for writing to file
        labels_data: Dict[str, str] = {}

        # Process files
        if max_workers == 1:
            # Sequential processing
            for i, inkml_path in enumerate(inkml_files, 1):
                success, filename, result = self.extract_single_label(inkml_path)

                if success:
                    stats.successful += 1
                    labels_data[filename] = result
                    self.logger.debug(f"Extracted label from {inkml_path.name}: {result}")
                else:
                    if "No label found" in result:
                        stats.no_label += 1
                    else:
                        stats.failed += 1
                    self.logger.error(result)

                # Progress update
                if i % 1000 == 0 or i == stats.total_files:
                    self.logger.info(f"Progress: {i}/{stats.total_files} files processed "
                                   f"({stats.successful} successful, {stats.failed} failed, {stats.no_label} no label)")
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.extract_single_label, inkml_path): inkml_path
                    for inkml_path in inkml_files
                }

                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_file):
                    inkml_path = future_to_file[future]
                    completed += 1

                    try:
                        success, filename, result = future.result()
                        if success:
                            stats.successful += 1
                            labels_data[filename] = result
                            self.logger.debug(f"Extracted label from {inkml_path.name}: {result}")
                        else:
                            if "No label found" in result:
                                stats.no_label += 1
                            else:
                                stats.failed += 1
                            self.logger.error(result)
                    except Exception as e:
                        stats.failed += 1
                        self.logger.error(f"Error processing {inkml_path.name}: {str(e)}")

                    # Progress update
                    if completed % 1000 == 0 or completed == stats.total_files:
                        self.logger.info(f"Progress: {completed}/{stats.total_files} files processed "
                                       f"({stats.successful} successful, {stats.failed} failed, {stats.no_label} no label)")

        # Write labels to output file
        if labels_data:
            self.logger.info(f"Writing {len(labels_data)} labels to {output_file}")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                for filename in sorted(labels_data.keys()):
                    label = labels_data[filename]
                    f.write(f"{filename} {label}\n")

            self.logger.info(f"Successfully wrote labels to {output_file}")
        else:
            self.logger.warning("No labels extracted, output file not created")

        stats.end_time = time.time()
        return stats

    def print_summary(self, stats: ExtractionStats):
        """Print extraction summary."""
        duration = stats.duration()
        rate = stats.total_files / duration if duration > 0 else 0

        print("\n" + "="*50)
        print("LABEL EXTRACTION SUMMARY")
        print("="*50)
        print(f"Total files:     {stats.total_files}")
        print(f"Successful:      {stats.successful}")
        print(f"Failed:          {stats.failed}")
        print(f"No label:        {stats.no_label}")
        print(f"Success rate:    {stats.successful/stats.total_files*100:.1f}%" if stats.total_files > 0 else "Success rate:    0%")
        print(f"Duration:        {duration:.2f} seconds")
        print(f"Processing rate: {rate:.1f} files/second")
        print("="*50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract LaTeX labels from MathWriting dataset InkML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="Input directory containing InkML files"
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help="Output file for labels (e.g., labels.txt)"
    )

    parser.add_argument(
        '--extension', '-e',
        type=str,
        default=".bmp",
        help="File extension to use in output (e.g., .bmp, .jpg)"
    )

    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=1,
        help="Number of parallel workers (1 for sequential processing)"
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

    # Validate paths
    input_dir = Path(args.input)
    output_file = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 1

    if not input_dir.is_dir():
        print(f"Error: Input path '{input_dir}' is not a directory")
        return 1

    # Validate extension format
    extension = args.extension
    if not extension.startswith('.'):
        extension = '.' + extension

    # Initialize extractor
    extractor = MathWritingLabelExtractor(output_extension=extension)

    print(f"Starting label extraction...")
    print(f"Input directory:    {input_dir}")
    print(f"Output file:        {output_file}")
    print(f"Output extension:   {extension}")
    print(f"Parallel workers:   {args.parallel}")

    # Run extraction
    stats = extractor.extract_batch(
        input_dir=input_dir,
        output_file=output_file,
        max_workers=args.parallel
    )

    # Print summary
    extractor.print_summary(stats)

    # Return appropriate exit code
    return 0 if stats.failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())