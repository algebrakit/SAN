"""
Symbol Distribution Analyzer Module

This module provides functionality to analyze the distribution of mathematical symbols
across InkML files for handwriting recognition datasets.
"""

from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd

from inkml_parser import parse_inkml_label, get_inkml_files


class SymbolDistributionAnalyzer:
    """Analyzer for symbol distribution in InkML datasets."""
    
    def __init__(self, symbols_directory: Path):
        """
        Initialize the analyzer with a symbols directory.
        
        Args:
            symbols_directory (Path): Path to directory containing InkML files
        """
        self.symbols_dir = Path(symbols_directory)
        self.symbol_counts = None
        self.all_labels = None
        self.total_processed = 0
        self.total_files = 0
        self.errors = 0
    
    def analyze_distribution(self, verbose: bool = True) -> Tuple[Counter, List[str], int]:
        """
        Analyze the distribution of symbols across all InkML files.
        
        Args:
            verbose (bool): Whether to print progress information
            
        Returns:
            Tuple[Counter, List[str], int]: Symbol counts, all labels, files processed
        """
        if verbose:
            print("Parsing all InkML files...")
        
        # Get all InkML files
        inkml_files = get_inkml_files(self.symbols_dir)
        self.total_files = len(inkml_files)
        
        if verbose:
            print(f"Found {self.total_files} InkML files")
        
        # Extract labels from all files
        labels = []
        processed = 0
        errors = 0
        
        for file_path in inkml_files:
            label = parse_inkml_label(file_path)
            if label:
                labels.append(label)
                processed += 1
            else:
                errors += 1
                
            # Progress indicator
            if verbose and (processed + errors) % 1000 == 0:
                print(f"Processed {processed + errors}/{self.total_files} files...")
        
        if verbose:
            print(f"\nCompleted processing:")
            print(f"- Total files: {self.total_files}")
            print(f"- Successfully parsed: {processed}")
            print(f"- Errors: {errors}")
        
        # Store results
        self.symbol_counts = Counter(labels)
        self.all_labels = labels
        self.total_processed = processed
        self.errors = errors
        
        return self.symbol_counts, labels, processed
    
    def get_basic_stats(self) -> Dict[str, any]:
        """
        Get basic statistics about the symbol distribution.
        
        Returns:
            Dict[str, any]: Dictionary containing basic statistics
        """
        if self.symbol_counts is None:
            raise ValueError("Must run analyze_distribution() first")
        
        return {
            'total_unique_symbols': len(self.symbol_counts),
            'total_samples': sum(self.symbol_counts.values()),
            'average_samples_per_symbol': sum(self.symbol_counts.values()) / len(self.symbol_counts),
            'total_files': self.total_files,
            'successfully_parsed': self.total_processed,
            'errors': self.errors
        }
    
    def get_top_symbols(self, n: int = 20) -> List[Tuple[str, int]]:
        """
        Get the top N most frequent symbols.
        
        Args:
            n (int): Number of top symbols to return
            
        Returns:
            List[Tuple[str, int]]: List of (symbol, count) pairs
        """
        if self.symbol_counts is None:
            raise ValueError("Must run analyze_distribution() first")
        
        return self.symbol_counts.most_common(n)
    
    def get_bottom_symbols(self, n: int = 20) -> List[Tuple[str, int]]:
        """
        Get the bottom N least frequent symbols.
        
        Args:
            n (int): Number of bottom symbols to return
            
        Returns:
            List[Tuple[str, int]]: List of (symbol, count) pairs
        """
        if self.symbol_counts is None:
            raise ValueError("Must run analyze_distribution() first")
        
        return self.symbol_counts.most_common()[-n:]
    
    def get_symbols_with_few_samples(self, threshold: int = 5) -> Dict[str, int]:
        """
        Get symbols with sample count below threshold.
        
        Args:
            threshold (int): Maximum sample count to consider as "few"
            
        Returns:
            Dict[str, int]: Dictionary of symbol->count for rare symbols
        """
        if self.symbol_counts is None:
            raise ValueError("Must run analyze_distribution() first")
        
        return {symbol: count for symbol, count in self.symbol_counts.items() 
                if count <= threshold}
    
    def save_results_to_csv(self, output_file: str = "symbol_distribution_analysis.csv") -> str:
        """
        Save analysis results to CSV file.
        
        Args:
            output_file (str): Name of output CSV file
            
        Returns:
            str: Path to saved file
        """
        if self.symbol_counts is None:
            raise ValueError("Must run analyze_distribution() first")
        
        df_distribution = pd.DataFrame([
            {
                'Symbol': symbol, 
                'Count': count, 
                'Percentage': (count / self.total_processed) * 100
            }
            for symbol, count in self.symbol_counts.most_common()
        ])
        
        output_path = self.symbols_dir / output_file
        df_distribution.to_csv(output_path, index=False)
        return str(output_path)
    
    def print_summary(self):
        """Print a formatted summary of the analysis."""
        if self.symbol_counts is None:
            raise ValueError("Must run analyze_distribution() first")
        
        stats = self.get_basic_stats()
        
        print(f"\n=== SYMBOL DISTRIBUTION ANALYSIS ===")
        print(f"Total unique symbols: {stats['total_unique_symbols']}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Average samples per symbol: {stats['average_samples_per_symbol']:.1f}")
        
        print(f"\n=== TOP 20 MOST FREQUENT SYMBOLS ===")
        for symbol, count in self.get_top_symbols(20):
            percentage = (count / self.total_processed) * 100
            print(f"{symbol:15} : {count:4d} samples ({percentage:5.1f}%)")