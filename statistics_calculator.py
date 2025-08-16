"""
Statistics Calculator Module

This module provides statistical analysis functions for symbol distribution data
in mathematical handwriting recognition datasets.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter


class StatisticsCalculator:
    """Calculator for statistical analysis of symbol distributions."""
    
    def __init__(self, symbol_counts: Counter):
        """
        Initialize with symbol count data.
        
        Args:
            symbol_counts (Counter): Counter object with symbol frequencies
        """
        self.symbol_counts = symbol_counts
        self.all_counts = list(symbol_counts.values())
        self.total_samples = sum(self.all_counts)
        self.num_symbols = len(self.all_counts)
    
    def calculate_percentiles(self, percentiles: List[int] = None) -> Dict[int, float]:
        """
        Calculate percentiles for sample counts.
        
        Args:
            percentiles (List[int]): List of percentile values to calculate
            
        Returns:
            Dict[int, float]: Dictionary mapping percentile to value
        """
        if percentiles is None:
            percentiles = [10, 25, 50, 75, 90, 95, 99]
        
        return {p: np.percentile(self.all_counts, p) for p in percentiles}
    
    def get_basic_statistics(self) -> Dict[str, float]:
        """
        Calculate basic statistical measures.
        
        Returns:
            Dict[str, float]: Dictionary with basic statistics
        """
        return {
            'mean': np.mean(self.all_counts),
            'median': np.median(self.all_counts),
            'std_deviation': np.std(self.all_counts),
            'variance': np.var(self.all_counts),
            'min': np.min(self.all_counts),
            'max': np.max(self.all_counts),
            'range': np.max(self.all_counts) - np.min(self.all_counts),
            'coefficient_of_variation': np.std(self.all_counts) / np.mean(self.all_counts)
        }
    
    def identify_rare_symbols(self, threshold: int = 5) -> Dict[str, int]:
        """
        Identify symbols with very few samples.
        
        Args:
            threshold (int): Maximum count to consider as rare
            
        Returns:
            Dict[str, int]: Dictionary of rare symbols and their counts
        """
        return {symbol: count for symbol, count in self.symbol_counts.items() 
                if count <= threshold}
    
    def identify_common_symbols(self, percentile: float = 95) -> Dict[str, int]:
        """
        Identify symbols with high sample counts.
        
        Args:
            percentile (float): Percentile threshold for common symbols
            
        Returns:
            Dict[str, int]: Dictionary of common symbols and their counts
        """
        threshold = np.percentile(self.all_counts, percentile)
        return {symbol: count for symbol, count in self.symbol_counts.items() 
                if count >= threshold}
    
    def calculate_distribution_metrics(self) -> Dict[str, any]:
        """
        Calculate distribution-specific metrics.
        
        Returns:
            Dict[str, any]: Dictionary with distribution metrics
        """
        # Calculate skewness (using scipy formula if available, else simple version)
        mean_val = np.mean(self.all_counts)
        std_val = np.std(self.all_counts)
        
        # Simple skewness calculation
        skewness = np.mean([(x - mean_val) / std_val for x in self.all_counts]) ** 3
        
        # Calculate entropy (information content)
        probabilities = np.array(self.all_counts) / self.total_samples
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small value to avoid log(0)
        
        # Gini coefficient (measure of inequality)
        sorted_counts = np.sort(self.all_counts)
        n = len(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        return {
            'skewness': skewness,
            'entropy': entropy,
            'gini_coefficient': gini,
            'concentration_ratio_top_10': self.calculate_concentration_ratio(10),
            'concentration_ratio_top_20': self.calculate_concentration_ratio(20)
        }
    
    def calculate_concentration_ratio(self, top_n: int) -> float:
        """
        Calculate concentration ratio for top N symbols.
        
        Args:
            top_n (int): Number of top symbols to consider
            
        Returns:
            float: Concentration ratio (0-1, where 1 means perfect concentration)
        """
        top_counts = [count for _, count in self.symbol_counts.most_common(top_n)]
        return sum(top_counts) / self.total_samples
    
    def get_cumulative_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate cumulative distribution of samples.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (ranks, cumulative_percentages)
        """
        sorted_counts = sorted(self.all_counts, reverse=True)
        cumulative_percentage = np.cumsum(sorted_counts) / self.total_samples * 100
        ranks = np.arange(1, len(sorted_counts) + 1)
        return ranks, cumulative_percentage
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a comprehensive summary DataFrame.
        
        Returns:
            pd.DataFrame: Summary statistics in tabular format
        """
        basic_stats = self.get_basic_statistics()
        distribution_metrics = self.calculate_distribution_metrics()
        rare_symbols = self.identify_rare_symbols()
        common_symbols = self.identify_common_symbols()
        
        summary_data = [
            ['Total unique symbols', self.num_symbols],
            ['Total samples', self.total_samples],
            ['Mean samples per symbol', basic_stats['mean']],
            ['Median samples per symbol', basic_stats['median']],
            ['Standard deviation', basic_stats['std_deviation']],
            ['Min samples per symbol', basic_stats['min']],
            ['Max samples per symbol', basic_stats['max']],
            ['Coefficient of variation', basic_stats['coefficient_of_variation']],
            ['Skewness', distribution_metrics['skewness']],
            ['Entropy (bits)', distribution_metrics['entropy']],
            ['Gini coefficient', distribution_metrics['gini_coefficient']],
            ['Symbols with ≤5 samples', len(rare_symbols)],
            ['Symbols with ≥95th percentile', len(common_symbols)],
            ['Top 10 concentration ratio', distribution_metrics['concentration_ratio_top_10']],
            ['Top 20 concentration ratio', distribution_metrics['concentration_ratio_top_20']]
        ]
        
        return pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    
    def print_detailed_statistics(self):
        """Print detailed statistical analysis."""
        print("\n=== DETAILED STATISTICS ===")
        
        # Basic statistics
        basic_stats = self.get_basic_statistics()
        print(f"\nBasic Statistics:")
        print(f"  Mean: {basic_stats['mean']:.2f}")
        print(f"  Median: {basic_stats['median']:.2f}")
        print(f"  Std Dev: {basic_stats['std_deviation']:.2f}")
        print(f"  Min: {basic_stats['min']:.0f}")
        print(f"  Max: {basic_stats['max']:.0f}")
        
        # Percentiles
        percentiles = self.calculate_percentiles()
        print("\nSample count percentiles:")
        for p, value in percentiles.items():
            print(f"  {p:2d}th percentile: {value:.1f} samples")
        
        # Distribution metrics
        dist_metrics = self.calculate_distribution_metrics()
        print(f"\nDistribution Metrics:")
        print(f"  Entropy: {dist_metrics['entropy']:.2f} bits")
        print(f"  Gini coefficient: {dist_metrics['gini_coefficient']:.3f}")
        print(f"  Top 10 concentration: {dist_metrics['concentration_ratio_top_10']:.1%}")
        print(f"  Top 20 concentration: {dist_metrics['concentration_ratio_top_20']:.1%}")
        
        # Data imbalance analysis
        rare_symbols = self.identify_rare_symbols()
        common_symbols = self.identify_common_symbols()
        print(f"\nData Balance Analysis:")
        print(f"  Symbols with ≤5 samples: {len(rare_symbols)} ({len(rare_symbols)/self.num_symbols*100:.1f}%)")
        print(f"  Symbols with ≥95th percentile: {len(common_symbols)}")