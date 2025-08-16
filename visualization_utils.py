"""
Visualization Utilities Module

This module provides plotting functions for visualizing symbol distribution data
in mathematical handwriting recognition datasets.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter


def plot_comprehensive_analysis(symbol_counts: Counter, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
    """
    Create a comprehensive 4-subplot visualization of symbol distribution.
    
    Args:
        symbol_counts (Counter): Counter object with symbol frequencies
        figsize (Tuple[int, int]): Figure size as (width, height)
        
    Returns:
        plt.Figure: The created figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Top 20 symbols bar chart
    top_20 = symbol_counts.most_common(20)
    symbols, counts = zip(*top_20)
    
    axes[0, 0].barh(range(len(symbols)), counts, color='skyblue')
    axes[0, 0].set_yticks(range(len(symbols)))
    axes[0, 0].set_yticklabels(symbols, fontsize=10)
    axes[0, 0].set_xlabel('Number of Samples')
    axes[0, 0].set_title('Top 20 Most Frequent Symbols')
    axes[0, 0].invert_yaxis()
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        axes[0, 0].text(count + max(counts)*0.01, i, str(count), 
                        va='center', fontsize=9)
    
    # 2. Distribution histogram
    all_counts = list(symbol_counts.values())
    axes[0, 1].hist(all_counts, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Number of Samples per Symbol')
    axes[0, 1].set_ylabel('Number of Symbols')
    axes[0, 1].set_title('Distribution of Sample Counts per Symbol')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    sorted_counts = sorted(all_counts, reverse=True)
    cumulative_percentage = np.cumsum(sorted_counts) / sum(sorted_counts) * 100
    axes[1, 0].plot(range(1, len(sorted_counts) + 1), cumulative_percentage, 
                    color='red', linewidth=2)
    axes[1, 0].set_xlabel('Symbol Rank (by frequency)')
    axes[1, 0].set_ylabel('Cumulative Percentage (%)')
    axes[1, 0].set_title('Cumulative Distribution of Samples')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Bottom 20 symbols
    bottom_20 = symbol_counts.most_common()[-20:]
    bottom_symbols, bottom_counts = zip(*bottom_20)
    
    axes[1, 1].barh(range(len(bottom_symbols)), bottom_counts, color='salmon')
    axes[1, 1].set_yticks(range(len(bottom_symbols)))
    axes[1, 1].set_yticklabels(bottom_symbols, fontsize=10)
    axes[1, 1].set_xlabel('Number of Samples')
    axes[1, 1].set_title('Bottom 20 Least Frequent Symbols')
    axes[1, 1].invert_yaxis()
    
    # Add count labels on bars
    for i, count in enumerate(bottom_counts):
        axes[1, 1].text(count + max(bottom_counts)*0.05, i, str(count), 
                        va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_top_symbols(symbol_counts: Counter, n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot horizontal bar chart of top N most frequent symbols.
    
    Args:
        symbol_counts (Counter): Counter object with symbol frequencies
        n (int): Number of top symbols to plot
        figsize (Tuple[int, int]): Figure size as (width, height)
        
    Returns:
        plt.Figure: The created figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    top_n = symbol_counts.most_common(n)
    symbols, counts = zip(*top_n)
    
    bars = ax.barh(range(len(symbols)), counts, color='skyblue')
    ax.set_yticks(range(len(symbols)))
    ax.set_yticklabels(symbols, fontsize=12)
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_title(f'Top {n} Most Frequent Symbols', fontsize=14)
    ax.invert_yaxis()
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        ax.text(count + max(counts)*0.01, i, str(count), 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_distribution_histogram(symbol_counts: Counter, bins: int = 30, 
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot histogram of sample count distribution.
    
    Args:
        symbol_counts (Counter): Counter object with symbol frequencies
        bins (int): Number of histogram bins
        figsize (Tuple[int, int]): Figure size as (width, height)
        
    Returns:
        plt.Figure: The created figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    all_counts = list(symbol_counts.values())
    
    n, bins_edges, patches = ax.hist(all_counts, bins=bins, color='lightgreen', 
                                   alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Samples per Symbol', fontsize=12)
    ax.set_ylabel('Number of Symbols', fontsize=12)
    ax.set_title('Distribution of Sample Counts per Symbol', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = np.mean(all_counts)
    median_val = np.median(all_counts)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_cumulative_distribution(symbol_counts: Counter, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot cumulative distribution of samples.
    
    Args:
        symbol_counts (Counter): Counter object with symbol frequencies
        figsize (Tuple[int, int]): Figure size as (width, height)
        
    Returns:
        plt.Figure: The created figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    all_counts = list(symbol_counts.values())
    sorted_counts = sorted(all_counts, reverse=True)
    cumulative_percentage = np.cumsum(sorted_counts) / sum(sorted_counts) * 100
    ranks = np.arange(1, len(sorted_counts) + 1)
    
    ax.plot(ranks, cumulative_percentage, color='red', linewidth=2)
    ax.set_xlabel('Symbol Rank (by frequency)', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title('Cumulative Distribution of Samples', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add reference lines
    ax.axhline(50, color='gray', linestyle='--', alpha=0.7, label='50%')
    ax.axhline(80, color='gray', linestyle='--', alpha=0.7, label='80%')
    ax.axhline(95, color='gray', linestyle='--', alpha=0.7, label='95%')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_pareto_chart(symbol_counts: Counter, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create a Pareto chart showing symbol frequencies and cumulative percentages.
    
    Args:
        symbol_counts (Counter): Counter object with symbol frequencies
        figsize (Tuple[int, int]): Figure size as (width, height)
        
    Returns:
        plt.Figure: The created figure object
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Get data
    symbols_counts = symbol_counts.most_common()
    symbols, counts = zip(*symbols_counts)
    
    # Calculate cumulative percentages
    total = sum(counts)
    cumulative_pct = np.cumsum(counts) / total * 100
    
    # Bar chart
    bars = ax1.bar(range(len(symbols)), counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Symbols (ranked by frequency)', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Line chart for cumulative percentage
    ax2 = ax1.twinx()
    line = ax2.plot(range(len(symbols)), cumulative_pct, color='red', 
                   marker='o', linewidth=2, markersize=3)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 100)
    
    # Add 80% line (Pareto principle)
    ax2.axhline(80, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='80%')
    ax2.legend()
    
    plt.title('Pareto Chart: Symbol Frequency Distribution', fontsize=14)
    plt.tight_layout()
    return fig


def save_all_plots(symbol_counts: Counter, output_dir: str = ".") -> List[str]:
    """
    Generate and save all visualization plots.
    
    Args:
        symbol_counts (Counter): Counter object with symbol frequencies
        output_dir (str): Directory to save plots
        
    Returns:
        List[str]: List of saved file paths
    """
    saved_files = []
    
    # Comprehensive analysis
    fig1 = plot_comprehensive_analysis(symbol_counts)
    path1 = f"{output_dir}/comprehensive_analysis.png"
    fig1.savefig(path1, dpi=300, bbox_inches='tight')
    saved_files.append(path1)
    plt.close(fig1)
    
    # Top symbols
    fig2 = plot_top_symbols(symbol_counts)
    path2 = f"{output_dir}/top_symbols.png"
    fig2.savefig(path2, dpi=300, bbox_inches='tight')
    saved_files.append(path2)
    plt.close(fig2)
    
    # Distribution histogram
    fig3 = plot_distribution_histogram(symbol_counts)
    path3 = f"{output_dir}/distribution_histogram.png"
    fig3.savefig(path3, dpi=300, bbox_inches='tight')
    saved_files.append(path3)
    plt.close(fig3)
    
    # Cumulative distribution
    fig4 = plot_cumulative_distribution(symbol_counts)
    path4 = f"{output_dir}/cumulative_distribution.png"
    fig4.savefig(path4, dpi=300, bbox_inches='tight')
    saved_files.append(path4)
    plt.close(fig4)
    
    # Pareto chart
    fig5 = plot_pareto_chart(symbol_counts)
    path5 = f"{output_dir}/pareto_chart.png"
    fig5.savefig(path5, dpi=300, bbox_inches='tight')
    saved_files.append(path5)
    plt.close(fig5)
    
    return saved_files