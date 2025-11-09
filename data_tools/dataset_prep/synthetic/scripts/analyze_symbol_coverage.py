#!/usr/bin/env python3
"""
Symbol Coverage Analysis

This script analyzes which LaTeX symbols from target expressions are covered
by the available handwritten symbol InkML library.

Usage:
    python3 analyze_symbol_coverage.py --index symbol_index.json --latex latex.txt --boxes boxes.jsonl
"""

import json
import argparse
import re
from pathlib import Path
from typing import Set, List, Dict, Tuple
from collections import Counter


class SymbolCoverageAnalyzer:
    """Analyzes coverage of LaTeX symbols by the InkML library."""

    def __init__(self, symbol_index_path: Path):
        """
        Initialize the coverage analyzer.

        Args:
            symbol_index_path: Path to symbol_index.json
        """
        self.symbol_index_path = Path(symbol_index_path)
        self.available_symbols: Set[str] = set()
        self.load_symbol_index()

    def load_symbol_index(self) -> None:
        """Load the symbol index and extract available symbols."""
        try:
            with open(self.symbol_index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            self.available_symbols = set(index_data['symbols'].keys())
            print(f"Loaded {len(self.available_symbols)} unique symbols from index")

        except Exception as e:
            print(f"Error loading symbol index: {e}")
            raise

    def tokenize_latex(self, latex_str: str) -> List[str]:
        """
        Tokenize a LaTeX string into individual symbols/commands.

        This is based on the tokenizer from mathwriting-book.ipynb.

        Args:
            latex_str: LaTeX expression string

        Returns:
            List of tokens
        """
        # Pattern matches LaTeX commands and single characters
        command_pattern = re.compile(
            r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)'
        )

        tokens = []
        s = latex_str

        while s:
            if s[0] == '\\':
                # Match LaTeX command
                match = command_pattern.match(s)
                if match:
                    tokens.append(match.group(0))
                    s = s[len(match.group(0)):]
                else:
                    # Couldn't match, just take the backslash
                    tokens.append(s[0])
                    s = s[1:]
            elif s[0] in '{}^_':
                # Skip structural characters (not symbols to render)
                s = s[1:]
            else:
                # Regular character
                tokens.append(s[0])
                s = s[1:]

        return tokens

    def analyze_latex_file(self, latex_file: Path) -> Tuple[Set[str], Counter]:
        """
        Analyze LaTeX expressions from a text file.

        Args:
            latex_file: Path to file with LaTeX expressions (one per line)

        Returns:
            Tuple of (unique_symbols, symbol_frequency_counter)
        """
        unique_symbols = set()
        symbol_counter = Counter()

        try:
            with open(latex_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                tokens = self.tokenize_latex(line)
                unique_symbols.update(tokens)
                symbol_counter.update(tokens)

            print(f"Analyzed {len(lines)} expressions from {latex_file.name}")

        except Exception as e:
            print(f"Error analyzing LaTeX file: {e}")

        return unique_symbols, symbol_counter

    def analyze_boxes_file(self, boxes_file: Path) -> Tuple[Set[str], Counter]:
        """
        Analyze symbols from a bounding boxes JSONL file.

        Args:
            boxes_file: Path to boxes.jsonl file

        Returns:
            Tuple of (unique_symbols, symbol_frequency_counter)
        """
        unique_symbols = set()
        symbol_counter = Counter()

        try:
            with open(boxes_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                if not line.strip():
                    continue

                data = json.loads(line)

                # Get tokens from bboxes
                for bbox in data.get('bboxes', []):
                    token = bbox.get('token', '')
                    if token:
                        unique_symbols.add(token)
                        symbol_counter[token] += 1

            print(f"Analyzed {len(lines)} expressions from {boxes_file.name}")

        except Exception as e:
            print(f"Error analyzing boxes file: {e}")

        return unique_symbols, symbol_counter

    def compute_coverage(
        self,
        required_symbols: Set[str],
        symbol_frequency: Counter
    ) -> Dict:
        """
        Compute coverage statistics.

        Args:
            required_symbols: Set of symbols needed
            symbol_frequency: Counter of symbol frequencies

        Returns:
            Dictionary with coverage statistics
        """
        covered = required_symbols & self.available_symbols
        missing = required_symbols - self.available_symbols

        # Count total occurrences
        total_occurrences = sum(symbol_frequency.values())
        covered_occurrences = sum(symbol_frequency[s] for s in covered)
        missing_occurrences = sum(symbol_frequency[s] for s in missing)

        return {
            "total_unique_symbols": len(required_symbols),
            "covered_symbols": len(covered),
            "missing_symbols": len(missing),
            "coverage_percentage": 100 * len(covered) / len(required_symbols) if required_symbols else 0,
            "total_occurrences": total_occurrences,
            "covered_occurrences": covered_occurrences,
            "missing_occurrences": missing_occurrences,
            "occurrence_coverage_percentage": 100 * covered_occurrences / total_occurrences if total_occurrences else 0,
            "covered_list": sorted(covered),
            "missing_list": sorted(missing),
        }

    def print_coverage_report(
        self,
        coverage: Dict,
        symbol_frequency: Counter,
        top_missing_n: int = 20
    ) -> None:
        """
        Print a detailed coverage report.

        Args:
            coverage: Coverage statistics dictionary
            symbol_frequency: Counter of symbol frequencies
            top_missing_n: Number of top missing symbols to show
        """
        print("\n" + "="*70)
        print("SYMBOL COVERAGE ANALYSIS")
        print("="*70)

        print("\nOVERALL COVERAGE:")
        print(f"  Total unique symbols needed:    {coverage['total_unique_symbols']}")
        print(f"  Symbols available in library:   {coverage['covered_symbols']}")
        print(f"  Symbols missing from library:   {coverage['missing_symbols']}")
        print(f"  Coverage by unique symbols:     {coverage['coverage_percentage']:.1f}%")

        print("\nOCCURRENCE-WEIGHTED COVERAGE:")
        print(f"  Total symbol occurrences:       {coverage['total_occurrences']}")
        print(f"  Covered occurrences:            {coverage['covered_occurrences']}")
        print(f"  Missing occurrences:            {coverage['missing_occurrences']}")
        print(f"  Coverage by occurrences:        {coverage['occurrence_coverage_percentage']:.1f}%")

        if coverage['missing_list']:
            print("\n" + "="*70)
            print(f"TOP {top_missing_n} MISSING SYMBOLS (by frequency)")
            print("="*70)

            # Get missing symbols sorted by frequency
            missing_with_freq = [
                (symbol, symbol_frequency[symbol])
                for symbol in coverage['missing_list']
            ]
            missing_with_freq.sort(key=lambda x: x[1], reverse=True)

            for i, (symbol, freq) in enumerate(missing_with_freq[:top_missing_n], 1):
                display_symbol = symbol if len(symbol) <= 30 else symbol[:27] + "..."
                print(f"{i:2d}. {display_symbol:30s} : {freq:4d} occurrences")

            if len(missing_with_freq) > top_missing_n:
                print(f"\n... and {len(missing_with_freq) - top_missing_n} more missing symbols")

        print("\n" + "="*70)

    def save_coverage_report(self, coverage: Dict, output_path: Path) -> None:
        """
        Save coverage report to a JSON file.

        Args:
            coverage: Coverage statistics
            output_path: Path to save report
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coverage, f, indent=2, ensure_ascii=False)

        print(f"\nCoverage report saved to: {output_path}")

    def save_markdown_report(
        self,
        coverage: Dict,
        symbol_frequency: Counter,
        output_path: Path
    ) -> None:
        """
        Save coverage report as a markdown file.

        Args:
            coverage: Coverage statistics
            symbol_frequency: Symbol frequency counter
            output_path: Path to save markdown report
        """
        lines = []

        lines.append("# Symbol Coverage Report\n")
        lines.append("## Summary\n")
        lines.append(f"- **Total unique symbols needed**: {coverage['total_unique_symbols']}")
        lines.append(f"- **Symbols available in library**: {coverage['covered_symbols']}")
        lines.append(f"- **Symbols missing from library**: {coverage['missing_symbols']}")
        lines.append(f"- **Coverage percentage**: {coverage['coverage_percentage']:.1f}%\n")

        lines.append("## Occurrence-Weighted Coverage\n")
        lines.append(f"- **Total occurrences**: {coverage['total_occurrences']}")
        lines.append(f"- **Covered occurrences**: {coverage['covered_occurrences']}")
        lines.append(f"- **Missing occurrences**: {coverage['missing_occurrences']}")
        lines.append(f"- **Occurrence coverage**: {coverage['occurrence_coverage_percentage']:.1f}%\n")

        if coverage['missing_list']:
            lines.append("## Missing Symbols\n")
            lines.append("| # | Symbol | Occurrences |")
            lines.append("|---|--------|-------------|")

            missing_with_freq = [
                (symbol, symbol_frequency[symbol])
                for symbol in coverage['missing_list']
            ]
            missing_with_freq.sort(key=lambda x: x[1], reverse=True)

            for i, (symbol, freq) in enumerate(missing_with_freq, 1):
                # Escape pipes in symbol for markdown table
                symbol_escaped = symbol.replace('|', '\\|')
                lines.append(f"| {i} | `{symbol_escaped}` | {freq} |")

            lines.append("")

        if coverage['covered_list']:
            lines.append("## Available Symbols\n")
            lines.append("The following symbols are available in the library:\n")
            lines.append("```")
            for symbol in coverage['covered_list']:
                lines.append(symbol)
            lines.append("```\n")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Markdown report saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze symbol coverage for synthesis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--index',
        type=str,
        default='symbol_index.json',
        help="Path to symbol index JSON file"
    )

    parser.add_argument(
        '--latex',
        type=str,
        help="Path to LaTeX expressions text file (one per line)"
    )

    parser.add_argument(
        '--boxes',
        type=str,
        help="Path to bounding boxes JSONL file"
    )

    parser.add_argument(
        '--output',
        type=str,
        default='coverage_report.json',
        help="Output JSON report path"
    )

    parser.add_argument(
        '--markdown',
        type=str,
        default='coverage_report.md',
        help="Output markdown report path"
    )

    args = parser.parse_args()

    # Validate that at least one input source is provided
    if not args.latex and not args.boxes:
        print("Error: Must provide at least one of --latex or --boxes")
        return 1

    # Load symbol index
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Error: Symbol index file '{index_path}' not found")
        return 1

    analyzer = SymbolCoverageAnalyzer(index_path)

    # Collect symbols from all sources
    all_symbols = set()
    all_frequencies = Counter()

    if args.latex:
        latex_path = Path(args.latex)
        if latex_path.exists():
            symbols, freq = analyzer.analyze_latex_file(latex_path)
            all_symbols.update(symbols)
            all_frequencies.update(freq)
        else:
            print(f"Warning: LaTeX file '{latex_path}' not found, skipping")

    if args.boxes:
        boxes_path = Path(args.boxes)
        if boxes_path.exists():
            symbols, freq = analyzer.analyze_boxes_file(boxes_path)
            all_symbols.update(symbols)
            all_frequencies.update(freq)
        else:
            print(f"Warning: Boxes file '{boxes_path}' not found, skipping")

    # Compute coverage
    coverage = analyzer.compute_coverage(all_symbols, all_frequencies)

    # Print report
    analyzer.print_coverage_report(coverage, all_frequencies)

    # Save reports
    analyzer.save_coverage_report(coverage, Path(args.output))
    analyzer.save_markdown_report(coverage, all_frequencies, Path(args.markdown))

    print("\n✓ Coverage analysis complete!")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
