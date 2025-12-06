#!/usr/bin/env python3
"""
Filter script for removing unsupported LaTeX expressions from label files.
Removes lines with matrices and strips font style commands from expressions.
"""

import os
import re
import sys
import argparse
import logging
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_tools.dataset_prep.LatexNormalizer.latex_normalizer import normalize_latex, LaTeXError
from data_tools.dataset_prep.config import (
    FORBIDDEN_COMMANDS,
    FONT_COMMANDS,
    VARIANT_REPLACEMENTS,
    DETECT_COMMANDS
)

def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def should_skip_line(latex_str: str) -> bool:
    """
    Check if a line should be skipped entirely because of forbidden symbols.

    Args:
        latex_str: The LaTeX expression string

    Returns:
        True if the line should be skipped
    """
    return any(cmd in latex_str for cmd in FORBIDDEN_COMMANDS)


def remove_font_commands(latex_str: str) -> str:
    """
    Remove font style commands but keep their content.

    Args:
        latex_str: The LaTeX expression string

    Returns:
        LaTeX string with font commands removed
    """
    result = latex_str
    for cmd in FONT_COMMANDS:
        # Pattern to match command with braces: \cmd{content}
        pattern = cmd + r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'

        def replace_match(match):
            return ' '+match.group(1)+' '  # Return just the content inside braces

        result = re.sub(pattern, replace_match, result)

        # also handle without arguments, e.g. \scriptstyle x
        # Use regex to remove command followed by optional space
        pattern_standalone = cmd + r'(?![a-zA-Z])'
        result = re.sub(pattern_standalone, ' ', result)

    return result


def replace_variant_symbols(latex_str: str) -> str:
    """
    Replace variant symbols with their standard equivalents.

    Args:
        latex_str: The LaTeX expression string

    Returns:
        LaTeX string with variant symbols replaced
    """
    # first get all keys in order from longest to shortest to avoid partial replacements
    sorted_keys = sorted(VARIANT_REPLACEMENTS.keys(), key=len, reverse=True)

    result = latex_str
    for key in sorted_keys:

        if key[-1].isalpha():            
            # Add negative lookahead to prevent matching within longer commands
            pattern = key + r'(?![a-zA-Z])'
        else:
            pattern = key
        value = VARIANT_REPLACEMENTS[key] + ' '
        if(value[0].isalpha()): value = ' ' + value
        # add space to prevent concatenation with next token .e.g \scriptstyle\mathbf{E} --> \scriptstyleE
        result = re.sub(pattern, value, result)

    return result

def detect_commands(latex_str: str) -> str:
    """
    Detect LaTeX commands in the string.
    E.g. 2\\cdot log_2(n) -> 2\\cdot\\log_2(n)

    Args:
        latex_str: The LaTeX expression string

    """
    for cmd in DETECT_COMMANDS:
        pattern = r'(?<![\\a-zA-Z])' + cmd + r'(?![a-zA-Z])'
        latex_str = re.sub(pattern, r'\\' + cmd + ' ', latex_str)

    return latex_str

def handle_matrices(latex_expr: str) -> str:
    """
    Transform:
     -  \\begin{pmatrix}..\\..\\end{pmatrix} --> ( \\stack{.. \\ ...} )
     -  Similar for bmatrix, Bmatrix, vmatrix, Vmatrix
     -  \\begin{array}{ccc}..\\..\\end{array} --> \\stack{.. \\ ...}
     -  \\binom{n}{k} into (\\stack{a \\ b})
     -  \\begin{cases}...\\end{cases} into \\{\\stack{.. \\ ..}
    """
    
    # handle matrices
    res = latex_expr.replace("\\begin{pmatrix}", "(\\stack{")
    res = res.replace("\\begin{matrix}", "\\stack{")
    res = res.replace("\\begin{vmatrix}", "|\\stack{")
    res = res.replace("\\begin{Vmatrix}", "\\Vert\\stack{")
    res = res.replace("\\begin{bmatrix}", "[\\stack{")
    res = res.replace("\\begin{Bmatrix}", "\\{\\stack{")
    res = res.replace("\\begin{cases}", "\\{\\stack{")
    res = res.replace("\\end{matrix}", "}")
    res = res.replace("\\end{pmatrix}", "})")
    res = res.replace("\\end{bmatrix}", "}]")
    res = res.replace("\\end{Bmatrix}", "}\\}")
    res = res.replace("\\end{vmatrix}", "}|")
    res = res.replace("\\end{Vmatrix}", "}\\Vert")
    res = res.replace("\\end{cases}", "}")

    # handle array. Remove first argument block with alignment indicators
    pattern = r'(\\begin\{array\})\{[^\}]+\}'
    res = re.sub(pattern, r'\\stack{', res)
    res = res.replace("\\end{array}", "}")

    return res

def process_file(input_file: str, output_file: str):
    """
    Process the input file and write filtered output.
    Skips lines containing matrices and removes font style commands.

    Args:
        input_file: Path to input labels file
        output_file: Path to output file
    """
    skipped_count = 0
    processed_count = 0
    normalization_errors = 0
    forbidden_errors = 0

    logging.info(f"Processing {input_file} -> {output_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # Split filename and LaTeX expression
                    parts = line.split('\t', 1)
                    if len(parts) != 2:
                        # Try space separation as fallback
                        parts = line.split(' ', 1)

                    if len(parts) != 2:
                        logging.warning(f"Line {line_num} has unexpected format: {line}")
                        continue

                    filename, latex_expr = parts

                    # Step 1: Replace variant symbols and normalize spacing commands to '\ '
                    latex_expr = replace_variant_symbols(latex_expr)

                    # Step 2: Handle matrices
                    latex_expr = handle_matrices(latex_expr)

                    # Step 3: Remove font style commands (this may introduce spaces)
                    filtered_latex = remove_font_commands(latex_expr)

                    # Step 4: Normalize LaTeX expression with error handling
                    try:
                        filtered_latex = normalize_latex(filtered_latex)
                    except LaTeXError as e:
                        skipped_count += 1
                        normalization_errors += 1
                        logging.debug(f"Skipped line {line_num}: normalization error - {e}")
                        continue
                    except Exception as e:
                        skipped_count += 1
                        normalization_errors += 1
                        logging.error(f"Skipped line {line_num}: unexpected normalization error - {e}")
                        continue

                    # Skip lines with forbidden symbols
                    if should_skip_line(filtered_latex):
                        skipped_count += 1
                        forbidden_errors += 1
                        logging.debug(f"Skipped line {line_num}: contains forbidden symbol")
                        continue

                    # Step 6: Detect commands
                    filtered_latex = detect_commands(filtered_latex)
                    filtered_latex = filtered_latex.strip()
                    
                    # Write the result
                    outfile.write(f"{filename}\t{filtered_latex}\n")
                    processed_count += 1

        logging.info(f"Processed {processed_count} lines")
        logging.info(f"Skipped {skipped_count} lines ({normalization_errors} normalization errors, {forbidden_errors} forbidden symbols)")

    except Exception as e:
        logging.error(f"Failed to process file: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Filter LaTeX labels file.")
    parser.add_argument("input_file", help="Path to input labels file")
    parser.add_argument("output_file", help="Path to output file")
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()

    setup_logging(args.log_file, args.verbose)

    try:
        process_file(args.input_file, args.output_file)
    except FileNotFoundError:
        logging.error(f"Input file '{args.input_file}' not found")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()