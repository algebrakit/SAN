#!/usr/bin/env python3
"""
LaTeX tokenizer script for preparing training labels.
Converts LaTeX expressions into space-separated tokens.
"""

import re
import sys
import argparse
import logging
import os
from typing import List, Optional, Set

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_tools.dataset_prep.config import ALWAYS_VALID_TOKENS

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

def tokenize_latex(latex_str: str) -> Optional[List[str]]:
    """
    Tokenize a LaTeX expression into individual commands and symbols.
    Filters out font style commands.

    Args:
        latex_str: The LaTeX expression string

    Returns:
        List of tokens or None if invalid
    """
    tokens = []
    i = 0

    while i < len(latex_str):
        char = latex_str[i]

        # Skip whitespace
        if char.isspace():
            i += 1
            continue

        # Handle LaTeX commands (starting with backslash)
        if char == '\\':
            # Find the end of the command
            j = i + 1
            while j < len(latex_str) and latex_str[j].isalpha():
                j += 1

            if j > i + 1:  # We found a command
                command = latex_str[i:j]
                if not command in ['\\left', '\\right', '\\big', '\\Big', '\\bigg', '\\Bigg']:
                    tokens.append(command)
                i = j

            elif i < len(latex_str) - 1:
                two_char = latex_str[i:i+2]
                if two_char in ['\\\\', '\\{', '\\}', '\\$', '\\&', '\\%', '\\#', '\\_', '\\^', '\\~', '\\|', '\\ ']:
                    tokens.append(two_char)
                    i += 2
                else:
                    return None  # Invalid command
            else:  
                tokens.append(char)
                i += 1

        else:
            # Single character
            tokens.append(char)
            i += 1

    return tokens

def special_cases(latex_str: str) -> str:
    # x ^ {'}  --> x'
    # x ^ {''}  --> x''
    # x ^ {''' }  --> x'''
    latex_str = re.sub(r"\^\s*\{(\s*')+\s*\}", lambda m: "' " * (m.group(0).count("'")), latex_str)

    return latex_str

def process_file(input_file: str, output_file: str):
    """
    Process the input file and write tokenized output.

    Args:
        input_file: Path to input labels file
        output_file: Path to output file
        valid_words: Optional set of valid words to filter against
    """
    processed_count = 0
    skipped_count = 0
    
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

                    # Tokenize the LaTeX expression
                    tokens = tokenize_latex(latex_expr)
                    if tokens is None:
                        skipped_count += 1
                        logging.debug(f"Skipped line {line_num}: invalid tokens")
                        continue
                        
                    # Write the result
                    tokenized_latex = ' '.join(tokens)
                    tokenized_latex = special_cases(tokenized_latex).strip()
                    # replace any multiple spaces with a single space
                    tokenized_latex = re.sub(r'\s+', ' ', tokenized_latex)
                    outfile.write(f"{filename}\t{tokenized_latex}\n")
                    processed_count += 1

        logging.info(f"Processed {processed_count} lines")
        logging.info(f"Skipped {skipped_count} lines")

    except Exception as e:
        logging.error(f"Failed to process file: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Tokenize LaTeX labels file.")
    parser.add_argument("input_file", help="Path to input labels file")
    parser.add_argument("output_file", help="Path to output file")
    parser.add_argument("--word-file", help="Path to valid words file (optional)")
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