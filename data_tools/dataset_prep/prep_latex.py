#!/usr/bin/env python3
"""
LaTeX tokenizer script for preparing training labels.
Converts LaTeX expressions into space-separated tokens.
"""

import re
import sys
from typing import List


def tokenize_latex(latex_str: str) -> List[str]:
    """
    Tokenize a LaTeX expression into individual commands and symbols.

    Args:
        latex_str: The LaTeX expression string

    Returns:
        List of tokens
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
                tokens.append(latex_str[i:j])
                i = j
            else:  # Just a backslash followed by non-alpha
                tokens.append(char)
                i += 1

        # Handle multi-character operators
        elif i < len(latex_str) - 1:
            two_char = latex_str[i:i+2]
            if two_char in ['\\\\', '\\{', '\\}', '\\$', '\\&', '\\%', '\\#', '\\_', '\\^', '\\~']:
                tokens.append(two_char)
                i += 2
            elif two_char in ['\\ne', '\\le', '\\ge', '\\ll', '\\gg', '\\pm', '\\mp', '\\to', '\\in', '\\ni']:
                # These should be handled by the backslash case above, but just in case
                tokens.append(two_char)
                i += 2
            else:
                tokens.append(char)
                i += 1
        else:
            # Single character
            tokens.append(char)
            i += 1

    return tokens


def process_file(input_file: str, output_file: str):
    """
    Process the input file and write tokenized output.

    Args:
        input_file: Path to input labels file
        output_file: Path to output file
    """
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
                    print(f"Warning: Line {line_num} has unexpected format: {line}", file=sys.stderr)
                    continue

                filename, latex_expr = parts

                # Tokenize the LaTeX expression
                tokens = tokenize_latex(latex_expr)

                # Write the result
                tokenized_latex = ' '.join(tokens)
                outfile.write(f"{filename}\t{tokenized_latex}\n")


def main():
    if len(sys.argv) != 3:
        print("Usage: python prep_latex.py <input_file> <output_file>")
        print("Example: python prep_latex.py labels.txt labels_tokenized.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        process_file(input_file, output_file)
        print(f"Successfully processed {input_file} -> {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()