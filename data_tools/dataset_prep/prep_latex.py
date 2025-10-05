#!/usr/bin/env python3
"""
LaTeX tokenizer script for preparing training labels.
Converts LaTeX expressions into space-separated tokens.
"""

import re
import sys
from typing import List



def tokenize_latex(latex_str: str) -> List[str] | None:
    """
    Tokenize a LaTeX expression into individual commands and symbols.
    Filters out font style commands.

    Args:
        latex_str: The LaTeX expression string

    Returns:
        List of tokens
    """
    # Font style commands to filter out (remove command but keep content)

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
                if two_char in ['\\\\', '\\{', '\\}', '\\$', '\\&', '\\%', '\\#', '\\_', '\\^', '\\~', '\\|']:
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

def filter_to_words(tokens: List[str], valid_words: set) -> bool:
    """
    Filter tokens to only include those in the valid words set.

    Args:
        tokens: List of tokens
        valid_words: Set of valid words
    """
    skip = False
    for token in tokens:
        if token not in valid_words:
            skip = True
            break
    return skip

def process_file(input_file: str, output_file: str, valid_words: set = None):
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
                if tokens is None:
                    continue
                if valid_words is not None:
                    skip = filter_to_words(tokens, valid_words)
                    if skip:
                        print(f"Skipping file {input_file} due to unknown tokens", file=sys.stderr)
                        continue

                # Write the result
                tokenized_latex = ' '.join(tokens)
                tokenized_latex = special_cases(tokenized_latex).strip()
                # replace any multiple spaces with a single space
                tokenized_latex = re.sub(r'\s+', ' ', tokenized_latex)
                outfile.write(f"{filename}\t{tokenized_latex}\n")


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python prep_latex.py <input_file> <output_file> <word_file>(optional)")
        print("Example: python prep_latex.py labels.txt labels_tokenized.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    if len(sys.argv) > 3:
        word_file = sys.argv[3]
        with open(word_file, 'r', encoding='utf-8') as wf:
            valid_words = set(line.strip() for line in wf if line.strip())
            valid_words.update(['{', '}', '[', ']', '^', '_'])  # Always allow these
    else:
        valid_words = None
    try:
        process_file(input_file, output_file, valid_words)
        print(f"Successfully processed {input_file} -> {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()