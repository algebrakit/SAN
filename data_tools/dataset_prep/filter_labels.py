#!/usr/bin/env python3
"""
Filter script for removing unsupported LaTeX expressions from label files.
Removes lines with matrices and strips font style commands from expressions.
"""

import re
import sys

from LatexParser.latex_normalizer import normalize_latex, LaTeXError

def should_skip_line(latex_str: str) -> bool:
    """
    Check if a line should be skipped entirely (e.g., contains matrices).

    Args:
        latex_str: The LaTeX expression string

    Returns:
        True if the line should be skipped
    """
    matrix_list = ['\\begin{matrix}', '\\begin{pmatrix}', '\\begin{bmatrix}', '\\begin{Bmatrix}',
                   '\\begin{vmatrix}', '\\begin{Vmatrix}', '\\begin{array}']
    command_list = [
        '\\binom', '\\tbinom', '\\choose', '\\atop', '\\brace', '\\brack', '\\cfrac','\\limits',
        '\\aleph','\\cong','\\oplus','\\mapsto', '\\bot', '\\vdash', '\\lnot', '\\models', '\\doteq', '*',
        '\\bigcap', '\\bigcup', '\\biguplus', '\\bigwedge', '\\bigvee', '\\coprod', '\\bigoplus', '\\bigcirc',
        '\\propto', '\\setminus', '\\langle', '\\rangle','\\Z', '\\R', '\\N', '\\Im', '\\Re', '\\wp', '\\Lambda',
        ';', '\\xi', '\\zeta', '\\mp', '\\dagger', '\\star', '\\simeq', '\\bullet', '\\oint', '\\ominus', '\\mathfrak'
        ]
    accents_list = ['\\vec', '\\dot', '\\ddot', '\\tilde', '\\hat', '\\bar', '\\breve', '\\acute', '\\grave', 
                    '\\mathring', '\\underline', '\\overline', '\\widehat', '\\widetilde','\\odot','\\hbar']
    
    filter_list = matrix_list + command_list + accents_list

    return any(cmd in latex_str for cmd in filter_list)


def remove_font_commands(latex_str: str) -> str:
    """
    Remove font style commands but keep their content.

    Args:
        latex_str: The LaTeX expression string

    Returns:
        LaTeX string with font commands removed
    """
    # Font style commands to remove
    font_commands = [r'\\boldsymbol', r'\\mathbf', r'\\mathrm', r'\\mathbb', r'\\operatorname', r'\\boldsymbol',
                     r'\\textstyle', r'\\scriptstyle', r'\\mbox']

    result = latex_str
    for cmd in font_commands:
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
    replacements = {
        r'\\varsigma': r'\\sigma',
        r'\\vartheta': r'\\theta',
        r'\\varepsilon': r'\\epsilon',
        r'\\varphi': r'\\phi',
        r'\\varpi': r'\\pi',
        r'\\varrho': r'\\rho',
        r'\\varnothing': r'\\emptyset',
        r'\\kappa': r'k',
        r'\\Upsilon': r'Y',
        r'\\neq': r'\\ne',
        r'\\ell': r'l',
        r'\\nu': r'v',
        r'\\eta': r'n',
        r'\\chi': r'x',
        r'\\iint': r'\\int\\int',
        r'\\prime': "'",
        r'\\big': r'',
        r'\\bigl': r'',
        r'\\bigr': r'',
        r'\\Big': r'',
        r'\\Bigl': r'',
        r'\\Bigr': r'',
        r'\\bigg': r'',
        r'\\biggl': r'',
        r'\\biggr': r'',
        r'\\Bigg': r'',
        r'\\tfrac': r'\\frac',
        r'\\to': r'\\rightarrow',
        r'\\,': r' ',
        r'\\;': r' ',
        r'\\:': r' ',
        r'\\>': r' ',
        r'\\!': r' ',
        r'\\ ': r' ',
        r'~': r' ',

    }

    result = latex_str
    for variant, standard in replacements.items():
        if variant[-1].isalpha():            
            # Add negative lookahead to prevent matching within longer commands
            pattern = variant + r'(?![a-zA-Z])'
        else:
            pattern = variant
        # add space to prevent concatenation with next token .e.g \scriptstyle\mathbf{E} --> \scriptstyleE
        result = re.sub(pattern, standard+' ', result)

    return result

def detect_commands(latex_str: str) -> str:
    """
    Detect LaTeX commands in the string.
    E.g. 2\cdot log_2(n) -> 2\cdot\log_2(n)

    Args:
        latex_str: The LaTeX expression string

    """
    cmd_list = ['log', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh',
                'tanh', 'coth', 'ln', 'exp', 'sum', 'prod', 'lim', 'max', 'min', 'inf', 'sup', 'det', 'dim', 'gcd', 'lcm',
                'mod', 'arg', 'div']
    for cmd in cmd_list:
        pattern = r'(?<![\\a-zA-Z])' + cmd + r'(?![a-zA-Z])'
        latex_str = re.sub(pattern, r'\\' + cmd + ' ', latex_str)

    return latex_str

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

                # Normalize LaTeX expression with error handling
                try:
                    latex_expr = normalize_latex(latex_expr)
                except LaTeXError as e:
                    skipped_count += 1
                    print(f"Skipped line {line_num}: normalization error - {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    skipped_count += 1
                    print(f"Skipped line {line_num}: unexpected normalization error - {e}", file=sys.stderr)
                    continue

                # Skip lines with matrices
                if should_skip_line(latex_expr):
                    skipped_count += 1
                    print(f"Skipped line {line_num}: contains matrix", file=sys.stderr)
                    continue

                # Remove font style commands and replace variant symbols
                filtered_latex = remove_font_commands(latex_expr)
                filtered_latex = replace_variant_symbols(filtered_latex)
                filtered_latex = detect_commands(filtered_latex)
                filtered_latex = filtered_latex.strip()
                # Write the result
                outfile.write(f"{filename}\t{filtered_latex}\n")
                processed_count += 1

    print(f"Processed {processed_count} lines, skipped {skipped_count} lines with matrices", file=sys.stderr)


def main():
    if len(sys.argv) != 3:
        print("Usage: python filter_labels.py <input_file> <output_file>")
        print("Example: python filter_labels.py labels.txt labels_filtered.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        process_file(input_file, output_file)
        print(f"Successfully filtered {input_file} -> {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()