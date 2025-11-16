#!/usr/bin/env python3
"""
Filter script for removing unsupported LaTeX expressions from label files.
Removes lines with matrices and strips font style commands from expressions.
"""

import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from LatexNormalizer.latex_normalizer import normalize_latex, LaTeXError
from utils.Expression import Expression

def should_skip_line(latex_str: str) -> bool:
    """
    Check if a line should be skipped entirely because of forbidden symbols.

    Args:
        latex_str: The LaTeX expression string

    Returns:
        True if the line should be skipped
    """
    forbidden_command_list = [
        '\\limits', '\\aleph','\\oplus', '\\models', '\\biguplus', '\\bigwedge', '\\bigvee', '\\coprod', 
        '\\bigoplus', '\\propto', '\\Im', '\\Re', '\\wp', '\\xi', '\\zeta', '\\Xi', '\\iota', '\\mp', '\\dagger', '\\star', '\\bullet', 
        '\\oint', '\\ominus', '\\mathfrak','\\odot','\\hbar','\\triangleleft','\\triangleq','\\triangleleft',
        '\\supseteq','\\subsetneq','\\sqsubseteq','\\rightleftharpoons', '\\Vdash','\\lg','\\pmod','\\tbinom',
        # '\\\\', '\\choose', # to handle later
        # forbidden accents
        '\\breve', '\\acute', '\\grave', '\\mathring'   
        ]
    return any(cmd in latex_str for cmd in forbidden_command_list)


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
                     r'\\mathtt',r'\\mathsf', r'\\bold',
                     r'\\textstyle', r'\\scriptstyle', r'\\scriptscriptstyle', r'\\mbox']

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
        r'\\kappa': r'k',
        r'\\Upsilon': r'Y',
        r'\\upsilon': r'v',
        r'\\Pi': r'\\prod',
        r'\\Sigma': r'\\sum',
        r'\\neq': r'\\ne',
        r'\\varnothing': r'\\emptyset',
        r'\\backslash': r'\\emptyset',
        r'\\lnot': r'\\neg',
        r'\\mapsto': r'\\rightarrow',
        r'\\cong': r'\\simeq',
        r'\\bigcirc': r'\\circ',
        r'\\smallsetminus': r'\\setminus',
        r'\\ell': r'l',
        r'\\nu': r'v',
        r'\\eta': r'n',
        r'\\chi': r'x',
        r'\\iint': r'\\int\\int',
        r'\\ll': r'< < ',
        r'\\gg': r'> > ',
        r'\\bar': r'\\overline',
        r'\\vec': r'\\overrightarrow',
        r'\\widehat': r'\\hat',
        r'\\widetilde': r'\\tilde',
        r'\\rVert': r'\\Vert',
        r'\\lVert': r'\\Vert',
        r'\\parallel': r'\\Vert',
        r'\\mid': r'| ',
        r'\\vert': r'| ',
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
        r'\\vee': r'\\lor',
        r'\\wedge': r'\\land',
        r'\\tfrac': r'\\frac',
        r'\\dfrac': r'\\frac',
        r'\\cfrac': r'\\frac',
        r'\\dbinom': r'\\binom',
        r'\\tbinom': r'\\binom',
        r'\\bmod': r'\\mod',
        r'\\hookrightarrow': r'\\rightarrow',
        r'\\longrightarrow': r'\\rightarrow',
        r'\\to': r'\\rightarrow',
        r'\\gets': r'\\leftarrow',
        r'\\iff': r'\\Leftrightarrow',
        r'\\lbrack': r'[',
        r'\\rbrack': r']',
        r'\\lbrace': r'\{',
        r'\\rbrace': r'\}',
        r'\\dots': r'. . . ',
        r'\\cdots': r'. . . ',
        r'\\ldots': r'. . . ',
        r'\\dotsb': r'. . . ',
        r'\\dotsc': r'. . . ',
        r'\\colon': r': ',
        r'\\,': r'\\ ',
        r'\\;': r'\\ ',
        r'\\:': r'\\ ',
        r'\\>': r'\\ ',
        r'\\!': r'',
        r'(?<!\\)\\ ': r'\\ ',
        r'<': r'\\lt',
        r'>': r'\\gt',
        r'~': r' ',
        r'\\degree': r'^ { o }',

    }

    # first get all keys in order from longest to shortest to avoid partial replacements
    sorted_keys = sorted(replacements.keys(), key=len, reverse=True)

    result = latex_str
    for key in sorted_keys:

        if key[-1].isalpha():            
            # Add negative lookahead to prevent matching within longer commands
            pattern = key + r'(?![a-zA-Z])'
        else:
            pattern = key
        value = replacements[key] + ' '
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
    cmd_list = ['log', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh',
                'tanh', 'coth', 'ln', 'exp', 'sum', 'prod', 'lim', 'max', 'min', 'inf', 'sup', 'det', 'dim', 'gcd', 'lcm',
                'mod', 'arg', 'div','alpha','beta','gamma','delta','epsilon','theta','pi','rho','sigma','tau','phi','omega',
                'Gamma','Delta','Theta','Lambda','Sigma','Phi','Omega','over']
    for cmd in cmd_list:
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

    # # handle binom
    # res = re.sub(r'\\binom\{([^}]+)\}\{([^}]+)\}', r'(\\stack{\1 \\\\ \2})', res)

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
                    print(f"Skipped line {line_num}: normalization error - {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    skipped_count += 1
                    print(f"Skipped line {line_num}: unexpected normalization error - {e}", file=sys.stderr)
                    continue

                # Skip lines with forbidden symbols
                if should_skip_line(filtered_latex):
                    skipped_count += 1
                    print(f"Skipped line {line_num}: contains forbidden symbol", file=sys.stderr)
                    continue

                # Step 6: Detect commands
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