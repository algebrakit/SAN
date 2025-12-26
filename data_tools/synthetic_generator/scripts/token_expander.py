"""Token expansion utilities for LaTeX processing.

This module handles the expansion of LaTeX tokens to match the glyphs
that appear in DVI output. Multi-character commands like \\cos are
expanded to individual character tokens ['c', 'o', 's'].
"""

from typing import List

from utils import tokenize_latex


# Function names that render as individual letters in DVI output
# NOTE: Only include built-in LaTeX function names here, NOT custom macros
FUNCTION_NAMES = {
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'sinh', 'cosh', 'tanh', 'coth',
    'arcsin', 'arccos', 'arctan',
    'log', 'ln', 'exp',
    'lim', 'sup', 'inf',
    'max', 'min',
    'det', 'dim', 'deg',
    'gcd', 'arg'
}

# Sizing commands that don't produce glyphs
SIZING_COMMANDS = {'left', 'right', 'big', 'Big', 'bigg', 'Bigg', 'bigl', 'bigr', 'Bigl', 'Bigr'}

# Custom macro expansions (defined in LaTeX preamble)
# These macros expand to other symbols that exist in the symbol library
MACRO_EXPANSIONS = {
    'degree': '\\circ',  # \degree is defined as ^\circ in preamble
}


def expand_latex_tokens(tokens: List[str]) -> List[str]:
    """
    Expand multi-character LaTeX commands to match actual DVI glyphs.

    This function handles the mismatch between LaTeX tokens and DVI glyphs:
    - Function names (\\cos) expand to individual letters ['c', 'o', 's']
    - Sizing commands (\\left, \\right) are removed (no glyphs)
    - Custom macros are expanded to their definitions

    Examples:
        \\cos -> ['c', 'o', 's']
        \\left( -> ['(']
        \\sin -> ['s', 'i', 'n']
        x -> ['x']
        \\degree -> ['\\circ']

    Args:
        tokens: List of LaTeX tokens from tokenize_latex()

    Returns:
        Expanded list of tokens matching DVI glyphs
    """
    expanded = []

    for token in tokens:
        # Handle LaTeX commands
        if token.startswith('\\'):
            cmd_name = token[1:]  # Remove backslash

            # Skip sizing commands (they don't produce glyphs)
            if cmd_name in SIZING_COMMANDS:
                continue

            # Expand custom macros
            if cmd_name in MACRO_EXPANSIONS:
                expanded.append(MACRO_EXPANSIONS[cmd_name])
            # Expand function names to individual letters
            elif cmd_name in FUNCTION_NAMES:
                expanded.extend(list(cmd_name))
            else:
                # Keep as-is (Greek letters, special symbols, etc.)
                expanded.append(token)
        else:
            # Regular characters, keep as-is
            expanded.append(token)

    return expanded


def tokenize_and_expand(latex: str) -> List[str]:
    """
    Tokenize LaTeX expression and expand to DVI glyph tokens.

    This is a convenience function that combines tokenize_latex()
    and expand_latex_tokens().

    Args:
        latex: LaTeX math expression string

    Returns:
        List of expanded tokens matching DVI glyphs
    """
    tokens = tokenize_latex(latex)
    return expand_latex_tokens(tokens)
