"""Examples of LaTeX to hybrid syntax conversion."""

import sys
import os

# Allow running as script or module
if __name__ == "__main__" and __package__ is None:
    # Running as script - add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from utils.Expression import Expression
else:
    # Running as module
    from .expression import Expression


def print_hybrid(latex: str, description: str = ""):
    """Parse LaTeX and print hybrid syntax."""
    if description:
        print(f"\n{description}")
    print(f"LaTeX: {latex}")
    print("Hybrid:")

    expr = Expression.fromLatex(latex)
    if expr:
        hybrid = expr.to_hybrid()
        for line in hybrid:
            print(' '.join(map(str, line)))
    else:
        print("Failed to parse")


def main():
    """Run examples from the documentation."""

    print("=" * 70)
    print("Hybrid Syntax Examples")
    print("=" * 70)

    # Example 1: Simple expression
    print_hybrid('9 + 5', 'Example 1: Simple expression')

    # Example 2: Fraction with right region
    print_hybrid(r'\frac { a + b } { c } + 4', 'Example 2: Fraction with right region')

    # Example 3: Square root without right region
    print_hybrid(r'\sqrt [ 2 ] { x }', 'Example 3: Square root without right region')

    # Example 4: Square root with right region
    print_hybrid(r'\sqrt [ 2 ] { x } + 1', 'Example 4: Square root with right region')

    # Example 5: Symbol with superscript
    print_hybrid(r'x ^ { 2 }', 'Example 5: Symbol with superscript')

    # Example 6: Symbol with subscript and superscript
    print_hybrid(r'x _ { i } ^ { 2 }', 'Example 6: Symbol with subscript and superscript')

    # Example 7: Sum with limits
    print_hybrid(r'\sum _ { i = 1 } ^ { n } x _ { i }', 'Example 7: Sum with limits')

    # Example 8: Product
    print_hybrid(r'\prod _ { k = 1 } ^ { m } a _ { k }', 'Example 8: Product')

    # Example 9: Nested fraction
    print_hybrid(r'\frac { \frac { 1 } { 2 } } { 3 }', 'Example 9: Nested fraction')

    # Example 10: Complex expression
    print_hybrid(
        r'\frac { \sqrt [ 2 ] { x } + 1 } { 1 - x ^ { 2 } }',
        'Example 10: Complex expression with fraction and sqrt'
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
