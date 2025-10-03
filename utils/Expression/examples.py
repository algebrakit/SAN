"""Example usage and tests for the Expression package."""

from .expression import Expression
from .constructs import Symbol, FractionConstruct, SqrtConstruct, AboveBelowConstruct


def main():
    """Run example expressions."""

    # Example 1: \\frac{\\sqrt[2]{x}+1}{1-x^2}
    print("Example 1: Fraction with square root")
    expr1 = Expression([
        FractionConstruct(
            construct_type='\\frac',
            above=Expression([
                SqrtConstruct(
                    construct_type='\\sqrt',
                    inside=Expression([Symbol('x')]),
                    l_sup=Expression([Symbol('2')])
                ),
                Symbol('+'),
                Symbol('1')
            ]),
            below=Expression([
                Symbol('1'),
                Symbol('-'),
                Symbol('x', sup=Expression([Symbol('2')]))
            ])
        )
    ])
    print(f"Direct construction: {expr1.toLatex()}")

    # Parse from LaTeX
    latex1 = r'\frac { \sqrt [ 2 ] { x } + 1 } { 1 - x ^ { 2 } }'
    parsed1 = Expression.fromLatex(latex1)
    print(f"Parsed from LaTeX:   {parsed1.toLatex()}")
    print()

    # Example 2: \\sum_{i=1}^{n} x_i^2
    print("Example 2: Summation")
    sum_expr = Expression([
        AboveBelowConstruct(
            construct_type="\\sum",
            below=Expression([
                Symbol('i'),
                Symbol('='),
                Symbol('1')
            ]),
            above=Expression([Symbol('n')])
        ),
        Symbol('x', sub=Expression([Symbol('i')]), sup=Expression([Symbol('2')]))
    ])
    print(f"Output: {sum_expr.toLatex()}")
    print()

    # Example 3: \\prod_{k=1}^{m} a_k
    print("Example 3: Product")
    prod_expr = Expression([
        AboveBelowConstruct(
            construct_type="\\prod",
            below=Expression([
                Symbol('k'),
                Symbol('='),
                Symbol('1')
            ]),
            above=Expression([Symbol('m')])
        ),
        Symbol('a', sub=Expression([Symbol('k')]))
    ])
    print(f"Output: {prod_expr.toLatex()}")
    print()

    # Example 4: \\int_{0}^{\\infty} f(x) dx
    print("Example 4: Integral")
    integral_expr = Expression([
        AboveBelowConstruct(
            construct_type="\\int",
            below=Expression([Symbol('0')]),
            above=Expression([Symbol('\\infty')])
        ),
        Symbol('f'),
        Symbol('('),
        Symbol('x'),
        Symbol(')'),
        Symbol('d'),
        Symbol('x')
    ])
    print(f"Output: {integral_expr.toLatex()}")
    print()

    # Example 5: Parse various LaTeX expressions
    print("Example 5: Parsing various LaTeX expressions")
    test_cases = [
        r'x ^ { 2 } + y ^ { 2 }',
        r'\sqrt { a ^ { 2 } + b ^ { 2 } }',
        r'\frac { 1 } { 2 }',
        r'a _ { i j } ^ { k }',
    ]

    for latex in test_cases:
        parsed = Expression.fromLatex(latex)
        if parsed:
            print(f"  {latex} -> {parsed.toLatex()}")
        else:
            print(f"  {latex} -> Failed to parse")


if __name__ == "__main__":
    main()
