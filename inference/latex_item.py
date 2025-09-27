from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union


class LatexItem(ABC):
    """Abstract base class for all LaTeX mathematical expression items."""

    def __init__(self, parent: Optional['Expression'] = None):
        self.parent = parent

    @abstractmethod
    def toLatex(self) -> str:
        """Convert this item to its LaTeX string representation."""
        pass

    def get_children(self) -> Dict[str, 'Expression']:
        """Get all child Expression objects mapped by relation type."""
        return {}


class Expression:
    """Container for a sequence of LaTeX items."""

    def __init__(self, items: Optional[List[LatexItem]] = None, parent: Optional['LatexItem'] = None):
        self.items = items or []
        self.parent = parent
        # Set parent reference for all items
        for item in self.items:
            item.parent = self

    def add_item(self, item: LatexItem) -> None:
        """Add an item to this expression."""
        item.parent = self
        self.items.append(item)

    def toLatex(self) -> str:
        """Convert this expression to LaTeX by concatenating all items."""
        return ' '.join(item.toLatex() for item in self.items)

    def get_children(self) -> List[LatexItem]:
        """Get all items contained in this expression."""
        return self.items


class Symbol(LatexItem):
    """Represents a single symbol (letter, number, operator, etc.) with optional sub/superscripts."""

    def __init__(self,
                 value: str,
                 sub: Optional[Expression] = None,
                 sup: Optional[Expression] = None,
                 parent: Optional[Expression] = None):
        super().__init__(parent)
        self.value = value
        self.sub = sub
        self.sup = sup

        # Set parent references
        if self.sub:
            self.sub.parent = self
        if self.sup:
            self.sup.parent = self

    def toLatex(self) -> str:
        """Convert symbol to LaTeX with optional subscript and superscript."""
        result = self.value

        if self.sub:
            result += f" _ {{ {self.sub.toLatex()} }}"

        if self.sup:
            result += f" ^ {{ {self.sup.toLatex()} }}"

        return result

    def get_children(self) -> Dict[str, Expression]:
        """Get subscript and superscript expressions if they exist."""
        children = {}
        if self.sub:
            children['sub'] = self.sub
        if self.sup:
            children['sup'] = self.sup
        return children


class Construct(LatexItem):
    """Abstract base class for mathematical constructs like \\frac, \\sqrt, \\sum, etc."""

    def __init__(self,
                 sup: Optional[Expression] = None,
                 parent: Optional[Expression] = None):
        super().__init__(parent)
        self.sup = sup

        if self.sup:
            self.sup.parent = self

    @abstractmethod
    def toLatex(self) -> str:
        """Convert construct to LaTeX representation."""
        pass


class FractionConstruct(Construct):
    """Represents a fraction: \\frac{above}{below}"""

    def __init__(self,
                 above: Optional[Expression] = None,
                 below: Optional[Expression] = None,
                 sup: Optional[Expression] = None,
                 parent: Optional[Expression] = None):
        super().__init__(sup, parent)
        self.above = above
        self.below = below

        # Set parent references
        if self.above:
            self.above.parent = self
        if self.below:
            self.below.parent = self

    def toLatex(self) -> str:
        """Convert fraction to LaTeX: \\frac{numerator}{denominator}^{superscript}"""
        above_latex = self.above.toLatex() if self.above else ""
        below_latex = self.below.toLatex() if self.below else ""
        result = f"\\frac {{ {above_latex} }} {{ {below_latex} }}"

        if self.sup:
            result += f" ^ {{ {self.sup.toLatex()} }}"

        return result

    def get_children(self) -> Dict[str, Expression]:
        """Get numerator, denominator, and optional superscript."""
        children = {}
        if self.above:
            children['above'] = self.above
        if self.below:
            children['below'] = self.below
        if self.sup:
            children['sup'] = self.sup
        return children


class SqrtConstruct(Construct):
    """Represents a square root: \\sqrt[l_sup]{inside}"""

    def __init__(self,
                 inside: Optional[Expression] = None,
                 l_sup: Optional[Expression] = None,
                 sup: Optional[Expression] = None,
                 parent: Optional[Expression] = None):
        super().__init__(sup, parent)
        self.inside = inside
        self.l_sup = l_sup  # degree/index (the [n] in \\sqrt[n])

        # Set parent references
        if self.inside:
            self.inside.parent = self
        if self.l_sup:
            self.l_sup.parent = self

    def toLatex(self) -> str:
        """Convert square root to LaTeX: \\sqrt[degree]{radicand}^{superscript}"""
        inside_latex = self.inside.toLatex() if self.inside else ""

        if self.l_sup:
            result = f"\\sqrt [ {self.l_sup.toLatex()} ] {{ {inside_latex} }}"
        else:
            result = f"\\sqrt{{ {inside_latex} }}"

        if self.sup:
            result += f" ^ {{ {self.sup.toLatex()} }}"

        return result

    def get_children(self) -> Dict[str, Expression]:
        """Get radicand, optional degree, and optional superscript."""
        children = {}
        if self.inside:
            children['inside'] = self.inside
        if self.l_sup:
            children['l_sup'] = self.l_sup
        if self.sup:
            children['sup'] = self.sup
        return children


class AboveBelowConstruct(Construct):
    """Represents constructs with above/below relations: \\sum, \\prod, \\int, etc."""

    def __init__(self,
                 construct_type: str,
                 below: Optional[Expression] = None,
                 above: Optional[Expression] = None,
                 sup: Optional[Expression] = None,
                 parent: Optional[Expression] = None):
        super().__init__(sup, parent)
        self.construct_type = construct_type  # e.g., "\\sum", "\\prod", "\\int"
        self.below = below
        self.above = above

        # Set parent references
        if self.below:
            self.below.parent = self
        if self.above:
            self.above.parent = self

    def toLatex(self) -> str:
        """Convert construct to LaTeX: \\construct_{below}^{above}"""
        result = self.construct_type

        if self.below:
            result += f" _ {{ {self.below.toLatex()} }}"

        if self.above:
            result += f" ^ {{ {self.above.toLatex()} }}"

        if self.sup:
            # Additional superscript after the construct
            result += f" ^ {{ {self.sup.toLatex()} }}"

        return result

    def get_children(self) -> Dict[str, Expression]:
        """Get lower limit, upper limit, and optional superscript."""
        children = {}
        if self.below:
            children['below'] = self.below
        if self.above:
            children['above'] = self.above
        if self.sup:
            children['sup'] = self.sup
        return children


# Example usage and test
if __name__ == "__main__":
    # Example: \\frac{\\sqrt[2]{x}+1}{1-x^2}
    expr = Expression([
        FractionConstruct(
            above=Expression([
                SqrtConstruct(
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

    print("Example expression:")
    print(expr.toLatex())
    # Output: \\frac{\\sqrt[2]{x}+1}{1-x^{2}}

    # Example: \\sum_{i=1}^{n} x_i^2
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

    print("\nSum example:")
    print(sum_expr.toLatex())
    # Output: \\sum_{i=1}^{n}x_{i}^{2}

    # Example: \\prod_{k=1}^{m} a_k
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

    print("\nProduct example:")
    print(prod_expr.toLatex())
    # Output: \\prod_{k=1}^{m}a_{k}

    # Example: \\int_{0}^{\\infty} f(x) dx
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

    print("\nIntegral example:")
    print(integral_expr.toLatex())
    # Output: \\int_{0}^{\\infty}f ( x ) d x