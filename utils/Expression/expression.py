"""Expression container class."""

from typing import List, Optional

from .base import LatexItem


class Expression:
    """Container for a sequence of LaTeX items."""

    def __init__(self, items: Optional[List[LatexItem]] = None, parent: Optional['LatexItem'] = None):
        self.items = items or []
        self.parent = parent
        # Set parent reference for all items
        for item in self.items:
            item.parent = self

    @staticmethod
    def fromLatex(latex: str) -> 'Expression | None':
        """Parse an Expression object from LaTeX syntax.

        Args:
            latex: LaTeX string with space-separated tokens and braced arguments

        Returns:
            Expression object or None if parsing fails
        """
        from .parser import parse_latex
        return parse_latex(latex)

    def add_item(self, item: LatexItem) -> None:
        """Add an item to this expression."""
        item.parent = self
        self.items.append(item)

    def toLatex(self) -> str:
        """Convert this expression to LaTeX by concatenating all items."""
        return ' '.join(item.toLatex() for item in self.items)

    def to_hybrid(self) -> str[]:
        pass
    
    def get_children(self) -> List[LatexItem]:
        """Get all items contained in this expression."""
        return self.items
