"""Expression container class."""

from typing import List, Optional, Union
from .base import LatexItem, LatexOptions
from .constructs import Symbol, StackConstruct


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

    def from_gtd_list(self, gtd_list) -> Optional['Expression']:
        """Create an Expression from a GTD list representation.
        Args:
            gtd_list: List of GTD entries, where each entry is:
                [symbol, id, parent_id, parent_symbol] if parent is a symbol
                [symbol, id, parent_id, region] if parent is a construct (e.g., '\\frac', '\\underline', etc)
        """
        from .gtd_parser import parse_gtd
        return parse_gtd(gtd_list)

    def toLatex(self, options: Optional[LatexOptions] = None) -> str:
        """Convert this expression to LaTeX by concatenating all items.

        Args:
            options: Options controlling the output format. If None, uses default options.
        """
        if options is None:
            options = LatexOptions()

        if options.convertMatrices:
            _items = _handleStackConstructs(self.items)
        else:
            _items = self.items
        return ' '.join(item.toLatex(options) for item in _items)

    def to_hybrid(self) -> List[List[Union[int, str, None]]]:
        """Convert this expression to hybrid syntax representation.

        Returns:
            List of hybrid lines, where each line is:
            [id, symbol, parent_id, parent_symbol, above, below, sub, sup, L-sup, inside, right]

        Example:
            >>> expr = Expression.fromLatex('x ^ { 2 }')
            >>> lines = expr.to_hybrid()
            >>> # Returns:
            >>> # [[1, 'x', 0, '<sos>', None, None, None, None, None, None, None],
            >>> #  [2, 'struct', 1, 'x', None, None, None, 'sup', None, None, None],
            >>> #  [3, '2', 2, 'sup', None, None, None, None, None, None, None],
            >>> #  [4, '<eos>', 3, '2', None, None, None, None, None, None, None],
            >>> #  [5, '<eos>', 4, '<eos>', None, None, None, None, None, None, None]]
        """
        from .hybrid import expression_to_hybrid
        return expression_to_hybrid(self)

    def get_children(self) -> List[LatexItem]:
        """Get all items contained in this expression."""
        return self.items


# ----------------------------------
def _handleStackConstructs(items) -> List[LatexItem]:
    _items = items.copy()
    ii = 1
    while ii < len(_items):
        item = _items[ii]
        if isinstance(item, StackConstruct):
            #get the bracket types to determine the kind of matrix
            matrix_type = None
            prev_item, next_item = None, None
            if ii > 0: prev_item = _items[ii - 1]
            if ii + 1 < len(_items): next_item = _items[ii + 1]

            if not(prev_item is None and next_item is None):
                if isinstance(prev_item, Symbol) and isinstance(next_item, Symbol):
                    prev_symbol = prev_item.value
                    next_symbol = next_item.value
                    if prev_symbol == '(' and next_symbol == ')':
                        matrix_type = 'pmatrix'
                    elif prev_symbol == '[' and next_symbol == ']':
                        matrix_type = 'bmatrix'
                    elif prev_symbol == r'\{' and next_symbol == r'\}':
                        matrix_type = 'Bmatrix'
                    elif prev_symbol == '|' and next_symbol == '|':
                        if ii > 1 and ii+2 < len(_items) \
                            and isinstance(_items[ii - 2], Symbol) and isinstance(_items[ii+2], Symbol) \
                            and _items[ii - 2].value == _items[ii+2].value == '|':
                                matrix_type = 'Vmatrix'
                                _items.pop(ii+1)
                                _items.pop(ii-1)
                                ii -= 1
                        else:
                            matrix_type = 'vmatrix'
                    elif prev_symbol == r'\Vert' and next_symbol == r'\Vert':
                        matrix_type = 'Vmatrix'

                    if not (matrix_type is None):
                        _items.pop(ii+1)
                        _items.pop(ii-1)
                        ii -= 1


            if matrix_type is None and not (prev_item is None):
                # might be a system of equations: \begin{cases} .. \\ .. \end{cases}
                prev_symbol = prev_item.value
                if prev_symbol == r'\{':
                    matrix_type = 'cases'
                    _items.pop(ii-1)
                    ii -= 1

            if matrix_type is None:
                matrix_type = 'matrix'

            item.set_matrix_type(matrix_type)

        ii += 1
    return _items    

