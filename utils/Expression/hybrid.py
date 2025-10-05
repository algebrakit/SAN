"""Hybrid syntax generator for LaTeX expressions."""

from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .expression import Expression
    from .base import LatexItem
    from .constructs import Symbol, Construct


class HybridGenerator:
    """Generates hybrid syntax representation from Expression tree."""

    def __init__(self):
        self.lines: List[List[Union[int, str, None]]] = []
        self.current_id = 1
        self.last_id = 0
        self.last_symbol = '<sos>'

    def generate(self, expression: 'Expression') -> List[List[Union[int, str, None]]]:
        """Generate hybrid syntax lines from an Expression.

        Returns:
            List of lines, where each line is:
            [id, symbol, parent_id, parent_symbol, above, below, sub, sup, L-sup, inside, right]
        """
        self.lines = []
        self.current_id = 1
        self.last_id = 0
        self.last_symbol = '<sos>'

        self._process_expression(expression)

        return self.lines

    def _emit(self, symbol: str, parent_id: int, parent_symbol: str,
              above=None, below=None, sub=None, sup=None, l_sup=None, inside=None, right=None):
        """Emit a single hybrid line."""
        line = [
            self.current_id,
            symbol,
            parent_id,
            parent_symbol,
            above,
            below,
            sub,
            sup,
            l_sup,
            inside,
            right
        ]
        self.lines.append(line)
        self.last_id = self.current_id
        self.last_symbol = symbol
        self.current_id += 1

    def _process_expression(self, expression: 'Expression'):
        """Process an Expression and all its items."""
        from .constructs import Symbol, Construct

        items = expression.items

        for i, item in enumerate(items):
            has_right = i < len(items) - 1  # More items follow this one

            if isinstance(item, (Symbol, Construct)):
                regions = item.get_regions()

                if not regions:
                    # Simple symbol with no regions
                    if isinstance(item, Symbol):
                        self._emit(item.value, self.last_id, self.last_symbol)
                    elif isinstance(item, Construct):
                        self._emit(item.construct_type, self.last_id, self.last_symbol)
                    if i == len(items) - 1:
                        # Last item in the expression, end symbols concatenation with <eos>
                        self._emit('<eos>', self.last_id, self.last_symbol)

                else:
                    # Item with regions - emit the base symbol/construct first
                    if isinstance(item, Symbol):
                        base_symbol = item.value
                        multiple_region_info = None
                    elif isinstance(item, Construct):
                        base_symbol = item.construct_type
                        multiple_region_info = item.get_region_lists()
                    else:
                        raise ValueError("Unknown item type")
                    
                    
                    self._emit(base_symbol, self.last_id, self.last_symbol)
                    construct_id = self.last_id
                    construct_symbol = self.last_symbol

                    # Emit struct line with region indicators
                    region_indicators = self._build_region_indicators(regions, has_right)
                    self._emit('struct', construct_id, construct_symbol, *region_indicators)
                    struct_id = self.last_id

                    # Process each region
                    for region_name, region_expr in regions:
                        if region_expr and region_expr.items:
                            # First item in region references the struct with region name
                            self.last_id = struct_id
                            self.last_symbol = region_name

                            self._process_expression(region_expr)
                        # Handle multiple regions if applicable
                        if multiple_region_info and region_name == multiple_region_info[0]:
                            for _expr in multiple_region_info[1]:
                                self.last_id = struct_id
                                self.last_symbol = region_name
                                self._process_expression(_expr)

                    # Process "right" region if there are more items
                    if has_right:
                        # Set up for right region
                        self.last_id = struct_id
                        self.last_symbol = 'right'



    def _build_region_indicators(self, regions: List, has_right: bool) -> tuple:
        """Build the 7 region indicator values for a struct line.

        Returns: (above, below, sub, sup, L-sup, inside, right)
        """
        region_names = [r[0] for r in regions]

        above = 'above' if 'above' in region_names else None
        below = 'below' if 'below' in region_names else None
        sub = 'sub' if 'sub' in region_names else None
        sup = 'sup' if 'sup' in region_names else None
        l_sup = 'L-sup' if 'L-sup' in region_names else None
        inside = 'inside' if 'inside' in region_names else None
        right = 'right' if has_right else None

        return (above, below, sub, sup, l_sup, inside, right)


def expression_to_hybrid(expression: 'Expression') -> List[List[Union[int, str, None]]]:
    """Convert an Expression to hybrid syntax.

    Args:
        expression: Expression object to convert

    Returns:
        List of hybrid lines, where each line is:
        [id, symbol, parent_id, parent_symbol, above, below, sub, sup, L-sup, inside, right]
    """
    generator = HybridGenerator()
    return generator.generate(expression)
