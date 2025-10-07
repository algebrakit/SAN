"""LaTeX parser for converting LaTeX strings to Expression objects."""

from typing import List, Optional
from .defs import ACCENT_COMMANDS_ABOVE, ACCENT_COMMANDS_BELOW, ABOVE_BELOW_COMMANDS
from .base import LatexItem
from .expression import Expression
from .constructs import AccentConstruct, RowConstruct, Symbol, Construct, FractionConstruct, SqrtConstruct, AboveBelowConstruct, StackConstruct

def parse_latex(latex: str) -> Optional[Expression]:
    """Parse an Expression object from LaTeX syntax.

    Args:
        latex: LaTeX string with space-separated tokens and braced arguments

    Returns:
        Expression object or None if parsing fails
    """
    if not latex or not latex.strip():
        return Expression([])

    tokens = latex.strip().split()

    class Parser:
        def __init__(self, tokens: List[str]):
            self.tokens = tokens
            self.pos = 0

        def current(self) -> Optional[str]:
            if self.pos < len(self.tokens):
                return self.tokens[self.pos]
            return None

        def advance(self) -> None:
            self.pos += 1

        def parse_expression(self, stop_at_brace: bool = False) -> Expression:
            """Parse a sequence of items until end or closing brace."""
            items = []

            while self.pos < len(self.tokens):
                token = self.current()

                if token == '}':
                    if stop_at_brace:
                        break
                    else:
                        # Unexpected closing brace
                        self.advance()
                        continue

                item = self.parse_item()
                if item:
                    items.append(item)

            return Expression(items)

        def parse_item(self) -> Optional[LatexItem]:
            """Parse a single LaTeX item (symbol or construct)."""
            token = self.current()
            if not token:
                return None

            # Handle opening brace - creates nested expression
            if token == '{':
                self.advance()
                self.parse_expression(stop_at_brace=True)
                if self.current() == '}':
                    self.advance()
                # Braces are structural, not items
                return None

            # Handle constructs
            if token == '\\frac':
                return self.parse_frac()
            elif token == '\\sqrt':
                return self.parse_sqrt()
            elif token == '\\stack':
                return self.parse_stack()
            elif token == '\\row':
                return self.parse_row()
            elif token in ABOVE_BELOW_COMMANDS:
                return self.parse_above_below(token)
            elif token in ACCENT_COMMANDS_ABOVE:
                return self.parse_accent(token, True)
            elif token in ACCENT_COMMANDS_BELOW:
                return self.parse_accent(token, False)

            # Regular symbol
            self.advance()
            symbol = Symbol(token)

            # Check for subscript or superscript
            symbol = self.apply_sub_sup(symbol)

            return symbol

        def apply_sub_sup(self, item):
            """Apply subscript and/or superscript to an item.
            
            For multiple superscripts (e.g., a^{b}^{c}), they are nested right-to-left:
            the rightmost superscript becomes the superscript of the previous one.
            """
            sub_expr = None
            sup_expr = None

            while self.current() in ('_', '^'):
                op = self.current()
                self.advance()

                # Expect opening brace
                if self.current() == '{':
                    self.advance()
                    expr = self.parse_expression(stop_at_brace=True)
                    if self.current() == '}':
                        self.advance()

                    if op == '_':
                        # For multiple subscripts, nest them right-to-left
                        if sub_expr is not None:
                            # Current sub_expr becomes base, new expr is its subscript
                            # We need to wrap sub_expr as a symbol and give it expr as subscript
                            # But this is complex - for now, just chain them in the last item
                            last_item = sub_expr.items[-1] if sub_expr.items else None
                            if last_item and isinstance(last_item, Symbol):
                                last_item.sub = expr
                                expr.parent = last_item
                            else:
                                # Fallback: just use the new one
                                sub_expr = expr
                        else:
                            sub_expr = expr
                    else:  # '^'
                        # For multiple superscripts, nest them right-to-left
                        if sup_expr is not None:
                            # Current sup_expr becomes base, new expr is its superscript
                            # Find the last symbol in sup_expr and attach new expr as its superscript
                            last_item = sup_expr.items[-1] if sup_expr.items else None
                            if last_item and isinstance(last_item, Symbol):
                                last_item.sup = expr
                                expr.parent = last_item
                            else:
                                # Fallback: just use the new one (overwrite)
                                sup_expr = expr
                        else:
                            sup_expr = expr

            # Apply to Symbol
            if isinstance(item, Symbol):
                if sub_expr:
                    item.sub = sub_expr
                    sub_expr.parent = item
                if sup_expr:
                    item.sup = sup_expr
                    sup_expr.parent = item
            # Apply to Construct
            elif isinstance(item, Construct):
                if sup_expr:
                    item.sup = sup_expr
                    sup_expr.parent = item
            if isinstance(item, AccentConstruct):        
                if sub_expr:
                    item.sub = sub_expr
                    sub_expr.parent = item

            return item

        def parse_braced_expression(self) -> Optional[Expression]:
            """Parse a braced expression { ... }."""
            if self.current() != '{':
                return None

            self.advance()
            expr = self.parse_expression(stop_at_brace=True)

            if self.current() == '}':
                self.advance()

            return expr

        def parse_frac(self):
            """Parse \\frac { above } { below }."""
            self.advance()  # skip '\frac'

            above = self.parse_braced_expression()
            below = self.parse_braced_expression()

            frac = FractionConstruct(construct_type='\\frac', above=above, below=below)

                # Check for superscript on the fraction
            return self.apply_sub_sup(frac)

        def parse_stack(self):
            self.advance()  # skip '\stack'
            child = self.parse_braced_expression()
            if child is None:
                return
            
            return StackConstruct(construct_type='\\stack', inside=child)

        def parse_row(self):
            self.advance()  # skip '\row'
            child = self.parse_braced_expression()
            if child is None:
                return
            
            return RowConstruct(inside=child)

        def parse_sqrt(self):
            """Parse \\sqrt { inside } or \\sqrt [ degree ] { inside }."""
            self.advance()  # skip '\sqrt'

            l_sup = None

            # Check for optional degree in brackets
            if self.current() == '[':
                self.advance()
                # Parse content until ]
                items = []
                while self.current() and self.current() != ']':
                    item = self.parse_item()
                    if item:
                        items.append(item)
                l_sup = Expression(items)

                if self.current() == ']':
                    self.advance()

            inside = self.parse_braced_expression()

            sqrt = SqrtConstruct(construct_type='\\sqrt', inside=inside, l_sup=l_sup)

            # Check for superscript on the sqrt
            return self.apply_sub_sup(sqrt)

        def parse_accent(self, construct_type, is_above):
            """Parse constructs like \\bar{..} and \\underline{..}"""
            self.advance()  # skip construct token
            child = self.parse_braced_expression()
            if child is None:
                return None
             
            accent = AccentConstruct(construct_type=construct_type, is_above=is_above, child=child)
            return self.apply_sub_sup(accent)

        def parse_above_below(self, construct_type: str):
            """Parse constructs like \\sum, \\prod, \\int with optional limits."""
            self.advance()  # skip construct token

            below = None
            above = None

            # Handle subscript and superscript in any order
            while self.current() in ('_', '^'):
                if self.current() == '_':
                    self.advance()
                    below = self.parse_braced_expression()
                elif self.current() == '^':
                    self.advance()
                    above = self.parse_braced_expression()

            construct = AboveBelowConstruct(
                construct_type=construct_type,
                below=below,
                above=above
            )

            # Check for additional superscript
            return self.apply_sub_sup(construct)

    try:
        parser = Parser(tokens)
        return parser.parse_expression()
    except Exception:
        return None
