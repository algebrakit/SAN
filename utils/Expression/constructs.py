"""LaTeX construct classes (Symbol, Fraction, Sqrt, etc.)."""

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .base import LatexItem

if TYPE_CHECKING:
    from .expression import Expression


class Symbol(LatexItem):
    """Represents a single symbol (letter, number, operator, etc.) with optional sub/superscripts."""

    def __init__(self,
                 value: str,
                 sub: Optional['Expression'] = None,
                 sup: Optional['Expression'] = None,
                 parent: Optional['Expression'] = None):
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

    def get_regions(self) -> List[Tuple[str, 'Expression']]:
        """Get ordered list of (region_name, Expression) tuples for hybrid syntax.

        Returns regions in order: sub, sup (only if they exist).
        """
        regions = []
        if self.sub:
            regions.append(('sub', self.sub))
        if self.sup:
            regions.append(('sup', self.sup))
        return regions


class Construct(LatexItem):
    """Abstract base class for mathematical constructs like \\frac, \\sqrt, \\sum, etc."""

    def __init__(self,
                 construct_type: str,
                 sup: Optional['Expression'] = None,
                 parent: Optional['Expression'] = None):
        super().__init__(parent)
        self.construct_type = construct_type
        self.sup = sup

        if self.sup:
            self.sup.parent = self

    @abstractmethod
    def toLatex(self) -> str:
        """Convert construct to LaTeX representation."""
        pass

    @abstractmethod
    def get_regions(self) -> List[Tuple[str, 'Expression']]:
        """Get ordered list of (region_name, Expression) tuples for hybrid syntax.

        Returns regions in order: above, below, sup (only if they exist).
        Note: sup is additional superscript after the fraction.
        """
        pass

class AccentConstruct(Construct):
    """Represents an accent above or below an expressoin, such as \\bar{AB} or \\underline{x}"""
    def __init__(self,
                 construct_type: str,
                 is_above: bool,
                 child: 'Expression',
                 sup: Optional['Expression'] = None,
                 sub: Optional['Expression'] = None,
                 parent: Optional['Expression'] = None):
        super().__init__(construct_type, parent)
        self.child = child
        self.sup = sup
        self.sub = sub
        self.is_above = is_above

        self.child.parent = self
        if self.sub:
            self.sub.parent = self
        if self.sup:
            self.sup.parent = self

    def toLatex(self) -> str:
        child_latex = self.child.toLatex()
        result = f"{self.construct_type} {{ {child_latex} }}"

        if self.sup:
            sup_latex = self.sup.toLatex() if self.sup else ""
            result += f" ^ {{ {sup_latex} }}"
        if self.sub:
            sub_latex = self.sub.toLatex() if self.sub else ""
            result += f" _ {{ {sub_latex} }}"

        return result
        
    def get_regions(self) -> List[Tuple[str, 'Expression']]:
        """Get ordered list of (region_name, Expression) tuples for hybrid syntax.
            Returns regions in order: above, below, sup (only if they exist).
        """
        regions = []
        if self.is_above:
            regions.append(('below', self.child))
        else:
            regions.append(('above', self.child))
        if self.sub:
            regions.append(('sub', self.sub))
        if self.sup:
            regions.append(('sup', self.sup))
        return regions
    
class RowConstruct(Construct):
    """Represents a row in a \\stack{\\row{..&..} \\row{..&..} }
       Rows have content in the 'inside' region. 
       Each row is in the 'below' region of every previous row.
    """

    def __init__(self,
                 inside: 'Expression',
                 below: Optional['Expression'] = None,
                 parent: Optional['Expression'] = None):
        super().__init__(r"\row", None, parent)
        self.inside = inside
        self.below = below
        inside.parent = self

    def toLatex(self) -> str:
        """Convert row to LaTeX: \\row{..&..}"""
        inside_latex = self.inside.toLatex() if self.inside else ""
        result = f"{self.construct_type} {{ {inside_latex} }}"
        if self.below:
            result += self.below.toLatex()
        return result

    def toLatex_matrixForm(self) -> str:
        """Convert row to LaTeX without \\rows but using '\\': ..&..  \\ ..&.. """
        inside_latex = self.inside.toLatex() if self.inside else ""
        result = inside_latex
        if self.below:
            next_row = self.below.get_children()[0]
            result += r" \\ " + next_row.toLatex_matrixForm()
        return result

    def get_regions(self) -> List[Tuple[str, 'Expression']]:
        """Get ordered list of (region_name, Expression) tuples for hybrid syntax.

        Returns regions in order: inside
        """
        regions = []
        if self.inside:
            regions.append(('inside', self.inside))
        if self.below:
            regions.append(('below', self.below))
        return regions
    
class StackConstruct(Construct):
    """Represents a matrix or vector \\stack{\\row{..&..} \\row{..&..} }
       Content resides in the 'inside' region and consists of the 1st row.
       Following rows appended 'below' regions.
       So:
       - stack
           .inside = 
              - row1
                  .inside = [item, item, item]
                  .below = 
                      -row2
                         .inside = ...
       This way the parents of each row are the parents above (vertically) and the items in each row are aligned horizontally.
    """

    def __init__(self,
                 construct_type: str,
                 inside: 'Expression',
                 parent: Optional['Expression'] = None):
        super().__init__(construct_type, None, parent)
        from .expression import Expression

        self.parent = parent
        rows = inside.get_children() # List[RowConstruct]
        first_row_expr = Expression([rows[0]])
        self.inside = first_row_expr
        inside.parent = self
        self.mtype = None

        lastRow = rows[0]
        ii=1
        while ii < len(rows):
            row_expr = Expression([rows[ii]])
            lastRow.below = row_expr
            row_expr.parent = lastRow
            lastRow = rows[ii]
            ii += 1


    def set_matrix_type(self, mtype:str):
        self.mtype = mtype

    def toLatex(self) -> str:
        """Convert stack to LaTeX: \\stack{..&.. \\ ..&.. }^{superscript}"""

        if self.mtype is None:
            child_latex = self.inside.toLatex() if self.inside else ""
            result = f"{self.construct_type} {{ {child_latex} }}"
        else:
            child_latex = self.inside.get_children()[0].toLatex_matrixForm()
            result = f"\\begin{{{self.mtype}}} {child_latex} \\end{{{self.mtype}}}"
            
        if self.sup:
            result += f" ^ {{ {self.sup.toLatex()} }}"

        return result

    def get_regions(self) -> List[Tuple[str, 'Expression']]:
        """Get numerator, denominator, and optional superscript."""
        regions = []

        regions.append(('inside', self.inside))

        return regions

class FractionConstruct(Construct):
    """Represents a fraction: \\frac{above}{below}"""

    def __init__(self,
                 construct_type: str,
                 above: Optional['Expression'] = None,
                 below: Optional['Expression'] = None,
                 sup: Optional['Expression'] = None,
                 parent: Optional['Expression'] = None):
        super().__init__(construct_type, sup, parent)
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
        result = f"{self.construct_type} {{ {above_latex} }} {{ {below_latex} }}"

        if self.sup:
            result += f" ^ {{ {self.sup.toLatex()} }}"

        return result

    def get_regions(self) -> List[Tuple[str, 'Expression']]:
        """Get ordered list of (region_name, Expression) tuples for hybrid syntax.

        Returns regions in order: above, below, sup (only if they exist).
        Note: sup is additional superscript after the fraction.
        """
        regions = []
        if self.above:
            regions.append(('above', self.above))
        if self.below:
            regions.append(('below', self.below))
        if self.sup:
            regions.append(('sup', self.sup))
        return regions


class SqrtConstruct(Construct):
    """Represents a square root: \\sqrt[l_sup]{inside}"""

    def __init__(self,
                 construct_type: str,
                 inside: Optional['Expression'] = None,
                 l_sup: Optional['Expression'] = None,
                 sup: Optional['Expression'] = None,
                 parent: Optional['Expression'] = None):
        super().__init__(construct_type, sup, parent)
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
            result = f"{self.construct_type} [ {self.l_sup.toLatex()} ] {{ {inside_latex} }}"
        else:
            result = f"{self.construct_type} {{ {inside_latex} }}"

        if self.sup:
            result += f" ^ {{ {self.sup.toLatex()} }}"

        return result

    def get_regions(self) -> List[Tuple[str, 'Expression']]:
        """Get ordered list of (region_name, Expression) tuples for hybrid syntax.

        Returns regions in order: L-sup, inside, sup (only if they exist).
        """
        regions = []
        if self.l_sup:
            regions.append(('L-sup', self.l_sup))
        if self.inside:
            regions.append(('inside', self.inside))
        if self.sup:
            regions.append(('sup', self.sup))
        return regions

class LogConstruct(Construct):
    """Represents a logarithm, including the Dutch notation: \\lognl[l_sup]"""

    def __init__(self,
                 construct_type: str,
                 l_sup: Optional['Expression'] = None,
                 sup: Optional['Expression'] = None,
                 sub: Optional['Expression'] = None,
                 parent: Optional['Expression'] = None):
        super().__init__(construct_type, sup, parent)
        self.l_sup = l_sup  # base (the [n] in \\lognl[n])
        self.sub = sub

        # Set parent references
        if self.l_sup:
            self.l_sup.parent = self
        if self.sub:
            self.sub.parent = self
        if self.sup:
            self.sup.parent = self

    def toLatex(self) -> str:
        """Convert square root to LaTeX: \\log[base]^{superscript}"""

        if self.l_sup:
            result = f"\\lognl [ {self.l_sup.toLatex()} ]"
            # sub not allowed
        else:
            result = f"\\log"
            if self.sub:
                result+= f" _ {{ {self.sub.toLatex()} }}"

        if self.sup:
            result += f" ^ {{ {self.sup.toLatex()} }}"

        return result

    def get_regions(self) -> List[Tuple[str, 'Expression']]:
        """Get ordered list of (region_name, Expression) tuples for hybrid syntax.

        Returns regions in order: L-sup, inside, sup (only if they exist).
        """
        regions = []
        if self.l_sup:
            regions.append(('L-sup', self.l_sup))
        elif self.sub:
            regions.append(('sub', self.sub))

        return regions


class AboveBelowConstruct(Construct):
    """Represents constructs with above/below relations: \\sum, \\prod, \\int, etc."""

    def __init__(self,
                 construct_type: str,
                 below: Optional['Expression'] = None,
                 above: Optional['Expression'] = None,
                 sup: Optional['Expression'] = None,
                 parent: Optional['Expression'] = None):
        super().__init__(construct_type, sup, parent)
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

    def get_regions(self) -> List[Tuple[str, 'Expression']]:
        """Get ordered list of (region_name, Expression) tuples for hybrid syntax.

        Returns regions in order: below, above, sup (only if they exist).
        """
        regions = []
        if self.below:
            regions.append(('below', self.below))
        if self.above:
            regions.append(('above', self.above))
        if self.sup:
            regions.append(('sup', self.sup))
        return regions
