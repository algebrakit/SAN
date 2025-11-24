"""Base classes for LaTeX items."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .expression import Expression


@dataclass
class LatexOptions:
    """Options for controlling LaTeX output format."""

    convertMatrices: bool = True
    """Convert stack constructs to standard LaTeX matrix environments."""

    convertLog: bool = False
    """Convert logarithm notation from Dutch (\\lognl) to standard (\\log) format."""


class LatexItem(ABC):
    """Abstract base class for all LaTeX mathematical expression items."""

    def __init__(self, parent: Optional['Expression'] = None):
        self.parent = parent

    @abstractmethod
    def toLatex(self, options: Optional['LatexOptions'] = None) -> str:
        """Convert this item to its LaTeX string representation.

        Args:
            options: Options controlling the output format. If None, uses default options.
        """
        pass

    def get_children(self) -> Dict[str, 'Expression']:
        """Get all child Expression objects mapped by relation type."""
        return {}
