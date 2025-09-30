"""Base classes for LaTeX items."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .expression import Expression


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
