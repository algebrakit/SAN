"""
Utility functions for parsing LaTeX symbols.
"""
import re
from typing import List, Dict, Set


def extract_basic_symbols(symbol: str) -> List[str]:
    """Extract basic symbols from a combined/compound LaTeX symbol.

    Examples:
        'BD' -> ['B', 'D']
        'A_{0}' -> ['A', '0']
        'x^{2}' -> ['x', '2']
        '\\gamma' -> ['\\gamma']
        '\\sin' -> ['\\sin']
        'A_{12}' -> ['A', '1', '2']

    Args:
        symbol: A LaTeX symbol string that may contain combined symbols.

    Returns:
        List of basic symbols extracted from the input.
    """
    if not symbol:
        return []

    symbols = []
    i = 0

    while i < len(symbol):
        char = symbol[i]

        # Handle LaTeX commands (e.g., \gamma, \sin)
        if char == '\\':
            # Find the end of the command
            j = i + 1
            while j < len(symbol) and symbol[j].isalpha():
                j += 1
            if j > i + 1:
                # Valid LaTeX command
                symbols.append(symbol[i:j])
                i = j
            else:
                # Backslash followed by non-alpha (e.g., \{, \})
                # Skip the backslash and the next char
                i = j + 1 if j < len(symbol) else j

        # Skip structural characters
        elif char in '{}_^':
            i += 1

        # Handle letters and digits as individual symbols
        elif char.isalnum():
            symbols.append(char)
            i += 1

        # Skip other characters (spaces, etc.)
        else:
            i += 1

    return symbols


def expand_symbol_adjustments(
    adjustments: List[Dict],
    vocabulary: Set[str]
) -> List[Dict]:
    """Expand combined symbols in adjustments to their basic symbols.

    For each adjustment, extracts basic symbols and creates new adjustment
    entries for each one. Only includes symbols that exist in the vocabulary.
    If the same symbol appears multiple times with different offsets,
    keeps the one with the strongest effect (highest boost or lowest penalty).

    Args:
        adjustments: List of dicts with 'symbol' and 'offset' keys.
        vocabulary: Set of valid symbol strings.

    Returns:
        List of expanded adjustments with only vocabulary-valid symbols.
    """
    if not adjustments:
        return []

    # Offset priority: higher value = stronger boost, lower value = stronger penalty
    # For conflicts, we keep the strongest effect
    OFFSET_VALUES = {
        'DISABLE': -100,
        'STRONG_PENALIZE': -5,
        'PENALIZE': -3,
        'BOOST': 3,
        'STRONG_BOOST': 5,
    }

    # Track best offset for each symbol
    symbol_offsets: Dict[str, str] = {}

    for adj in adjustments:
        original_symbol = adj.get('symbol', '')
        offset = adj.get('offset', '')

        if not original_symbol or offset not in OFFSET_VALUES:
            continue

        # Extract basic symbols
        basic_symbols = extract_basic_symbols(original_symbol)

        for sym in basic_symbols:
            # Only process if symbol is in vocabulary
            if sym not in vocabulary:
                continue

            if sym not in symbol_offsets:
                symbol_offsets[sym] = offset
            else:
                # Keep the strongest effect
                current_value = OFFSET_VALUES[symbol_offsets[sym]]
                new_value = OFFSET_VALUES[offset]

                # For boosts, keep higher value; for penalties, keep lower value
                if new_value > 0 and current_value > 0:
                    # Both are boosts, keep stronger
                    if new_value > current_value:
                        symbol_offsets[sym] = offset
                elif new_value < 0 and current_value < 0:
                    # Both are penalties, keep stronger (more negative)
                    if new_value < current_value:
                        symbol_offsets[sym] = offset
                elif new_value < 0:
                    # New is penalty, current is boost - penalty wins
                    symbol_offsets[sym] = offset
                # else: current is penalty, new is boost - keep penalty

    # Convert back to list of adjustments
    return [{'symbol': sym, 'offset': off} for sym, off in symbol_offsets.items()]
