"""Shared utilities for the synthetic expression generator pipeline.

This module provides common functions used across multiple scripts:
- LaTeX tokenization
- InkML file parsing
- File I/O utilities
- Statistics formatting
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import xml.etree.ElementTree as ET


# =============================================================================
# LaTeX Tokenization
# =============================================================================

# Pattern matches LaTeX commands and single characters
# Handles: \mathbb{X}, \begin{env}, \end{env}, \operatorname*, \command, \X
_COMMAND_PATTERN = re.compile(
    r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)'
)


def tokenize_latex(latex: str) -> List[str]:
    """
    Tokenize a LaTeX string into individual symbols/commands.

    Handles:
    - LaTeX commands (\\cos, \\frac, etc.)
    - Single characters (digits, letters, operators)
    - Skips structural chars ({, }, ^, _) and whitespace

    Args:
        latex: LaTeX expression string

    Returns:
        List of tokens (commands like '\\frac' and characters like 'x')

    Examples:
        >>> tokenize_latex("x^2")
        ['x', '2']
        >>> tokenize_latex("\\frac{1}{2}")
        ['\\frac', '1', '2']
        >>> tokenize_latex("\\sqrt{x+1}")
        ['\\sqrt', 'x', '+', '1']
    """
    tokens = []
    s = latex

    while s:
        if s[0] == '\\':
            # Match LaTeX command
            match = _COMMAND_PATTERN.match(s)
            if match:
                tokens.append(match.group(0))
                s = s[len(match.group(0)):]
            else:
                tokens.append(s[0])
                s = s[1:]
        elif s[0] in '{}^_':
            # Structural characters (not rendered as glyphs)
            s = s[1:]
        elif s[0].isspace():
            # Skip whitespace
            s = s[1:]
        else:
            # Regular character
            tokens.append(s[0])
            s = s[1:]

    return tokens


# =============================================================================
# InkML Parsing
# =============================================================================

# InkML namespace used in all InkML files
INKML_NS = '{http://www.w3.org/2003/InkML}'


def parse_inkml_label(file_path: Path) -> Optional[str]:
    """
    Parse an InkML file and extract the symbol label.

    Looks for an annotation element with type="label".

    Args:
        file_path: Path to the InkML file

    Returns:
        The symbol label (LaTeX format) or None if not found/error
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Find annotation with type="label"
        for annotation in root.findall(f'.//{INKML_NS}annotation'):
            if annotation.get('type') == 'label':
                return annotation.text

        return None
    except Exception:
        return None


def parse_inkml_traces(file_path: Path) -> List[List[List[float]]]:
    """
    Parse InkML file to extract stroke traces.

    Each stroke is a list of points, where each point is [x, y] or [x, y, t].

    Args:
        file_path: Path to InkML file

    Returns:
        List of strokes, where each stroke is a list of points

    Example:
        >>> traces = parse_inkml_traces(Path("symbol.inkml"))
        >>> traces[0]  # First stroke
        [[0.0, 0.0, 0.0], [1.0, 2.0, 10.0], ...]
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    strokes = []
    for trace in root.findall(f'.//{INKML_NS}trace'):
        points_text = trace.text
        if not points_text:
            continue

        # Parse points: "x1 y1 t1, x2 y2 t2, ..."
        points = []
        for point_str in points_text.split(','):
            coords = point_str.strip().split()
            if len(coords) >= 2:
                point = [float(coords[0]), float(coords[1])]
                if len(coords) >= 3:
                    point.append(float(coords[2]))
                points.append(point)

        if points:
            strokes.append(points)

    return strokes


def get_inkml_bounds(file_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Get bounding box of all strokes in an InkML file.

    Args:
        file_path: Path to the InkML file

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) or None if error/empty
    """
    try:
        traces = parse_inkml_traces(file_path)

        if not traces:
            return None

        all_x = []
        all_y = []

        for stroke in traces:
            for point in stroke:
                all_x.append(point[0])
                all_y.append(point[1])

        if not all_x or not all_y:
            return None

        return (min(all_x), min(all_y), max(all_x), max(all_y))

    except Exception:
        return None


# =============================================================================
# File I/O Utilities
# =============================================================================

def read_lines_from_file(
    file_path: Path,
    skip_comments: bool = True,
    skip_empty: bool = True
) -> List[str]:
    """
    Read lines from a text file.

    Args:
        file_path: Path to text file
        skip_comments: If True, skip lines starting with '#'
        skip_empty: If True, skip empty lines

    Returns:
        List of stripped lines
    """
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if skip_empty and not line:
                continue
            if skip_comments and line.startswith('#'):
                continue
            lines.append(line)
    return lines


def write_jsonl(data: List[Dict[str, Any]], file_path: Path) -> None:
    """
    Write list of dictionaries to JSONL (JSON Lines) file.

    Each dictionary is written as a single JSON line.

    Args:
        data: List of dictionaries to write
        file_path: Output file path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Read JSONL (JSON Lines) file into list of dictionaries.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON content as dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Path, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        file_path: Output file path
        indent: Indentation level for pretty printing
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# =============================================================================
# Statistics Formatting
# =============================================================================

def print_stats_header(title: str, width: int = 60) -> None:
    """
    Print a statistics section header.

    Args:
        title: Title text to display
        width: Width of separator lines
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_stats_footer(width: int = 60) -> None:
    """
    Print a statistics section footer.

    Args:
        width: Width of separator line
    """
    print("=" * width)
