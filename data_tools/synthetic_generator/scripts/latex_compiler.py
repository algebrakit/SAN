"""LaTeX compilation utilities.

This module handles the compilation of LaTeX expressions to DVI format,
including document creation and error handling.
"""

import subprocess
from pathlib import Path
from typing import Optional


def create_latex_document(expression: str) -> str:
    """
    Create a minimal LaTeX document containing the expression.

    The document includes necessary packages (amsmath, amssymb) and
    custom command definitions (lognl, degree).

    Args:
        expression: LaTeX math expression

    Returns:
        Complete LaTeX document as string
    """
    return f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{textcomp}}
\\pagestyle{{empty}}
\\newcommand\\lognl[1][]{{\\mathop{{ {{}}^{{#1}}\\mathrm{{log}} }} }}
\\newcommand\\degree{{^\\circ}}
\\begin{{document}}
${expression}$
\\end{{document}}
"""


def compile_to_dvi(latex_content: str, work_dir: Path) -> Optional[Path]:
    """
    Compile LaTeX content to DVI file.

    Args:
        latex_content: Complete LaTeX document
        work_dir: Working directory for compilation

    Returns:
        Path to DVI file, or None if compilation failed
    """
    # Write LaTeX file
    tex_file = work_dir / "document.tex"
    tex_file.write_text(latex_content, encoding='utf-8')

    # Find LaTeX binary
    latex_cmd = '/Library/TeX/texbin/latex'
    if not Path(latex_cmd).exists():
        # Fallback to PATH
        latex_cmd = 'latex'

    # Compile with latex
    try:
        result = subprocess.run(
            [latex_cmd, '-interaction=nonstopmode', 'document.tex'],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )

        dvi_file = work_dir / "document.dvi"
        log_file = work_dir / "document.log"

        # Check for LaTeX compilation errors in log file
        if log_file.exists():
            log_content = log_file.read_text(encoding='utf-8', errors='ignore')

            # Check for critical errors
            error_patterns = [
                '! Undefined control sequence',
                '! LaTeX Error:',
                '! Missing',
                '! Emergency stop'
            ]

            for pattern in error_patterns:
                if pattern in log_content:
                    # Extract context around the error for better diagnostics
                    lines = log_content.split('\n')
                    for i, line in enumerate(lines):
                        if pattern in line:
                            print(f"  LaTeX compilation error detected:")
                            print(f"  {pattern}")
                            if '! Undefined control sequence' in pattern and i + 1 < len(lines):
                                # Try to extract the undefined command
                                next_line = lines[i + 1]
                                if next_line.strip():
                                    print(f"  {next_line.strip()}")
                            break
                    return None

            # Check for warnings about missing characters
            warning_patterns = [
                'Missing character:',
                'Some font shapes were not available',
            ]

            for pattern in warning_patterns:
                if pattern in log_content:
                    print(f"  LaTeX compilation warning detected:")
                    lines = log_content.split('\n')
                    warning_lines = [line.strip() for line in lines if pattern in line]
                    # Show first few warnings for context
                    for warning in warning_lines[:3]:
                        print(f"  {warning}")
                    if len(warning_lines) > 3:
                        print(f"  ... and {len(warning_lines) - 3} more warnings")
                    return None

        if dvi_file.exists():
            return dvi_file
        else:
            return None

    except subprocess.TimeoutExpired:
        print(f"  Compilation timeout")
        return None
    except Exception as e:
        print(f"  Compilation error: {e}")
        return None
