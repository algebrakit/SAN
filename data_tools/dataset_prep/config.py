"""
Configuration constants for the dataset preprocessing pipeline.
"""

# Commands that cause a line to be skipped entirely
FORBIDDEN_COMMANDS = [
    '\\limits', '\\aleph', '\\oplus', '\\models', '\\biguplus', '\\bigwedge', '\\bigvee', '\\coprod',
    '\\bigoplus', '\\propto', '\\Im', '\\Re', '\\wp', '\\xi', '\\zeta', '\\Xi', '\\iota', '\\mp', '\\dagger', '\\star', '\\bullet',
    '\\oint', '\\ominus', '\\mathfrak', '\\odot', '\\hbar', '\\triangleleft', '\\triangleq', '\\triangleleft',
    '\\supseteq', '\\subsetneq', '\\sqsubseteq', '\\rightleftharpoons', '\\Vdash', '\\lg', '\\pmod', '\\tbinom', '\\#', '\\&',
    # '\\\\', '\\choose', # to handle later
    # forbidden accents
    '\\breve', '\\acute', '\\grave', '\\mathring'
]

# Font style commands to remove (keeping content)
FONT_COMMANDS = [
    r'\\boldsymbol', r'\\mathbf', r'\\mathrm', r'\\mathbb', r'\\operatorname', r'\\boldsymbol',
    r'\\mathtt', r'\\mathsf', r'\\bold',
    r'\\textstyle', r'\\scriptstyle', r'\\scriptscriptstyle', r'\\mbox'
]

# Variant symbol replacements
VARIANT_REPLACEMENTS = {
    r'\\varsigma': r'\\sigma',
    r'\\vartheta': r'\\theta',
    r'\\varepsilon': r'\\epsilon',
    r'\\varphi': r'\\phi',
    r'\\varpi': r'\\pi',
    r'\\varrho': r'\\rho',
    r'\\kappa': r'k',
    r'\\Upsilon': r'Y',
    r'\\upsilon': r'v',
    r'\\Pi': r'\\prod',
    r'\\Sigma': r'\\sum',
    r'\\neq': r'\\ne',
    r'\\varnothing': r'\\emptyset',
    r'\\backslash': r'\\emptyset',
    r'\\lnot': r'\\neg',
    r'\\mapsto': r'\\rightarrow',
    r'\\cong': r'\\simeq',
    r'\\bigcirc': r'\\circ',
    r'\\smallsetminus': r'\\setminus',
    r'\\ell': r'l',
    r'\\nu': r'v',
    r'\\eta': r'n',
    r'\\chi': r'x',
    r'\\iint': r'\\int\\int',
    r'\\ll': r'< < ',
    r'\\gg': r'> > ',
    r'\\bar': r'\\overline',
    r'\\vec': r'\\overrightarrow',
    r'\\widehat': r'\\hat',
    r'\\widetilde': r'\\tilde',
    r'\\\|': r'\\Vert',
    r'\\rVert': r'\\Vert',
    r'\\lVert': r'\\Vert',
    r'\\parallel': r'\\Vert',
    r'\\mid': r'| ',
    r'\\vert': r'| ',
    r'\\prime': "'",
    r'\\big': r'',
    r'\\bigl': r'',
    r'\\bigr': r'',
    r'\\Big': r'',
    r'\\Bigl': r'',
    r'\\Bigr': r'',
    r'\\bigg': r'',
    r'\\biggl': r'',
    r'\\biggr': r'',
    r'\\Bigg': r'',
    r'\\vee': r'\\lor',
    r'\\wedge': r'\\land',
    r'\\tfrac': r'\\frac',
    r'\\dfrac': r'\\frac',
    r'\\cfrac': r'\\frac',
    r'\\dbinom': r'\\binom',
    r'\\tbinom': r'\\binom',
    r'\\bmod': r'\\mod',
    r'\\hookrightarrow': r'\\rightarrow',
    r'\\longrightarrow': r'\\rightarrow',
    r'\\to': r'\\rightarrow',
    r'\\gets': r'\\leftarrow',
    r'\\iff': r'\\Leftrightarrow',
    r'\\lbrack': r'[',
    r'\\rbrack': r']',
    r'\\lbrace': r'\{',
    r'\\rbrace': r'\}',
    r'\\dots': r'. . . ',
    r'\\cdots': r'. . . ',
    r'\\ldots': r'. . . ',
    r'\\dotsb': r'. . . ',
    r'\\dotsc': r'. . . ',
    r'\\colon': r': ',
    r'\\,': r'\\ ',
    r'\\;': r'\\ ',
    r'\\:': r'\\ ',
    r'\\>': r'\\ ',
    r'\\!': r'',
    r'(?<!\\)\\ ': r'\\ ',
    r'<': r'\\lt',
    r'>': r'\\gt',
    r'~': r' ',
    r'\\degree': r'^ { o }'
}

# Commands to detect and ensure spacing around
DETECT_COMMANDS = [
    'log', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh',
    'tanh', 'coth', 'ln', 'exp', 'sum', 'prod', 'lim', 'max', 'min', 'inf', 'sup', 'det', 'dim', 'gcd', 'lcm',
    'mod', 'arg', 'div', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'pi', 'rho', 'sigma', 'tau', 'phi', 'omega',
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Sigma', 'Phi', 'Omega', 'over'
]

# Special tokens that are always valid
ALWAYS_VALID_TOKENS = ['{', '}', '[', ']', '^', '_']
