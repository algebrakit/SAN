# Expression Module Reference

This module provides a tree-based representation of mathematical LaTeX expressions with bidirectional conversion between LaTeX strings, parse trees (GTD), and hybrid syntax for ML training.

## Quick Start

```python
from utils.Expression import Expression, Symbol, FractionConstruct, LatexOptions

# Parse LaTeX to Expression
expr = Expression.fromLatex(r'\frac { a + b } { c }')

# Convert Expression to LaTeX (with default options)
latex = expr.toLatex()

# Convert with custom options
options = LatexOptions(convertMatrices=False, convertLog=True)
latex = expr.toLatex(options)

# Convert to hybrid syntax (for ML model)
hybrid = expr.to_hybrid()

# Parse GTD list to Expression
gtd = [['\\frac', 1, 0, '<sos>'], ['a', 2, 1, 'above'], ['c', 3, 1, 'below']]
expr = Expression().from_gtd_list(gtd)
```

## Architecture Overview

### Core Design
- **Tree-based representation**: Mathematical expressions as parent-child trees
- **Bidirectional conversion**: LaTeX ↔ Expression ↔ GTD ↔ Hybrid
- **Strong typing**: Each construct has dedicated class
- **Parent tracking**: Every node maintains `.parent` reference

### Class Hierarchy

```
LatexItem (abstract base)
├── Symbol - Atomic symbols with optional sub/superscripts
└── Construct (abstract)
    ├── FractionConstruct - \frac{above}{below}
    ├── SqrtConstruct - \sqrt[L-sup]{inside}
    ├── AboveBelowConstruct - \sum_{below}^{above}
    ├── AccentConstruct - \bar{x}, \vec{v}, \underline{AB}
    ├── StackConstruct - Matrix/vector rows
    ├── RowConstruct - Matrix row items
    └── LogConstruct - \log with base notation
```

## LatexOptions

**File**: `base.py`

The `LatexOptions` class controls how expressions are converted to LaTeX format:

```python
@dataclass
class LatexOptions:
    convertMatrices: bool = True    # Convert stack constructs to \begin{matrix}...\end{matrix}
    convertLog: bool = False        # Convert \lognl to \log format (not yet implemented)
```

**Usage**:
```python
from utils.Expression import Expression, LatexOptions

expr = Expression.fromLatex(r'( \stack { \row { 1 & 2 } \row { 3 & 4 } } )')

# Use default options
latex1 = expr.toLatex()  # Converts to \begin{pmatrix}...\end{pmatrix}

# Disable matrix conversion
options = LatexOptions(convertMatrices=False)
latex2 = expr.toLatex(options)  # Keeps as \stack{\row{...}}

# Custom options
options = LatexOptions(convertMatrices=True, convertLog=True)
latex3 = expr.toLatex(options)
```

**Note**: All `toLatex()` methods accept an optional `LatexOptions` parameter and pass it to child expressions. If `None`, default options are used.

## Key Classes

### Expression Container
**File**: `expression.py`

Container for a sequence of LatexItem objects. Serves as:
- Root of expression tree
- Horizontal sequence of items
- Child region for constructs (e.g., numerator of fraction)

```python
class Expression:
    items: List[LatexItem]  # Sequence of symbols/constructs
    parent: Optional[LatexItem]  # Parent construct/None for root
```

### Symbol
**File**: `constructs.py`

Represents atomic symbols with optional subscript/superscript:

```python
class Symbol(LatexItem):
    value: str  # The symbol itself
    sub: Optional[Expression]  # Subscript
    sup: Optional[Expression]  # Superscript
```

**Example**: `x_i^2` has value='x', sub=Expression([Symbol('i')]), sup=Expression([Symbol('2')])

**Regions**: `sub`, `sup` (in that order)

### Constructs

All constructs support optional superscript on the entire construct.

#### FractionConstruct
**LaTeX**: `\frac{numerator}{denominator}`

```python
above: Expression  # Numerator
below: Expression  # Denominator
sup: Optional[Expression]  # Superscript on entire fraction
```

**Regions**: `above`, `below`, `sup`

#### SqrtConstruct
**LaTeX**: `\sqrt{radicand}` or `\sqrt[degree]{radicand}`

```python
inside: Expression  # Radicand
l_sup: Optional[Expression]  # Degree (left-superscript)
sup: Optional[Expression]  # Superscript on entire sqrt
```

**Regions**: `L-sup`, `inside`, `sup`

#### AboveBelowConstruct
**LaTeX**: `\sum_{lower}^{upper}`, `\prod`, `\int`, `\lim`, etc.

```python
construct_type: str  # '\\sum', '\\prod', '\\int', etc.
below: Optional[Expression]  # Lower limit
above: Optional[Expression]  # Upper limit
sup: Optional[Expression]  # Additional superscript
```

**Regions**: `below`, `above`, `sup`

#### AccentConstruct
**LaTeX**: `\bar{x}`, `\vec{v}`, `\underline{AB}`, etc.

```python
construct_type: str  # '\\bar', '\\vec', '\\underline', etc.
is_above: bool  # True for accents above, False for below
child: Expression  # Content under/over accent
sub: Optional[Expression]
sup: Optional[Expression]
```

**Above accents**: `\vec`, `\dot`, `\ddot`, `\bar`, `\tilde`, `\hat`, `\overline`, `\widehat`, `\widetilde`

**Below accents**: `\underline`

**Regions**: For above: `below`, `sub`, `sup`; For below: `above`, `sub`, `sup`

#### StackConstruct & RowConstruct
**LaTeX**: `\stack{\row{a&b}\row{c&d}}` (internal representation)

Matrices/vectors use row-chaining via `below` region:

```python
# StackConstruct
inside: Expression  # First row
mtype: Optional[str]  # 'pmatrix', 'bmatrix', etc.

# RowConstruct
inside: Expression  # Row items (separated by &)
below: Optional[Expression]  # Next row
```

**Matrix structure**:
```
StackConstruct
└── inside: RowConstruct(row1)
    ├── inside: [a, &, b]
    └── below: RowConstruct(row2)
        ├── inside: [c, &, d]
        └── below: None
```

#### LogConstruct
**LaTeX**: `\lognl[base]` (Dutch notation) or `\log_base`

```python
l_sup: Optional[Expression]  # Base in \lognl[base]
sub: Optional[Expression]  # Base in \log_base
sup: Optional[Expression]
```

**Regions**: `L-sup` or `sub`

## Format Conversions

### LaTeX Format
**File**: `parser.py`

**Syntax**: Space-separated tokens with braces for grouping

```latex
\frac { a + b } { c }
\sqrt [ 2 ] { x }
x ^ { 2 }
```

**Important**:
- Space-separated tokens required
- Braces `{ }` for grouping
- Brackets `[ ]` for optional arguments
- Multiple superscripts nest right-to-left: `a^{b}^{c}` → `a^(b^c)`

### GTD (Gene Tree Description) Format
**File**: `gtd_parser.py`

List representation: `[symbol, id, parent_id, parent_symbol_or_region]`

```python
[
    ['<sos>', 0, -1, None],
    ['\\frac', 1, 0, '<sos>'],
    ['a', 2, 1, 'above'],
    ['c', 3, 1, 'below']
]
```

### Hybrid Syntax Format
**File**: `hybrid.py`, **Documentation**: `hybrid_syntax.md`

Linearized tree with spatial regions for ML training:

```
<id> <symbol> <parent_id> <parent_symbol> [above below sub sup L-sup inside right]
```

**Key concepts**:
- Default left-to-right sequencing
- `struct` operation declares spatial regions
- `<eos>` marks end of regions
- Region order: above, below, sub, sup, L-sup, inside, right

**Example**: `\frac{a+b}{c}+4`
```
1 \frac 0 <sos>   None None None None None None None
2 struct 1 \frac  above below None None None None right
3 a 2 above       None None None None None None None
4 + 3 a           None None None None None None None
5 b 4 +           None None None None None None None
6 <eos> 5 b       None None None None None None None
7 c 2 below       None None None None None None None
8 <eos> 7 c       None None None None None None None
9 + 2 right       None None None None None None None
10 4 9 +          None None None None None None None
11 <eos> 10 4     None None None None None None None
```

## CRITICAL: Region Ordering Convention

**Fixed region order** (enforced everywhere):
```
below, above, sub, sup, L-sup, inside, right
```

This order MUST be maintained in:
- `get_regions()` methods on all constructs
- Hybrid syntax generator
- Any code processing regions

**Why**: The ML model expects regions in this exact order. Changing it breaks training/inference.

## Parent-Child Relationships

Bidirectional tree structure:

```python
# Parent → Children
expression.items  # List of LatexItem objects

# Child → Parent
item.parent  # Reference to containing Expression

# Construct → Regions
fraction.above  # Expression for numerator
fraction.below  # Expression for denominator

# Region → Construct
fraction.above.parent  # Points back to fraction
```

**Example tree** for `\frac{a+b}{c}`:
```
Expression (root)
└── FractionConstruct
    ├── above: Expression
    │   ├── Symbol('a')  [.parent → above]
    │   ├── Symbol('+')  [.parent → above]
    │   └── Symbol('b')  [.parent → above]
    └── below: Expression
        └── Symbol('c')  [.parent → below]
```

## Matrix Detection Algorithm

**File**: `expression.py:75-130` (`_handleStackConstructs()`)

Automatically detects matrix types based on surrounding brackets:

```python
( \stack{...} )      → pmatrix
[ \stack{...} ]      → bmatrix
\{ \stack{...} \}    → Bmatrix
| \stack{...} |      → vmatrix
|| \stack{...} ||    → Vmatrix
\{ \stack{...}       → cases (left bracket only)
\stack{...}          → matrix (no brackets)
```

**Algorithm**:
1. Find StackConstruct items
2. Check previous/next items for brackets
3. For `||` notation, check two positions back/forward
4. Remove bracket symbols from list
5. Set matrix type via `set_matrix_type()`

**Usage**:
```python
expr = Expression.fromLatex(r'( \stack { \row { 1 & 2 } \row { 3 & 4 } } )')
latex = expr.toLatex(convertMatrices=True)
# Result: '\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}'
```

## Common Usage Patterns

### Parse and Convert LaTeX
```python
# Parse LaTeX
expr = Expression.fromLatex(r'\frac { a + b } { c }')

# Convert back to LaTeX
latex = expr.toLatex()

# Convert matrices
latex = expr.toLatex(convertMatrices=True)
```

### Manual Construction
```python
from utils.Expression import Expression, Symbol, FractionConstruct

# Build expression manually
expr = Expression([
    FractionConstruct(
        construct_type='\\frac',
        above=Expression([Symbol('a'), Symbol('+'), Symbol('b')]),
        below=Expression([Symbol('c')])
    )
])

latex = expr.toLatex()  # '\frac { a + b } { c }'
```

### GTD to LaTeX
```python
gtd_list = [
    ['<sos>', 0, -1, None],
    ['\\frac', 1, 0, '<sos>'],
    ['a', 2, 1, 'above'],
    ['+', 3, 2, 'a'],
    ['b', 4, 3, '+'],
    ['c', 5, 1, 'below']
]

expr = Expression().from_gtd_list(gtd_list)
latex = expr.toLatex()  # '\frac { a + b } { c }'
```

### Expression to Hybrid (ML Training)
```python
expr = Expression.fromLatex(r'x ^ { 2 }')
hybrid_lines = expr.to_hybrid()
# Returns list of [id, symbol, parent_id, parent_symbol, regions...]
```

### Tree Traversal
```python
def visit_all_symbols(expr: Expression):
    """Recursively visit all symbols in expression."""
    for item in expr.items:
        if isinstance(item, Symbol):
            print(f"Symbol: {item.value}")

        # Visit all child regions
        for region_name, region_expr in item.get_children().items():
            visit_all_symbols(region_expr)
```

## Important Implementation Details

### 1. Space-Separated LaTeX Tokens
LaTeX parsing requires **space-separated tokens**:
- ✅ Correct: `\frac { a + b } { c }`
- ❌ Wrong: `\frac{a+b}{c}` (will fail to parse)

### 2. Multiple Superscripts
Multiple consecutive superscripts nest right-to-left:
```python
# a^{b}^{c} becomes a^(b^c)
expr = Expression.fromLatex('a ^ { b } ^ { c }')
# Tree: Symbol('a') with sup=Expression([Symbol('b')])
#       where Symbol('b') has sup=Expression([Symbol('c')])
```

### 3. Region Order is Sacred
Always return regions in the canonical order from `get_regions()`:
```python
def get_regions(self):
    return [
        ('below', self.below),
        ('above', self.above),
        ('sub', self.sub),
        ('sup', self.sup),
        # ... etc in fixed order
    ]
```

### 4. Parent References Must Be Set
When constructing expressions manually, set parent references:
```python
expr = Expression([Symbol('a')])
# Parent is automatically set in Expression.__init__()
```

### 5. Matrix Row Chaining
Matrices chain rows via `below` region, not a list:
```python
# First row points to second via below
row1 = RowConstruct(inside=Expression([...]), below=Expression([row2]))
row2 = RowConstruct(inside=Expression([...]), below=None)
```

## Common Gotchas

1. **Forgetting spaces in LaTeX**: Parser requires space-separated tokens
2. **Wrong region order**: Breaking canonical order breaks hybrid syntax
3. **Missing parent references**: Can cause traversal issues
4. **Matrix detection requires brackets**: StackConstruct alone won't detect type
5. **Multiple superscripts**: They nest, not replace each other

## File Organization

```
utils/Expression/
├── CLAUDE.md           # This file
├── README.md           # User documentation
├── hybrid_syntax.md    # Detailed hybrid syntax spec
├── base.py            # LatexItem abstract base
├── constructs.py      # Symbol and all Construct classes
├── expression.py      # Expression container class
├── parser.py          # LaTeX → Expression parser
├── gtd_parser.py      # GTD → Expression parser
├── hybrid.py          # Expression → Hybrid converter
└── utils.py           # Utility functions

```

## Testing

Test files demonstrate usage patterns:
- Check test files for examples of parsing and conversion
- Look for edge cases in test data (nested constructs, matrices, etc.)

## References

- **Main documentation**: README.md
- **Hybrid syntax spec**: hybrid_syntax.md
- **CVPR 2022 paper**: For theoretical background on syntax-aware decoding

## Quick Reference

| Task | Code |
|------|------|
| Parse LaTeX | `Expression.fromLatex(latex_str)` |
| To LaTeX (default) | `expr.toLatex()` |
| To LaTeX (custom) | `expr.toLatex(LatexOptions(convertMatrices=False))` |
| From GTD | `Expression().from_gtd_list(gtd)` |
| To Hybrid | `expr.to_hybrid()` |
| Get children | `item.get_children()` |
| Get parent | `item.parent` |
| Create options | `LatexOptions(convertMatrices=True, convertLog=False)` |

---

**When modifying this module**: Always maintain region order, preserve parent references, and test all format conversions.
