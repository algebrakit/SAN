# Hybrid syntax for math expressions

The ML model for recognizing math expressions works with a domain-specific syntax called a 'hybrid tree'. A math expression is written as a sequence of symbols, 'struct' operations, and 'end of sequence' symbols (`<eos>`).

## Structure

Each line in the hybrid tree has the following format:
```
<id> <symbol> <parent_id> <parent_symbol> [above below sub sup L-sup inside right]
```

Where:
- **`<id>`**: Integer line number (starts at 1)
- **`<symbol>`**: Either a math symbol, `struct`, or `<eos>`
- **`<parent_id>`**: ID of the immediately preceding symbol in depth-first traversal
- **`<parent_symbol>`**: The symbol at `<parent_id>` (or region name for region entry)
- **Position indicators**: Seven values in fixed order: `above below sub sup L-sup inside right`
  - Non-None only on `struct` lines to declare which regions are active
  - All `None` for regular symbols and `<eos>` symbols

## Key Rules

### 1. Symbol Sequencing
By default, symbols are concatenated left-to-right. Each symbol references the immediately preceding symbol as its parent.

### 2. Struct Operation
The `struct` symbol is used to define spatial regions for sub-expressions:
- **Regions**: `above`, `below`, `sub`, `sup`, `L-sup` (left-superscript for radicals), `inside`, `right`
- The position indicators on a `struct` line declare which regions are active (non-None values)
- First symbol entering a region uses the **region name** as `<parent_symbol>` and the **struct's id** as `<parent_id>`

### 3. Region Traversal
- Symbols within a region reference each other normally (parent = previous symbol)
- Each declared region ends with `<eos>` that references the last symbol in that region

### 4. End of Sequence (`<eos>`)
- Each active region ends with `<eos>`
- **If struct has "right" region**: Final `<eos>` appears after the right region completes
- **If struct has NO "right" region**: The last region's `<eos>` ends the expression (no additional `<eos>`)
- Top-level expressions end with `<eos>` referencing the last symbol

## Examples

### Example 1: Simple Expression
**LaTeX**: `9 + 5`

**Hybrid**:
```
1 9 0 <sos>   None None None None None None None
2 + 1 9       None None None None None None None
3 5 2 +       None None None None None None None
4 <eos> 3 5   None None None None None None None
```

**Explanation**:
- Line 1: Symbol `9` starts the expression (parent is `<sos>`)
- Line 2: Symbol `+` follows `9` (parent is `1 9`)
- Line 3: Symbol `5` follows `+` (parent is `2 +`)
- Line 4: Expression ends with `<eos>` (parent is `3 5`)

---

### Example 2: Fraction with Right Region
**LaTeX**: `\frac{a + b}{c} + 4`

**Hybrid**:
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

**Explanation**:
- Line 1: `\frac` construct starts
- Line 2: `struct` declares three regions: `above`, `below`, `right`
- Lines 3-6: "above" region contains `a + b`, ends with `<eos>`
  - Line 3: `a` enters "above" region (parent is `2 above`)
  - Line 4: `+` follows `a` (parent is `3 a`)
  - Line 5: `b` follows `+` (parent is `4 +`)
  - Line 6: `<eos>` ends "above" region (parent is `5 b`)
- Lines 7-8: "below" region contains `c`, ends with `<eos>`
  - Line 7: `c` enters "below" region (parent is `2 below`)
  - Line 8: `<eos>` ends "below" region (parent is `7 c`)
- Lines 9-11: "right" region contains `+ 4`, ends with final `<eos>`
  - Line 9: `+` enters "right" region (parent is `2 right`)
  - Line 10: `4` follows `+` (parent is `9 +`)
  - Line 11: Final `<eos>` ends expression (parent is `10 4`)

---

### Example 3: Square Root without Right Region
**LaTeX**: `\sqrt[2]{x}`

**Hybrid**:
```
1 \sqrt 0 <sos>   None None None None None None None
2 struct 1 \sqrt  None None None None L-sup inside None
3 2 2 L-sup       None None None None None None None
4 <eos> 3 2       None None None None None None None
5 x 2 inside      None None None None None None None
6 <eos> 5 x       None None None None None None None
```

**Explanation**:
- Line 2: `struct` declares `L-sup` and `inside` regions (NO "right" region)
- Lines 3-4: "L-sup" region contains `2`
- Lines 5-6: "inside" region contains `x`
- Line 6: Final `<eos>` ends expression (no additional `<eos>` needed since there's no "right" region)

---

### Example 4: Square Root with Right Region
**LaTeX**: `\sqrt[2]{x} + 1`

**Hybrid**:
```
1 \sqrt 0 <sos>   None None None None None None None
2 struct 1 \sqrt  None None None None L-sup inside right
3 2 2 L-sup       None None None None None None None
4 <eos> 3 2       None None None None None None None
5 x 2 inside      None None None None None None None
6 <eos> 5 x       None None None None None None None
7 + 2 right       None None None None None None None
8 1 7 +           None None None None None None None
9 <eos> 8 1       None None None None None None None
```

**Explanation**:
- Line 2: `struct` declares `L-sup`, `inside`, AND `right` regions
- Lines 3-6: "L-sup" and "inside" regions as before
- Lines 7-9: "right" region contains `+ 1`
- Line 9: Final `<eos>` ends expression (appears after "right" region completes)
