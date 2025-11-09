# Synthetic Data Generation System

This system generates synthetic handwritten mathematical expressions by compiling LaTeX to DVI format, extracting glyph positions and bounding boxes, then replacing each symbol with real handwritten strokes from a library of 6,423 InkML files.

---

## Quick Start Commands

**Generate bounding boxes from LaTeX**:
```bash
cd data_tools/dataset_prep/synthetic
python3 scripts/generate_boxes.py --input latex.txt --output output/boxes.jsonl
```

**Generate complete dataset (LaTeX → InkML)**:
```bash
python3 scripts/generate_dataset.py \
    --input latex.txt \
    --output-dir output/dataset \
    --seed 42
```

**Index symbol library**:
```bash
python3 scripts/index_symbols.py --symbols-dir symbols/ --output output/symbol_index.json
```

**Render InkML to image**:
```bash
python3 scripts/render_inkml.py --input output/dataset/inkml/expr_0001.inkml --output preview.png
```

---

## Pipeline Overview

```
LaTeX Expression (e.g., "x^2+\frac{1}{2}")
    ↓
[generate_boxes.py] Compile to DVI → Parse glyphs & boxes → Map to tokens
    ↓
Bounding Boxes JSONL: [{"token": "x", "xMin": 0, "yMin": -10, ...}, ...]
    ↓
[synthesize_expression.py] Load random symbol variants → Fit to bboxes
    ↓
Combined InkML: All symbols positioned in expression layout
    ↓
Save as .inkml file with <annotation type="label">LaTeX</annotation>
```

**Key Scripts**:
- `generate_boxes.py`: LaTeX → DVI → bounding boxes (via `LaTeXToDVIBoxes` class)
- `synthesize_expression.py`: Bounding boxes + symbol library → positioned strokes
- `stroke_transformer.py`: Scale/translate strokes to fit bboxes
- `generate_dataset.py`: End-to-end orchestration
- `index_symbols.py`: Build searchable symbol library index

---

## Critical Technical Details

### DVI Coordinate System (IMPORTANT!)

**DVI uses inverted Y-axis**:
- Origin at top-left
- X increases right (normal)
- **Y increases downward** (inverted!)
- Baseline at y ≈ 0
- Superscripts have positive y (~8-10)
- Subscripts have negative y

**Always negate y when creating bboxes**:
```python
bbox = {
    "xMin": float(x),
    "yMin": float(-y - height),  # Negate to flip axis
    "xMax": float(x + width),
    "yMax": float(-y)
}
```

### Token/Glyph Mapping Challenge

**LaTeX tokens ≠ DVI glyphs** (1:1 mapping impossible):
- `\cos` (1 token) → `c`, `o`, `s` (3 glyphs)
- `\frac` (1 token) → no glyph, only a box
- `\sqrt{x}` (2 tokens) → 2 glyphs + 1 box
- `^` (1 token) → no glyph, only position shift
- `{`, `}`, `_` → stripped by tokenizer

**Solution**: Sequential mapping with special cases for complex constructs.

---

## Special Cases

### Square Root (`\sqrt`)

**Challenge**: `\sqrt{x+1}` produces:
- 1 glyph: radical (√) at elevated y (~8.24)
- 1 box: overline over radicand
- N glyphs: radicand content at baseline (y ≈ 0)

**Solution** (lines 398-460 in `generate_boxes.py`):
1. Pre-scan boxes to find sqrt overlines (high y-position)
2. When processing `\sqrt` token:
   - Use overline box width to determine radicand extent
   - Count glyphs within `box_x` to `box_x + box_width` at baseline
   - Create single bbox covering radical + radicand
   - **Skip radical glyph**: `glyph_idx += 1`
   - **Skip radicand glyphs**: `glyph_idx += radicand_glyph_count`
   - **Skip radicand tokens**: `token_idx += radicand_glyph_count`
   - **Skip spacing tokens**: `\\ `, `\,`, etc.

**Recent Bug Fix**: Radicand glyphs were being processed twice (once in sqrt bbox, once as separate tokens). Now properly skipped to prevent duplicates.

### Fractions (`\frac`)

**Detection**: `classify_box()` finds horizontal boxes with glyphs above AND below

**Processing**:
- Token `\frac` is skipped (no glyph)
- DVI box becomes `\frac` token with bbox for the fraction bar
- Numerator/denominator glyphs processed normally

### Case Environment (`\begin{cases}`)

**Processing** (lines 358-396):
- Map `\begin{cases}` → `\{` token
- Extend bbox vertically to span all case lines
- Skip `\\` and `\end{cases}` tokens (no glyphs)

### Custom Macro (`\lognl`)

**Definition**: `\newcommand\lognl[1][]{{\mathop{{^{#1}\mathrm{log}}}}}`

**Processing** (lines 462-517):
- 4 glyphs: superscript digit + `l` + `o` + `g`
- Create separate bbox for superscript using actual character
- Normalize heights for `l`, `o`, `g`
- Skip bracket tokens `[`, digit, `]`

---

## Debugging Pitfalls

**1. Token/Glyph Mismatch**:
- New LaTeX commands may expand to multiple glyphs
- Check `expand_latex_tokens()` or add special handling
- Verify `token_idx` and `glyph_idx` stay synchronized

**2. Duplicate Bboxes** (like sqrt bug):
- Glyphs consumed without incrementing indices
- Add explicit counting and skipping for complex constructs
- Use `continue` to skip common bbox append

**3. DVI Y-Axis Confusion**:
- Y increases downward in DVI (inverted!)
- Always negate: `yMin = -y - height`, `yMax = -y`
- Superscripts have positive DVI y but negative math y

**4. Box Misclassification**:
- Sqrt overlines confused with generic overlines
- Use position heuristics: `y > 5.0` for sqrt
- Context-aware classification with token list

**5. Structural Tokens**:
- Don't produce glyphs: `^`, `_`, `{`, `}`, `\frac`, `\left`, `\right`, spacing
- Already skipped at lines 367-370 in `generate_boxes.py`

**6. Missing Symbols**:
- Check `symbol_index.json` coverage
- Run `analyze_symbol_coverage.py`
- Use `--skip-missing` flag to continue despite gaps

---

## File Structure

```
data_tools/dataset_prep/synthetic/
├── CLAUDE.md                           # This file
├── scripts/
│   ├── generate_boxes.py               # LaTeX → DVI → bboxes (main algorithm)
│   ├── generate_dataset.py             # End-to-end pipeline
│   ├── synthesize_expression.py        # Bbox → positioned strokes
│   ├── stroke_transformer.py           # Geometric transformations
│   ├── index_symbols.py                # Build symbol library index
│   ├── render_inkml.py                 # InkML → PNG rendering
│   └── analyze_symbol_coverage.py      # Check symbol availability
├── symbols/                            # 6,423 handwritten symbol InkML files
├── latex.txt                           # Input: LaTeX expressions (1 per line)
└── output/
    ├── boxes_generated.jsonl           # Generated bboxes
    ├── symbol_index.json               # Symbol library index
    └── dataset/
        ├── inkml/                      # Generated .inkml files
        └── labels.txt                  # InkML filename → LaTeX mapping
```

---

## Integration with SAN Model

Generated InkML files can be converted to the SAN model's hybrid tree format using:
```bash
cd ../..  # Back to data_tools/dataset_prep/
python3 prepare_synthetic_data.py  # (if such script exists)
```

The synthetic data augments real handwritten datasets (CROHME, MathWriting) to improve model generalization.
