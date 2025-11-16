# Handwritten Math Expression Synthesis Pipeline

This directory contains tools for generating synthetic handwritten mathematical expressions from LaTeX by replacing DVI bounding boxes with real handwritten symbol strokes, output as InkML files.

## Directory Structure

```
synthetic/
├── scripts/                      # Python scripts
│   ├── index_symbols.py         # Indexes symbol InkML files
│   ├── analyze_symbol_coverage.py # Analyzes symbol coverage
│   └── mathwriting-book.ipynb   # Reference notebook
├── input/                        # Input files
│   └── latex.tex                # LaTeX source files
├── output/                       # Generated outputs
│   ├── symbol_index.json        # Symbol library index
│   ├── coverage_report.json     # Coverage analysis (JSON)
│   └── coverage_report.md       # Coverage analysis (Markdown)
├── symbols/                      # Symlink to InkML symbol library (6,423 files)
├── boxes.jsonl                  # Bounding box data
├── latex.txt                    # Sample LaTeX expressions
├── SYNTHESIS_BACKLOG.md         # Project backlog/roadmap
└── PHASE1_SUMMARY.md            # Phase 1 completion summary
```

## Quick Start

### Complete Dataset Generation (End-to-End)

Generate a complete dataset with one command:

1. create symbols index
   `python3 scripts/index_symbols.py --symbols-dir symbols --output output/symbol_index.json`

2. create dataset
   `python3 scripts/generate_dataset.py --input latex.txt --output-dir output/dataset`

# Output:
#   output/dataset/inkml/        - InkML files
#   output/dataset/labels.txt    - InkML-LaTeX mappings

---

## Individual Pipeline Steps

If you need more control, you can run each step separately:

### 1. Index Symbol Library

Index all 6,423 handwritten symbol InkML files:

```bash
cd /Users/martijnslob/github/SAN/data_tools/dataset_prep/synthetic

python3 scripts/index_symbols.py \
  --symbols-dir symbols \
  --output output/symbol_index.json
```

**Output**: `output/symbol_index.json` with 229 unique symbols

### 2. Analyze Symbol Coverage

Check which symbols are available for your LaTeX expressions:

```bash
python3 scripts/analyze_symbol_coverage.py \
  --index output/symbol_index.json \
  --latex latex.txt \
  --boxes boxes.jsonl \
  --output output/coverage_report.json \
  --markdown output/coverage_report.md
```

**Outputs**:
- `output/coverage_report.json` - Machine-readable coverage data
- `output/coverage_report.md` - Human-readable report

## Current Status

### Phase 1: Infrastructure Setup - ✓ COMPLETE
- [x] Symbol library indexer
- [x] Symbol coverage analyzer
- [x] 82.4% coverage on test expressions

### Phase 2: Core Synthesis Engine - ✓ COMPLETE
- [x] InkML stroke transformer
- [x] Composite InkML generator
- [x] Successfully synthesized sample expressions

### Phase 3: DVI Processing - ✓ COMPLETE
- [x] LaTeX to DVI bounding box generator
- [x] DVI parser for existing files
- [x] Full automation working
- [x] End-to-end testing successful

**Result**: Successfully generated bounding boxes from LaTeX and synthesized handwritten expressions. See `PHASE3_COMPLETE.md` for details.

### Phase 4: End-to-End Pipeline - ✓ COMPLETE
- [x] InkML file generation
- [x] Label generation integration
- [x] Automation script
- [x] Complete dataset generation in one command

**Result**: Single script generates InkML files and labels from LaTeX expressions. See `PHASE4_COMPLETE.md` for details.

## All Phases Complete! 🎉

The handwritten math expression synthesis pipeline is fully operational. See individual phase summaries for details.

## Symbol Library Statistics

- **Total InkML files**: 6,423
- **Unique LaTeX labels**: 229
- **Average variants per symbol**: 28
- **Coverage on test expressions**: 82.4%

Top symbols by variant count:
1. `\frac` - 60 variants
2. `\sqrt` - 58 variants
3. `c` - 51 variants
4. `E` - 48 variants

## Usage Examples

### Index with custom paths

```bash
python3 scripts/index_symbols.py \
  --symbols-dir /path/to/symbols \
  --output output/my_index.json \
  --verbose
```

### Analyze coverage from multiple sources

```bash
python3 scripts/analyze_symbol_coverage.py \
  --index output/symbol_index.json \
  --boxes boxes.jsonl \
  --latex input/my_expressions.txt
```

### Synthesize handwritten expressions

```bash
# Basic synthesis
python3 scripts/synthesize_expression.py \
  --index output/symbol_index.json \
  --symbols-dir symbols \
  --boxes boxes.jsonl \
  --output output/synthesized

# With reproducible seed
python3 scripts/synthesize_expression.py \
  --index output/symbol_index.json \
  --symbols-dir symbols \
  --boxes boxes.jsonl \
  --output output/synthesized \
  --seed 42
```

### Render InkML to image

```bash
# Render single expression
python3 scripts/render_inkml.py \
  output/synthesized/expr_0001_3x^2=1.inkml \
  output/preview.bmp

# Custom size
python3 scripts/render_inkml.py \
  output/synthesized/expr_0001_3x^2=1.inkml \
  output/preview.bmp \
  --size 1600x800
```

## Next Steps

1. Build stroke transformation utilities (Phase 2-1)
2. Create composite InkML generator (Phase 2-2)
3. Integrate with rendering pipeline (Phase 4)

See `SYNTHESIS_BACKLOG.md` for complete implementation plan.

## Dependencies

- Python 3.7+
- `xml.etree.ElementTree` (built-in)
- `tqdm` (for progress bars)

## Notes

- The `symbols/` directory is a symlink to the MathWriting dataset
- Missing symbols (like `\lim`, `\to`) can be added later or skipped
- Spaces in expressions are handled by bounding box positions
