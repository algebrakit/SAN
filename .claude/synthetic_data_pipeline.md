# Synthetic Data Generation Pipeline

## Overview

The synthetic data generation system creates handwritten mathematical expression images from LaTeX syntax. It uses a clever approach: compile LaTeX to get perfect spatial layouts, then replace rendered symbols with real handwritten samples.

## File Location
`data_tools/dataset_prep/synthetic/scripts/generate_dataset.py`

## Pipeline Architecture

### Step 1: LaTeX Input
- Reads LaTeX expressions from text file (one per line)
- Example input: `\frac{a}{b}`, `2x_0`, `\int_{0}^{1} x^2 dx`

### Step 2: Bounding Box Generation
**Component**: `LaTeXToDVIBoxes` (from `generate_boxes.py`)
- Compiles LaTeX expressions using the LaTeX compiler
- Generates DVI (DeVice Independent) file format
- Parses DVI file to extract bounding boxes for each symbol
- Returns structure containing:
  - Symbol positions (x, y coordinates)
  - Symbol dimensions (width, height)
  - Symbol identifiers (character codes)
  - Original LaTeX label

**Configuration**:
- DPI: 72 (dots per inch)
- `keep_temp=False` (cleanup temporary files)

### Step 3: Handwritten Synthesis
**Component**: `ExpressionSynthesizer` (from `synthesize_expression.py`)
- Takes bounding box data as input
- For each symbol in the layout:
  1. Looks up handwritten samples from symbol library
  2. Selects a random handwritten instance
  3. Transforms/scales it to fit the bounding box
  4. Places it at the correct position

**Parameters**:
- `preserve_aspect=True`: Maintains symbol aspect ratios
- `skip_missing=True`: Skips symbols not in library (instead of failing)

**Symbol Library**:
- Symbol index: `output/symbol_index.json`
- Symbol files: InkML format in `symbols/` directory
- Contains real handwritten samples from datasets

### Step 4: InkML Output
**Component**: `StrokeSet` (from `stroke_transformer.py`)
- Represents handwritten data as stroke sequences
- Each stroke is a sequence of (x, y, time) points
- Saves as InkML XML format

**File naming**:
```python
filename = f"expr_{i:04d}_{safe_expr}.inkml"
# Example: expr_0001_frac_a_b.inkml
```

### Step 5: Labels File
**Output**: `labels.txt`
**Format**: `filename.inkml\tlatex_expression`
```
expr_0001_frac_a_b.inkml	\frac{a}{b}
expr_0002_2x_0.inkml	2x_0
```

## Statistics Tracking

The generator monitors:
- `total_expressions`: Total LaTeX expressions processed
- `successful`: Successfully generated InkML files
- `failed_bbox`: Failed at bounding box generation stage
- `failed_synthesis`: Failed at handwriting synthesis stage
- `failed_saving`: Failed at InkML saving stage
- Success rate percentage

## Dependencies

### External Modules
- `generate_boxes.py`: LaTeX→DVI→BBox pipeline
- `synthesize_expression.py`: Symbol substitution logic
- `stroke_transformer.py`: Stroke data structures and InkML I/O

### External Tools
- LaTeX compiler (required for DVI generation)
- DVI parser (to extract bounding boxes)

### Data Dependencies
- `symbol_index.json`: Index mapping LaTeX symbols to handwritten samples
- `symbols/`: Directory of handwritten symbol InkML files
- Typically sourced from datasets like CROHME or MathWriting-2024

## Usage

```bash
python3 generate_dataset.py \
  --input latex.txt \
  --output-dir output \
  --symbol-index output/symbol_index.json \
  --symbols-dir symbols \
  --seed 42
```

## Key Design Decisions

1. **Why LaTeX→DVI→BBox?**
   - LaTeX provides perfect typesetting and spatial layout
   - DVI format contains exact positioning information
   - Easier than manually calculating math layout rules

2. **Why InkML output?**
   - InkML preserves stroke ordering and timing
   - Compatible with existing HMER datasets (CROHME)
   - Can be rendered to images later

3. **Why symbol substitution?**
   - Reuses real handwriting samples for realism
   - Avoids need for handwriting generation models
   - Maintains natural variation from human writers

4. **Why skip_missing=True?**
   - Allows partial synthesis when some symbols unavailable
   - Better than failing entire expression
   - Useful for iterative symbol library building

## Integration with SAN Model

This synthetic data can be used to:
1. Augment training data for the SAN model
2. Generate specific expression types that are rare in real datasets
3. Test the model on controlled variations
4. Pre-train before fine-tuning on real data

The output InkML files need to be:
1. Converted to images (rendering step)
2. Converted to hybrid tree labels (using SAN's data prep scripts)
3. Converted to pickle format for training

## Next Steps for Investigation

To fully understand the pipeline, need to examine:
- [ ] `generate_boxes.py` - How DVI parsing works
- [ ] `synthesize_expression.py` - Symbol lookup and transformation logic
- [ ] `stroke_transformer.py` - StrokeSet data structure
- [ ] `parse_dvi_to_boxes.py` - DVI file format parsing
- [ ] Symbol indexing system (`index_symbols.py`)
