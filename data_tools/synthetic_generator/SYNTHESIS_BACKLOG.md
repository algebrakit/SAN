# Handwritten Math Expression Synthesis - Implementation Backlog

## Project Goal
Generate synthetic handwritten mathematical expression images from LaTeX by:
1. Extracting symbol positions from LaTeX/DVI compilation
2. Replacing each symbol with real handwritten inkml strokes
3. Rendering composite images and generating labels for SAN training

---

## Available Resources
- **6,423 handwritten symbol inkml files** in `data_tools/mathwriting-tools/symbols/`
- **Bounding box data** in `boxes.jsonl` (LaTeX expressions with DVI symbol positions)
- **LaTeX expressions** in `latex.txt` (sample expressions)
- **Existing utilities**:
  - InkML parser (`stroke_processing/inkml_parser.py`)
  - Stroke-to-image converter (`stroke_processing/stroke2img.py`)
  - Hybrid label generator (`gen_hybrid_data.py`)
- **Reference notebook**: `mathwriting-book.ipynb` with rendering examples

---

## Backlog Items

### PHASE 1: Infrastructure Setup

#### [P1-1] Symbol Library Indexer
**Status**: Not Started
**Priority**: High
**Effort**: Small (2-4 hours)

**Description**: Create a tool to index all 6,423 symbol inkml files by their LaTeX label.

**Tasks**:
- [ ] Create `data_tools/dataset_prep/synthetic/index_symbols.py`
- [ ] Parse all inkml files in `symbols/` directory
- [ ] Extract label from `<annotation type="label">` tag
- [ ] Extract stroke data and compute bounding box
- [ ] Group symbols by label (multiple variants per symbol)
- [ ] Output `symbol_index.json` with structure:
  ```json
  {
    "\\alpha": ["0005e477f85ab99f.inkml", "..."],
    "0": ["...", "..."],
    "x": ["...", "..."]
  }
  ```
- [ ] Add statistics reporting (symbols per label, total unique labels)

**Deliverables**:
- `index_symbols.py` script
- `symbol_index.json` output file
- Symbol coverage report

---

#### [P1-2] Symbol Coverage Analysis
**Status**: Not Started
**Priority**: Medium
**Effort**: Small (1-2 hours)

**Description**: Analyze which symbols from typical LaTeX expressions are covered by the inkml library.

**Tasks**:
- [ ] Load `symbol_index.json`
- [ ] Parse LaTeX expressions from `latex.txt` and `boxes.jsonl`
- [ ] Tokenize LaTeX (use existing tokenizer from `mathwriting-book.ipynb`)
- [ ] Report coverage: which symbols are available vs. missing
- [ ] Identify fallback strategy for missing symbols

**Deliverables**:
- `analyze_symbol_coverage.py` script
- Coverage report (markdown table)

---

### PHASE 2: Core Synthesis Engine

#### [P2-1] InkML Stroke Transformer
**Status**: Not Started
**Priority**: High
**Effort**: Medium (4-6 hours)

**Description**: Utility to transform inkml strokes (scale, translate, rotate).

**Tasks**:
- [ ] Create `data_tools/stroke_processing/stroke_transformer.py`
- [ ] Implement functions:
  - `normalize_strokes(strokes)` - normalize to unit bounding box
  - `scale_strokes(strokes, width, height)` - scale to target size
  - `translate_strokes(strokes, dx, dy)` - translate position
  - `get_bounding_box(strokes)` - compute bbox
  - `combine_strokes(strokes_list)` - merge multiple stroke sets
- [ ] Add tests with sample inkml
- [ ] Preserve stroke ordering and structure

**Deliverables**:
- `stroke_transformer.py` module
- Unit tests or validation script

---

#### [P2-2] Composite InkML Generator
**Status**: Not Started
**Priority**: High
**Effort**: Large (8-12 hours)

**Description**: Main synthesis engine that creates handwritten expression inkml from bounding boxes.

**Tasks**:
- [ ] Create `data_tools/dataset_prep/synthesize_handwritten_expression.py`
- [ ] Input: Bounding box JSON (from `boxes.jsonl` format)
- [ ] For each symbol/token:
  - [ ] Look up symbol in `symbol_index.json`
  - [ ] Select random variant (or specific variant)
  - [ ] Load inkml file
  - [ ] Extract strokes
  - [ ] Normalize, scale, translate to target bbox
  - [ ] Add to composite stroke collection
- [ ] Handle special cases:
  - [ ] Missing symbols (skip or use placeholder)
  - [ ] Structural symbols (`\frac`, `\sqrt`)
  - [ ] Multi-character tokens
- [ ] Output composite inkml file with all strokes
- [ ] Preserve metadata (original LaTeX, normalizedLabel)

**Deliverables**:
- `synthesize_handwritten_expression.py` script
- Sample output inkml files
- Error handling for edge cases

---

### PHASE 3: DVI Processing (Optional - Can use existing boxes.jsonl)

#### [P3-1] LaTeX to DVI Bounding Box Extractor
**Status**: Not Started
**Priority**: Low
**Effort**: Large (12-16 hours)

**Description**: Extract symbol bounding boxes from LaTeX compilation (if not using existing boxes.jsonl).

**Tasks**:
- [ ] Research DVI parsing libraries (`dviread`, `dviasm`, or custom)
- [ ] Create `data_tools/dataset_prep/latex_to_dvi_boxes.py`
- [ ] Compile LaTeX to DVI using `latex` command
- [ ] Parse DVI to extract boxes for each character/symbol
- [ ] Map DVI font positions to LaTeX tokens
- [ ] Output JSON in same format as `boxes.jsonl`
- [ ] Handle math mode, commands, special symbols

**Note**: Can skip this phase initially and use existing `boxes.jsonl` data.

**Deliverables**:
- `latex_to_dvi_boxes.py` script (if needed)
- DVI parsing documentation

---

### PHASE 4: End-to-End Pipeline

#### [P4-1] Image Rendering Pipeline
**Status**: Not Started
**Priority**: High
**Effort**: Small (2-4 hours)

**Description**: Integrate synthesis with existing rendering tools.

**Tasks**:
- [ ] Use existing `inkml_to_images.py` to render composite inkml
- [ ] Test rendering quality and parameters:
  - [ ] Stroke width
  - [ ] Image size
  - [ ] Padding
  - [ ] Background/foreground colors
- [ ] Batch processing for multiple expressions
- [ ] Validate output image quality

**Deliverables**:
- Rendered sample images
- Rendering parameter recommendations

---

#### [P4-2] Label Generation Integration
**Status**: Not Started
**Priority**: High
**Effort**: Small (2-4 hours)

**Description**: Generate hybrid tree labels for synthesized expressions.

**Tasks**:
- [ ] Use existing `gen_hybrid_data.py` pipeline
- [ ] Input: LaTeX expressions (from bounding box JSON)
- [ ] Generate hybrid tree labels
- [ ] Create image-label pairs in expected format
- [ ] Validate label correctness

**Deliverables**:
- Label files in hybrid format
- Validation that labels match SAN training format

---

#### [P4-3] End-to-End Automation Script
**Status**: Not Started
**Priority**: Medium
**Effort**: Medium (4-6 hours)

**Description**: Create master script to run entire synthesis pipeline.

**Tasks**:
- [ ] Create `data_tools/dataset_prep/run_synthesis_pipeline.py`
- [ ] Chain all steps:
  1. Index symbols (if needed)
  2. Load bounding boxes
  3. Synthesize inkml
  4. Render images
  5. Generate labels
- [ ] Add command-line interface with options:
  - Input bounding box file
  - Output directory
  - Number of variants per expression
  - Random seed for reproducibility
- [ ] Progress reporting and logging
- [ ] Error handling and recovery

**Deliverables**:
- `run_synthesis_pipeline.py` master script
- User documentation

---

### PHASE 5: Testing and Validation

#### [P5-1] Sample Generation Test
**Status**: Not Started
**Priority**: High
**Effort**: Small (2-3 hours)

**Description**: Generate sample dataset and validate quality.

**Tasks**:
- [ ] Run pipeline on 10-20 expressions from `latex.txt`
- [ ] Visual inspection of generated images
- [ ] Verify stroke quality and positioning
- [ ] Check label correctness
- [ ] Compare with real handwritten samples

**Deliverables**:
- Sample dataset (images + labels)
- Quality assessment report

---

#### [P5-2] Training Integration Test
**Status**: Not Started
**Priority**: Medium
**Effort**: Medium (4-6 hours)

**Description**: Test synthesized data with SAN training pipeline.

**Tasks**:
- [ ] Convert synthesized images/labels to pickle format
- [ ] Add to training data or create separate synthetic dataset
- [ ] Run training for few epochs
- [ ] Verify no errors in data loading
- [ ] Check if model can learn from synthetic data

**Deliverables**:
- Training test results
- Integration documentation

---

### PHASE 6: Documentation and Deployment

#### [P6-1] User Documentation
**Status**: Not Started
**Priority**: Medium
**Effort**: Small (2-3 hours)

**Description**: Create comprehensive documentation.

**Tasks**:
- [ ] Create `SYNTHESIS_README.md` with:
  - [ ] Overview and motivation
  - [ ] Installation requirements
  - [ ] Usage examples
  - [ ] Configuration options
  - [ ] Troubleshooting guide
- [ ] Add inline code documentation
- [ ] Create usage examples

**Deliverables**:
- `SYNTHESIS_README.md`
- Example command snippets

---

#### [P6-2] Dataset Generation at Scale
**Status**: Not Started
**Priority**: Low
**Effort**: Variable

**Description**: Generate large-scale synthetic dataset for training augmentation.

**Tasks**:
- [ ] Determine target dataset size
- [ ] Optimize pipeline for batch processing
- [ ] Generate multiple variants per expression
- [ ] Organize output directory structure
- [ ] Create dataset statistics and metadata

**Deliverables**:
- Large-scale synthetic dataset
- Dataset statistics report

---

## Open Questions

1. **DVI Processing**: Should we implement DVI parsing or use existing `boxes.jsonl`?
   - **Recommendation**: Start with existing `boxes.jsonl`, add DVI parser later if needed

2. **Missing Symbols**: How to handle symbols not in inkml library?
   - Options: Skip expression, use placeholder, synthesize from font
   - **Recommendation**: Skip for now, document missing symbols

3. **Structural Elements**: How to render `\frac` bars, `\sqrt` radicals, etc.?
   - These are typically drawn as lines/curves, not from inkml library
   - **Recommendation**: Phase 2 - consider synthesizing or using generic strokes

4. **Variant Strategy**: How many handwritten variants per expression?
   - **Recommendation**: Start with 1, make configurable for future

5. **Dataset Size**: How many expressions to generate?
   - **Recommendation**: Start with 100-500 for validation, scale based on results

---

## Priority Execution Order

### Milestone 1: Proof of Concept (Week 1)
1. [P1-1] Symbol Library Indexer
2. [P2-1] InkML Stroke Transformer
3. [P2-2] Composite InkML Generator (basic version)
4. [P5-1] Sample Generation Test (5-10 expressions)

### Milestone 2: Working Pipeline (Week 2)
1. [P4-1] Image Rendering Pipeline
2. [P4-2] Label Generation Integration
3. [P1-2] Symbol Coverage Analysis
4. [P4-3] End-to-End Automation Script

### Milestone 3: Production Ready (Week 3)
1. [P5-2] Training Integration Test
2. [P6-1] User Documentation
3. [P6-2] Dataset Generation at Scale (optional)

---

## Success Criteria

- [ ] Successfully synthesize handwritten images from LaTeX expressions
- [ ] Generated images are visually similar to real handwriting
- [ ] Labels are correct and compatible with SAN training
- [ ] Pipeline can process batch expressions automatically
- [ ] Code is documented and maintainable
- [ ] Synthetic data improves or maintains model performance

---

## Notes
- Start with simple expressions (no fractions/matrices initially)
- Use existing `boxes.jsonl` data before implementing DVI parser
- Focus on getting end-to-end pipeline working before optimization
- Test incrementally at each phase
