# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the official PyTorch implementation of **SAN (Syntax-Aware Network)** for Handwritten Mathematical Expression Recognition, published at CVPR 2022. The system uses a hierarchical attention mechanism with syntax-aware decoding to recognize handwritten mathematical expressions.

## Key Commands

### Installation
```bash
pip3 install -r requirement.txt
```

### Data Preparation (CROHME 2016)
```bash
cd data
# Step 1: Prepare image and label files
python3 prepare_crohme_data.py

# Step 2: Generate hybrid tree labels
python3 gen_hybrid_data.py

# Step 3: Convert to pickle format
python3 convert_hybrid_to_pkl.py
```

### Training
```bash
python3 train.py --config config.yaml
```

### Inference
```bash
python3 inference.py --config 14.yaml --image_path data/14_test_images --label_path data/test_caption.txt
```

## Architecture Overview

The codebase implements a deep learning model for recognizing handwritten mathematical expressions with the following key components:

### Core Model Architecture
- **Backbone** (`models/Backbone.py`): Main model wrapper that combines encoder and decoder
- **Encoder**: DenseNet-based CNN encoder (`models/CNN/densenet.py`) that extracts visual features from handwritten images
- **Decoder**: SAN decoder (`models/Hierarchical_attention/decoder.py`) that uses hierarchical attention and syntax-aware decoding
- **Attention Mechanism** (`models/Hierarchical_attention/attention.py`): Implements the attention mechanism for focusing on relevant image regions

### Key Configuration
- Model configurations are in YAML files (`config.yaml`, `14.yaml`)
- Image dimensions: configurable, default 1600×320 or 3200×400 pixels
- Uses GRU cells for sequence modeling
- Implements both word-level and structure-level predictions

### Data Processing
- Dataset handling in `dataset.py`
- Data generation scripts in `data/` directory:
  - `gen_hybrid_data.py`: Generate hybrid tree data
  - `gen_pkl.py`: Convert data to pickle format
  - `gen_symbols_struct_dict.py`: Generate symbol structure dictionary

### Training Pipeline
- Main training script: `train.py`
- Training logic: `training.py`
- Utilities: `utils.py`
- Checkpoints saved to `checkpoints/` directory
- TensorBoard logs saved to `logs/` directory

### Inference
- Inference script: `inference.py`
- Separate inference modules in `infer/` directory

## Important Notes
- The model requires CUDA-compatible GPU (originally uses PyTorch 1.6.0+cu101)
- For Python 3.11+, use `requirements_python311.txt` for compatible package versions
- On macOS with Apple Silicon, PyTorch will use MPS (Metal Performance Shaders) instead of CUDA
- Pre-trained checkpoint available at `checkpoints/SAN_decoder/best.pth`
- Supports CROHME and HME100K datasets
- No explicit test suite - evaluation is done during training/inference