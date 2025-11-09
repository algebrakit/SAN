# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview
This is the official PyTorch implementation of **SAN (Syntax-Aware Network)** for Handwritten Mathematical Expression Recognition, published at CVPR 2022. The system uses a hierarchical attention mechanism with syntax-aware decoding to recognize handwritten mathematical expressions.

- **model**: Implementation of the Syntax-Aware Encoder-Decoder model
- **data**: Datasets that can be used for training
- **data_tools**: Tools to preprocess data or generate synthetic data and create a new dataset
  - **`data_tools/synthetic_generator/CLAUDE.md`** - Complete pipeline documentation
  - **`data_tools/synthetic_generator/README.md`** - Quick start guide
- **app**: A webservice for inference and web-component to capture strokes.

See also README.md for general information.
