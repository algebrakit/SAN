# Code Reorganization Summary

## Overview
The SAN codebase has been reorganized from a mixed structure into a clean, modular architecture that clearly separates concerns and improves maintainability.

## New Structure

### Main Directories

1. **`app/`** - Web Application Layer
   - `backend/` - Flask API server for stroke-based inference
   - `frontend/` - Stencil web components for handwriting input

2. **`san_model/`** - Core ML Model Package
   - `backbone.py` - Main model architecture
   - `encoder/` - CNN encoder modules (DenseNet)
   - `decoder/` - Hierarchical attention decoder
   - `utils.py` - Model utilities

3. **`training/`** - Training Pipeline
   - `train.py` - Main training script
   - `training.py` - Training logic
   - `dataset.py` - Dataset loaders
   - `config/` - YAML configuration files

4. **`inference/`** - Standalone Inference Tools
   - `inference.py` - CLI inference tool
   - `Backbone.py` - Inference-optimized model
   - `san_decoder.py` - Inference decoder

5. **`data_tools/`** - Data Processing Utilities
   - `stroke_processing/` - Stroke-to-image conversion tools
   - `dataset_prep/` - Dataset preparation scripts

6. **`data/`** - Datasets (unchanged)

7. **`checkpoints/`** - Model checkpoints

8. **`tests/`** - Test suite

9. **`docs/`** - Documentation

## Key Changes Made

1. **Separated Concerns**:
   - Application code (API/UI) moved to `app/`
   - ML model code isolated in `san_model/` package
   - Training code separated from inference and application

2. **Created Proper Python Packages**:
   - Added `__init__.py` files for all modules
   - Created `setup.py` for package installation
   - Updated import paths to use relative imports

3. **Improved Modularity**:
   - Stroke processing tools can be used by both training and inference
   - Model package can be imported independently
   - Clear separation between web service and ML components

## Benefits

1. **Better Organization**: Clear separation of application, model, and tools
2. **Reusability**: Components can be imported and used independently
3. **Deployment**: Easier to deploy specific components (e.g., just the API)
4. **Development**: Clearer where to find and add new functionality
5. **Testing**: Easier to test individual components

## Migration Notes

### For Existing Code
- The original `backend/` folder still exists with all original files
- The original `frontend/` folder still exists
- All data and checkpoints remain in place
- The reorganized code is in the new structure

### Import Path Updates
- Model imports: `from models.Backbone import Backbone` → `from san_model import Backbone`
- Stroke tools: `from strokes.stroke2img import ...` → `from data_tools.stroke_processing.stroke2img import ...`
- Training utils: `from utils import ...` → `from san_model.utils import ...`

## Next Steps

1. **Remove Old Directories** (after verification):
   ```bash
   rm -rf backend/ frontend/ data-util/ tools/
   ```

2. **Install as Package**:
   ```bash
   pip install -e .
   ```

3. **Update Documentation**:
   - Update CLAUDE.md with new structure
   - Update deployment guides

4. **Test All Components**:
   - Test training pipeline
   - Test inference tools
   - Test web application

## File Mapping

| Original Location | New Location |
|------------------|--------------|
| `backend/models/` | `san_model/` |
| `backend/strokes/` | `data_tools/stroke_processing/` |
| `backend/train.py` | `training/train.py` |
| `backend/server.py` | `app/backend/server.py` |
| `frontend/` | `app/frontend/` |
| `data-util/` | `data_tools/dataset_prep/` |
| `backend/infer/` | `inference/` |