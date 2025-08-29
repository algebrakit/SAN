# Stroke-Aware SAN Implementation

This directory contains a complete implementation of stroke-aware SAN for handwritten mathematical expression recognition using InkML stroke data.

## 🎯 Key Features

- **Direct stroke processing**: No image conversion required
- **Expression-level normalization**: Preserves mathematical relationships
- **Hybrid architecture**: LSTM + Transformer + CNN for optimal performance
- **Compatible with existing SAN decoder**: Reuses proven attention mechanism
- **Batch processing**: Efficient training with 396,989 InkML files

## 📁 New Files Created

### Core Components
- `dataset_stroke.py` - InkML data loading and stroke normalization
- `models/stroke_encoder.py` - Stroke-aware encoder architecture
- `models/Backbone_stroke.py` - Modified backbone for stroke processing
- `training_stroke.py` - Training functions for stroke data
- `train_stroke.py` - Main training script

### Configuration
- `config_stroke.yaml` - Stroke-specific configuration parameters

## 🏗️ Architecture Overview

```
InkML Files (.inkml)
    ↓
StrokeNormalizer (Expression-level normalization)
    ↓
[batch, max_strokes, max_points, 3] (x,y,t coordinates)
    ↓
PointSequenceLSTM (Process each stroke's points)
    ↓
StrokeRelationTransformer (Model stroke relationships) 
    ↓
SpatialFeatureGenerator (Create 2D feature maps)
    ↓
[batch, 684, 20, 100] (Same as DenseNet output!)
    ↓
Existing SAN Decoder (Hierarchical attention)
    ↓
LaTeX Output
```

## 🚀 Usage

### Quick Test
```bash
# Test the setup (code check mode)
python train_stroke.py --config config_stroke.yaml --check
```

### Full Training  
```bash
# Start training with 396,989 InkML files
python train_stroke.py --config config_stroke.yaml
```

### Configuration Parameters

Key stroke-specific parameters in `config_stroke.yaml`:

```yaml
# Stroke Processing
stroke_target_width: 4.0       # Normalized coordinate range [-2, 2]
stroke_target_height: 1.0      # Normalized coordinate range [-0.5, 0.5] 
max_strokes: 20                 # Max strokes per expression
max_points_per_stroke: 100      # Max points per stroke

# Architecture
point_lstm_hidden: 128          # Point LSTM hidden size
transformer_heads: 8            # Stroke transformer attention heads
transformer_layers: 3           # Stroke transformer layers

# Data
inkml_folder: 'synthetic'       # Folder with InkML files
```

## 📊 Data Flow

### Input Processing
1. **Parse InkML**: Extract strokes and LaTeX labels
2. **Expression Normalization**: Scale entire expression to consistent coordinate range
3. **Temporal Encoding**: Preserve stroke order and drawing dynamics  
4. **Sequence Padding**: Fixed tensor dimensions for batching

### Model Architecture
1. **Point LSTM**: Process temporal sequence within each stroke
2. **Stroke Transformer**: Model relationships between strokes
3. **Spatial Generator**: Create 2D feature maps from stroke features
4. **SAN Decoder**: Existing hierarchical attention decoder (unchanged)

## 🔧 Key Advantages

### vs Original Image-Based SAN:
- ✅ **No scale variation**: Expression-level normalization eliminates character size differences
- ✅ **Temporal information**: Preserves stroke order and drawing dynamics
- ✅ **Vector precision**: No rasterization artifacts
- ✅ **Direct tablet input**: Can process real-time pen strokes

### vs Stroke-Level Normalization:
- ✅ **Mathematical structure preserved**: Spatial relationships maintained
- ✅ **Decoder compatibility**: Same output format as DenseNet
- ✅ **Training stability**: Consistent input representations

## 📈 Expected Training Process

With 396,989 InkML files:
- **Dataset loading**: ~5-10 minutes (parsing InkML files)
- **Training time**: ~12-20 hours (200 epochs, depending on GPU)
- **Memory usage**: ~8GB GPU memory (batch_size=8)
- **Checkpoints**: Saved to `checkpoints_stroke/`
- **Logs**: TensorBoard logs in `logs_stroke/`

## 🎯 End-to-End Workflow

### Training Phase:
```
396,015 InkML files → Stroke normalization → LSTM+Transformer → Feature maps → SAN decoder → LaTeX
```

### Inference Phase (Future):
```
Tablet pen input → Real-time strokes → Same pipeline → LaTeX output
```

## 🚨 Important Notes

1. **Label Processing**: Currently uses placeholder labels - will need proper LaTeX→hybrid tree conversion
2. **Memory Requirements**: Large dataset requires sufficient RAM for loading
3. **GPU Recommended**: Training 396K samples benefits significantly from GPU acceleration
4. **Batch Size**: Start with smaller batch_size if encountering memory issues

## 📝 Next Steps

1. **Run code check**: `python train_stroke.py --config config_stroke.yaml --check`
2. **Implement label parsing**: Convert LaTeX to proper hybrid tree format
3. **Start training**: Full training run with stroke data
4. **Monitor progress**: Use TensorBoard to track training metrics
5. **Inference pipeline**: Create real-time stroke recognition interface

This stroke-aware implementation provides a direct path from pen strokes to LaTeX without any image processing steps!