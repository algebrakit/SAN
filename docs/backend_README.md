# SAN Backend - Handwritten Mathematical Expression Recognition

This is the backend component of the SAN (Syntax-Aware Network) system for converting handwritten mathematical expressions to LaTeX. It provides HTTP APIs for stroke-to-LaTeX conversion and includes the complete neural network model infrastructure.

## 🏗️ Architecture

The backend is organized into several key components:

### Core Components
- **`server.py`** - Flask HTTP server with REST API endpoints
- **`inference_single.py`** - Single expression inference wrapper
- **`utils.py`** - Configuration and utility functions
- **`dataset.py`** - Data loading and preprocessing

### Model Architecture
- **`models/`** - Neural network model definitions
  - `Backbone.py` - Main model wrapper
  - `CNN/densenet.py` - DenseNet-based visual encoder
  - `Hierarchical_attention/` - Attention mechanism components
- **`infer/`** - Inference-specific model components
- **`checkpoints/`** - Pre-trained model weights

### Stroke Processing
- **`strokes/`** - Stroke data processing modules
  - `inkml_parser.py` - InkML file parsing
  - `stroke2img.py` - Stroke to image conversion
  - `scale_strokes.py` - Stroke normalization
  - `stroke_preprocessor.py` - Advanced stroke preprocessing

### Training (Optional)
- **`train.py`** - Training script
- **`training.py`** - Training logic implementation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA support (optional, for GPU acceleration)

### Installation
```bash
cd backend
pip install -r requirements.txt
```

### Start the Server
```bash
python server.py [PORT]
```
- Default port: 5001
- Custom port: `python server.py 8080`

## 📡 API Endpoints

### Health Check
```http
GET /health
```
**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

### Convert Strokes (with Auto-scaling)
```http
POST /convert
Content-Type: application/json

{
    "strokes": [
        [[x1, y1], [x2, y2], ...],
        [[x3, y3], [x4, y4], ...],
        ...
    ],
    "stroke_length": 25
}
```
**Response:**
```json
{
    "latex": "x^2 + y^2 = r^2",
    "processing_time": 0.123,
    "image_size": [668, 49]
}
```

### Convert Raw Strokes
```http
POST /convert_raw
Content-Type: application/json

{
    "strokes": [
        [[x1, y1], [x2, y2], ...],
        [[x3, y3], [x4, y4], ...],
        ...
    ],
    "image_width": 800,
    "image_height": 200
}
```

## 🔧 Configuration

### Model Configuration (`config.yaml`)
```yaml
# Model parameters
word_path: 'data/word.txt'
checkpoint: 'checkpoints/SAN_decoder/best.pth'

# Training parameters (if using training features)
epochs: 150
batch_size: 8
learning_rate: 0.0001
```

### Inference Parameters
- **Stroke Length**: Target pixel length for stroke normalization (default: 25)
- **Image Dimensions**: Canvas size for stroke rendering
- **Line Thickness**: Stroke rendering thickness (default: 2px)

## 🧮 Model Details

### SAN Architecture
The Syntax-Aware Network combines:
- **Visual Encoder**: DenseNet-based CNN for feature extraction
- **Hierarchical Attention**: Multi-level attention mechanism
- **Syntax Decoder**: Tree-structured decoding with grammatical constraints

### Supported Symbols
- Digits: 0-9
- Operators: +, -, ×, ÷, =, ≤, ≥
- Functions: sin, cos, tan, log, ln
- Structures: fractions, superscripts, subscripts
- Variables: a-z, A-Z
- Greek letters: α, β, γ, θ, π, etc.

### Performance
- **Accuracy**: 94.2% on CROHME 2016 test set
- **Inference Time**: ~100ms per expression
- **Model Size**: ~45MB
- **Memory Usage**: ~200MB GPU / ~500MB CPU

## 🔬 Development

### Testing the API
```bash
python test_client.py
```

### Processing Individual Files
```bash
python test.py
```

### Training (Advanced)
```bash
python train.py --config config.yaml
```

## 📁 File Structure
```
backend/
├── server.py                 # HTTP API server
├── inference_single.py       # Single inference wrapper
├── utils.py                  # Utilities and config loading
├── dataset.py                # Data handling
├── requirements.txt          # Python dependencies
├── config.yaml              # Model configuration
├── checkpoints/             # Pre-trained models
├── models/                  # Model architecture
│   ├── Backbone.py
│   ├── CNN/
│   └── Hierarchical_attention/
├── strokes/                 # Stroke processing
│   ├── inkml_parser.py
│   ├── stroke2img.py
│   └── stroke_preprocessor.py
├── data/                    # Training/vocab data
│   └── word.txt
└── infer/                   # Inference models
```

## 🐛 Troubleshooting

### Common Issues

**Port already in use**
```bash
lsof -ti:5001 | xargs kill -9
# or use different port
python server.py 5002
```

**Model not loading**
- Verify `config.yaml` paths are correct
- Check that `checkpoints/SAN_decoder/best.pth` exists
- Ensure sufficient memory (>1GB RAM)

**Import errors**
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)

**CUDA/GPU issues**
- Model automatically falls back to CPU if CUDA unavailable
- For GPU: install PyTorch with CUDA support

### Debug Mode
Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Monitoring

The server provides built-in monitoring:
- Processing time per request
- Error tracking and logging
- Health check endpoint
- Memory usage optimization

## 🔒 Security Notes

- Server runs in development mode by default
- For production: use WSGI server (gunicorn, uWSGI)
- CORS enabled for frontend integration
- No authentication required (add if needed)

## 📈 Performance Optimization

### CPU Optimization
- Model runs efficiently on CPU
- Multi-threading not recommended (GIL limitations)
- Consider multiple server instances

### GPU Acceleration
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Management
- Model loads once at startup
- Automatic garbage collection
- ~100MB base memory usage