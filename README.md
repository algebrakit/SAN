# SAN: Syntax-Aware Network for Handwritten Mathematical Expression Recognition

Official PyTorch implementation of **SAN** published at CVPR 2022.

## Project Structure

The codebase has been reorganized for better modularity and maintainability:

```
SAN/
├── app/                      # Web application
│   ├── backend/             # Flask API server for stroke-based inference
│   └── frontend/            # Web UI for handwriting input
├── model/                    # Core ML model package (inference + training)
│   ├── san_model/           # Core model architecture
│   │   ├── backbone.py     # Main model combining encoder+decoder
│   │   ├── encoder/        # CNN encoder (DenseNet)
│   │   └── decoder/        # Hierarchical attention decoder
│   ├── training/            # Training pipeline
│   │   ├── train.py        # Main training script
│   │   ├── dataset.py      # Dataset loaders
│   │   └── training.py     # Training loop
│   ├── inference/           # Inference-specific modules
│   │   ├── Backbone.py     # Inference-optimized backbone
│   │   └── san_decoder.py  # Inference-optimized decoder
│   ├── utils/               # Model utilities
│   ├── inference.py         # CLI inference tool
│   └── config.yaml          # Main inference configuration
├── data_tools/              # Data processing utilities
│   ├── dataset_prep/        # Dataset preparation scripts (CROHME)
│   ├── synthetic_generator/ # Synthetic data generation system
│   └── stroke_processing/   # Stroke-to-image conversion
├── utils/                    # Shared utilities
│   └── Expression/          # LaTeX expression parsing
├── data/                    # Datasets (train/test pickles)
├── checkpoints/             # Model checkpoints
└── logs/                    # TensorBoard logs
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/SAN.git
cd SAN

# Install dependencies
pip install -r requirements.txt

# For development installation
pip install -e .
```

## Quick Start

### Web Application
```bash
# Start the backend server
cd app/backend
python server.py

# In another terminal, start the frontend
cd app/frontend
npm install
npm start
```

### Docker Deployment
```bash
# Build and run with Docker
cd app/backend
./build-docker.sh
docker run -p 8080:8080 san-backend:latest
```

### Training
```bash
# Prepare data
cd data_tools/dataset_prep
python prepare_crohme_data.py
python gen_hybrid_data.py
python gen_pkl.py

# Train the model
cd ../../model/training
python train.py --config config.yaml
```

### Inference
```bash
cd model
python inference.py --config config.yaml --image_path ../data/test_images --label_path ../data/test_caption.txt
```

## Model Architecture

SAN uses a hierarchical attention mechanism with syntax-aware decoding:
- **Encoder**: DenseNet-based CNN for visual feature extraction
- **Decoder**: Hierarchical attention with word-level and structure-level predictions
- **Attention**: Multi-level attention mechanism for focusing on relevant image regions

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{san2022cvpr,
  title={Syntax-Aware Network for Handwritten Mathematical Expression Recognition},
  author={...},
  booktitle={CVPR},
  year={2022}
}
```

## License

This project is licensed under the MIT License.