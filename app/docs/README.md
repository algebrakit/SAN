# SAN - Syntax-Aware Network for Handwritten Mathematical Expression Recognition

A complete system for recognizing handwritten mathematical expressions using deep learning. This implementation combines the original PyTorch research code with a modern web interface for interactive drawing and conversion.

![SAN Overview](overview.png)

## 🎯 Project Structure

This repository now includes both the original research implementation and a complete web application:

```
SAN/
├── frontend/              # TypeScript/StencilJS web application
│   ├── src/components/    # Drawing canvas and UI components
│   ├── package.json       # Frontend dependencies
│   └── README.md          # Frontend documentation
├── backend/               # Python/PyTorch backend (original code + API server)
│   ├── server.py          # HTTP API server
│   ├── models/            # Neural network architecture
│   ├── strokes/           # Stroke processing modules
│   ├── checkpoints/       # Pre-trained model weights
│   ├── train.py          # Training script
│   ├── inference_single.py # Single expression inference
│   └── README.md          # Backend documentation
└── data/                  # Training and test data (shared)
```

## 🚀 Quick Start - Web Application

### Prerequisites
- **Python 3.8+** with PyTorch
- **Node.js 16+** with npm

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python server.py 5001
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm start
```

### 3. Open Application
Visit `http://localhost:3334` to draw mathematical expressions and convert them to LaTeX!

## 🔬 Research Usage (Original Implementation)

### Environment
```
python==3.8.5
numpy==1.22.2
opencv-python==4.5.5.62
PyYAML==6.0
tensorboardX==2.5
torch==1.6.0+cu101
torchvision==0.7.0+cu101
tqdm==4.64.0
```

### Train
```bash
cd backend
python train.py --config path_to_config_yaml
```

### Inference
```bash
cd backend
python inference.py --config path_to_config_yaml --image_path path_to_image_folder --label_path path_to_label_folder
```

Example:
```bash
python inference.py --config 14.yaml --image_path data/14_test_images --label_path data/test_caption.txt
```

### Dataset

**CROHME**: 
```
Download from: https://github.com/JianshuZhang/WAP/tree/master/data
```

**HME100K**:
```
Download from: https://ai.100tal.com/dataset
```

### Citation

If you find this dataset helpful for your research, please cite the following paper:

```
@inproceedings{yuan2022syntax,
  title={Syntax-Aware Network for Handwritten Mathematical Expression Recognition},
  author={Yuan, Ye and Liu, Xiao and Dikubab, Wondimu and Liu, Hui and Ji, Zhilong and Wu, Zhongqin and Bai, Xiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4553--4562},
  year={2022}
}
```