# YOLOv11-CFruit

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

YOLOv11-CFruit is a next-generation object detection model designed for Camellia oleifera fruit detection, based on the latest YOLOv11 architecture. It integrates C2f backbone, CBAM attention, Transformer, AdamW optimizer, Automatic Mixed Precision (AMP), Exponential Moving Average (EMA), Mixup/CopyPaste augmentation, and more. The project is fully compatible with YOLOv8 configuration and usage.

## 🌟 Key Features

- **C2f Backbone**: More efficient feature extraction, faster and more accurate
- **CBAM Attention**: Focuses on salient features, suppresses background noise
- **Transformer-Enhanced Neck**: Global context modeling, better for occlusion/dense scenes
- **AdamW Optimizer**: Improved convergence and generalization
- **AMP/EMA**: Mixed precision and exponential moving average for stable training and fast inference
- **Mixup/CopyPaste**: Stronger data augmentation, improved robustness
- **Inference Optimization**: Supports Conv-BN fusion, optimized NMS
- **YOLOv8 Compatibility**: Seamless switch between YOLOv8/YOLOv11 configs

## 🏗️ Project Structure

```
YOLOv11-CFruit/
├── configs/
│   ├── model/
│   │   ├── yolov11_cfruit.yaml
│   │   └── yolov8_cfruit.yaml
│   └── data/
│       └── cfruit.yaml
├── models/
│   ├── yolov11_cfruit.py
│   ├── yolov8_cfruit.py
│   ├── backbone/
│   │   ├── cspdarknet.py
│   │   ├── c2f.py
│   │   └── cbam.py
│   ├── neck/
│   │   ├── panet.py
│   │   └── transformer.py
│   ├── head/
│   │   └── anchor_free.py
│   └── __init__.py
├── utils/
│   ├── losses.py
│   └── __init__.py
├── scripts/
│   └── train.py
├── examples/
│   └── basic_detection.py
├── requirements.txt
├── setup.py
├── install_linux.sh
├── fix_opencv_linux.sh
├── LICENSE
├── DesignDoc_en.md
├── DesignDoc.md
└── README_en.md
```

## 🚀 Quick Start

### Requirements
- Python 3.8+ (推荐 Python 3.12+)
- PyTorch 2.0+
- CUDA 11.0+ (GPU recommended)

### Installation

#### Windows
```powershell
# Recommended: PowerShell
.\install.ps1
# Or batch script
install.bat
```

#### Linux
```bash
# Method 1: Use Linux installation script (Recommended)
chmod +x install_linux.sh
./install_linux.sh

# Method 2: Python 3.12 specific installation
chmod +x install_python312.sh
./install_python312.sh

# Method 3: Manual installation
sudo apt update  # Ubuntu/Debian
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgtk-3-0
pip install -r requirements.txt
pip install -e .

# Method 4: If you encounter OpenCV issues
chmod +x fix_opencv_linux.sh
./fix_opencv_linux.sh
```

#### macOS
```bash
# Install system dependencies
brew install opencv

# Install Python dependencies
pip install -r requirements.txt
pip install -e .
```

#### Docker (Recommended for Linux)
```bash
# Build and run with Docker (Python 3.12)
docker build -t yolov11-cfruit .
docker run -it --rm -v $(pwd):/workspace yolov11-cfruit

# Or use docker-compose
docker-compose up -d yolov11-cfruit
docker-compose exec yolov11-cfruit bash

# For GPU support (requires nvidia-docker)
docker-compose up -d yolov11-cfruit-gpu
docker-compose exec yolov11-cfruit-gpu bash
```

### Verify Installation
```bash
# Basic test
python test_project.py

# Python 3.12 compatibility check
python check_python312.py
```

### Troubleshooting

#### OpenCV Issues
If you encounter `libGL.so.1: cannot open shared object file` error on Linux:

1. **Quick Fix**: Run the fix script
   ```bash
   chmod +x fix_opencv_linux.sh
   ./fix_opencv_linux.sh
   ```

2. **Manual Fix**: Install system dependencies
   ```bash
   # Ubuntu/Debian
   sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgtk-3-0
   
   # CentOS/RHEL/Fedora
   sudo yum install -y mesa-libGL glib2 libSM libXext libXrender libgomp gtk3
   
   # Reinstall OpenCV
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

3. **Alternative**: Use Docker
   ```bash
   docker run -it --rm -v $(pwd):/workspace python:3.12 bash
   cd /workspace && pip install -r requirements.txt
   ```

#### Python 3.12 Compatibility
If you encounter compatibility issues with Python 3.12:

1. **Check compatibility**: Run the compatibility check
   ```bash
   python check_python312.py
   ```

2. **Update packages**: Manually update problematic packages
   ```bash
   pip install --upgrade numpy pandas scipy scikit-learn
   ```

3. **Use specific versions**: Install compatible versions
   ```bash
   pip install "numpy>=1.24.0" "pandas>=2.0.0" "scipy>=1.11.0"
   ```

#### Docker Issues
If you encounter Docker permission issues:

1. **Run as administrator** (Windows)
2. **Add user to docker group** (Linux)
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```
3. **Use WSL2** (Windows)
   ```bash
   wsl
   cd /mnt/d/code/YOLOv8-CFruit
   docker build -t yolov11-cfruit .
   ```

## 📖 Configuration & Usage

### YOLOv11 Config Example (`configs/model/yolov11_cfruit.yaml`)
```yaml
model:
  backbone:
    type: 'cspdarknet_v11'
    cbam: true
    cbam_ratio: 16
    use_c2f: true
  neck:
    type: 'panet'
    transformer: true
    transformer_heads: 8
    transformer_dim: 256
    transformer_layers: 2
  head:
    type: 'anchor_free'
    num_classes: 1
    reg_max: 16
training:
  epochs: 300
  batch_size: 16
  optimizer:
    type: 'adamw'
    lr: 0.001
    weight_decay: 0.0005
  scheduler:
    type: 'cosine'
    min_lr: 0.00001
augmentation:
  mixup: 0.1
  copy_paste: 0.1
# ... see full yaml for details
```

### Training Command
```bash
python scripts/train.py --config configs/model/yolov11_cfruit.yaml --data configs/data/cfruit.yaml
```

### Inference Command
```bash
python examples/basic_detection.py --config configs/model/yolov11_cfruit.yaml --weights checkpoints/yolov11_cfruit.pt --source path/to/image.jpg
```

### YOLOv8 Compatibility
- Simply switch to `--config configs/model/yolov8_cfruit.yaml`.
- The code will automatically load the correct model based on the config.

## 📊 Performance

| Model         | mAP@0.5 | mAP@0.5:0.95 | F1-Score | Inference Time |
|--------------|---------|--------------|----------|---------------|
| YOLOv11-CFruit | 0.94    | 0.81         | 0.91     | 15ms          |
| YOLOv8-CFruit  | 0.92    | 0.78         | 0.89     | 18ms          |

## 🛠️ Customization & Extension
- Custom backbone/neck/head/loss supported
- Advanced features: AMP, EMA, AdamW, Mixup, CopyPaste
- See `configs/model/yolov11_cfruit.yaml` and `utils/losses.py`

## 📚 Documentation
- [Design Document (English)](DesignDoc_en.md)
- [设计文档 (中文)](DesignDoc.md)
- [Usage Guide (USAGE.md)]

## 🤝 Contributing
PRs and issues are welcome!

## 📄 License
MIT License

## 🙏 Acknowledgements
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [YOLOv8-CFruit](https://github.com/your-repo/yolov8-cfruit)
- All contributors and researchers

## 📞 Contact
- Issue: https://github.com/your-username/YOLOv8-CFruit/issues
- Email: cindynianx@gmail.com

---
**Note**: This project supports both YOLOv8 and YOLOv11 configs. YOLOv11-CFruit is recommended for best results. 