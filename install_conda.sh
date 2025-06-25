#!/bin/bash

# YOLOv11-CFruit Anaconda安装脚本
# 适用于Azure Linux虚拟机

set -e

echo "=== YOLOv11-CFruit Anaconda安装脚本 ==="
echo "开始安装配置..."

# 检查是否已安装Anaconda
if command -v conda &> /dev/null; then
    echo "✓ Anaconda已安装"
else
    echo "正在安装Anaconda..."
    
    # 下载Anaconda安装脚本
    wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -O anaconda.sh
    
    # 安装Anaconda
    bash anaconda.sh -b -p $HOME/anaconda3
    
    # 初始化conda
    $HOME/anaconda3/bin/conda init bash
    
    # 重新加载bash配置
    source ~/.bashrc
    
    echo "✓ Anaconda安装完成"
fi

# 创建conda环境
echo "正在创建conda环境..."
conda env create -f conda_setup.yml

# 激活环境
echo "激活conda环境..."
source activate yolov11-cfruit

# 验证安装
echo "验证安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision版本: {torchvision.__version__}')"
python -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')"

echo "=== 安装完成 ==="
echo "使用方法:"
echo "1. 激活环境: conda activate yolov11-cfruit"
echo "2. 运行训练: python scripts/train.py"
echo "3. 运行检测: python examples/basic_detection.py" 