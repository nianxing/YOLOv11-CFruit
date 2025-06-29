# YOLOv11-CFruit Anaconda安装配置指南

## 📋 概述
本指南将帮助您使用Anaconda安装配置YOLOv11-CFruit项目，避免Python包依赖冲突问题。

---

**最后更新：2024年6月**  
**文档版本：v1.0**

---

## 🖥️ 系统要求
- Linux/Windows/macOS系统
- 至少4GB RAM
- 至少20GB可用磁盘空间
- 网络连接

## 🚀 快速安装

### Linux系统
```bash
# 1. 下载并运行安装脚本
chmod +x install_conda.sh
./install_conda.sh
```

### Windows系统
```powershell
# 1. 以管理员身份运行PowerShell
# 2. 设置执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. 运行安装脚本
./install_conda.ps1
```

## 📦 手动安装步骤

### 1. 安装Anaconda

#### Linux
```bash
# 下载Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# 安装
bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -p $HOME/anaconda3

# 初始化
$HOME/anaconda3/bin/conda init bash
source ~/.bashrc
```

#### Windows
```powershell
# 下载Anaconda
Invoke-WebRequest -Uri "https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Windows-x86_64.exe" -OutFile "anaconda.exe"

# 安装
Start-Process -FilePath "anaconda.exe" -ArgumentList "/S /D=$env:USERPROFILE\anaconda3" -Wait
```

### 2. 创建conda环境
```bash
# 使用环境配置文件
conda env create -f conda_setup.yml

# 或者手动创建
conda create -n yolov11-cfruit python=3.11
conda activate yolov11-cfruit
```

### 3. 安装PyTorch
```bash
# 激活环境
conda activate yolov11-cfruit

# 安装PyTorch (CPU版本)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 或者GPU版本 (如果有CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 4. 安装其他依赖
```bash
# 基础科学计算包
conda install numpy scipy scikit-learn pandas matplotlib seaborn

# 图像处理
conda install pillow opencv

# 其他工具
conda install pyyaml tensorboard tqdm requests pytest

# 通过pip安装特殊包
pip install albumentations timm thop black flake8
```

## ✅ 验证安装

### 1. 检查环境
```bash
conda activate yolov11-cfruit
conda list
```

### 2. 测试关键包
```python
import torch
import torchvision
import cv2
import numpy as np
import albumentations as A

print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
```

### 3. 测试项目
```bash
# 运行测试脚本
python test_project.py

# 运行基础检测示例
python examples/basic_detection.py
```

## 🔧 常见问题解决

### 1. 包冲突问题
```bash
# 清理conda缓存
conda clean --all

# 重新创建环境
conda env remove -n yolov11-cfruit
conda env create -f conda_setup.yml
```

### 2. 网络问题
```bash
# 使用国内镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

### 3. 权限问题
```bash
# Linux: 确保脚本有执行权限
chmod +x install_conda.sh

# Windows: 以管理员身份运行PowerShell
```

## 🛠️ 使用说明

### 激活环境
```bash
conda activate yolov11-cfruit
```

### 运行训练
```bash
# 改进版训练（推荐）
python scripts/train_improved.py --config configs/model/yolov11_cfruit_improved.yaml --data configs/data/cfruit.yaml

# 简化训练
python scripts/simple_train.py --config configs/model/yolov11_cfruit.yaml --data configs/data/cfruit.yaml
```

### 运行检测
```bash
python examples/basic_detection.py --model checkpoints/best.pt --image path/to/image.jpg
```

### 退出环境
```bash
conda deactivate
```

## ⭐ 优势

使用Anaconda的优势：
1. **依赖管理**: conda自动处理包依赖关系
2. **环境隔离**: 避免与系统Python包冲突
3. **版本控制**: 精确控制包版本
4. **跨平台**: 支持Linux、Windows、macOS
5. **预编译包**: 减少编译时间和错误

## ⚠️ 注意事项

1. 确保系统有足够的磁盘空间
2. 建议使用Python 3.11版本以获得最佳兼容性
3. 如果使用GPU，确保安装对应的CUDA版本
4. 定期更新conda和包版本
5. 备份重要的conda环境配置

## 📞 技术支持

如果遇到问题，请：
1. 检查错误日志
2. 确认系统要求
3. 尝试重新创建环境
4. 查看项目文档

## 🔗 相关链接

- [快速开始指南](QUICK_START.md)
- [使用说明](USAGE.md)
- [Docker设置指南](DOCKER_WINDOWS_SETUP.md) 