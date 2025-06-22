#!/bin/bash

# Linux OpenCV 问题修复脚本
echo "=== Linux OpenCV 问题修复脚本 ==="
echo ""

# 检测系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
    echo "检测到系统: $OS $VER"
else
    echo "无法检测系统信息"
    exit 1
fi

echo ""

# 方法1: 安装系统依赖
echo "方法1: 安装系统依赖..."
if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    sudo apt update
    sudo apt install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgthread-2.0-0 \
        libgtk-3-0 \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev
elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Fedora"* ]]; then
    if command -v dnf &> /dev/null; then
        sudo dnf install -y mesa-libGL glib2 libSM libXext libXrender libgomp gtk3
    else
        sudo yum install -y mesa-libGL glib2 libSM libXext libXrender libgomp gtk3
    fi
fi

echo "系统依赖安装完成"
echo ""

# 方法2: 重新安装OpenCV
echo "方法2: 重新安装OpenCV..."
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-python-headless

echo "OpenCV重新安装完成"
echo ""

# 方法3: 测试OpenCV
echo "方法3: 测试OpenCV..."
python3 -c "
import cv2
print('OpenCV版本:', cv2.__version__)
print('OpenCV路径:', cv2.__file__)
print('OpenCV导入成功!')
"

if [ $? -eq 0 ]; then
    echo "✓ OpenCV测试成功！"
else
    echo "✗ OpenCV测试失败"
    echo ""
    echo "方法4: 检查库依赖..."
    python3 -c "import cv2; print(cv2.__file__)" 2>/dev/null | xargs ldd 2>/dev/null | grep -E "(libGL|libglib|libsm|libxext|libxrender|libgomp)" || echo "无法检查库依赖"
fi

echo ""
echo "如果问题仍然存在，请尝试以下方法："
echo "1. 使用Docker容器运行"
echo "2. 在虚拟环境中安装"
echo "3. 联系系统管理员安装缺失的库"
echo ""
echo "Docker示例："
echo "docker run -it --rm -v \$(pwd):/workspace python:3.9 bash"
echo "cd /workspace && pip install -r requirements.txt" 