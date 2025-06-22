#!/bin/bash

# YOLOv11-CFruit Python 3.12 安装脚本
echo "正在安装 YOLOv11-CFruit 依赖 (Python 3.12)..."
echo ""

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "检测到Python版本: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo "警告: 当前Python版本不是3.12，建议使用Python 3.12"
    echo "继续安装可能会遇到兼容性问题"
    read -p "是否继续? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查是否为root用户
if [ "$EUID" -eq 0 ]; then
    echo "检测到root权限，将安装系统级依赖"
    SUDO=""
else
    echo "使用sudo安装系统级依赖"
    SUDO="sudo"
fi

# 检测Linux发行版
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    echo "无法检测Linux发行版"
    exit 1
fi

echo "检测到系统: $OS $VER"
echo ""

# 安装系统依赖
echo "安装系统依赖..."

if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    # Ubuntu/Debian
    echo "使用apt安装依赖..."
    $SUDO apt update
    $SUDO apt install -y \
        python3.12 \
        python3.12-pip \
        python3.12-dev \
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
        libtiff-dev \
        libatlas-base-dev \
        gfortran \
        wget \
        curl \
        git

elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Fedora"* ]]; then
    # CentOS/RHEL/Fedora
    echo "使用yum/dnf安装依赖..."
    if command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
    else
        PKG_MANAGER="yum"
    fi
    
    $SUDO $PKG_MANAGER update -y
    $SUDO $PKG_MANAGER install -y \
        python3.12 \
        python3.12-pip \
        python3.12-devel \
        mesa-libGL \
        glib2 \
        libSM \
        libXext \
        libXrender \
        libgomp \
        gtk3 \
        ffmpeg-devel \
        libjpeg-turbo-devel \
        libpng-devel \
        libtiff-devel \
        atlas-devel \
        gcc \
        gcc-c++ \
        gfortran \
        wget \
        curl \
        git

elif [[ "$OS" == *"Arch"* ]]; then
    # Arch Linux
    echo "使用pacman安装依赖..."
    $SUDO pacman -Syu --noconfirm
    $SUDO pacman -S --noconfirm \
        python \
        python-pip \
        mesa \
        glib2 \
        libsm \
        libxext \
        libxrender \
        gtk3 \
        ffmpeg \
        libjpeg-turbo \
        libpng \
        libtiff \
        blas \
        lapack \
        gcc \
        gfortran \
        wget \
        curl \
        git

else
    echo "不支持的Linux发行版: $OS"
    echo "请手动安装以下依赖:"
    echo "- python3.12"
    echo "- libgl1-mesa-glx"
    echo "- libglib2.0-0"
    echo "- libsm6"
    echo "- libxext6"
    echo "- libxrender-dev"
    echo "- libgomp1"
    echo "- libgtk-3-0"
    echo "- ffmpeg相关库"
    echo "- 图像处理库 (libjpeg, libpng, libtiff)"
    echo ""
    read -p "是否继续安装Python依赖? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "系统依赖安装完成！"
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.12"
    exit 1
fi

echo "Python版本:"
python3 --version
echo ""

# 升级pip
echo "升级pip..."
python3 -m pip install --upgrade pip

# 安装基础依赖
echo "安装基础依赖..."
python3 -m pip install PyYAML torch torchvision

# 安装OpenCV (使用headless版本避免GUI依赖)
echo "安装OpenCV (headless版本)..."
python3 -m pip install opencv-python-headless

# 安装项目依赖
echo "安装项目依赖..."
python3 -m pip install -r requirements.txt

# 安装项目
echo "安装项目..."
python3 -m pip install -e .

echo ""
echo "安装完成！"
echo ""
echo "运行测试:"
echo "python3 test_project.py"
echo ""
echo "Python 3.12 兼容性说明:"
echo "- 所有依赖包已更新到支持Python 3.12的版本"
echo "- 如果遇到兼容性问题，请检查具体错误信息"
echo "- 某些旧版本的包可能需要手动更新"
echo "" 