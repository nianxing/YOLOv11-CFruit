#!/bin/bash

# Azure GPU快速修复脚本
echo "=== Azure GPU快速修复 ==="

# 检查当前状态
echo "[INFO] 检查当前状态..."
echo "内核版本: $(uname -r)"
echo "GPU检测:"
lspci | grep -i nvidia

# 卸载现有驱动
echo "[INFO] 卸载现有NVIDIA驱动..."
sudo apt remove --purge nvidia* -y
sudo apt autoremove -y

# 清理DKMS
echo "[INFO] 清理DKMS..."
sudo dkms remove nvidia --all 2>/dev/null || true

# 更新系统
echo "[INFO] 更新系统..."
sudo apt update

# 安装适合Azure的驱动
echo "[INFO] 安装NVIDIA驱动..."
if sudo apt install nvidia-driver-535 -y; then
    echo "[SUCCESS] 驱动安装成功"
else
    echo "[ERROR] 驱动安装失败，尝试其他版本..."
    sudo apt install nvidia-driver-525 -y
fi

# 安装CUDA
echo "[INFO] 安装CUDA工具包..."
sudo apt install nvidia-cuda-toolkit -y

# 安装持久化服务（明确指定版本）
echo "[INFO] 安装NVIDIA持久化服务..."
if sudo apt install nvidia-compute-utils-535 -y; then
    echo "[SUCCESS] 持久化服务安装成功"
else
    echo "[WARNING] 535版本安装失败，尝试其他版本..."
    if sudo apt install nvidia-compute-utils-525 -y; then
        echo "[SUCCESS] 525版本持久化服务安装成功"
    else
        echo "[WARNING] 持久化服务安装失败，继续其他步骤..."
    fi
fi

# 更新initramfs
echo "[INFO] 更新initramfs..."
sudo update-initramfs -u

# 设置环境变量
echo "[INFO] 设置环境变量..."
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# NVIDIA CUDA Environment Variables" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
fi

echo "[SUCCESS] 修复完成！"
echo "[WARNING] 请重启系统: sudo reboot"
echo ""
echo "重启后运行以下命令验证:"
echo "1. nvidia-smi"
echo "2. nvcc --version"
echo "3. python3 -c \"import torch; print('CUDA可用:', torch.cuda.is_available())\"" 