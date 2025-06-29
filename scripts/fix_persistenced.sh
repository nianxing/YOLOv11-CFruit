#!/bin/bash

# 修复nvidia-persistenced问题
echo "=== 修复nvidia-persistenced问题 ==="

# 检查当前安装的NVIDIA驱动版本
echo "[INFO] 检查当前NVIDIA驱动版本..."
nvidia_version=$(dpkg -l | grep nvidia-driver | head -1 | awk '{print $2}' | cut -d'-' -f3)
echo "检测到驱动版本: $nvidia_version"

if [ -z "$nvidia_version" ]; then
    echo "[ERROR] 未检测到NVIDIA驱动，请先安装驱动"
    exit 1
fi

# 安装对应版本的compute-utils
echo "[INFO] 安装nvidia-compute-utils-$nvidia_version..."
if sudo apt install nvidia-compute-utils-$nvidia_version -y; then
    echo "[SUCCESS] nvidia-compute-utils-$nvidia_version 安装成功"
else
    echo "[WARNING] 安装失败，尝试其他版本..."
    
    # 尝试其他常见版本
    for version in 535 525 470 465; do
        echo "[INFO] 尝试安装 nvidia-compute-utils-$version..."
        if sudo apt install nvidia-compute-utils-$version -y; then
            echo "[SUCCESS] nvidia-compute-utils-$version 安装成功"
            break
        fi
    done
fi

# 启动nvidia-persistenced服务
echo "[INFO] 启动nvidia-persistenced服务..."
sudo systemctl enable nvidia-persistenced
sudo systemctl start nvidia-persistenced

# 检查服务状态
echo "[INFO] 检查服务状态..."
sudo systemctl status nvidia-persistenced --no-pager

echo "[SUCCESS] 修复完成！"
echo "[INFO] 现在可以测试: nvidia-smi" 