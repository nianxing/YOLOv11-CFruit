#!/bin/bash

# Azure GPU修复脚本
# 专门针对Azure云服务器的NVIDIA GPU配置

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Azure环境
check_azure_environment() {
    log_info "检查Azure环境..."
    
    # 检查是否为Azure
    if [ -f /sys/hypervisor/uuid ] && grep -q "microsoft" /sys/hypervisor/uuid; then
        log_success "检测到Azure环境"
    else
        log_warning "可能不是Azure环境"
    fi
    
    # 检查内核版本
    kernel_version=$(uname -r)
    log_info "内核版本: $kernel_version"
    
    # 检查是否为Azure内核
    if [[ $kernel_version == *"azure"* ]]; then
        log_info "检测到Azure内核"
    else
        log_warning "不是Azure内核"
    fi
}

# 检查当前驱动状态
check_current_driver_status() {
    log_info "检查当前驱动状态..."
    
    # 检查已安装的NVIDIA包
    nvidia_packages=$(dpkg -l | grep nvidia | awk '{print $2}')
    if [ -n "$nvidia_packages" ]; then
        log_info "已安装的NVIDIA包:"
        echo "$nvidia_packages"
    else
        log_error "未找到NVIDIA包"
    fi
    
    # 检查驱动模块
    if lsmod | grep nvidia > /dev/null; then
        log_success "NVIDIA驱动模块已加载"
        lsmod | grep nvidia
    else
        log_error "NVIDIA驱动模块未加载"
    fi
    
    # 检查DKMS状态
    if command -v dkms > /dev/null; then
        log_info "DKMS状态:"
        sudo dkms status
    else
        log_warning "DKMS未安装"
    fi
}

# 安装适合Azure的NVIDIA驱动
install_azure_nvidia_driver() {
    log_info "安装适合Azure的NVIDIA驱动..."
    
    # 检查是否有推荐的驱动
    if command -v ubuntu-drivers > /dev/null; then
        log_info "检查推荐的驱动..."
        recommended_driver=$(ubuntu-drivers devices | grep recommended | awk '{print $3}')
        if [ -n "$recommended_driver" ]; then
            log_info "推荐驱动: $recommended_driver"
        fi
    fi
    
    # 卸载现有驱动
    log_info "卸载现有NVIDIA驱动..."
    sudo apt remove --purge nvidia* -y
    sudo apt autoremove -y
    
    # 清理DKMS
    sudo dkms remove nvidia --all 2>/dev/null || true
    
    # 更新包列表
    log_info "更新包列表..."
    sudo apt update
    
    # 安装适合Azure的驱动
    log_info "安装NVIDIA驱动..."
    
    # 尝试安装最新驱动
    if sudo apt install nvidia-driver-545 -y; then
        log_success "安装nvidia-driver-545成功"
    elif sudo apt install nvidia-driver-535 -y; then
        log_success "安装nvidia-driver-535成功"
    elif sudo apt install nvidia-driver-525 -y; then
        log_success "安装nvidia-driver-525成功"
    else
        log_error "无法安装NVIDIA驱动"
        return 1
    fi
    
    # 安装CUDA工具包
    log_info "安装CUDA工具包..."
    sudo apt install nvidia-cuda-toolkit -y
    
    # 安装NVIDIA持久化服务（明确指定版本）
    log_info "安装NVIDIA持久化服务..."
    if sudo apt install nvidia-compute-utils-535 -y; then
        log_success "安装nvidia-compute-utils-535成功"
    elif sudo apt install nvidia-compute-utils-525 -y; then
        log_success "安装nvidia-compute-utils-525成功"
    else
        log_warning "持久化服务安装失败，但继续其他步骤"
    fi
}

# 配置内核模块
configure_kernel_modules() {
    log_info "配置内核模块..."
    
    # 更新initramfs
    log_info "更新initramfs..."
    sudo update-initramfs -u
    
    # 配置模块加载
    log_info "配置模块自动加载..."
    echo "nvidia" | sudo tee -a /etc/modules
    echo "nvidia_uvm" | sudo tee -a /etc/modules
    echo "nvidia_drm" | sudo tee -a /etc/modules
    echo "nvidia_modeset" | sudo tee -a /etc/modules
    
    # 创建模块配置
    sudo tee /etc/modprobe.d/nvidia.conf > /dev/null <<EOF
options nvidia NVreg_UsePageAttributeTable=1
options nvidia NVreg_EnablePCIeGen3=1
options nvidia NVreg_InitializeSystemMemoryAllocations=1
EOF
}

# 设置环境变量
setup_environment_variables() {
    log_info "设置环境变量..."
    
    # 添加到bashrc
    if ! grep -q "CUDA_HOME" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# NVIDIA CUDA Environment Variables" >> ~/.bashrc
        echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
        echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
    fi
    
    # 立即生效
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=0
    
    log_success "环境变量设置完成"
}

# 安装PyTorch GPU版本
install_pytorch_gpu() {
    log_info "安装PyTorch GPU版本..."
    
    # 卸载CPU版本
    pip3 uninstall torch torchvision torchaudio -y 2>/dev/null || true
    
    # 安装GPU版本
    log_info "安装PyTorch CUDA 11.8版本..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    log_success "PyTorch GPU版本安装完成"
}

# 重启后验证
post_reboot_verification() {
    log_info "重启后验证步骤..."
    
    echo ""
    echo "=== 重启后需要执行的命令 ==="
    echo ""
    echo "1. 检查NVIDIA驱动:"
    echo "   nvidia-smi"
    echo ""
    echo "2. 检查CUDA:"
    echo "   nvcc --version"
    echo ""
    echo "3. 检查PyTorch:"
    echo "   python3 -c \"import torch; print('CUDA可用:', torch.cuda.is_available())\""
    echo ""
    echo "4. 运行GPU测试:"
    echo "   python3 scripts/quick_gpu_test.py"
    echo ""
    echo "5. 开始训练:"
    echo "   python3 scripts/train.py --device 0 --batch-size 32"
    echo ""
}

# 主函数
main() {
    echo "=== Azure GPU修复工具 ==="
    echo ""
    
    log_info "检测到Azure环境，开始专门修复..."
    echo ""
    
    # 检查Azure环境
    check_azure_environment
    echo ""
    
    # 检查当前状态
    check_current_driver_status
    echo ""
    
    # 询问是否继续
    echo "是否继续安装/修复NVIDIA驱动？(y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "操作已取消"
        exit 0
    fi
    
    # 安装驱动
    if install_azure_nvidia_driver; then
        log_success "NVIDIA驱动安装成功"
    else
        log_error "NVIDIA驱动安装失败"
        exit 1
    fi
    
    # 配置内核模块
    configure_kernel_modules
    echo ""
    
    # 设置环境变量
    setup_environment_variables
    echo ""
    
    # 安装PyTorch
    install_pytorch_gpu
    echo ""
    
    # 显示重启后验证步骤
    post_reboot_verification
    
    log_success "Azure GPU环境配置完成！"
    log_warning "请重启系统以完成配置: sudo reboot"
}

# 运行主函数
main "$@" 