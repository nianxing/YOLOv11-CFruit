# YOLOv11-CFruit Anaconda安装脚本 (Windows版本)
# 适用于Azure Windows虚拟机

Write-Host "=== YOLOv11-CFruit Anaconda安装脚本 ===" -ForegroundColor Green
Write-Host "开始安装配置..." -ForegroundColor Yellow

# 检查是否已安装Anaconda
if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "✓ Anaconda已安装" -ForegroundColor Green
} else {
    Write-Host "正在安装Anaconda..." -ForegroundColor Yellow
    
    # 下载Anaconda安装程序
    $anacondaUrl = "https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Windows-x86_64.exe"
    $anacondaInstaller = "anaconda_installer.exe"
    
    Write-Host "下载Anaconda安装程序..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $anacondaUrl -OutFile $anacondaInstaller
    
    # 安装Anaconda (静默安装)
    Write-Host "安装Anaconda..." -ForegroundColor Yellow
    Start-Process -FilePath $anacondaInstaller -ArgumentList "/S /D=$env:USERPROFILE\anaconda3" -Wait
    
    # 清理安装文件
    Remove-Item $anacondaInstaller -Force
    
    # 刷新环境变量
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    Write-Host "✓ Anaconda安装完成" -ForegroundColor Green
}

# 创建conda环境
Write-Host "正在创建conda环境..." -ForegroundColor Yellow
conda env create -f conda_setup.yml

# 激活环境
Write-Host "激活conda环境..." -ForegroundColor Yellow
conda activate yolov11-cfruit

# 验证安装
Write-Host "验证安装..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision版本: {torchvision.__version__}')"
python -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')"

Write-Host "=== 安装完成 ===" -ForegroundColor Green
Write-Host "使用方法:" -ForegroundColor Cyan
Write-Host "1. 激活环境: conda activate yolov11-cfruit" -ForegroundColor White
Write-Host "2. 运行训练: python scripts/train.py" -ForegroundColor White
Write-Host "3. 运行检测: python examples/basic_detection.py" -ForegroundColor White 