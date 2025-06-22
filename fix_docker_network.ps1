# Docker网络问题解决脚本 (Windows)
Write-Host "=== Docker网络问题解决脚本 (Windows) ===" -ForegroundColor Green
Write-Host ""

# 检查Docker是否运行
try {
    docker info | Out-Null
    Write-Host "✓ Docker正在运行" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker未运行，请先启动Docker Desktop" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 方法1: 测试网络连接
Write-Host "方法1: 测试网络连接..." -ForegroundColor Yellow
try {
    Test-NetConnection -ComputerName docker.io -Port 443 | Out-Null
    Write-Host "✓ 网络连接正常" -ForegroundColor Green
} catch {
    Write-Host "⚠ 网络连接可能有问题" -ForegroundColor Yellow
}

# 方法2: 配置Docker镜像源
Write-Host ""
Write-Host "方法2: 配置Docker镜像源..." -ForegroundColor Yellow
Write-Host "建议使用国内镜像源加速下载" -ForegroundColor Cyan

# 检查Docker Desktop设置
Write-Host "请在Docker Desktop中配置镜像源:" -ForegroundColor Cyan
Write-Host "1. 打开Docker Desktop" -ForegroundColor White
Write-Host "2. 进入 Settings -> Docker Engine" -ForegroundColor White
Write-Host "3. 添加以下配置到JSON中:" -ForegroundColor White
Write-Host "   {" -ForegroundColor Gray
Write-Host "     ""registry-mirrors"": [" -ForegroundColor Gray
Write-Host "       ""https://docker.mirrors.ustc.edu.cn""," -ForegroundColor Gray
Write-Host "       ""https://hub-mirror.c.163.com""," -ForegroundColor Gray
Write-Host "       ""https://mirror.baidubce.com""" -ForegroundColor Gray
Write-Host "     ]" -ForegroundColor Gray
Write-Host "   }" -ForegroundColor Gray
Write-Host "4. 点击 Apply & Restart" -ForegroundColor White

# 方法3: 使用不同的基础镜像
Write-Host ""
Write-Host "方法3: 使用不同的基础镜像..." -ForegroundColor Yellow
Write-Host "已更新Dockerfile使用python:3.11-slim" -ForegroundColor Green
Write-Host "如果仍有问题，可以尝试:" -ForegroundColor Cyan
Write-Host "- python:3.10-slim" -ForegroundColor White
Write-Host "- python:3.9-slim" -ForegroundColor White

# 方法4: 手动拉取镜像
Write-Host ""
Write-Host "方法4: 手动拉取镜像..." -ForegroundColor Yellow
Write-Host "尝试手动拉取镜像:" -ForegroundColor Cyan
Write-Host "docker pull python:3.11-slim" -ForegroundColor White

# 方法5: 清理Docker缓存
Write-Host ""
Write-Host "方法5: 清理Docker缓存..." -ForegroundColor Yellow
Write-Host "如果问题持续，可以清理Docker缓存:" -ForegroundColor Cyan
Write-Host "docker system prune -a" -ForegroundColor White

# 方法6: 使用WSL2
Write-Host ""
Write-Host "方法6: 使用WSL2..." -ForegroundColor Yellow
Write-Host "如果Docker Desktop有问题，可以尝试WSL2:" -ForegroundColor Cyan
Write-Host "wsl" -ForegroundColor White
Write-Host "cd /mnt/d/code/YOLOv8-CFruit" -ForegroundColor White
Write-Host "docker build -t yolov11-cfruit ." -ForegroundColor White

Write-Host ""
Write-Host "=== 解决方案总结 ===" -ForegroundColor Green
Write-Host "1. 检查网络连接" -ForegroundColor White
Write-Host "2. 配置Docker Desktop镜像源" -ForegroundColor White
Write-Host "3. 使用python:3.11-slim (已更新)" -ForegroundColor White
Write-Host "4. 手动拉取镜像" -ForegroundColor White
Write-Host "5. 清理Docker缓存" -ForegroundColor White
Write-Host "6. 使用WSL2" -ForegroundColor White
Write-Host ""
Write-Host "建议按顺序尝试上述方法" -ForegroundColor Cyan 