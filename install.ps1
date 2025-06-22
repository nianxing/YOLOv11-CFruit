# YOLOv8-CFruit 安装脚本
Write-Host "正在安装 YOLOv8-CFruit 依赖..." -ForegroundColor Green
Write-Host ""

# 检查Python是否安装
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python版本: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "错误: 未找到Python，请先安装Python 3.8+" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

Write-Host ""

# 升级pip
Write-Host "升级pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 安装基础依赖
Write-Host "安装基础依赖..." -ForegroundColor Yellow
pip install PyYAML torch torchvision

# 安装项目依赖
Write-Host "安装项目依赖..." -ForegroundColor Yellow
pip install -r requirements.txt

# 安装项目
Write-Host "安装项目..." -ForegroundColor Yellow
pip install -e .

Write-Host ""
Write-Host "安装完成！" -ForegroundColor Green
Write-Host ""
Write-Host "运行测试:" -ForegroundColor Cyan
Write-Host "python test_project.py" -ForegroundColor White
Write-Host ""
Read-Host "按任意键退出" 