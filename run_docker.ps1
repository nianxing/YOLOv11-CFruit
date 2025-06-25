# 运行YOLOv11-CFruit Docker容器脚本

Write-Host "启动YOLOv11-CFruit Docker环境..." -ForegroundColor Green

# 检查是否在项目根目录
if (-not (Test-Path "Dockerfile")) {
    Write-Host "错误：请在项目根目录运行此脚本" -ForegroundColor Red
    exit 1
}

# 创建必要的目录
Write-Host "创建必要的目录..." -ForegroundColor Yellow
$directories = @("data", "checkpoints", "output")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        try {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "创建目录: $dir" -ForegroundColor Cyan
        } catch {
            Write-Host "创建目录失败: $dir" -ForegroundColor Red
        }
    } else {
        Write-Host "目录已存在: $dir" -ForegroundColor Gray
    }
}

# 检查Docker是否运行
Write-Host "检查Docker服务状态..." -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker服务未运行，请先启动Docker Desktop" -ForegroundColor Red
        exit 1
    }
    Write-Host "Docker服务运行正常" -ForegroundColor Green
} catch {
    Write-Host "无法连接到Docker服务" -ForegroundColor Red
    exit 1
}

# 选择运行模式
Write-Host "`n选择运行模式:" -ForegroundColor Yellow
Write-Host "1. CPU版本 (推荐用于测试)" -ForegroundColor White
Write-Host "2. GPU版本 (需要NVIDIA GPU和nvidia-docker)" -ForegroundColor White

$choice = Read-Host "请输入选择 (1 或 2)"

if ($choice -eq "2") {
    # GPU版本
    Write-Host "启动GPU版本容器..." -ForegroundColor Yellow
    try {
        docker-compose up -d yolov11-cfruit-gpu
        if ($LASTEXITCODE -eq 0) {
            Write-Host "GPU容器启动成功！" -ForegroundColor Green
            # 进入容器
            Write-Host "进入GPU容器..." -ForegroundColor Yellow
            docker exec -it yolov11-cfruit-gpu bash
        } else {
            Write-Host "GPU容器启动失败" -ForegroundColor Red
        }
    } catch {
        Write-Host "启动GPU容器时出错" -ForegroundColor Red
    }
} else {
    # CPU版本
    Write-Host "启动CPU版本容器..." -ForegroundColor Yellow
    try {
        docker-compose up -d yolov11-cfruit
        if ($LASTEXITCODE -eq 0) {
            Write-Host "CPU容器启动成功！" -ForegroundColor Green
            # 进入容器
            Write-Host "进入CPU容器..." -ForegroundColor Yellow
            docker exec -it yolov11-cfruit bash
        } else {
            Write-Host "CPU容器启动失败" -ForegroundColor Red
        }
    } catch {
        Write-Host "启动CPU容器时出错" -ForegroundColor Red
    }
}

Write-Host "`n容器已启动！" -ForegroundColor Green
Write-Host "在容器内，您可以运行以下命令：" -ForegroundColor Cyan
Write-Host "  python test_project.py          # 测试项目" -ForegroundColor White
Write-Host "  python scripts/prepare_data.py  # 准备数据" -ForegroundColor White
Write-Host "  python scripts/train.py         # 训练模型" -ForegroundColor White 