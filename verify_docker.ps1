# Docker安装验证脚本

Write-Host "验证Docker安装..." -ForegroundColor Green

# 检查Docker版本
Write-Host "`n检查Docker版本:" -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host $dockerVersion -ForegroundColor Green
    } else {
        throw "Docker命令执行失败"
    }
} catch {
    Write-Host "Docker未安装或未正确配置" -ForegroundColor Red
    Write-Host "请先安装Docker Desktop" -ForegroundColor Yellow
    exit 1
}

# 检查Docker Compose版本
Write-Host "`n检查Docker Compose版本:" -ForegroundColor Yellow
try {
    $composeVersion = docker-compose --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host $composeVersion -ForegroundColor Green
    } else {
        Write-Host "Docker Compose未安装" -ForegroundColor Red
    }
} catch {
    Write-Host "Docker Compose未安装" -ForegroundColor Red
}

# 运行测试容器
Write-Host "`n运行测试容器:" -ForegroundColor Yellow
try {
    $testResult = docker run --rm hello-world 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker安装成功！" -ForegroundColor Green
        Write-Host $testResult -ForegroundColor Cyan
    } else {
        throw "测试容器运行失败"
    }
} catch {
    Write-Host "Docker测试失败，请检查安装" -ForegroundColor Red
    Write-Host "确保Docker Desktop正在运行" -ForegroundColor Yellow
}

# 检查Docker服务状态
Write-Host "`n检查Docker服务状态:" -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker服务运行正常" -ForegroundColor Green
        # 显示一些关键信息
        $dockerInfo | Select-String -Pattern "Server Version|Operating System|Kernel Version" | ForEach-Object {
            Write-Host $_.ToString() -ForegroundColor Cyan
        }
    } else {
        throw "无法获取Docker信息"
    }
} catch {
    Write-Host "无法获取Docker信息" -ForegroundColor Red
    Write-Host "请确保Docker Desktop正在运行" -ForegroundColor Yellow
}

Write-Host "`n验证完成！" -ForegroundColor Green 