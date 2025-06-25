# Windows Docker环境设置脚本
# 以管理员身份运行此脚本

Write-Host "正在设置Windows Docker环境..." -ForegroundColor Green

# 检查管理员权限
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "错误：请以管理员身份运行此脚本" -ForegroundColor Red
    exit 1
}

# 启用Hyper-V功能
Write-Host "启用Hyper-V..." -ForegroundColor Yellow
try {
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All -NoRestart
    Write-Host "Hyper-V启用成功" -ForegroundColor Green
} catch {
    Write-Host "启用Hyper-V失败: $($_.Exception.Message)" -ForegroundColor Red
}

# 启用容器功能
Write-Host "启用容器功能..." -ForegroundColor Yellow
try {
    Enable-WindowsOptionalFeature -Online -FeatureName Containers -NoRestart
    Write-Host "容器功能启用成功" -ForegroundColor Green
} catch {
    Write-Host "启用容器功能失败: $($_.Exception.Message)" -ForegroundColor Red
}

# 启用Windows子系统for Linux (WSL2)
Write-Host "启用WSL2..." -ForegroundColor Yellow
try {
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    Write-Host "WSL2启用成功" -ForegroundColor Green
} catch {
    Write-Host "启用WSL2失败: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nWindows功能设置完成！" -ForegroundColor Green
Write-Host "请重启计算机后继续安装Docker Desktop。" -ForegroundColor Yellow
Write-Host "重启后，请运行以下命令来检查Docker是否正常工作：" -ForegroundColor Cyan
Write-Host "  .\verify_docker.ps1" -ForegroundColor White
Write-Host "  docker --version" -ForegroundColor White
Write-Host "  docker run hello-world" -ForegroundColor White 