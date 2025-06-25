# Windows Docker Environment Setup Script
# Run this script as Administrator

Write-Host "Setting up Windows Docker environment..." -ForegroundColor Green

# Check administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "Error: Please run this script as Administrator" -ForegroundColor Red
    exit 1
}

# Enable Hyper-V feature
Write-Host "Enabling Hyper-V..." -ForegroundColor Yellow
try {
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All -NoRestart
    Write-Host "Hyper-V enabled successfully" -ForegroundColor Green
} catch {
    Write-Host "Failed to enable Hyper-V: $($_.Exception.Message)" -ForegroundColor Red
}

# Enable Containers feature
Write-Host "Enabling Containers feature..." -ForegroundColor Yellow
try {
    Enable-WindowsOptionalFeature -Online -FeatureName Containers -NoRestart
    Write-Host "Containers feature enabled successfully" -ForegroundColor Green
} catch {
    Write-Host "Failed to enable Containers feature: $($_.Exception.Message)" -ForegroundColor Red
}

# Enable Windows Subsystem for Linux (WSL2)
Write-Host "Enabling WSL2..." -ForegroundColor Yellow
try {
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    Write-Host "WSL2 enabled successfully" -ForegroundColor Green
} catch {
    Write-Host "Failed to enable WSL2: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nWindows features setup completed!" -ForegroundColor Green
Write-Host "Please restart your computer and then install Docker Desktop." -ForegroundColor Yellow
Write-Host "After restart, run the following commands to check if Docker works properly:" -ForegroundColor Cyan
Write-Host "  .\verify_docker_en.ps1" -ForegroundColor White
Write-Host "  docker --version" -ForegroundColor White
Write-Host "  docker run hello-world" -ForegroundColor White 