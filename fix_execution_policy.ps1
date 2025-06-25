# 修复PowerShell执行策略脚本
# 需要以管理员身份运行

Write-Host "修复PowerShell执行策略..." -ForegroundColor Green

# 检查当前执行策略
Write-Host "当前执行策略: $(Get-ExecutionPolicy)" -ForegroundColor Yellow

# 临时设置为允许运行脚本
Write-Host "设置执行策略为RemoteSigned..." -ForegroundColor Yellow
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

Write-Host "执行策略已更新为: $(Get-ExecutionPolicy)" -ForegroundColor Green

# 现在可以运行设置脚本
Write-Host "`n现在可以运行Docker设置脚本了:" -ForegroundColor Cyan
Write-Host ".\setup_docker_windows.ps1" -ForegroundColor White

# 可选：恢复原始执行策略（更安全）
Write-Host "`n注意：为了安全起见，建议在完成后恢复执行策略:" -ForegroundColor Yellow
Write-Host "Set-ExecutionPolicy -ExecutionPolicy Restricted -Scope CurrentUser" -ForegroundColor White 