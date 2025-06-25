# 测试所有PowerShell脚本的语法
Write-Host "测试PowerShell脚本语法..." -ForegroundColor Green

$scripts = @(
    "setup_docker_windows.ps1",
    "verify_docker.ps1", 
    "run_docker.ps1",
    "fix_execution_policy.ps1"
)

foreach ($script in $scripts) {
    if (Test-Path $script) {
        try {
            $null = [System.Management.Automation.PSParser]::Tokenize((Get-Content $script -Raw), [ref]$null)
            Write-Host "✓ $script - 语法正确" -ForegroundColor Green
        } catch {
            Write-Host "✗ $script - 语法错误: $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        Write-Host "? $script - 文件不存在" -ForegroundColor Yellow
    }
}

Write-Host "`n语法检查完成！" -ForegroundColor Green 