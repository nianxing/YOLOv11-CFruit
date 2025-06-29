# YOLOv11-CFruit 项目清理脚本 (PowerShell版本)
Write-Host "=== 开始清理项目无用代码 ===" -ForegroundColor Blue

# 备份目录
$BACKUP_DIR = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $BACKUP_DIR -Force | Out-Null

Write-Host "[INFO] 创建备份目录: $BACKUP_DIR" -ForegroundColor Blue

# 1. 删除空的或损坏的文件
Write-Host "[INFO] 删除空的或损坏的文件..." -ForegroundColor Blue

$emptyFiles = @(
    "scripts/quick_gpu_test.py",
    "scripts/setup_gpu.sh", 
    "scripts/gpu_diagnosis.py",
    "scripts/fix_circle_labels.py",
    "scripts/debug_labels.py"
)

foreach ($file in $emptyFiles) {
    if (Test-Path $file) {
        Write-Host "[INFO] 删除空文件: $file" -ForegroundColor Blue
        Remove-Item $file -Force
    }
}

# 2. 删除重复的训练脚本
Write-Host "[INFO] 整理重复的训练脚本..." -ForegroundColor Blue

if (Test-Path "scripts/quick_train.py") {
    Write-Host "[INFO] 删除重复的训练脚本: scripts/quick_train.py" -ForegroundColor Blue
    Remove-Item "scripts/quick_train.py" -Force
}

# 3. 删除重复的GPU修复脚本
Write-Host "[INFO] 整理GPU修复脚本..." -ForegroundColor Blue

if (Test-Path "scripts/fix_nvidia_driver.sh") {
    Write-Host "[INFO] 删除重复的GPU修复脚本: scripts/fix_nvidia_driver.sh" -ForegroundColor Blue
    Move-Item "scripts/fix_nvidia_driver.sh" "$BACKUP_DIR/"
}

if (Test-Path "scripts/quick_fix_gpu.sh") {
    Write-Host "[INFO] 删除重复的GPU修复脚本: scripts/quick_fix_gpu.sh" -ForegroundColor Blue
    Move-Item "scripts/quick_fix_gpu.sh" "$BACKUP_DIR/"
}

# 4. 删除重复的数据准备脚本
Write-Host "[INFO] 整理数据准备脚本..." -ForegroundColor Blue

if (Test-Path "scripts/prepare_data.py") {
    Write-Host "[INFO] 备份旧版本数据准备脚本: scripts/prepare_data.py" -ForegroundColor Blue
    Move-Item "scripts/prepare_data.py" "$BACKUP_DIR/"
}

if (Test-Path "scripts/prepare_data_fixed.py") {
    Write-Host "[INFO] 备份旧版本数据准备脚本: scripts/prepare_data_fixed.py" -ForegroundColor Blue
    Move-Item "scripts/prepare_data_fixed.py" "$BACKUP_DIR/"
}

# 5. 删除重复的可视化脚本
Write-Host "[INFO] 整理可视化脚本..." -ForegroundColor Blue

if (Test-Path "scripts/quick_visualize.py") {
    Write-Host "[INFO] 删除重复的可视化脚本: scripts/quick_visualize.py" -ForegroundColor Blue
    Move-Item "scripts/quick_visualize.py" "$BACKUP_DIR/"
}

# 6. 删除重复的训练脚本
Write-Host "[INFO] 整理训练脚本..." -ForegroundColor Blue

if (Test-Path "scripts/train_memory_optimized.py") {
    Write-Host "[INFO] 删除重复的训练脚本: scripts/train_memory_optimized.py" -ForegroundColor Blue
    Move-Item "scripts/train_memory_optimized.py" "$BACKUP_DIR/"
}

# 7. 删除 __pycache__ 目录
Write-Host "[INFO] 删除Python缓存文件..." -ForegroundColor Blue

Get-ChildItem -Path . -Recurse -Directory -Name "__pycache__" | ForEach-Object {
    Remove-Item $_ -Recurse -Force -ErrorAction SilentlyContinue
}

Get-ChildItem -Path . -Recurse -Include "*.pyc", "*.pyo" | Remove-Item -Force

# 8. 删除临时文件
Write-Host "[INFO] 删除临时文件..." -ForegroundColor Blue

Get-ChildItem -Path . -Recurse -Include "*.tmp", "*.log", "*.pid" | Remove-Item -Force

# 9. 创建清理报告
Write-Host "[INFO] 生成清理报告..." -ForegroundColor Blue

$CLEANUP_REPORT = "cleanup_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

@"
YOLOv11-CFruit 项目清理报告
清理时间: $(Get-Date)

=== 删除的文件 ===
"@ | Out-File -FilePath $CLEANUP_REPORT -Encoding UTF8

foreach ($file in $emptyFiles) {
    "- $file (空文件)" | Out-File -FilePath $CLEANUP_REPORT -Append -Encoding UTF8
}

@"

=== 备份的文件 ===
"@ | Out-File -FilePath $CLEANUP_REPORT -Append -Encoding UTF8

if (Test-Path $BACKUP_DIR) {
    Get-ChildItem -Path $BACKUP_DIR -File | ForEach-Object {
        "- $($_.Name)" | Out-File -FilePath $CLEANUP_REPORT -Append -Encoding UTF8
    }
}

@"

=== 保留的核心文件 ===
- scripts/train_improved.py (主要训练脚本)
- scripts/simple_train.py (简化训练脚本)
- scripts/auto_train_and_visualize.sh (自动训练脚本)
- scripts/quick_auto_train.sh (快速测试脚本)
- scripts/prepare_data_circle_fixed.py (数据准备脚本)
- scripts/visualize_training.py (可视化脚本)
- scripts/fix_azure_gpu.sh (Azure GPU修复脚本)
- scripts/quick_azure_fix.sh (快速Azure修复脚本)
- utils/losses.py (完整损失函数)
- utils/simple_loss.py (简化损失函数)
"@ | Out-File -FilePath $CLEANUP_REPORT -Append -Encoding UTF8

Write-Host "[SUCCESS] 清理完成！" -ForegroundColor Green
Write-Host "[INFO] 清理报告: $CLEANUP_REPORT" -ForegroundColor Blue
Write-Host "[INFO] 备份目录: $BACKUP_DIR" -ForegroundColor Blue

# 显示当前项目结构
Write-Host ""
Write-Host "=== 清理后的项目结构 ===" -ForegroundColor Blue
Write-Host "scripts/ 目录:" -ForegroundColor Yellow
Get-ChildItem -Path "scripts" -Include "*.py", "*.sh" | Select-Object Name | Format-Table -AutoSize

Write-Host ""
Write-Host "utils/ 目录:" -ForegroundColor Yellow
Get-ChildItem -Path "utils" -Include "*.py" | Select-Object Name | Format-Table -AutoSize

Write-Host ""
Write-Host "[SUCCESS] 项目清理完成！" -ForegroundColor Green 