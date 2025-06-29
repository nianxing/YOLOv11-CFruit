#!/bin/bash

# YOLOv11-CFruit 项目清理脚本
echo "=== 开始清理项目无用代码 ==="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 备份目录
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

log_info "创建备份目录: $BACKUP_DIR"

# 1. 删除空的或损坏的文件
log_info "删除空的或损坏的文件..."

empty_files=(
    "scripts/quick_gpu_test.py"
    "scripts/setup_gpu.sh"
    "scripts/gpu_diagnosis.py"
    "scripts/fix_circle_labels.py"
    "scripts/debug_labels.py"
)

for file in "${empty_files[@]}"; do
    if [ -f "$file" ]; then
        log_info "删除空文件: $file"
        rm -f "$file"
    fi
done

# 2. 删除重复的训练脚本（保留最稳定的版本）
log_info "整理重复的训练脚本..."

# 保留 train_improved.py 和 simple_train.py，删除 quick_train.py
if [ -f "scripts/quick_train.py" ]; then
    log_info "删除重复的训练脚本: scripts/quick_train.py"
    rm -f "scripts/quick_train.py"
fi

# 3. 删除重复的GPU修复脚本（保留最有效的版本）
log_info "整理GPU修复脚本..."

# 保留 fix_azure_gpu.sh 和 quick_azure_fix.sh，删除其他重复的
if [ -f "scripts/fix_nvidia_driver.sh" ]; then
    log_info "删除重复的GPU修复脚本: scripts/fix_nvidia_driver.sh"
    mv "scripts/fix_nvidia_driver.sh" "$BACKUP_DIR/"
fi

if [ -f "scripts/quick_fix_gpu.sh" ]; then
    log_info "删除重复的GPU修复脚本: scripts/quick_fix_gpu.sh"
    mv "scripts/quick_fix_gpu.sh" "$BACKUP_DIR/"
fi

# 4. 删除重复的数据准备脚本（保留最新版本）
log_info "整理数据准备脚本..."

# 保留 prepare_data_circle_fixed.py，删除其他版本
if [ -f "scripts/prepare_data.py" ]; then
    log_info "备份旧版本数据准备脚本: scripts/prepare_data.py"
    mv "scripts/prepare_data.py" "$BACKUP_DIR/"
fi

if [ -f "scripts/prepare_data_fixed.py" ]; then
    log_info "备份旧版本数据准备脚本: scripts/prepare_data_fixed.py"
    mv "scripts/prepare_data_fixed.py" "$BACKUP_DIR/"
fi

# 5. 删除重复的可视化脚本（保留功能最全的版本）
log_info "整理可视化脚本..."

# 保留 visualize_training.py，删除其他版本
if [ -f "scripts/quick_visualize.py" ]; then
    log_info "删除重复的可视化脚本: scripts/quick_visualize.py"
    mv "scripts/quick_visualize.py" "$BACKUP_DIR/"
fi

# 6. 删除重复的训练脚本（保留改进版本）
log_info "整理训练脚本..."

# 保留 train_improved.py，删除其他版本
if [ -f "scripts/train_memory_optimized.py" ]; then
    log_info "删除重复的训练脚本: scripts/train_memory_optimized.py"
    mv "scripts/train_memory_optimized.py" "$BACKUP_DIR/"
fi

# 7. 删除重复的损失函数（保留简化版本用于测试）
log_info "整理损失函数..."

# 保留 losses.py 和 simple_loss.py，它们都有用途

# 8. 删除 __pycache__ 目录
log_info "删除Python缓存文件..."

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# 9. 删除临时文件
log_info "删除临时文件..."

find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.log" -delete 2>/dev/null || true
find . -name "*.pid" -delete 2>/dev/null || true

# 10. 创建清理后的文件列表
log_info "生成清理报告..."

CLEANUP_REPORT="cleanup_report_$(date +%Y%m%d_%H%M%S).txt"

cat > "$CLEANUP_REPORT" << EOF
YOLOv11-CFruit 项目清理报告
清理时间: $(date)

=== 删除的文件 ===
EOF

# 列出删除的文件
for file in "${empty_files[@]}"; do
    echo "- $file (空文件)" >> "$CLEANUP_REPORT"
done

echo "" >> "$CLEANUP_REPORT"
echo "=== 备份的文件 ===" >> "$CLEANUP_REPORT"
if [ -d "$BACKUP_DIR" ]; then
    find "$BACKUP_DIR" -type f | while read file; do
        echo "- $(basename "$file")" >> "$CLEANUP_REPORT"
    done
fi

echo "" >> "$CLEANUP_REPORT"
echo "=== 保留的核心文件 ===" >> "$CLEANUP_REPORT"
echo "- scripts/train_improved.py (主要训练脚本)" >> "$CLEANUP_REPORT"
echo "- scripts/simple_train.py (简化训练脚本)" >> "$CLEANUP_REPORT"
echo "- scripts/auto_train_and_visualize.sh (自动训练脚本)" >> "$CLEANUP_REPORT"
echo "- scripts/quick_auto_train.sh (快速测试脚本)" >> "$CLEANUP_REPORT"
echo "- scripts/prepare_data_circle_fixed.py (数据准备脚本)" >> "$CLEANUP_REPORT"
echo "- scripts/visualize_training.py (可视化脚本)" >> "$CLEANUP_REPORT"
echo "- scripts/fix_azure_gpu.sh (Azure GPU修复脚本)" >> "$CLEANUP_REPORT"
echo "- scripts/quick_azure_fix.sh (快速Azure修复脚本)" >> "$CLEANUP_REPORT"
echo "- utils/losses.py (完整损失函数)" >> "$CLEANUP_REPORT"
echo "- utils/simple_loss.py (简化损失函数)" >> "$CLEANUP_REPORT"

log_success "清理完成！"
log_info "清理报告: $CLEANUP_REPORT"
log_info "备份目录: $BACKUP_DIR"

# 显示当前项目结构
echo ""
log_info "=== 清理后的项目结构 ==="
echo "scripts/ 目录:"
ls -la scripts/ | grep -E "\.(py|sh)$" | head -20

echo ""
echo "utils/ 目录:"
ls -la utils/ | grep -E "\.(py)$"

echo ""
log_success "项目清理完成！" 