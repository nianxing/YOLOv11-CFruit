#!/bin/bash

# YOLOv11-CFruit 自动化训练和可视化脚本
# 适用于服务器环境，支持后台训练、日志记录和自动视频生成

set -e  # 遇到错误时退出

# 配置参数
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
VISUALIZATION_DIR="${PROJECT_DIR}/visualization"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/train_improved.py"
VISUALIZE_SCRIPT="${PROJECT_DIR}/scripts/visualize_training.py"
QUICK_VISUALIZE_SCRIPT="${PROJECT_DIR}/scripts/show_training_results.py"

# 训练参数
CONFIG_FILE="${PROJECT_DIR}/configs/model/yolov11_cfruit_improved.yaml"
DATA_CONFIG="${PROJECT_DIR}/configs/data/cfruit.yaml"
EPOCHS=100
BATCH_SIZE=4
WORKERS=4
LEARNING_RATE=0.001

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${TIMESTAMP}.pid"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi
    
    # 检查必要的Python包
    python -c "import torch, yaml, numpy" 2>/dev/null || {
        log_error "缺少必要的Python包 (torch, yaml, numpy)"
        exit 1
    }
    
    # 检查ffmpeg（用于视频生成）
    if ! command -v ffmpeg &> /dev/null; then
        log_warning "ffmpeg未安装，视频生成可能失败"
    fi
    
    log_success "依赖检查完成"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$CHECKPOINT_DIR"
    mkdir -p "$VISUALIZATION_DIR"
    
    log_success "目录创建完成"
}

# 检查数据
check_data() {
    log_info "检查训练数据..."
    
    if [ ! -f "$DATA_CONFIG" ]; then
        log_error "数据配置文件不存在: $DATA_CONFIG"
        exit 1
    fi
    
    # 检查数据目录是否存在
    python -c "
import yaml
with open('$DATA_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
train_dir = config['dataset']['train']
val_dir = config['dataset']['val']
import os
if not os.path.exists(train_dir):
    print(f'Train directory not found: {train_dir}')
    exit(1)
if not os.path.exists(val_dir):
    print(f'Val directory not found: {val_dir}')
    exit(1)
print('Data directories found')
"
    
    log_success "数据检查完成"
}

# 开始训练
start_training() {
    log_info "开始后台训练..."
    
    # 构建训练命令
    TRAIN_CMD="python $TRAIN_SCRIPT \
        --config $CONFIG_FILE \
        --data $DATA_CONFIG \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --workers $WORKERS \
        --lr $LEARNING_RATE \
        --output-dir $CHECKPOINT_DIR \
        --log-file $LOG_FILE"
    
    # 后台运行训练
    nohup $TRAIN_CMD > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    
    # 保存PID
    echo $TRAIN_PID > "$PID_FILE"
    
    log_success "训练已启动，PID: $TRAIN_PID"
    log_info "日志文件: $LOG_FILE"
    log_info "使用 'tail -f $LOG_FILE' 查看实时日志"
    log_info "使用 'kill $TRAIN_PID' 停止训练"
}

# 监控训练进度
monitor_training() {
    log_info "开始监控训练进度..."
    
    while kill -0 $TRAIN_PID 2>/dev/null; do
        # 检查训练是否还在运行
        if ! ps -p $TRAIN_PID > /dev/null; then
            break
        fi
        
        # 显示当前进度
        if [ -f "$LOG_FILE" ]; then
            CURRENT_EPOCH=$(grep -o "Epoch [0-9]*" "$LOG_FILE" | tail -1 | grep -o "[0-9]*" || echo "0")
            if [ "$CURRENT_EPOCH" != "0" ]; then
                echo -ne "\r${BLUE}[INFO]${NC} 当前训练进度: Epoch $CURRENT_EPOCH/$EPOCHS"
            fi
        fi
        
        sleep 30  # 每30秒检查一次
    done
    
    echo  # 换行
    log_info "训练监控结束"
}

# 等待训练完成
wait_for_training() {
    log_info "等待训练完成..."
    
    # 等待训练进程结束
    wait $TRAIN_PID
    
    # 检查训练是否成功完成
    if [ $? -eq 0 ]; then
        log_success "训练成功完成"
    else
        log_error "训练失败"
        exit 1
    fi
}

# 生成可视化视频
generate_visualization() {
    log_info "开始生成可视化视频..."
    
    # 检查是否有训练日志
    if [ ! -f "$LOG_FILE" ]; then
        log_warning "未找到训练日志文件，生成示例视频"
        python "$QUICK_VISUALIZE_SCRIPT" --output-dir "$VISUALIZATION_DIR"
    else
        # 检查是否有模型文件
        MODEL_FILE=$(find "$CHECKPOINT_DIR" -name "*.pt" -type f | head -1)
        
        if [ -n "$MODEL_FILE" ]; then
            log_info "使用真实训练数据生成视频"
            python "$VISUALIZE_SCRIPT" \
                --log-file "$LOG_FILE" \
                --model-path "$MODEL_FILE" \
                --config "$CONFIG_FILE" \
                --data "$DATA_CONFIG" \
                --output-dir "$VISUALIZATION_DIR" \
                --fps 2
        else
            log_warning "未找到模型文件，生成示例视频"
            python "$QUICK_VISUALIZE_SCRIPT" --output-dir "$VISUALIZATION_DIR"
        fi
    fi
    
    log_success "可视化视频生成完成"
}

# 生成训练报告
generate_report() {
    log_info "生成训练报告..."
    
    REPORT_FILE="${VISUALIZATION_DIR}/training_report_${TIMESTAMP}.txt"
    
    cat > "$REPORT_FILE" << EOF
YOLOv11-CFruit 训练报告
生成时间: $(date)
训练ID: ${TIMESTAMP}

=== 训练配置 ===
配置文件: ${CONFIG_FILE}
数据配置: ${DATA_CONFIG}
训练轮数: ${EPOCHS}
批次大小: ${BATCH_SIZE}
工作进程: ${WORKERS}
学习率: ${LEARNING_RATE}

=== 训练结果 ===
日志文件: ${LOG_FILE}
模型文件: ${CHECKPOINT_DIR}

=== 生成的文件 ===
EOF

    # 列出生成的文件
    if [ -d "$VISUALIZATION_DIR" ]; then
        find "$VISUALIZATION_DIR" -name "*.mp4" -o -name "*.png" -o -name "*.jpg" | while read file; do
            echo "- $(basename "$file")" >> "$REPORT_FILE"
        done
    fi
    
    # 添加训练统计信息
    if [ -f "$LOG_FILE" ]; then
        echo "" >> "$REPORT_FILE"
        echo "=== 训练统计 ===" >> "$REPORT_FILE"
        echo "最终训练损失: $(grep "Train Loss:" "$LOG_FILE" | tail -1 | grep -o "Train Loss: [0-9.]*" | cut -d' ' -f3 || echo "N/A")" >> "$REPORT_FILE"
        echo "最终验证损失: $(grep "Val Loss:" "$LOG_FILE" | tail -1 | grep -o "Val Loss: [0-9.]*" | cut -d' ' -f3 || echo "N/A")" >> "$REPORT_FILE"
        echo "训练时长: $(grep "Time:" "$LOG_FILE" | tail -1 | grep -o "Time: [0-9:]*" | cut -d' ' -f2 || echo "N/A")" >> "$REPORT_FILE"
    fi
    
    log_success "训练报告已生成: $REPORT_FILE"
}

# 清理临时文件
cleanup() {
    log_info "清理临时文件..."
    
    # 删除PID文件
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
    fi
    
    log_success "清理完成"
}

# 主函数
main() {
    log_info "=== YOLOv11-CFruit 自动化训练和可视化 ==="
    log_info "项目目录: $PROJECT_DIR"
    log_info "时间戳: $TIMESTAMP"
    
    # 检查依赖
    check_dependencies
    
    # 创建目录
    create_directories
    
    # 检查数据
    check_data
    
    # 开始训练
    start_training
    
    # 监控训练
    monitor_training &
    MONITOR_PID=$!
    
    # 等待训练完成
    wait_for_training
    
    # 停止监控
    kill $MONITOR_PID 2>/dev/null || true
    
    # 生成可视化
    generate_visualization
    
    # 生成报告
    generate_report
    
    # 清理
    cleanup
    
    log_success "=== 自动化训练和可视化完成 ==="
    log_info "结果保存在: $VISUALIZATION_DIR"
    log_info "训练报告: ${VISUALIZATION_DIR}/training_report_${TIMESTAMP}.txt"
}

# 信号处理
trap 'log_error "收到中断信号，正在清理..."; cleanup; exit 1' INT TERM

# 运行主函数
main "$@" 