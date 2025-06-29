#!/bin/bash

# 快速自动训练脚本 - 用于测试
echo "=== 快速自动训练测试 ==="

# 配置参数
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/simple_train.py"
CONFIG_FILE="${PROJECT_DIR}/configs/model/yolov11_cfruit_improved.yaml"
DATA_CONFIG="${PROJECT_DIR}/configs/data/cfruit.yaml"

# 训练参数（简化版本）
EPOCHS=3
BATCH_SIZE=2
WORKERS=2

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${PROJECT_DIR}/logs/quick_train_${TIMESTAMP}.log"

# 创建日志目录
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/checkpoints"

echo "[INFO] 开始快速训练测试..."
echo "[INFO] 训练轮数: $EPOCHS"
echo "[INFO] 批次大小: $BATCH_SIZE"
echo "[INFO] 工作进程: $WORKERS"
echo "[INFO] 日志文件: $LOG_FILE"

# 构建训练命令
TRAIN_CMD="python $TRAIN_SCRIPT \
    --config $CONFIG_FILE \
    --data $DATA_CONFIG \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --num-workers $WORKERS \
    --save-dir ${PROJECT_DIR}/checkpoints \
    --log-dir ${PROJECT_DIR}/logs"

echo "[INFO] 执行命令: $TRAIN_CMD"

# 运行训练
$TRAIN_CMD 2>&1 | tee "$LOG_FILE"

# 检查结果
if [ $? -eq 0 ]; then
    echo "[SUCCESS] 快速训练测试成功完成！"
    echo "[INFO] 检查日志文件: $LOG_FILE"
else
    echo "[ERROR] 快速训练测试失败"
    echo "[INFO] 查看错误日志: $LOG_FILE"
    exit 1
fi 