# Scripts 目录

本目录包含YOLOv11-CFruit项目的核心脚本。

## 📁 文件说明

### 核心训练脚本

#### `train_improved_v2.py`
主要的训练脚本，包含所有改进功能：
- 改进的YOLOv11架构
- 多GPU训练支持
- 内存优化（混合精度、梯度累积）
- 数据增强（Mixup、Mosaic等）
- 早停机制
- 学习率调度

**使用方法：**
```bash
python scripts/train_improved_v2.py \
    --config configs/model/yolov11_cfruit_improved.yaml \
    --data-config configs/data/cfruit.yaml \
    --batch-size 2 \
    --epochs 100 \
    --save-dir checkpoints
```

**参数说明：**
- `--config`: 模型配置文件路径
- `--data-config`: 数据配置文件路径
- `--batch-size`: 批次大小
- `--epochs`: 训练轮数
- `--save-dir`: 保存目录
- `--resume`: 恢复训练的检查点路径（可选）

### 数据检查脚本

#### `check_data.py`
检查数据路径和格式的脚本：
- 验证数据目录结构
- 检查图像和标签文件
- 统计数据集信息
- 创建示例数据（如果需要）

**使用方法：**
```bash
# 检查现有数据
python scripts/check_data.py

# 创建示例数据
python scripts/check_data.py --create-sample
```

### 模型评估脚本

#### `evaluate_model.py`
评估训练好的模型性能：
- 计算mAP、Precision、Recall等指标
- 生成评估报告
- 可视化检测结果

**使用方法：**
```bash
python scripts/evaluate_model.py \
    --model checkpoints/best.pt \
    --data-config configs/data/cfruit.yaml \
    --output-dir evaluation_results
```

**参数说明：**
- `--model`: 模型文件路径
- `--data-config`: 数据配置文件路径
- `--output-dir`: 输出目录
- `--conf-thresh`: 置信度阈值
- `--nms-thresh`: NMS阈值

## 🚀 快速开始

### 1. 检查数据
```bash
python scripts/check_data.py
```

### 2. 开始训练
```bash
python scripts/train_improved_v2.py \
    --config configs/model/yolov11_cfruit_improved.yaml \
    --data-config configs/data/cfruit.yaml \
    --batch-size 2 \
    --epochs 100 \
    --save-dir checkpoints
```

### 3. 评估模型
```bash
python scripts/evaluate_model.py \
    --model checkpoints/best.pt \
    --data-config configs/data/cfruit.yaml \
    --output-dir evaluation_results
```

## 📊 训练监控

### 查看训练日志
```bash
# 实时查看训练日志
tail -f training_*.log

# 查看GPU使用情况
nvidia-smi

# 查看进程
ps aux | grep train_improved
```

### 训练指标
训练过程中会记录以下指标：
- 训练损失 (Training Loss)
- 验证损失 (Validation Loss)
- 学习率 (Learning Rate)
- GPU内存使用情况

## 🔧 故障排除

### 常见问题

#### 1. GPU内存不足
```bash
# 减少批次大小
--batch-size 1

# 设置内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### 2. 数据路径错误
```bash
# 检查数据路径
python scripts/check_data.py
```

#### 3. 依赖包版本冲突
```bash
# 重新安装依赖
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📝 注意事项

1. **数据格式**：确保数据符合项目要求的格式
2. **GPU内存**：根据GPU内存大小调整批次大小
3. **环境变量**：建议设置内存优化环境变量
4. **日志监控**：定期查看训练日志和GPU使用情况
5. **模型保存**：训练过程中会自动保存最佳模型

## 🤝 贡献

如需添加新功能或修复问题，请：
1. 创建功能分支
2. 修改相关脚本
3. 更新文档
4. 提交Pull Request 