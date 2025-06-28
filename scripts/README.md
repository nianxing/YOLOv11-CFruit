# Scripts 目录说明

本目录包含了YOLOv11-CFruit项目的所有脚本文件，用于训练、评估、数据处理等任务。

## 📁 脚本分类

### 🎯 训练脚本

#### `train_improved.py` ⭐ (推荐)
- **功能**: 改进版训练脚本，包含早停、混合精度等高级功能
- **特点**: 
  - 自动设备检测（CPU/GPU）
  - 早停机制
  - 自动混合精度训练
  - 梯度累积
  - 详细的训练监控
- **使用**: `python scripts/train_improved.py`

#### `train.py`
- **功能**: 基础训练脚本
- **特点**: 简单易用，适合入门
- **使用**: `python scripts/train.py`

#### `train_memory_optimized.py`
- **功能**: 内存优化版训练脚本
- **特点**: 适用于显存较小的GPU
- **使用**: `python scripts/train_memory_optimized.py`

#### `quick_train.py`
- **功能**: 快速训练脚本
- **特点**: 预设参数，一键训练
- **使用**: `python scripts/quick_train.py`

### 📊 数据处理脚本

#### `prepare_data.py`
- **功能**: 数据预处理和准备
- **特点**: 
  - 自动数据增强
  - 格式转换
  - 数据验证
- **使用**: `python scripts/prepare_data.py --data-path /path/to/data`

#### `check_data.py`
- **功能**: 数据质量检查
- **特点**: 
  - 检查图像和标签一致性
  - 统计数据集信息
  - 检测异常数据
- **使用**: `python scripts/check_data.py --data-path /path/to/data`

### 🔍 评估脚本

#### `evaluate_model.py`
- **功能**: 模型评估和测试
- **特点**: 
  - 计算mAP等指标
  - 生成评估报告
  - 可视化检测结果
- **使用**: `python scripts/evaluate_model.py --model-path checkpoints/best.pt`

#### `quick_test.py`
- **功能**: 快速模型测试
- **特点**: 简单测试，快速验证
- **使用**: `python scripts/quick_test.py --model-path checkpoints/best.pt`

### 📈 可视化脚本

#### `visualize_training.py`
- **功能**: 训练过程可视化
- **特点**: 
  - 损失曲线
  - 学习率变化
  - 指标趋势
- **使用**: `python scripts/visualize_training.py --log-dir logs`

#### `quick_visualize.py`
- **功能**: 快速数据可视化
- **特点**: 
  - 数据集样本展示
  - 检测结果可视化
  - 批量图像处理
- **使用**: `python scripts/quick_visualize.py --data-path /path/to/data`

#### `show_training_results.py`
- **功能**: 训练结果展示
- **特点**: 
  - 训练日志分析
  - 性能指标展示
  - 结果对比
- **使用**: `python scripts/show_training_results.py --log-dir logs`

### 🔄 自动化脚本

#### `auto_train_and_visualize.sh`
- **功能**: 自动化训练和可视化
- **特点**: 
  - 一键完成训练
  - 自动生成可视化报告
  - 批量处理
- **使用**: `bash scripts/auto_train_and_visualize.sh`

## 🚀 使用指南

### 1. 环境准备
```bash
# 确保在项目根目录
cd YOLOv11-CFruit

# 检查Python环境
python --version

# 检查PyTorch
python -c "import torch; print(torch.__version__)"
```

### 2. 数据准备
```bash
# 准备数据集
python scripts/prepare_data.py --data-path /path/to/data

# 检查数据质量
python scripts/check_data.py --data-path /path/to/data
```

### 3. 模型训练
```bash
# 使用改进版训练脚本（推荐）
python scripts/train_improved.py --device auto --epochs 100

# 或使用快速训练
python scripts/quick_train.py
```

### 4. 模型评估
```bash
# 评估模型性能
python scripts/evaluate_model.py --model-path checkpoints/best.pt

# 快速测试
python scripts/quick_test.py --model-path checkpoints/best.pt
```

### 5. 结果可视化
```bash
# 可视化训练过程
python scripts/visualize_training.py --log-dir logs

# 可视化检测结果
python scripts/quick_visualize.py --data-path /path/to/test/data
```

## ⚙️ 参数说明

### 通用参数
- `--device`: 训练设备 (auto/cpu/cuda)
- `--batch-size`: 批次大小
- `--epochs`: 训练轮数
- `--save-dir`: 模型保存目录
- `--log-dir`: 日志保存目录

### 训练参数
- `--config`: 模型配置文件
- `--data`: 数据配置文件
- `--resume`: 从检查点恢复训练
- `--num-workers`: 数据加载工作进程数

### 评估参数
- `--model-path`: 模型文件路径
- `--test-data`: 测试数据路径
- `--conf-threshold`: 置信度阈值
- `--iou-threshold`: IoU阈值

## 📝 注意事项

1. **GPU内存**: 根据GPU显存调整批次大小
2. **数据路径**: 确保数据路径正确且可访问
3. **配置文件**: 检查配置文件格式和参数
4. **日志监控**: 定期查看训练日志和可视化结果
5. **模型保存**: 定期保存检查点，避免训练中断

## 🔧 故障排除

### 常见问题
1. **CUDA内存不足**: 减小批次大小或使用内存优化脚本
2. **数据加载错误**: 检查数据路径和格式
3. **训练不收敛**: 调整学习率和数据增强参数
4. **模型保存失败**: 检查磁盘空间和权限

### 调试技巧
1. 使用 `--device cpu` 测试脚本功能
2. 检查日志文件获取详细错误信息
3. 使用小数据集进行快速测试
4. 逐步增加训练参数复杂度

---

**最后更新**: 2024年12月
**版本**: v1.0 