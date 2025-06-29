# Scripts 目录说明

本目录包含了YOLOv11-CFruit项目的所有脚本文件，用于训练、评估、数据处理等任务。

---

**最后更新：2024年6月**  
**文档版本：v1.0**

---

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
- **使用**: `python scripts/train_improved.py --device cuda --batch-size 8 --save-dir checkpoints`

#### `simple_train.py` ⭐ (简化版)
- **功能**: 简化训练脚本，易于调试
- **特点**: 简单易用，适合入门和调试
- **使用**: `python scripts/simple_train.py --device cuda --batch-size 8 --save-dir checkpoints`

#### `train.py`
- **功能**: 基础训练脚本
- **特点**: 标准训练流程
- **使用**: `python scripts/train.py`

### 📊 数据处理脚本

#### `prepare_data_circle_fixed.py` ⭐ (推荐)
- **功能**: 数据预处理和准备（支持圆形标注）
- **特点**: 
  - 支持labelme JSON格式
  - 支持圆形标注转换为矩形框
  - 自动数据增强
  - 格式转换
  - 数据验证
- **使用**: `python scripts/prepare_data_circle_fixed.py --input-dir /path/to/data --output-dir data/cfruit --class-names cfruit`

#### `check_data.py`
- **功能**: 数据质量检查
- **特点**: 
  - 检查图像和标签一致性
  - 统计数据集信息
  - 检测异常数据
- **使用**: `python scripts/check_data.py --data-dir data/cfruit`

#### `quick_rename_labels.py`
- **功能**: 快速标签重命名
- **特点**: 
  - 批量重命名JSON标签
  - 支持备份和预览
- **使用**: `python scripts/quick_rename_labels.py --input-dir /path/to/json --old-label youcha --new-label cfruit`

#### `rename_labels.py`
- **功能**: 完整标签重命名工具
- **特点**: 
  - 递归处理目录
  - 支持多种标签格式
  - 详细日志记录
- **使用**: `python scripts/rename_labels.py --input-dir /path/to/json --old-label youcha --new-label cfruit`

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
- **使用**: `python scripts/quick_test.py --model checkpoints/best.pt --data-dir data/cfruit/val`

### 📈 可视化脚本

#### `visualize_training.py`
- **功能**: 训练过程可视化
- **特点**: 
  - 损失曲线
  - 学习率变化
  - 指标趋势
- **使用**: `python scripts/visualize_training.py --log-dir logs`

#### `show_training_results.py`
- **功能**: 训练结果展示
- **特点**: 
  - 训练日志分析
  - 性能指标展示
  - 结果对比
- **使用**: `python scripts/show_training_results.py --checkpoint checkpoints/best.pt`

### 🔄 自动化脚本

#### `auto_train_and_visualize.sh` ⭐ (推荐)
- **功能**: 自动化训练和可视化
- **特点**: 
  - 一键完成训练
  - 自动生成可视化报告
  - 批量处理
- **使用**: `bash scripts/auto_train_and_visualize.sh`

#### `quick_auto_train.sh`
- **功能**: 快速自动训练
- **特点**: 
  - 快速测试训练流程
  - 预设参数
- **使用**: `bash scripts/quick_auto_train.sh`

### 🛠️ GPU相关脚本

#### `fix_azure_gpu.sh`
- **功能**: Azure GPU环境修复
- **特点**: 
  - 修复NVIDIA驱动问题
  - 配置GPU环境
- **使用**: `bash scripts/fix_azure_gpu.sh`

#### `quick_azure_fix.sh`
- **功能**: 快速Azure GPU修复
- **特点**: 
  - 快速修复常见问题
  - 简化操作流程
- **使用**: `bash scripts/quick_azure_fix.sh`

#### `fix_persistenced.sh`
- **功能**: NVIDIA持久化服务修复
- **特点**: 
  - 修复持久化服务
  - 优化GPU性能
- **使用**: `bash scripts/fix_persistenced.sh`

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
# 准备数据集（支持圆形标注）
python scripts/prepare_data_circle_fixed.py --input-dir /path/to/data --output-dir data/cfruit --class-names cfruit

# 检查数据质量
python scripts/check_data.py --data-dir data/cfruit
```

### 3. 模型训练
```bash
# 使用改进版训练脚本（推荐）
python scripts/train_improved.py --device cuda --batch-size 8 --save-dir checkpoints

# 或使用简化训练
python scripts/simple_train.py --device cuda --batch-size 8 --save-dir checkpoints
```

### 4. 模型评估
```bash
# 评估模型性能
python scripts/evaluate_model.py --model-path checkpoints/best.pt

# 快速测试
python scripts/quick_test.py --model checkpoints/best.pt --data-dir data/cfruit/val
```

### 5. 结果可视化
```bash
# 可视化训练过程
python scripts/visualize_training.py --log-dir logs

# 显示训练结果
python scripts/show_training_results.py --checkpoint checkpoints/best.pt
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
1. **CUDA内存不足**: 减小批次大小或使用简化训练脚本
2. **数据加载错误**: 检查数据路径和格式
3. **训练不收敛**: 调整学习率和数据增强参数
4. **模型保存失败**: 检查磁盘空间和权限

### 调试技巧
1. 使用 `--device cpu` 测试脚本功能
2. 检查日志文件获取详细错误信息
3. 使用小数据集进行快速测试
4. 逐步增加训练参数复杂度

---

**最后更新**: 2024年6月  
**版本**: v1.0 