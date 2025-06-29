# YOLOv11-CFruit 快速开始指南

## 📋 概述

本指南将帮助您快速使用基于labelme标注的水果数据来训练YOLOv11-CFruit模型。

---

**最后更新：2024年6月**  
**文档版本：v1.0**

---

## 🚀 环境准备

### 1. 安装依赖

```bash
# 安装Python依赖包
pip install -r requirements.txt

# 或者使用conda
conda install pytorch torchvision torchaudio -c pytorch
pip install -r requirements.txt
```

### 2. 检查环境

```bash
python test_project.py
```

## 📊 数据准备

### 数据格式要求

您的数据应该包含：
- 图像文件：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` 格式
- 标注文件：labelme格式的 `.json` 文件

### 目录结构示例

```
your_data/
├── image1.jpg
├── image1.json
├── image2.jpg
├── image2.json
└── ...
```

### 快速数据准备

```bash
# 使用示例数据演示
python examples/prepare_and_train.py --create-sample

# 使用真实数据（支持圆形标注）
python scripts/prepare_data_circle_fixed.py --input-dir /path/to/your/data --output-dir data/cfruit --class-names cfruit
```

## 🎯 训练模型

### 一键训练（推荐）

```bash
# 使用自动训练脚本
./scripts/auto_train_and_visualize.sh

# 使用快速测试脚本
./scripts/quick_auto_train.sh
```

### 分步训练

#### 步骤1: 数据准备

```bash
python scripts/prepare_data_circle_fixed.py \
    --input-dir /path/to/your/data \
    --output-dir data/cfruit \
    --class-names cfruit
```

#### 步骤2: 开始训练

```bash
# 改进版训练（推荐）
python scripts/train_improved.py \
    --config configs/model/yolov11_cfruit_improved.yaml \
    --data configs/data/cfruit.yaml \
    --epochs 100 \
    --batch-size 16 \
    --save-dir checkpoints

# 简化训练
python scripts/simple_train.py \
    --config configs/model/yolov11_cfruit.yaml \
    --data configs/data/cfruit.yaml \
    --epochs 100 \
    --batch-size 16 \
    --save-dir checkpoints
```

## ⚙️ 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | 必需 | 包含图像和JSON文件的目录 |
| `--class-names` | cfruit | 类别名称列表 |
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 16 | 批次大小 |
| `--img-size` | 640 | 输入图像尺寸 |
| `--save-dir` | checkpoints | 模型保存目录 |
| `--device` | auto | 训练设备（cuda/cpu/auto） |

## 📈 监控训练

### TensorBoard

```bash
tensorboard --logdir logs
```

访问 http://localhost:6006 查看训练曲线。

### 训练日志

训练日志保存在 `logs/train.log`，包含：
- 损失值变化
- 学习率变化
- 验证指标
- 模型保存信息

## 🧪 模型测试

训练完成后，您可以测试模型：

```bash
python examples/basic_detection.py \
    --model checkpoints/best.pt \
    --image /path/to/test/image.jpg
```

## ❓ 常见问题

### Q: 如何处理不同尺寸的图像？
A: 训练时会自动调整到统一尺寸（默认640x640），保持宽高比。

### Q: 标注质量不好怎么办？
A: 建议：
1. 检查标注是否准确
2. 增加数据增强
3. 调整学习率
4. 增加训练轮数

### Q: 训练速度慢怎么办？
A: 可以：
1. 减少批次大小
2. 使用GPU训练
3. 减少图像尺寸
4. 使用预训练权重

### Q: 如何添加新的类别？
A: 修改 `--class-names` 参数，例如：
```bash
--class-names cfruit unripe_cfruit
```

## 🚀 性能优化建议

### 数据质量
- 确保标注准确性
- 平衡各类别样本数量
- 增加数据多样性

### 训练策略
- 使用预训练权重
- 调整学习率调度
- 使用混合精度训练
- 启用EMA（指数移动平均）

### 硬件优化
- 使用GPU训练
- 调整批次大小
- 优化数据加载
- 使用SSD存储

## 📝 完整示例

```bash
# 1. 创建示例数据并训练
python examples/prepare_and_train.py --create-sample --epochs 10

# 2. 使用真实数据训练
python scripts/prepare_data_circle_fixed.py \
    --input-dir /path/to/your/data \
    --output-dir data/cfruit \
    --class-names cfruit

python scripts/train_improved.py \
    --config configs/model/yolov11_cfruit_improved.yaml \
    --data configs/data/cfruit.yaml \
    --epochs 100 \
    --batch-size 16 \
    --save-dir checkpoints

# 3. 监控训练
tensorboard --logdir logs

# 4. 测试模型
python examples/basic_detection.py \
    --model checkpoints/best.pt \
    --image /path/to/test/image.jpg
```

## 🔗 下一步

训练完成后，您可以：
1. 测试模型性能
2. 部署到生产环境
3. 优化模型结构
4. 收集更多数据

详细的使用说明请参考 [docs/data_preparation.md](docs/data_preparation.md) 和 [USAGE.md](USAGE.md)。 