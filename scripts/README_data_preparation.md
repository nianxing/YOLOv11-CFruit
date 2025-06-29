# 数据重新准备指南

本指南将帮助你重新准备油茶果检测数据集。

## 📋 数据准备流程

### 1. 数据格式要求

你的原始数据应该包含：
- **图像文件**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` 等格式
- **标注文件**: Labelme格式的 `.json` 文件
- **文件对应关系**: 每个图像文件都有对应的同名JSON文件

```
原始数据目录结构:
your_data/
├── image1.jpg
├── image1.json
├── image2.jpg
├── image2.json
└── ...
```

### 2. 标注格式要求

JSON文件应该是Labelme格式，包含：
```json
{
  "version": "4.5.6",
  "flags": {},
  "shapes": [
    {
      "label": "cfruit",  // 标签名称
      "points": [[x1, y1], [x2, y2], ...],  // 坐标点
      "group_id": null,
      "shape_type": "polygon",  // 或 "rectangle"
      "flags": {}
    }
  ],
  "imagePath": "image1.jpg",
  "imageData": null,
  "imageHeight": 480,
  "imageWidth": 640
}
```

## 🚀 快速开始

### 方法1: 一键完成（推荐）

```bash
# 完整流程：数据准备 + 训练
python scripts/quick_train.py --input-dir /path/to/your/data --output-dir data/cfruit
```

### 方法2: 分步执行

```bash
# 步骤1: 数据准备
python scripts/prepare_data.py --input-dir /path/to/your/data --output-dir data/cfruit

# 步骤2: 开始训练
python scripts/train.py --config configs/model/yolov11_cfruit.yaml --data configs/data/cfruit.yaml
```

## 📁 输出目录结构

数据准备完成后，会生成以下结构：

```
data/cfruit/
├── train/
│   ├── images/     # 训练图像
│   └── labels/     # 训练标签（YOLO格式）
├── val/
│   ├── images/     # 验证图像
│   └── labels/     # 验证标签
└── test/
    ├── images/     # 测试图像
    └── labels/     # 测试标签

configs/data/cfruit.yaml  # 数据集配置文件
```

## 🔧 详细参数说明

### prepare_data.py 参数

```bash
python scripts/prepare_data.py \
    --input-dir /path/to/your/data \     # 输入目录
    --output-dir data/cfruit \           # 输出目录
    --class-names cfruit \               # 类别名称
    --train-ratio 0.7 \                  # 训练集比例
    --val-ratio 0.2 \                    # 验证集比例
    --test-ratio 0.1 \                   # 测试集比例
    --output-yaml configs/data/cfruit.yaml  # 配置文件路径
```

### quick_train.py 参数

```bash
python scripts/quick_train.py \
    --input-dir /path/to/your/data \     # 输入目录
    --output-dir data/cfruit \           # 输出目录
    --class-names cfruit \               # 类别名称
    --model-type yolov11 \               # 模型类型 (yolov8/yolov11)
    --epochs 100 \                       # 训练轮数
    --batch-size 16 \                    # 批次大小
    --img-size 640 \                     # 图像尺寸
    --device auto \                      # 训练设备
    --skip-data-prep \                   # 跳过数据准备
    --skip-train                         # 跳过训练
```

## 🔍 数据质量检查

### 1. 检查数据完整性

```bash
# 检查数据准备结果
python scripts/check_data.py --data-dir data/cfruit
```

### 2. 可视化标注结果

```bash
# 可视化训练数据
python scripts/quick_visualize.py --data-dir data/cfruit --split train --num-samples 10
```

### 3. 统计信息

```bash
# 显示数据集统计信息
python scripts/show_training_results.py --data-dir data/cfruit
```

## 🛠️ 常见问题解决

### 1. 标签名称不匹配

如果你的JSON文件中标签是"youcha"而不是"cfruit"：

```bash
# 先重命名标签
python scripts/rename_labels.py --directory /path/to/your/data --old-label youcha --new-label cfruit

# 然后准备数据
python scripts/prepare_data.py --input-dir /path/to/your/data --output-dir data/cfruit
```

### 2. 内存不足

如果数据量很大，使用内存优化版本：

```bash
# 使用内存优化的数据准备
python scripts/prepare_data.py --input-dir /path/to/your/data --output-dir data/cfruit --batch-size 8
```

### 3. 数据格式错误

检查JSON文件格式：

```bash
# 验证JSON文件格式
python -c "
import json
import sys
try:
    with open('your_file.json', 'r') as f:
        data = json.load(f)
    print('JSON格式正确')
except Exception as e:
    print(f'JSON格式错误: {e}')
"
```

## 📊 数据集分割比例

默认分割比例：
- **训练集**: 70% (用于模型训练)
- **验证集**: 20% (用于模型验证)
- **测试集**: 10% (用于最终评估)

你可以根据需要调整：

```bash
python scripts/prepare_data.py \
    --input-dir /path/to/your/data \
    --output-dir data/cfruit \
    --train-ratio 0.8 \
    --val-ratio 0.15 \
    --test-ratio 0.05
```

## 🔄 重新准备数据

如果你想重新准备数据：

1. **清理旧数据**:
```bash
rm -rf data/cfruit
rm -f configs/data/cfruit.yaml
```

2. **重新准备**:
```bash
python scripts/prepare_data.py --input-dir /path/to/your/data --output-dir data/cfruit
```

3. **验证结果**:
```bash
python scripts/check_data.py --data-dir data/cfruit
```

## 📈 下一步

数据准备完成后：

1. **开始训练**:
```bash
python scripts/train.py --config configs/model/yolov11_cfruit.yaml --data configs/data/cfruit.yaml
```

2. **监控训练**:
```bash
tensorboard --logdir logs
```

3. **评估模型**:
```bash
python scripts/evaluate_model.py --model-path checkpoints/best.pt
```

## 💡 提示

- 建议先使用小数据集测试流程
- 确保图像和JSON文件一一对应
- 检查标注质量，确保边界框准确
- 定期备份原始数据
- 使用试运行模式检查结果 