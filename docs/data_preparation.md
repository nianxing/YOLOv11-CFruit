# 数据准备和训练指南

## 📋 概述

本指南将帮助您使用基于labelme标注的水果数据来训练YOLOv11-CFruit模型。

---

**最后更新：2024年6月**  
**文档版本：v1.0**

---

## 📊 数据格式要求

### 输入数据格式

您的数据应该包含以下文件：
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

### labelme标注格式

每个JSON文件应包含以下结构：
```json
{
  "version": "4.5.6",
  "flags": {},
  "shapes": [
    {
      "label": "cfruit",
      "points": [[x1, y1], [x2, y2], ...],
      "group_id": null,
      "shape_type": "polygon",
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

### 方法1: 一键训练（推荐）

使用自动训练脚本，自动完成数据准备和训练：

```bash
# 自动训练和可视化
bash scripts/auto_train_and_visualize.sh

# 快速测试训练
bash scripts/quick_auto_train.sh
```

### 方法2: 分步执行

#### 步骤1: 数据准备

```bash
python scripts/prepare_data_circle_fixed.py \
    --input-dir /path/to/your/data \
    --output-dir data/cfruit \
    --class-names cfruit \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1
```

这将：
- 将数据集分割为训练/验证/测试集
- 将labelme格式转换为YOLO格式
- 支持圆形标注转换为矩形框
- 创建数据集配置文件

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

## 📊 数据准备详细说明

### 数据转换过程

1. **数据集分割**: 按指定比例分割为训练/验证/测试集
2. **格式转换**: 将labelme的JSON格式转换为YOLO的txt格式
3. **坐标转换**: 将多边形/矩形/圆形标注转换为边界框格式
4. **配置文件生成**: 创建数据集配置文件

### 支持的标注类型

- **polygon**: 多边形标注
- **rectangle**: 矩形标注
- **circle**: 圆形标注（转换为矩形框）
- **point**: 点标注（转换为小矩形）

### 输出格式

转换后的标签文件格式（每行一个目标）：
```
class_id x_center y_center width height
```

其中坐标已归一化到[0,1]范围。

## 🎯 训练配置

### 模型配置

项目支持两种模型配置：

#### YOLOv11-CFruit（推荐）
- 使用C2f骨干网络
- 改进的Transformer颈部
- AdamW优化器
- 高级数据增强
- 自动混合精度训练

#### YOLOv8-CFruit
- 使用CSPDarknet骨干网络
- PANet颈部网络
- 无锚点检测头
- 适合实时检测

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 100 | 训练轮数 |
| batch_size | 16 | 批次大小 |
| img_size | 640 | 输入图像尺寸 |
| lr | 0.001 | 学习率 |
| weight_decay | 0.0005 | 权重衰减 |

### 数据增强

训练时自动应用以下增强：
- 随机水平翻转
- 随机旋转
- 随机缩放
- 颜色抖动
- 马赛克增强（YOLOv11）

## 📈 监控训练

### TensorBoard

启动TensorBoard监控训练过程：
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

### 训练可视化

```bash
# 可视化训练过程
python scripts/visualize_training.py --log-dir logs

# 显示训练结果
python scripts/show_training_results.py --checkpoint checkpoints/best.pt
```

## 🧪 模型测试

训练完成后，测试模型性能：

```bash
# 评估模型
python scripts/evaluate_model.py --model-path checkpoints/best.pt

# 快速测试
python scripts/quick_test.py --model checkpoints/best.pt --data-dir data/cfruit/val
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

### Q: 如何处理圆形标注？
A: 使用 `prepare_data_circle_fixed.py` 脚本，它会自动将圆形标注转换为矩形框。

## 🔧 数据质量检查

```bash
# 检查数据质量
python scripts/check_data.py --data-dir data/cfruit
```

## 🔄 标签重命名

如果需要批量重命名标签：

```bash
# 快速重命名
python scripts/quick_rename_labels.py \
    --input-dir /path/to/json/files \
    --old-label youcha \
    --new-label cfruit

# 完整重命名工具
python scripts/rename_labels.py \
    --input-dir /path/to/json/files \
    --old-label youcha \
    --new-label cfruit
```

## 📝 完整示例

```bash
# 1. 准备数据
python scripts/prepare_data_circle_fixed.py \
    --input-dir /path/to/your/data \
    --output-dir data/cfruit \
    --class-names cfruit

# 2. 检查数据质量
python scripts/check_data.py --data-dir data/cfruit

# 3. 开始训练
python scripts/train_improved.py \
    --config configs/model/yolov11_cfruit_improved.yaml \
    --data configs/data/cfruit.yaml \
    --epochs 100 \
    --batch-size 16 \
    --save-dir checkpoints

# 4. 监控训练
tensorboard --logdir logs

# 5. 测试模型
python scripts/evaluate_model.py --model-path checkpoints/best.pt
```

## 🔗 相关链接

- [快速开始指南](../QUICK_START.md)
- [使用说明](../USAGE.md)
- [脚本说明](../scripts/README.md) 