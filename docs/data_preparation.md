# 数据准备和训练指南

## 概述

本指南将帮助您使用基于labelme标注的油茶果数据来训练YOLOv8/YOLOv11-CFruit模型。

## 数据格式要求

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

## 快速开始

### 方法1: 一键训练（推荐）

使用快速训练脚本，自动完成数据准备和训练：

```bash
python scripts/quick_train.py --input-dir /path/to/your/data --class-names cfruit
```

参数说明：
- `--input-dir`: 包含图像和JSON文件的目录
- `--class-names`: 类别名称（默认：cfruit）
- `--model-type`: 模型类型（yolov8/yolov11，默认：yolov8）
- `--epochs`: 训练轮数（默认：100）
- `--batch-size`: 批次大小（默认：16）
- `--img-size`: 输入图像尺寸（默认：640）

### 方法2: 分步执行

#### 步骤1: 数据准备

```bash
python scripts/prepare_data.py \
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
- 创建数据集配置文件

#### 步骤2: 开始训练

```bash
python scripts/train.py \
    --config configs/model/yolov8_cfruit.yaml \
    --data configs/data/cfruit.yaml \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640
```

## 数据准备详细说明

### 数据转换过程

1. **数据集分割**: 按指定比例分割为训练/验证/测试集
2. **格式转换**: 将labelme的JSON格式转换为YOLO的txt格式
3. **坐标转换**: 将多边形/矩形标注转换为边界框格式
4. **配置文件生成**: 创建数据集配置文件

### 支持的标注类型

- **polygon**: 多边形标注
- **rectangle**: 矩形标注
- **point**: 点标注（转换为小矩形）

### 输出格式

转换后的标签文件格式（每行一个目标）：
```
class_id x_center y_center width height
```

其中坐标已归一化到[0,1]范围。

## 训练配置

### 模型配置

项目支持两种模型配置：

#### YOLOv8-CFruit
- 使用CSPDarknet骨干网络
- PANet颈部网络
- 无锚点检测头
- 适合实时检测

#### YOLOv11-CFruit
- 使用C2f骨干网络
- 改进的Transformer颈部
- AdamW优化器
- 高级数据增强

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

## 监控训练

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

## 常见问题

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

## 性能优化建议

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

## 下一步

训练完成后，您可以：
1. 测试模型性能
2. 部署到生产环境
3. 优化模型结构
4. 收集更多数据

详细的使用说明请参考 [USAGE.md](../USAGE.md)。 