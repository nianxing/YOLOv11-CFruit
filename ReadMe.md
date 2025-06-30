# YOLOv11-CFruit: 油茶果检测模型

基于YOLOv11的油茶果（Camellia oleifera）检测模型，专门针对油茶果检测任务进行了优化和改进。

## 🚀 特性

- **改进的YOLOv11架构**：针对油茶果检测进行了专门优化
- **多GPU训练支持**：支持DataParallel和DistributedDataParallel
- **内存优化**：自动混合精度训练，梯度累积，内存管理优化
- **数据增强**：Mixup、Mosaic、旋转、剪切等增强策略
- **早停机制**：防止过拟合，自动保存最佳模型
- **学习率调度**：余弦退火调度器，自适应学习率调整

## 📋 目录结构

```
YOLOv11-CFruit/
├── README.md                 # 项目文档
├── LICENSE                   # MIT许可证
├── requirements.txt          # Python依赖
├── configs/                  # 配置文件
│   ├── data/cfruit.yaml     # 数据配置
│   └── model/yolov11_cfruit_improved.yaml # 模型配置
├── data/
│   └── dataset.py           # 数据集类
├── models/                   # 模型文件
│   ├── yolov11_cfruit.py    # YOLOv11模型
│   ├── backbone/            # 骨干网络
│   ├── neck/               # 颈部网络
│   └── head/               # 检测头
├── training/                # 训练模块
├── utils/                   # 工具函数
│   ├── losses.py           # 损失函数
│   └── transforms.py       # 数据变换
├── scripts/                 # 核心脚本
│   ├── train_improved_v2.py # 主要训练脚本
│   ├── check_data.py        # 数据检查
│   └── evaluate_model.py    # 模型评估
└── docs/                    # 详细文档
```

## 🛠️ 安装

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (GPU训练)
- 16GB+ GPU内存 (推荐)

### 快速安装

```bash
# 克隆项目
git clone https://github.com/your-repo/YOLOv11-CFruit.git
cd YOLOv11-CFruit

# 安装依赖
pip install -r requirements.txt

# 验证安装
python scripts/check_data.py
```

### Conda安装

```bash
# 创建conda环境
conda create -n yolov11-cfruit python=3.9
conda activate yolov11-cfruit

# 安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

## 📊 数据准备

### 数据格式

项目支持以下数据格式：
- **图像文件**：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- **标注文件**：Labelme格式的`.json`文件

### 目录结构

```
your_data/
├── image1.jpg
├── image1.json
├── image2.jpg
├── image2.json
└── ...
```

### 标注格式

JSON文件应为Labelme格式：
```json
{
  "version": "4.5.6",
  "flags": {},
  "shapes": [
    {
      "label": "cfruit",
      "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
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

### 数据检查

```bash
# 检查数据路径和格式
python scripts/check_data.py
```

## 🚀 训练

### 快速开始

```bash
# 创建示例数据（如果没有真实数据）
python scripts/check_data.py

# 开始训练
python scripts/train_improved_v2.py \
    --config configs/model/yolov11_cfruit_improved.yaml \
    --data-config configs/data/cfruit.yaml \
    --batch-size 2 \
    --epochs 100 \
    --save-dir checkpoints
```

### 多GPU训练

```bash
# 设置内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 使用4个GPU训练
nohup python scripts/train_improved_v2.py \
    --config configs/model/yolov11_cfruit_improved.yaml \
    --data-config configs/data/cfruit.yaml \
    --batch-size 2 \
    --epochs 100 \
    --save-dir checkpoints_improved \
    > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 从检查点恢复训练

```bash
python scripts/train_improved_v2.py \
    --config configs/model/yolov11_cfruit_improved.yaml \
    --data-config configs/data/cfruit.yaml \
    --batch-size 2 \
    --epochs 100 \
    --save-dir checkpoints_improved \
    --resume checkpoints/best.pt
```

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 模型配置文件 | `configs/model/yolov11_cfruit_improved.yaml` |
| `--data-config` | 数据配置文件 | `configs/data/cfruit.yaml` |
| `--batch-size` | 批次大小 | 2 |
| `--epochs` | 训练轮数 | 100 |
| `--save-dir` | 保存目录 | `checkpoints` |
| `--resume` | 恢复训练检查点 | 无 |

## 🔧 配置

### 模型配置

主要配置在 `configs/model/yolov11_cfruit_improved.yaml`：

```yaml
model:
  name: "YOLOv11-CFruit-Improved"
  input_size: 640
  num_classes: 1
  anchors: [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
  anchor_masks: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

training:
  optimizer: "AdamW"
  learning_rate: 0.001
  weight_decay: 0.0005
  batch_size: 2
  epochs: 100
  scheduler: "cosine"
  gradient_accumulation_steps: 4
  mixed_precision: true
  gradient_clip: 10.0
```

### 数据配置

数据配置在 `configs/data/cfruit.yaml`：

```yaml
dataset:
  train: 'data/cfruit/train/images'
  val: 'data/cfruit/val/images'
  train_labels: 'data/cfruit/train/labels'
  val_labels: 'data/cfruit/val/labels'
  nc: 1
  names: ['cfruit']
```

## 📈 监控训练

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

## 🎯 评估

### 模型评估

```bash
# 评估模型性能
python scripts/evaluate_model.py \
    --model checkpoints/best.pt \
    --data-config configs/data/cfruit.yaml \
    --output-dir evaluation_results
```

### 评估指标

- **mAP@0.5**: 平均精度 (IoU=0.5)
- **mAP@0.5:0.95**: 平均精度 (IoU=0.5:0.95)
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1分数

## 🔍 推理

### 单张图像推理

```python
from models.yolov11_cfruit import YOLOv11CFruit
import torch

# 加载模型
model = YOLOv11CFruit(config)
model.load_state_dict(torch.load('checkpoints/best.pt'))
model.eval()

# 推理
with torch.no_grad():
    predictions = model(image)
```

### 批量推理

```python
# 批量处理
results = []
for images in dataloader:
    with torch.no_grad():
        predictions = model(images)
        results.extend(predictions)
```

## 🐛 故障排除

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

### 性能优化

#### 内存优化
- 使用混合精度训练 (`mixed_precision: true`)
- 启用梯度累积 (`gradient_accumulation_steps: 4`)
- 设置内存管理环境变量

#### 训练优化
- 使用余弦退火学习率调度
- 启用早停机制
- 使用Focal Loss和标签平滑

## 📚 技术细节

### 模型架构

- **骨干网络**: CSPDarknet + CBAM注意力机制
- **颈部网络**: PANet + 特征金字塔
- **检测头**: Anchor-free检测头
- **损失函数**: YOLOv11Loss (EIoU + Focal Loss + DFL)

### 数据增强

- **几何增强**: 旋转、缩放、剪切、透视变换
- **颜色增强**: HSV调整、亮度对比度
- **高级增强**: Mixup、Mosaic、Copy-Paste

### 训练策略

- **学习率调度**: 余弦退火 + 预热
- **优化器**: AdamW + 权重衰减
- **正则化**: 标签平滑、梯度裁剪
- **早停**: 基于验证损失的早停机制

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/your-repo/YOLOv11-CFruit.git
cd YOLOv11-CFruit

# 安装开发依赖
pip install -r requirements.txt
pip install pytest black flake8

# 运行测试
pytest tests/
```

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- YOLOv11原始实现
- PyTorch团队
- 开源社区贡献者

## 📞 联系方式

- 项目主页: https://github.com/your-repo/YOLOv11-CFruit
- 问题反馈: https://github.com/your-repo/YOLOv11-CFruit/issues
- 邮箱: cindynianx@gmail.com

---

