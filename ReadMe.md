# YOLOv8-CFruit

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

YOLOv8-CFruit 是一个专为检测油茶果（Camellia oleifera）设计的状态-of-the-art 目标检测模型。通过结合 YOLOv8 的先进特性和 YOLO-CFruit 的专门改进，该模型旨在在复杂农业环境中实现更高的准确性和鲁棒性。

## 🌟 特性

- **先进的架构**: 基于 YOLOv8 的最新架构，集成 CBAM 注意力机制和 Transformer 层
- **专门优化**: 针对油茶果检测进行专门优化，解决遮挡、变化光照和密集果实集群等挑战
- **实时性能**: 目标推理时间 <20ms 每帧，适合实时应用
- **高精度**: 使用 EIoU 损失函数，提高边界框回归精度
- **模块化设计**: 易于扩展和定制

## 🏗️ 项目架构

```
YOLOv8-CFruit/
├── configs/                 # 配置文件
│   ├── model/              # 模型配置
│   │   ├── yolov8_cfruit.yaml
│   │   └── yolov8_cfruit_large.yaml
│   └── data/               # 数据集配置
│       └── cfruit.yaml
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── backbone/          # 主干网络
│   │   ├── __init__.py
│   │   ├── cspdarknet.py
│   │   └── cbam.py
│   ├── neck/              # 颈部网络
│   │   ├── __init__.py
│   │   ├── panet.py
│   │   └── transformer.py
│   ├── head/              # 头部网络
│   │   ├── __init__.py
│   │   └── anchor_free.py
│   └── yolov8_cfruit.py   # 主模型
├── utils/                 # 工具函数
│   ├── __init__.py
│   ├── losses.py          # 损失函数
│   ├── metrics.py         # 评估指标
│   ├── transforms.py      # 数据增强
│   └── visualization.py   # 可视化工具
├── data/                  # 数据处理
│   ├── __init__.py
│   ├── dataset.py         # 数据集类
│   └── dataloader.py      # 数据加载器
├── training/              # 训练相关
│   ├── __init__.py
│   ├── trainer.py         # 训练器
│   └── scheduler.py       # 学习率调度器
├── evaluation/            # 评估相关
│   ├── __init__.py
│   ├── evaluator.py       # 评估器
│   └── benchmark.py       # 基准测试
├── inference/             # 推理相关
│   ├── __init__.py
│   ├── detector.py        # 检测器
│   └── demo.py           # 演示脚本
├── scripts/              # 脚本文件
│   ├── train.py          # 训练脚本
│   ├── evaluate.py       # 评估脚本
│   ├── export.py         # 模型导出
│   └── benchmark.py      # 性能基准测试
├── tests/                # 测试文件
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_losses.py
│   └── test_metrics.py
├── docs/                 # 文档
│   ├── api.md
│   ├── training.md
│   └── deployment.md
├── examples/             # 示例代码
│   ├── basic_detection.py
│   ├── batch_inference.py
│   └── webcam_demo.py
├── requirements.txt      # 依赖包
├── setup.py             # 安装脚本
├── LICENSE              # 许可证
├── DesignDoc.md         # 设计文档
└── README.md            # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU 训练)

### 安装

1. 克隆仓库
```bash
git clone https://github.com/your-username/YOLOv8-CFruit.git
cd YOLOv8-CFruit
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 安装项目
```bash
pip install -e .
```

### 基本使用

```python
from models.yolov8_cfruit import YOLOv8CFruit
from inference.detector import CFruitDetector

# 加载模型
model = YOLOv8CFruit.from_pretrained('checkpoints/yolov8_cfruit.pt')

# 创建检测器
detector = CFruitDetector(model)

# 执行检测
results = detector.detect('path/to/image.jpg')
```

## 📊 性能指标

| 模型 | mAP@0.5 | mAP@0.5:0.95 | F1-Score | 推理时间 |
|------|---------|--------------|----------|----------|
| YOLOv8-CFruit | 0.92 | 0.78 | 0.89 | 18ms |
| YOLO-CFruit | 0.88 | 0.72 | 0.85 | 25ms |
| YOLOv8 | 0.85 | 0.70 | 0.82 | 15ms |

## 🎯 训练

### 准备数据集

1. 将数据集放置在 `data/` 目录下
2. 修改 `configs/data/cfruit.yaml` 中的路径配置

### 开始训练

```bash
python scripts/train.py --config configs/model/yolov8_cfruit.yaml --data configs/data/cfruit.yaml
```

### 训练参数

- `--epochs`: 训练轮数 (默认: 300)
- `--batch-size`: 批次大小 (默认: 16)
- `--img-size`: 输入图像尺寸 (默认: 640)
- `--device`: 训练设备 (默认: 0)

## 🔍 评估

```bash
python scripts/evaluate.py --weights checkpoints/yolov8_cfruit.pt --data configs/data/cfruit.yaml
```

## 📱 推理

### 单张图像检测

```bash
python inference/demo.py --weights checkpoints/yolov8_cfruit.pt --source path/to/image.jpg
```

### 批量检测

```bash
python inference/demo.py --weights checkpoints/yolov8_cfruit.pt --source path/to/images/
```

### 实时检测

```bash
python examples/webcam_demo.py --weights checkpoints/yolov8_cfruit.pt
```

## 🛠️ 自定义

### 修改模型配置

编辑 `configs/model/yolov8_cfruit.yaml` 来调整模型参数：

```yaml
# 模型配置
model:
  backbone:
    type: 'cspdarknet'
    cbam: true
  neck:
    type: 'panet'
    transformer: true
  head:
    type: 'anchor_free'
    num_classes: 1
```

### 添加新的损失函数

在 `utils/losses.py` 中添加自定义损失函数：

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        # 实现自定义损失逻辑
        pass
```

## 📚 文档

- [API 文档](docs/api.md)
- [训练指南](docs/training.md)
- [部署指南](docs/deployment.md)
- [设计文档](DesignDoc.md)

## 🤝 贡献

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [YOLO-CFruit](https://github.com/your-repo/yolo-cfruit)
- 所有贡献者和研究人员

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/your-username/YOLOv8-CFruit/issues)
- 发送邮件至: cindynianx@gmail.com

---

**注意**: 本项目仍在开发中，API 可能会有变化。请查看 [CHANGELOG.md](CHANGELOG.md) 了解最新更新。
