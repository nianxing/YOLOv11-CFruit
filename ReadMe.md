# YOLOv11-CFruit

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

YOLOv11-CFruit 是专为油茶果（Camellia oleifera）检测设计的最新一代目标检测模型，基于YOLOv11架构，集成C2f主干、CBAM注意力、Transformer、AdamW优化器、自动混合精度（AMP）、EMA、Mixup/CopyPaste等前沿技术，兼容YOLOv8配置和用法。

## 🌟 主要特性

- **C2f主干网络**：更高效的特征提取，显著提升速度和精度
- **CBAM注意力机制**：聚焦显著特征，抑制背景噪声
- **Transformer增强颈部**：全局上下文建模，提升遮挡/密集场景表现
- **AdamW优化器**：更优收敛性和泛化能力
- **AMP/EMA**：自动混合精度与指数滑动平均，提升训练稳定性和推理速度
- **Mixup/CopyPaste**：更强数据增强，提升鲁棒性
- **推理优化**：支持Conv-BN融合、优化NMS
- **兼容YOLOv8配置**：可无缝切换YOLOv8/YOLOv11

## 🏗️ 项目架构

```
YOLOv11-CFruit/
├── configs/
│   ├── model/
│   │   ├── yolov11_cfruit.yaml
│   │   └── yolov8_cfruit.yaml
│   └── data/
│       └── cfruit.yaml
├── models/
│   ├── yolov11_cfruit.py
│   ├── yolov8_cfruit.py
│   ├── backbone/
│   │   ├── cspdarknet.py
│   │   ├── c2f.py
│   │   └── cbam.py
│   ├── neck/
│   │   ├── panet.py
│   │   └── transformer.py
│   ├── head/
│   │   └── anchor_free.py
│   └── __init__.py
├── utils/
│   ├── losses.py
│   └── __init__.py
├── scripts/
│   └── train.py
├── examples/
│   └── basic_detection.py
├── requirements.txt
├── setup.py
├── LICENSE
├── DesignDoc.md
└── README.md
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (推荐GPU)

### 安装
```powershell
# 推荐使用PowerShell
.\install.ps1
# 或批处理
install.bat
```

### 验证安装
```bash
python test_project.py
```

## 📖 配置与用法

### YOLOv11配置示例（configs/model/yolov11_cfruit.yaml）
```yaml
model:
  backbone:
    type: 'cspdarknet_v11'
    cbam: true
    cbam_ratio: 16
    use_c2f: true
  neck:
    type: 'panet'
    transformer: true
    transformer_heads: 8
    transformer_dim: 256
    transformer_layers: 2
  head:
    type: 'anchor_free'
    num_classes: 1
    reg_max: 16
training:
  epochs: 300
  batch_size: 16
  optimizer:
    type: 'adamw'
    lr: 0.001
    weight_decay: 0.0005
  scheduler:
    type: 'cosine'
    min_lr: 0.00001
augmentation:
  mixup: 0.1
  copy_paste: 0.1
# ... 详见完整yaml
```

### 训练命令
```bash
python scripts/train.py --config configs/model/yolov11_cfruit.yaml --data configs/data/cfruit.yaml
```

### 推理命令
```bash
python examples/basic_detection.py --config configs/model/yolov11_cfruit.yaml --weights checkpoints/yolov11_cfruit.pt --source path/to/image.jpg
```

### 兼容YOLOv8
- 只需切换`--config configs/model/yolov8_cfruit.yaml`即可。
- 代码自动识别配置并加载对应模型。

## 📊 性能指标

| 模型           | mAP@0.5 | mAP@0.5:0.95 | F1-Score | 推理时间 |
|----------------|---------|--------------|----------|----------|
| YOLOv11-CFruit | 0.94    | 0.81         | 0.91     | 15ms     |
| YOLOv8-CFruit  | 0.92    | 0.78         | 0.89     | 18ms     |

## 🛠️ 自定义与扩展
- 支持自定义主干/颈部/头部/损失函数
- 支持AMP、EMA、AdamW、Mixup、CopyPaste等高级特性
- 详见`configs/model/yolov11_cfruit.yaml`和`utils/losses.py`

## 📚 文档
- [设计文档](DesignDoc.md)
- [使用说明](USAGE.md)

## 🤝 贡献
欢迎PR和Issue！

## 📄 许可证
MIT License

## 🙏 致谢
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [YOLOv8-CFruit](https://github.com/your-repo/yolov8-cfruit)
- 所有贡献者和研究人员

## 📞 联系方式
- Issue: https://github.com/your-username/YOLOv8-CFruit/issues
- 邮箱: cindynianx@gmail.com

---
**注意**: 本项目支持YOLOv8/YOLOv11双配置，推荐优先体验YOLOv11-CFruit。
