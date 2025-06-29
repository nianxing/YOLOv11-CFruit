# YOLOv11-CFruit: 基于YOLOv11的水果检测系统

[English](README_en.md) | 中文

## 📋 项目概述

YOLOv11-CFruit 是一个基于YOLOv11架构的水果检测系统，专用于识别和定位图像中的水果。项目结合了最新的YOLOv11技术，提供高效、准确的水果检测解决方案。

---

**最后更新：2024年6月**  
**文档版本：v1.0**

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+（可选，用于GPU加速）

### 安装方式

#### 方式1: 使用Docker（推荐）
```bash
# Windows
./run_docker.ps1

# Linux/Mac
docker-compose up -d
```

#### 方式2: 手动安装
```bash
# 克隆项目
git clone <repository-url>
cd YOLOv11-CFruit

# 安装依赖
pip install -r requirements.txt
```

#### 方式3: 使用Conda
```bash
# Windows
./install_conda.ps1

# Linux/Mac
./install_conda.sh
```

## 📁 项目结构

```
YOLOv11-CFruit/
├── configs/                # 配置文件
│   ├── data/               # 数据配置
│   └── model/              # 模型配置
├── data/                   # 数据处理模块
│   └── dataset.py
├── models/                 # 模型定义
│   ├── backbone/           # 主干网络
│   ├── neck/               # 颈部网络
│   ├── head/               # 头部网络
│   └── yolov11_cfruit.py
├── training/               # 训练模块
│   ├── trainer.py
│   └── scheduler.py
├── utils/                  # 工具函数
│   ├── losses.py
│   ├── simple_loss.py
│   └── transforms.py
├── scripts/                # 训练和评估脚本
│   ├── train_improved.py   # 改进版训练脚本
│   ├── simple_train.py     # 简化训练脚本
│   ├── auto_train_and_visualize.sh # 自动训练脚本
│   ├── quick_auto_train.sh # 快速测试脚本
│   ├── prepare_data_circle_fixed.py # 数据准备脚本（支持圆形标注）
│   ├── evaluate_model.py   # 模型评估
│   ├── visualize_training.py # 训练过程可视化
│   ├── check_data.py       # 数据质量检查
│   ├── quick_test.py       # 快速测试
│   ├── quick_rename_labels.py # 标签批量重命名
│   ├── rename_labels.py    # 标签重命名
│   └── ...
├── examples/               # 使用示例
│   ├── basic_detection.py
│   └── prepare_and_train.py
├── docs/                   # 详细文档
│   ├── README.md
│   └── data_preparation.md
├── tests/                  # 测试文件
├── inference/              # 推理模块
├── evaluation/             # 评估结果
├── requirements.txt        # 依赖包
├── README.md               # 项目说明
├── QUICK_START.md          # 快速开始指南
├── USAGE.md                # 使用说明
├── DesignDoc.md            # 设计文档
└── ...
```

## 🎯 主要功能

### 1. 模型架构
- **YOLOv11-CFruit**: 基于YOLOv11的改进架构
- **CSPDarknet**: 高效的主干网络
- **PANet**: 特征金字塔网络
- **Anchor-Free**: 无锚点检测头

### 2. 训练功能
- 自动混合精度训练
- 早停机制
- 学习率调度
- 阶梯累积
- 多GPU支持

### 3. 数据处理
- 自动数据增强
- 多格式支持
- 数据验证
- 可视化工具

## 🛠️ 使用方法

### 1. 数据准备
```bash
python scripts/prepare_data_circle_fixed.py --input-dir /path/to/your/data --output-dir data/cfruit --class-names cfruit
```

### 2. 模型训练
```bash
# 改进版训练（推荐）
python scripts/train_improved.py --device cuda --batch-size 8 --save-dir checkpoints

# 简化训练
python scripts/simple_train.py --device cuda --batch-size 8 --save-dir checkpoints
```

### 3. 模型评估
```bash
python scripts/evaluate_model.py --model-path checkpoints/best.pt
```

### 4. 推理检测
```python
from examples.basic_detection import detect_fruits

# 检测图像中的水果
results = detect_fruits("path/to/image.jpg")
```

## 📊 性能指标

| 模型           | mAP@0.5 | 推理速度 | 模型大小 |
|----------------|---------|----------|----------|
| YOLOv11-CFruit | 0.85+   | 30ms     | 45MB     |
| YOLOv8-CFruit  | 0.82    | 25ms     | 42MB     |

## ⚙️ 配置说明

### 模型配置
配置文件位于 `configs/model/` 目录：
- `yolov11_cfruit.yaml`：基础配置
- `yolov11_cfruit_improved.yaml`：改进配置

### 数据配置
配置文件位于 `configs/data/` 目录：
- `cfruit.yaml`：水果数据集配置

## 📚 详细文档

- [快速开始指南](QUICK_START.md)
- [使用说明](USAGE.md)
- [设计文档](DesignDoc.md)
- [数据准备指南](docs/data_preparation.md)
- [Docker设置指南](DOCKER_WINDOWS_SETUP.md)

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- YOLOv11 团队
- PyTorch 社区
- 所有贡献者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 参与讨论

---

**注意**: 本项目仍在积极开发中，API 可能会有变化。 