# YOLOv11-CFruit 文档中心

欢迎来到YOLOv11-CFruit项目的文档中心！这里包含了项目的所有详细文档和指南。

## 📚 文档目录

### 🚀 快速开始
- [项目概述](../README.md) - 项目简介和快速开始
- [快速开始指南](../QUICK_START.md) - 详细的安装和使用指南
- [快速开始指南(英文)](../QUICK_START_EN.md) - English version

### 📖 使用指南
- [使用说明](../USAGE.md) - 详细的使用方法和API文档
- [数据准备指南](data_preparation.md) - 数据集准备和处理指南

### 🏗️ 技术文档
- [设计文档](../DesignDoc.md) - 系统架构和设计思路
- [设计文档(英文)](../DesignDoc_en.md) - English version

### 🐳 部署指南
- [Docker设置指南](../DOCKER_WINDOWS_SETUP.md) - Docker环境配置
- [Conda环境设置](../ANACONDA_SETUP.md) - Conda环境配置

### 📁 项目结构

```
docs/
├── README.md                    # 本文档
├── data_preparation.md          # 数据准备指南
└── api/                         # API文档（待添加）
    ├── models.md               # 模型API
    ├── training.md             # 训练API
    └── utils.md                # 工具函数API
```

## 🔍 文档导航

### 按用户类型分类

#### 👶 新手用户
1. 阅读 [项目概述](../README.md)
2. 按照 [快速开始指南](../QUICK_START.md) 安装环境
3. 运行 [基础检测示例](../examples/basic_detection.py)

#### 👨‍💻 开发者
1. 查看 [设计文档](../DesignDoc.md) 了解架构
2. 阅读 [使用说明](../USAGE.md) 了解API
3. 参考 [数据准备指南](data_preparation.md) 处理数据

#### 🚀 高级用户
1. 自定义模型配置 (`configs/model/`)
2. 优化训练参数 (`scripts/train_improved.py`)
3. 部署到生产环境

### 按功能分类

#### 🎯 模型训练
- [训练脚本](../scripts/train_improved.py) - 改进版训练脚本
- [训练配置](../configs/model/) - 模型配置文件
- [训练监控](../scripts/visualize_training.py) - 训练过程可视化

#### 🔍 模型推理
- [基础检测](../examples/basic_detection.py) - 简单推理示例
- [批量检测](../scripts/evaluate_model.py) - 批量评估脚本

#### 📊 数据处理
- [数据准备](../scripts/prepare_data.py) - 数据预处理脚本
- [数据验证](../scripts/check_data.py) - 数据质量检查
- [数据可视化](../scripts/quick_visualize.py) - 数据可视化工具

## 📝 文档贡献

如果您发现文档中的错误或有改进建议，请：

1. 提交 Issue 描述问题
2. 创建 Pull Request 修复问题
3. 参与文档讨论

## 🔗 相关链接

- [项目主页](../README.md)
- [GitHub仓库](https://github.com/your-repo/YOLOv11-CFruit)
- [问题反馈](https://github.com/your-repo/YOLOv11-CFruit/issues)
- [讨论区](https://github.com/your-repo/YOLOv11-CFruit/discussions)

---

**最后更新**: 2024年12月
**文档版本**: v1.0 