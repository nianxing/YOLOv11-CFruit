# YOLOv11-CFruit 使用说明

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU训练推荐)

### 2. 安装

#### Windows 用户

**方法1: 使用批处理脚本**
```cmd
install.bat
```

**方法2: 使用PowerShell脚本**
```powershell
./install.ps1
```

**方法3: 手动安装**
```cmd
# 升级pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

#### Linux/Mac 用户
```bash
# 升级pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 3. 验证安装

运行测试脚本验证项目是否正确安装：

```bash
python test_project.py
```

如果看到 "🎉 所有测试通过！" 表示安装成功。

---

**最后更新：2024年6月**  
**文档版本：v1.0**

---

## 📁 项目结构

```
YOLOv11-CFruit/
├── configs/                 # 配置文件
│   ├── model/              # 模型配置
│   └── data/               # 数据集配置
├── models/                 # 模型定义
│   ├── backbone/          # 主干网络
│   ├── neck/              # 颈部网络
│   ├── head/              # 头部网络
│   └── yolov11_cfruit.py   # 主模型
├── utils/                 # 工具函数
│   ├── losses.py          # 损失函数
│   ├── simple_loss.py     # 简化损失函数
│   └── transforms.py      # 数据变换
├── scripts/              # 训练脚本
│   ├── train_improved.py  # 改进版训练脚本
│   ├── simple_train.py    # 简化训练脚本
│   ├── prepare_data_circle_fixed.py # 数据准备脚本
│   └── ...
├── examples/             # 示例代码
├── requirements.txt      # 依赖包
└── README.md            # 项目说明
```

## 🛠️ 基本使用

### 1. 模型创建

```python
import yaml
from models.yolov11_cfruit import YOLOv11CFruit

# 加载配置
with open('configs/model/yolov11_cfruit.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建模型
model = YOLOv11CFruit(config)

# 获取模型信息
model_info = model.get_model_info()
print(f"模型参数数量: {model_info['total_params']:,}")
```

### 2. 基本推理

```python
import torch
from models.yolov11_cfruit import YOLOv11CFruit

# 加载预训练模型
model = YOLOv11CFruit.from_pretrained('checkpoints/yolov11_cfruit.pt')
model.eval()

# 准备输入
x = torch.randn(1, 3, 640, 640)

# 推理
with torch.no_grad():
    bboxes = model.inference(x, conf_thresh=0.25, nms_thresh=0.45)

print(f"检测到 {len(bboxes)} 个目标")
```

### 3. 训练模型

```bash
# 改进版训练（推荐）
python scripts/train_improved.py --config configs/model/yolov11_cfruit_improved.yaml --data configs/data/cfruit.yaml

# 简化训练
python scripts/simple_train.py --config configs/model/yolov11_cfruit.yaml --data configs/data/cfruit.yaml

# 指定GPU训练
python scripts/train_improved.py --device 0 --batch-size 16 --save-dir checkpoints

# 恢复训练
python scripts/train_improved.py --resume checkpoints/last.pt
```

### 4. 评估模型

```bash
python scripts/evaluate_model.py --model-path checkpoints/yolov11_cfruit.pt --data configs/data/cfruit.yaml
```

## ⚙️ 配置文件说明

### 模型配置 (configs/model/yolov11_cfruit.yaml)

```yaml
model:
  backbone:
    type: 'cspdarknet'
    cbam: true
    cbam_ratio: 16
  neck:
    type: 'panet'
    transformer: true
    transformer_heads: 8
    transformer_dim: 256
  head:
    type: 'anchor_free'
    num_classes: 1
    reg_max: 16

training:
  epochs: 300
  batch_size: 16
  img_size: 640
  optimizer:
    type: 'adam'
    lr: 0.001
    weight_decay: 0.0005
```

### 数据集配置 (configs/data/cfruit.yaml)

```yaml
dataset:
  train: 'data/cfruit/train/images'
  val: 'data/cfruit/val/images'
  nc: 1
  names: ['cfruit']

dataloader:
  batch_size: 16
  num_workers: 8
```

## 🔧 自定义配置

### 1. 修改模型架构

编辑 `configs/model/yolov11_cfruit.yaml`：

```yaml
model:
  backbone:
    cbam_ratio: 8  # 修改CBAM比例
  neck:
    transformer_heads: 4  # 修改Transformer头数
    transformer_dim: 128  # 修改Transformer维度
```

### 2. 修改训练参数

```yaml
training:
  epochs: 500  # 增加训练轮数
  batch_size: 32  # 增加批次大小
  optimizer:
    lr: 0.0005  # 调整学习率
```

## 📊 数据处理

### 1. 数据准备

```bash
# 支持圆形标注的数据准备
python scripts/prepare_data_circle_fixed.py \
    --input-dir /path/to/your/data \
    --output-dir data/cfruit \
    --class-names cfruit
```

### 2. 数据验证

```bash
# 检查数据质量
python scripts/check_data.py --data-dir data/cfruit
```

### 3. 标签重命名

```bash
# 批量重命名标签
python scripts/quick_rename_labels.py \
    --input-dir /path/to/json/files \
    --old-label youcha \
    --new-label cfruit
```

## 🎯 训练监控

### 1. TensorBoard

```bash
tensorboard --logdir logs
```

访问 http://localhost:6006 查看训练曲线。

### 2. 训练可视化

```bash
# 可视化训练过程
python scripts/visualize_training.py --log-dir logs
```

### 3. 训练结果展示

```bash
# 显示训练结果
python scripts/show_training_results.py --checkpoint checkpoints/best.pt
```

## 🧪 模型测试

### 1. 基础检测

```bash
python examples/basic_detection.py \
    --model checkpoints/best.pt \
    --image /path/to/test/image.jpg
```

### 2. 快速测试

```bash
python scripts/quick_test.py \
    --model checkpoints/best.pt \
    --data-dir data/cfruit/val
```

## 🔗 相关链接

- [快速开始指南](QUICK_START.md)
- [数据准备指南](docs/data_preparation.md)
- [设计文档](DesignDoc.md)
- [Docker设置指南](DOCKER_WINDOWS_SETUP.md)

## 📞 技术支持

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 参与讨论

---

**注意**: 本项目仍在积极开发中，API 可能会有变化。 