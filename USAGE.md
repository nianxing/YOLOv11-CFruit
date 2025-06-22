# YOLOv8-CFruit 使用说明

## 快速开始

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
.\install.ps1
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

## 项目结构

```
YOLOv8-CFruit/
├── configs/                 # 配置文件
│   ├── model/              # 模型配置
│   └── data/               # 数据集配置
├── models/                 # 模型定义
│   ├── backbone/          # 主干网络
│   ├── neck/              # 颈部网络
│   ├── head/              # 头部网络
│   └── yolov8_cfruit.py   # 主模型
├── utils/                 # 工具函数
├── scripts/              # 训练脚本
├── examples/             # 示例代码
├── requirements.txt      # 依赖包
└── README.md            # 项目说明
```

## 基本使用

### 1. 模型创建

```python
import yaml
from models.yolov8_cfruit import YOLOv8CFruit

# 加载配置
with open('configs/model/yolov8_cfruit.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建模型
model = YOLOv8CFruit(config)

# 获取模型信息
model_info = model.get_model_info()
print(f"模型参数数量: {model_info['total_params']:,}")
```

### 2. 基本推理

```python
import torch
from models.yolov8_cfruit import YOLOv8CFruit

# 加载预训练模型
model = YOLOv8CFruit.from_pretrained('checkpoints/yolov8_cfruit.pt')
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
# 基本训练
python scripts/train.py --config configs/model/yolov8_cfruit.yaml --data configs/data/cfruit.yaml

# 指定GPU训练
python scripts/train.py --device 0 --batch-size 16

# 恢复训练
python scripts/train.py --resume checkpoints/last.pt
```

### 4. 评估模型

```bash
python scripts/evaluate.py --weights checkpoints/yolov8_cfruit.pt --data configs/data/cfruit.yaml
```

## 配置文件说明

### 模型配置 (configs/model/yolov8_cfruit.yaml)

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

## 自定义配置

### 1. 修改模型架构

编辑 `configs/model/yolov8_cfruit.yaml`：

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

### 3. 修改损失权重

```yaml
training:
  loss_weights:
    cls: 0.3  # 分类损失权重
    box: 10.0  # 边界框损失权重
    dfl: 2.0  # DFL损失权重
```

## 常见问题

### 1. 内存不足

- 减少 `batch_size`
- 减少 `img_size`
- 使用梯度累积

### 2. 训练速度慢

- 使用GPU训练
- 增加 `num_workers`
- 使用混合精度训练

### 3. 模型不收敛

- 检查学习率设置
- 检查数据标注质量
- 调整损失权重

### 4. 导入错误

- 确保已安装所有依赖
- 检查Python路径设置
- 重新安装项目

## 技术支持

如有问题，请通过以下方式联系：

- 提交 [Issue](https://github.com/your-username/YOLOv8-CFruit/issues)
- 发送邮件至: cindynianx@gmail.com

## 许可证

本项目采用 MIT 许可证。 