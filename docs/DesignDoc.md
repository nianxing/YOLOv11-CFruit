# YOLOv11-CFruit 设计文档（DesignDoc）

## 1. 项目简介

YOLOv11-CFruit 是一个基于 YOLOv11 架构的油茶果（Camellia oleifera）目标检测系统，针对油茶果果实的自动检测与定位任务进行了专门优化。系统集成了多种改进策略，包括骨干网络增强、数据增强、损失函数优化、训练流程自动化等，适用于学术研究和实际生产场景。

## 2. 系统架构

系统整体架构如下：

- **数据准备**：支持Labelme格式标注，自动转换为YOLO格式，数据增强（Mosaic、Mixup、旋转、剪切等）。
- **模型结构**：
  - Backbone: CSPDarknet + CBAM注意力机制
  - Neck: PANet + FPN
  - Head: Anchor-Free 检测头
- **训练流程**：
  - 支持多GPU训练与混合精度
  - 梯度累积与梯度裁剪
  - 余弦退火学习率调度
  - 早停机制与自动保存最佳模型
- **评估与推理**：支持mAP、Precision、Recall等指标评估，支持批量与单张推理

## 3. 主要模块说明

### 3.1 数据模块（data/）
- `dataset.py`：实现CFruitDataset类，支持图像与标签的加载、预处理、数据增强。

### 3.2 模型模块（models/）
- `yolov11_cfruit.py`：主模型定义，集成CSPDarknet、PANet、Anchor-Free Head。
- `backbone/`、`neck/`、`head/`：分别实现主干、颈部、检测头结构。

### 3.3 工具模块（utils/）
- `losses.py`：实现EIoU、Focal Loss、DFL等损失函数。
- `transforms.py`：实现多种数据增强方法。

### 3.4 训练与评估脚本（scripts/）
- `train_improved_v2.py`：主训练脚本，支持多GPU、混合精度、早停、自动保存。
- `check_data.py`：数据检查与示例数据生成。
- `evaluate_model.py`：模型评估与结果可视化。

## 4. 改进点与创新

- **骨干网络增强**：引入CBAM注意力机制提升特征表达能力。
- **Anchor-Free检测头**：提升小目标检测能力，简化正负样本分配。
- **损失函数优化**：采用EIoU、Focal Loss、DFL等组合，提升收敛速度与鲁棒性。
- **数据增强**：集成Mosaic、Mixup、Copy-Paste等高级增强策略。
- **训练流程优化**：支持混合精度、梯度累积、自动早停、余弦退火调度。
- **高效推理**：支持批量推理与高效NMS。

## 5. 实验流程

1. **数据准备**：
   - 标注数据（Labelme格式）
   - 使用 `scripts/check_data.py` 检查并生成数据集
2. **模型训练**：
   - 配置 `configs/model/yolov11_cfruit_improved.yaml` 和 `configs/data/cfruit.yaml`
   - 运行 `scripts/train_improved_v2.py` 进行训练
3. **模型评估**：
   - 使用 `scripts/evaluate_model.py` 评估模型性能
   - 记录mAP、Precision、Recall等指标
4. **推理与可视化**：
   - 使用训练好的模型进行单张或批量推理
   - 可视化检测结果

## 6. 主要配置说明

- `configs/model/yolov11_cfruit_improved.yaml`：模型结构与训练超参数
- `configs/data/cfruit.yaml`：数据集路径与类别信息

## 7. Reference

1. Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934.
2. Wang, C. Y., Bochkovskiy, A., & Liao, H. Y. M. (2023). YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696.
3. Liu, S., Qi, L., Qin, H., Shi, J., & Jia, J. (2018). Path Aggregation Network for Instance Segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
4. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. In Proceedings of the European Conference on Computer Vision (ECCV).
5. Zhang, S., Chi, C., Yao, Y., Lei, Z., & Li, S. Z. (2020). Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
6. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).
7. Ge, Z., Liu, S., Wang, F., Li, Z., & Sun, J. (2021). YOLOX: Exceeding YOLO Series in 2021. arXiv preprint arXiv:2107.08430.
8. Glenn Jocher et al. (2023). YOLO by Ultralytics. https://github.com/ultralytics/yolov5

---

如需引用本项目，请注明：YOLOv11-CFruit: Camellia oleifera Fruit Detection System, 2025. 