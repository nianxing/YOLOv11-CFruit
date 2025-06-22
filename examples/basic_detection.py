#!/usr/bin/env python3
"""
YOLOv8-CFruit 基本检测示例
"""

import os
import sys
import torch
import cv2
import numpy as np
import yaml
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov8_cfruit import YOLOv8CFruit


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_image(image_path, img_size=640):
    """预处理图像"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整大小
    h, w = image.shape[:2]
    scale = min(img_size / w, img_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # 填充到正方形
    padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # 转换为张量
    tensor = torch.from_numpy(padded).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0)  # 添加批次维度
    
    return tensor, image, (scale, (new_w, new_h))


def postprocess_bboxes(bboxes, scale, pad_size, orig_size):
    """后处理边界框"""
    if len(bboxes) == 0:
        return []
    
    # 缩放回原始尺寸
    bboxes[:, :4] /= scale
    
    # 裁剪到原始图像范围
    orig_w, orig_h = orig_size
    bboxes[:, 0] = torch.clamp(bboxes[:, 0], 0, orig_w)
    bboxes[:, 1] = torch.clamp(bboxes[:, 1], 0, orig_h)
    bboxes[:, 2] = torch.clamp(bboxes[:, 2], 0, orig_w)
    bboxes[:, 3] = torch.clamp(bboxes[:, 3], 0, orig_h)
    
    return bboxes


def draw_bboxes(image, bboxes, class_names=['cfruit']):
    """绘制边界框"""
    image = image.copy()
    
    for bbox in bboxes:
        x1, y1, x2, y2, conf, cls_id = bbox
        
        # 转换为整数坐标
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"{class_names[cls_id]}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image


def main():
    """主函数"""
    # 配置
    config_path = 'configs/model/yolov8_cfruit.yaml'
    model_path = 'checkpoints/yolov8_cfruit.pt'  # 需要训练好的模型
    image_path = 'data/sample.jpg'  # 测试图像路径
    output_path = 'output.jpg'
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型或下载预训练权重")
        return
    
    if not os.path.exists(image_path):
        print(f"测试图像不存在: {image_path}")
        return
    
    # 加载配置
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = YOLOv8CFruit.from_pretrained(model_path, config)
    model.to(device)
    model.eval()
    
    # 预处理图像
    print(f"处理图像: {image_path}")
    tensor, orig_image, (scale, pad_size) = preprocess_image(image_path)
    tensor = tensor.to(device)
    
    # 推理
    print("执行推理...")
    with torch.no_grad():
        bboxes = model.inference(tensor, conf_thresh=0.25, nms_thresh=0.45)
    
    # 后处理
    orig_h, orig_w = orig_image.shape[:2]
    bboxes = postprocess_bboxes(bboxes, scale, pad_size, (orig_w, orig_h))
    
    # 绘制结果
    print(f"检测到 {len(bboxes)} 个目标")
    result_image = draw_bboxes(orig_image, bboxes)
    
    # 保存结果
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存到: {output_path}")
    
    # 显示检测结果
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, conf, cls_id = bbox
        print(f"目标 {i+1}: 位置({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
              f"置信度: {conf:.3f}, 类别: {cls_id}")


if __name__ == '__main__':
    main() 