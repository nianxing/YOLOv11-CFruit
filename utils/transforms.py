#!/usr/bin/env python3
"""
数据变换模块 - 包含数据增强和预处理功能
"""

import cv2
import numpy as np
import torch
import random
import math
from typing import List, Tuple, Dict, Any


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, bboxes=None):
        for transform in self.transforms:
            image, bboxes = transform(image, bboxes)
        return image, bboxes


class Resize:
    """调整图像尺寸"""
    
    def __init__(self, size):
        # 处理size参数，支持单个整数或元组
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
    
    def __call__(self, image, bboxes=None):
        h, w = image.shape[:2]
        new_h, new_w = self.size
        
        # 计算缩放比例
        scale_x = new_w / w
        scale_y = new_h / h
        
        # 调整图像
        image = cv2.resize(image, (new_w, new_h))
        
        # 调整边界框
        if bboxes is not None and len(bboxes) > 0:
            bboxes = np.array(bboxes)
            bboxes[:, 1::2] *= scale_x  # x坐标
            bboxes[:, 2::2] *= scale_y  # y坐标
        
        return image, bboxes


class Normalize:
    """标准化图像"""
    
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def __call__(self, image, bboxes=None):
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image, bboxes


class RandomHorizontalFlip:
    """随机水平翻转"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, bboxes=None):
        if random.random() < self.p:
            h, w = image.shape[:2]
            image = cv2.flip(image, 1)
            
            if bboxes is not None and len(bboxes) > 0:
                bboxes = np.array(bboxes)
                bboxes[:, 1::2] = w - bboxes[:, 1::2]  # 翻转x坐标
        
        return image, bboxes


class RandomVerticalFlip:
    """随机垂直翻转"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, bboxes=None):
        if random.random() < self.p:
            h, w = image.shape[:2]
            image = cv2.flip(image, 0)
            
            if bboxes is not None and len(bboxes) > 0:
                bboxes = np.array(bboxes)
                bboxes[:, 2::2] = h - bboxes[:, 2::2]  # 翻转y坐标
        
        return image, bboxes


class RandomRotation:
    """随机旋转"""
    
    def __init__(self, degrees=10):
        self.degrees = degrees
    
    def __call__(self, image, bboxes=None):
        if random.random() < 0.5:
            angle = random.uniform(-self.degrees, self.degrees)
            h, w = image.shape[:2]
            
            # 计算旋转矩阵
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 旋转图像
            image = cv2.warpAffine(image, matrix, (w, h))
            
            # 旋转边界框
            if bboxes is not None and len(bboxes) > 0:
                bboxes = np.array(bboxes)
                new_bboxes = []
                
                for bbox in bboxes:
                    class_id, x1, y1, x2, y2 = bbox
                    
                    # 转换边界框四个角点
                    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                    points = points.reshape(-1, 1, 2)
                    
                    # 旋转点
                    rotated_points = cv2.transform(points, matrix)
                    rotated_points = rotated_points.reshape(-1, 2)
                    
                    # 计算新的边界框
                    x_min = np.min(rotated_points[:, 0])
                    y_min = np.min(rotated_points[:, 1])
                    x_max = np.max(rotated_points[:, 0])
                    y_max = np.max(rotated_points[:, 1])
                    
                    # 确保边界框在图像范围内
                    x_min = max(0, min(w, x_min))
                    y_min = max(0, min(h, y_min))
                    x_max = max(0, min(w, x_max))
                    y_max = max(0, min(h, y_max))
                    
                    if x_max > x_min and y_max > y_min:
                        new_bboxes.append([class_id, x_min, y_min, x_max, y_max])
                
                bboxes = np.array(new_bboxes) if new_bboxes else np.zeros((0, 5))
        
        return image, bboxes


class RandomScale:
    """随机缩放"""
    
    def __init__(self, scale_range=(0.5, 1.5)):
        self.scale_range = scale_range
    
    def __call__(self, image, bboxes=None):
        if random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            h, w = image.shape[:2]
            
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # 缩放图像
            image = cv2.resize(image, (new_w, new_h))
            
            # 缩放边界框
            if bboxes is not None and len(bboxes) > 0:
                bboxes = np.array(bboxes)
                bboxes[:, 1:] *= scale
        
        return image, bboxes


class ColorJitter:
    """颜色抖动"""
    
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, image, bboxes=None):
        # 转换为HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 亮度调整
        if random.random() < 0.5:
            hsv[:, :, 2] *= random.uniform(1 - self.brightness, 1 + self.brightness)
        
        # 饱和度调整
        if random.random() < 0.5:
            hsv[:, :, 1] *= random.uniform(1 - self.saturation, 1 + self.saturation)
        
        # 色调调整
        if random.random() < 0.5:
            hsv[:, :, 0] += random.uniform(-self.hue * 180, self.hue * 180)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
        
        # 转换回RGB
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return image, bboxes


class Mosaic:
    """马赛克增强"""
    
    def __init__(self, p=0.5, size=640):
        self.p = p
        self.size = size
    
    def __call__(self, image, bboxes=None):
        if random.random() < self.p:
            # 这里简化实现，实际应该从数据集中随机选择其他图像
            # 完整的马赛克增强需要访问整个数据集
            pass
        
        return image, bboxes


class Mixup:
    """Mixup增强"""
    
    def __init__(self, p=0.5, alpha=0.2):
        self.p = p
        self.alpha = alpha
    
    def __call__(self, image, bboxes=None):
        if random.random() < self.p:
            # 这里简化实现，实际需要另一张图像
            # 完整的Mixup需要访问数据集
            pass
        
        return image, bboxes


def get_transforms(img_size=640, is_training=True, **kwargs):
    """
    获取数据变换
    
    Args:
        img_size: 图像尺寸
        is_training: 是否为训练模式
        **kwargs: 其他参数
    
    Returns:
        变换函数
    """
    transforms = []
    
    if is_training:
        # 训练时增强
        if kwargs.get('fliplr', 0) > 0:
            transforms.append(RandomHorizontalFlip(p=kwargs['fliplr']))
        
        if kwargs.get('flipud', 0) > 0:
            transforms.append(RandomVerticalFlip(p=kwargs['flipud']))
        
        if kwargs.get('degrees', 0) > 0:
            transforms.append(RandomRotation(degrees=kwargs['degrees']))
        
        if kwargs.get('scale', 1.0) != 1.0:
            scale_range = (1 - kwargs['scale'], 1 + kwargs['scale'])
            transforms.append(RandomScale(scale_range=scale_range))
        
        # 颜色增强
        if any([kwargs.get('hsv_h', 0), kwargs.get('hsv_s', 0), kwargs.get('hsv_v', 0)]):
            transforms.append(ColorJitter(
                hue=kwargs.get('hsv_h', 0),
                saturation=kwargs.get('hsv_s', 0),
                brightness=kwargs.get('hsv_v', 0)
            ))
    
    # 调整尺寸
    transforms.append(Resize((img_size, img_size)))
    
    # 标准化
    transforms.append(Normalize())
    
    return Compose(transforms)


def test_transforms():
    """测试变换"""
    import matplotlib.pyplot as plt
    
    # 创建测试图像
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bboxes = np.array([[0, 100, 100, 200, 200], [0, 300, 300, 400, 400]])
    
    # 创建变换
    transform = get_transforms(img_size=640, is_training=True, 
                             fliplr=0.5, degrees=10, scale=0.2)
    
    # 应用变换
    transformed_image, transformed_bboxes = transform(image, bboxes)
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原图
    ax1.imshow(image)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[1:]
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
    ax1.set_title('原图')
    ax1.axis('off')
    
    # 变换后
    ax2.imshow(transformed_image)
    for bbox in transformed_bboxes:
        x1, y1, x2, y2 = bbox[1:]
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
    ax2.set_title('变换后')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_transforms() 