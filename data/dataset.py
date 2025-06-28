#!/usr/bin/env python3
"""
油茶果数据集类
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging


class CFruitDataset(Dataset):
    """油茶果数据集类"""
    
    def __init__(self, img_dir, label_dir, transform=None, img_size=640):
        """
        初始化数据集
        
        Args:
            img_dir: 图像目录路径
            label_dir: 标签目录路径
            transform: 数据变换
            img_size: 输入图像尺寸
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.img_size = img_size
        
        # 获取所有图像文件
        self.img_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            self.img_files.extend(self.img_dir.glob(f'*{ext}'))
            self.img_files.extend(self.img_dir.glob(f'*{ext.upper()}'))
        
        self.img_files = sorted(self.img_files)
        
        logging.info(f"找到 {len(self.img_files)} 个图像文件")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法加载图像: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 加载标签
        label_path = self.label_dir / f"{img_path.stem}.txt"
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # 转换为像素坐标
                            x_center *= w
                            y_center *= h
                            width *= w
                            height *= h
                            
                            # 转换为左上角和右下角坐标
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            
                            labels.append([class_id, x1, y1, x2, y2])
        
        # 应用变换
        if self.transform:
            img, labels = self.transform(img, labels)
        
        # 转换为张量
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # 处理标签
        if len(labels) > 0:
            labels = np.array(labels)
            # 确保标签格式正确 [class_id, x1, y1, x2, y2]
            labels = torch.from_numpy(labels).float()
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        
        return {
            'image': img,
            'labels': labels,
            'img_path': str(img_path),
            'img_size': (h, w)
        }


class CFruitDataLoader:
    """数据加载器工厂类"""
    
    @staticmethod
    def create_dataloader(img_dir, label_dir, batch_size=16, num_workers=8, 
                         transform=None, img_size=640, shuffle=True):
        """
        创建数据加载器
        
        Args:
            img_dir: 图像目录
            label_dir: 标签目录
            batch_size: 批次大小
            num_workers: 工作进程数
            transform: 数据变换
            img_size: 图像尺寸
            shuffle: 是否打乱数据
        
        Returns:
            DataLoader实例
        """
        from torch.utils.data import DataLoader
        
        dataset = CFruitDataset(img_dir, label_dir, transform, img_size)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=CFruitDataLoader.collate_fn
        )
        
        return dataloader
    
    @staticmethod
    def collate_fn(batch):
        """
        自定义批处理函数
        
        Args:
            batch: 批次数据
        
        Returns:
            处理后的批次数据
        """
        images = []
        labels = []
        img_paths = []
        img_sizes = []
        
        for sample in batch:
            images.append(sample['image'])
            labels.append(sample['labels'])
            img_paths.append(sample['img_path'])
            img_sizes.append(sample['img_size'])
        
        images = torch.stack(images, dim=0)
        
        # 处理标签 - 将列表转换为张量
        # 由于每张图像的标签数量可能不同，我们需要填充到相同长度
        max_labels = max(len(label) for label in labels)
        if max_labels > 0:
            # 创建填充后的标签张量
            padded_labels = torch.zeros(len(labels), max_labels, 5, dtype=torch.float32)
            for i, label in enumerate(labels):
                if len(label) > 0:
                    padded_labels[i, :len(label)] = label
        else:
            # 如果没有标签，创建空的张量
            padded_labels = torch.zeros(len(labels), 0, 5, dtype=torch.float32)
        
        return {
            'images': images,
            'labels': padded_labels,
            'img_paths': img_paths,
            'img_sizes': img_sizes
        }


def visualize_dataset(dataset, num_samples=5, save_dir=None):
    """
    可视化数据集样本
    
    Args:
        dataset: 数据集实例
        num_samples: 可视化样本数量
        save_dir: 保存目录
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img = sample['image'].permute(1, 2, 0).numpy()
        labels = sample['labels']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        
        # 绘制边界框
        for label in labels:
            class_id, x1, y1, x2, y2 = label
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # 添加类别标签
            ax.text(x1, y1-5, f'cfruit', color='red', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title(f'样本 {i+1}: {len(labels)} 个目标')
        ax.axis('off')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'), 
                       bbox_inches='tight', dpi=150)
        else:
            plt.show()
        
        plt.close()


def get_dataset_stats(dataset):
    """
    获取数据集统计信息
    
    Args:
        dataset: 数据集实例
    
    Returns:
        统计信息字典
    """
    total_objects = 0
    class_counts = {}
    img_sizes = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample['labels']
        img_size = sample['img_size']
        
        total_objects += len(labels)
        img_sizes.append(img_size)
        
        for label in labels:
            class_id = int(label[0])
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    # 计算图像尺寸统计
    img_sizes = np.array(img_sizes)
    width_stats = {
        'mean': np.mean(img_sizes[:, 0]),
        'std': np.std(img_sizes[:, 0]),
        'min': np.min(img_sizes[:, 0]),
        'max': np.max(img_sizes[:, 0])
    }
    height_stats = {
        'mean': np.mean(img_sizes[:, 1]),
        'std': np.std(img_sizes[:, 1]),
        'min': np.min(img_sizes[:, 1]),
        'max': np.max(img_sizes[:, 1])
    }
    
    stats = {
        'total_images': len(dataset),
        'total_objects': total_objects,
        'avg_objects_per_image': total_objects / len(dataset),
        'class_counts': class_counts,
        'width_stats': width_stats,
        'height_stats': height_stats
    }
    
    return stats


if __name__ == '__main__':
    # 测试数据集
    import yaml
    
    # 加载配置
    with open('configs/data/cfruit.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建数据集
    dataset = CFruitDataset(
        config['dataset']['train'],
        config['dataset']['train_labels'],
        img_size=config['preprocessing']['img_size']
    )
    
    # 获取统计信息
    stats = get_dataset_stats(dataset)
    print("数据集统计信息:")
    print(f"总图像数: {stats['total_images']}")
    print(f"总目标数: {stats['total_objects']}")
    print(f"平均每张图像目标数: {stats['avg_objects_per_image']:.2f}")
    print(f"类别分布: {stats['class_counts']}")
    print(f"图像宽度统计: {stats['width_stats']}")
    print(f"图像高度统计: {stats['height_stats']}")
    
    # 可视化样本
    visualize_dataset(dataset, num_samples=3, save_dir='data/visualization') 