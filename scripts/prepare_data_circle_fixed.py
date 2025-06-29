#!/usr/bin/env python3
"""
数据准备脚本 - 支持圆形标注的修复版本
将labelme标注的JSON文件转换为YOLO格式，支持circle、polygon、rectangle、point等形状
"""

import os
import json
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split


class LabelmeToYOLO:
    """将labelme格式转换为YOLO格式"""
    
    def __init__(self, class_names):
        """
        初始化转换器
        
        Args:
            class_names: 类别名称列表，如 ['cfruit']
        """
        self.class_names = class_names
        self.class_to_id = {name: i for i, name in enumerate(class_names)}
        print(f"类别映射: {self.class_to_id}")
    
    def convert_annotation(self, json_path, img_path, output_dir):
        """
        转换单个标注文件
        
        Args:
            json_path: labelme JSON文件路径
            img_path: 对应图像文件路径
            output_dir: 输出目录
        """
        try:
            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取图像尺寸
            img_height = data['imageHeight']
            img_width = data['imageWidth']
            
            # 创建标签文件
            label_filename = Path(json_path).stem + '.txt'
            label_path = os.path.join(output_dir, label_filename)
            
            annotations_written = 0
            
            with open(label_path, 'w') as f:
                for shape in data['shapes']:
                    label = shape['label']
                    
                    # 检查标签是否在类别列表中
                    if label not in self.class_names:
                        print(f"警告: 未知类别 '{label}' 在文件 {json_path}")
                        continue
                    
                    class_id = self.class_to_id[label]
                    points = shape['points']
                    shape_type = shape['shape_type']
                    
                    print(f"处理标注: 类别={label}({class_id}), 类型={shape_type}, 点数={len(points)}")
                    
                    # 转换多边形或矩形为边界框
                    if shape_type in ['polygon', 'rectangle']:
                        # 计算边界框
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        # 转换为YOLO格式 (中心点坐标和宽高，归一化)
                        x_center = (x_min + x_max) / 2.0 / img_width
                        y_center = (y_min + y_max) / 2.0 / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        
                        # 确保坐标在[0, 1]范围内
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        # 写入标签文件
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        annotations_written += 1
                        print(f"  写入标注: {class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    
                    elif shape_type == 'circle':
                        # 处理圆形标注
                        if len(points) >= 2:
                            # 圆心和半径点
                            center_x, center_y = points[0]
                            radius_x, radius_y = points[1]
                            
                            # 计算半径
                            radius = ((radius_x - center_x) ** 2 + (radius_y - center_y) ** 2) ** 0.5
                            
                            # 转换为边界框
                            x_min = max(0, center_x - radius)
                            y_min = max(0, center_y - radius)
                            x_max = min(img_width, center_x + radius)
                            y_max = min(img_height, center_y + radius)
                            
                            # 转换为YOLO格式
                            x_center = (x_min + x_max) / 2.0 / img_width
                            y_center = (y_min + y_max) / 2.0 / img_height
                            width = (x_max - x_min) / img_width
                            height = (y_max - y_min) / img_height
                            
                            # 确保坐标在[0, 1]范围内
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))
                            
                            # 写入标签文件
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            annotations_written += 1
                            print(f"  写入圆形标注: {class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                        else:
                            print(f"  警告: 圆形标注点数不足，需要至少2个点（圆心和半径点）")
                    
                    elif shape_type == 'point':
                        # 处理点标注（转换为小矩形）
                        point = points[0]
                        x, y = point[0], point[1]
                        
                        # 创建小矩形（例如10x10像素）
                        size = 10
                        x_min = max(0, x - size/2)
                        y_min = max(0, y - size/2)
                        x_max = min(img_width, x + size/2)
                        y_max = min(img_height, y + size/2)
                        
                        # 转换为YOLO格式
                        x_center = (x_min + x_max) / 2.0 / img_width
                        y_center = (y_min + y_max) / 2.0 / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        annotations_written += 1
                        print(f"  写入点标注: {class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    
                    else:
                        print(f"警告: 不支持的形状类型 '{shape_type}' 在文件 {json_path}")
            
            print(f"文件 {json_path} 处理完成，写入 {annotations_written} 个标注")
            return annotations_written > 0
            
        except Exception as e:
            print(f"处理文件 {json_path} 时出错: {e}")
            return False


def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    分割数据集为训练、验证和测试集
    
    Args:
        data_dir: 包含图像和JSON文件的目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    # 收集所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(data_dir).glob(f'*{ext}'))
        image_files.extend(Path(data_dir).glob(f'*{ext.upper()}'))
    
    # 过滤出有对应JSON文件的图像
    valid_files = []
    for img_path in image_files:
        json_path = img_path.with_suffix('.json')
        if json_path.exists():
            valid_files.append((img_path, json_path))
    
    print(f"找到 {len(valid_files)} 个有效的图像-标注对")
    
    if len(valid_files) == 0:
        print("错误: 没有找到有效的图像-标注对")
        return
    
    # 随机打乱并分割
    random.shuffle(valid_files)
    
    n_total = len(valid_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:n_train + n_val]
    test_files = valid_files[n_train + n_val:]
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    print(f"测试集: {len(test_files)} 个文件")
    
    # 复制文件到对应目录
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        print(f"\n处理 {split_name} 集...")
        for img_path, json_path in tqdm(files):
            # 复制图像
            dst_img = os.path.join(output_dir, split_name, 'images', img_path.name)
            shutil.copy2(img_path, dst_img)
            
            # 复制JSON文件（用于调试）
            dst_json = os.path.join(output_dir, split_name, 'labels', json_path.name)
            shutil.copy2(json_path, dst_json)


def convert_labels(data_dir, output_dir, class_names):
    """
    转换所有标签文件
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    converter = LabelmeToYOLO(class_names)
    total_converted = 0
    total_files = 0
    
    for split in ['train', 'val', 'test']:
        labels_dir = os.path.join(data_dir, split, 'labels')
        if not os.path.exists(labels_dir):
            continue
            
        print(f"\n转换 {split} 集标签...")
        
        # 处理JSON文件
        json_files = list(Path(labels_dir).glob('*.json'))
        split_converted = 0
        
        for json_path in tqdm(json_files):
            # 查找对应的图像文件
            img_name = json_path.stem
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            img_path = None
            
            for ext in img_extensions:
                potential_img = json_path.parent.parent / 'images' / f"{img_name}{ext}"
                if potential_img.exists():
                    img_path = potential_img
                    break
                potential_img = json_path.parent.parent / 'images' / f"{img_name}{ext.upper()}"
                if potential_img.exists():
                    img_path = potential_img
                    break
            
            if img_path is None:
                print(f"警告: 找不到图像文件 {img_name}")
                continue
            
            # 转换标注
            if converter.convert_annotation(json_path, img_path, labels_dir):
                split_converted += 1
            
            total_files += 1
        
        print(f"{split} 集: 转换了 {split_converted}/{len(json_files)} 个文件")
        total_converted += split_converted
    
    print(f"\n总计: 转换了 {total_converted}/{total_files} 个文件")
    
    # 删除JSON文件（可选）
    # for split in ['train', 'val', 'test']:
    #     labels_dir = os.path.join(data_dir, split, 'labels')
    #     if os.path.exists(labels_dir):
    #         json_files = list(Path(labels_dir).glob('*.json'))
    #         for json_file in json_files:
    #             json_file.unlink()


def create_dataset_yaml(output_dir, class_names, output_yaml_path):
    """
    创建数据集配置文件
    
    Args:
        output_dir: 数据目录
        class_names: 类别名称列表
        output_yaml_path: 输出YAML文件路径
    """
    # 统计文件数量
    train_count = len(list((Path(output_dir) / 'train' / 'images').glob('*')))
    val_count = len(list((Path(output_dir) / 'val' / 'images').glob('*')))
    test_count = len(list((Path(output_dir) / 'test' / 'images').glob('*')))
    
    # 统计标签文件数量
    train_labels = len(list((Path(output_dir) / 'train' / 'labels').glob('*.txt')))
    val_labels = len(list((Path(output_dir) / 'val' / 'labels').glob('*.txt')))
    test_labels = len(list((Path(output_dir) / 'test' / 'labels').glob('*.txt')))
    
    config = {
        'dataset': {
            'train': str(Path(output_dir) / 'train' / 'images'),
            'val': str(Path(output_dir) / 'val' / 'images'),
            'test': str(Path(output_dir) / 'test' / 'images'),
            'train_labels': str(Path(output_dir) / 'train' / 'labels'),
            'val_labels': str(Path(output_dir) / 'val' / 'labels'),
            'test_labels': str(Path(output_dir) / 'test' / 'labels'),
            'nc': len(class_names),
            'names': class_names,
            'total_images': train_count + val_count + test_count,
            'train_images': train_count,
            'val_images': val_count,
            'test_images': test_count,
            'train_labels_count': train_labels,
            'val_labels_count': val_labels,
            'test_labels_count': test_labels
        },
        'dataloader': {
            'batch_size': 16,
            'num_workers': 8,
            'pin_memory': True,
            'shuffle': True
        },
        'preprocessing': {
            'img_size': 640,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225]
        },
        'augmentation': {
            'train': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0
            },
            'val': {
                'hsv_h': 0.0,
                'hsv_s': 1.0,
                'hsv_v': 1.0,
                'degrees': 0.0,
                'translate': 0.0,
                'scale': 1.0,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.0,
                'mosaic': 0.0,
                'mixup': 0.0,
                'copy_paste': 0.0
            }
        }
    }
    
    # 写入YAML文件
    import yaml
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"数据集配置文件已保存到: {output_yaml_path}")
    print(f"图像统计: 训练={train_count}, 验证={val_count}, 测试={test_count}")
    print(f"标签统计: 训练={train_labels}, 验证={val_labels}, 测试={test_labels}")


def main():
    parser = argparse.ArgumentParser(description='准备油茶果数据集（支持圆形标注）')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='包含图像和labelme JSON文件的输入目录')
    parser.add_argument('--output-dir', type=str, default='data/cfruit',
                       help='输出目录')
    parser.add_argument('--class-names', type=str, nargs='+', default=['cfruit'],
                       help='类别名称列表')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='测试集比例')
    parser.add_argument('--output-yaml', type=str, default='configs/data/cfruit.yaml',
                       help='输出数据集配置文件路径')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式，显示详细信息')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== 油茶果数据集准备（支持圆形标注） ===")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"类别: {args.class_names}")
    print(f"调试模式: {args.debug}")
    print("支持的形状类型: polygon, rectangle, circle, point")
    
    # 步骤1: 分割数据集
    print("\n步骤1: 分割数据集...")
    split_dataset(args.input_dir, args.output_dir, 
                 args.train_ratio, args.val_ratio, args.test_ratio)
    
    # 步骤2: 转换标签格式
    print("\n步骤2: 转换标签格式...")
    convert_labels(args.output_dir, args.output_dir, args.class_names)
    
    # 步骤3: 创建配置文件
    print("\n步骤3: 创建数据集配置文件...")
    create_dataset_yaml(args.output_dir, args.class_names, args.output_yaml)
    
    print("\n=== 数据集准备完成 ===")
    print(f"数据集已保存到: {args.output_dir}")
    print(f"配置文件已保存到: {args.output_yaml}")
    print("\n下一步:")
    print("1. 检查数据集质量: python scripts/check_data.py --data-dir data/cfruit")
    print("2. 可视化标注: python scripts/quick_visualize.py --data-dir data/cfruit --split train --num-samples 5")
    print("3. 运行训练脚本: python scripts/train.py")


if __name__ == '__main__':
    main()