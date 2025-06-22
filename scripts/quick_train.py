#!/usr/bin/env python3
"""
快速开始训练脚本 - 一键完成数据准备和训练
"""

import os
import sys
import argparse
import yaml
import torch
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_data import main as prepare_data
from scripts.train import main as train_model


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def check_requirements():
    """检查环境要求"""
    try:
        import torch
        import cv2
        import numpy as np
        import yaml
        from tqdm import tqdm
        print("✓ 所有依赖包已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖包: {e}")
        print("请运行: pip install -r requirements.txt")
        return False


def check_data_structure(input_dir):
    """检查数据目录结构"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"✗ 输入目录不存在: {input_dir}")
        return False
    
    # 检查是否有图像文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"✗ 在 {input_dir} 中未找到图像文件")
        return False
    
    # 检查是否有对应的JSON文件
    json_files = list(input_path.glob('*.json'))
    if not json_files:
        print(f"✗ 在 {input_dir} 中未找到labelme JSON文件")
        return False
    
    # 检查匹配的图像-标注对
    valid_pairs = 0
    for img_file in image_files:
        json_file = img_file.with_suffix('.json')
        if json_file.exists():
            valid_pairs += 1
    
    print(f"✓ 找到 {len(image_files)} 个图像文件")
    print(f"✓ 找到 {len(json_files)} 个JSON文件")
    print(f"✓ 有效图像-标注对: {valid_pairs}")
    
    return valid_pairs > 0


def create_training_config(output_dir, class_names, model_type='yolov8'):
    """创建训练配置文件"""
    config = {
        'model': {
            'type': model_type,
            'backbone': {
                'type': 'cspdarknet',
                'depth_multiple': 1.0,
                'width_multiple': 1.0
            },
            'neck': {
                'type': 'panet',
                'depth_multiple': 1.0,
                'width_multiple': 1.0
            },
            'head': {
                'type': 'anchor_free',
                'num_classes': len(class_names),
                'depth_multiple': 1.0,
                'width_multiple': 1.0
            }
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'device': 'auto',
            'workers': 8,
            'optimizer': {
                'type': 'adam',
                'lr': 0.001,
                'weight_decay': 0.0005
            },
            'scheduler': {
                'type': 'cosine',
                'warmup_epochs': 3
            },
            'loss': {
                'type': 'yolov8_loss',
                'box_weight': 0.05,
                'cls_weight': 0.5,
                'dfl_weight': 1.0
            }
        },
        'save': {
            'save_dir': 'checkpoints',
            'save_interval': 10,
            'log_dir': 'logs'
        }
    }
    
    # 保存配置文件
    config_path = os.path.join(output_dir, f'{model_type}_config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config_path


def main():
    parser = argparse.ArgumentParser(description='快速开始油茶果检测训练')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='包含图像和labelme JSON文件的输入目录')
    parser.add_argument('--output-dir', type=str, default='data/cfruit',
                       help='输出目录')
    parser.add_argument('--class-names', type=str, nargs='+', default=['cfruit'],
                       help='类别名称列表')
    parser.add_argument('--model-type', type=str, default='yolov8',
                       choices=['yolov8', 'yolov11'],
                       help='模型类型')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--img-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备 (auto/cpu/0/1/...)')
    parser.add_argument('--skip-data-prep', action='store_true',
                       help='跳过数据准备步骤')
    parser.add_argument('--skip-train', action='store_true',
                       help='跳过训练步骤')
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("=== 油茶果检测快速训练 ===")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"类别: {args.class_names}")
    print(f"模型类型: {args.model_type}")
    
    # 检查环境
    if not check_requirements():
        return
    
    # 检查数据
    if not check_data_structure(args.input_dir):
        return
    
    # 步骤1: 数据准备
    if not args.skip_data_prep:
        print("\n=== 步骤1: 数据准备 ===")
        
        # 准备数据参数
        prepare_args = argparse.Namespace(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            class_names=args.class_names,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            output_yaml=f'configs/data/cfruit_{args.model_type}.yaml'
        )
        
        try:
            prepare_data()
            print("✓ 数据准备完成")
        except Exception as e:
            print(f"✗ 数据准备失败: {e}")
            return
    else:
        print("✓ 跳过数据准备步骤")
    
    # 步骤2: 创建训练配置
    print("\n=== 步骤2: 创建训练配置 ===")
    
    config_path = create_training_config(
        args.output_dir, 
        args.class_names, 
        args.model_type
    )
    print(f"✓ 训练配置文件已创建: {config_path}")
    
    # 步骤3: 开始训练
    if not args.skip_train:
        print("\n=== 步骤3: 开始训练 ===")
        
        # 训练参数
        train_args = argparse.Namespace(
            config=config_path,
            data=f'configs/data/cfruit_{args.model_type}.yaml',
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            workers=8,
            lr=0.001,
            weight_decay=0.0005,
            warmup_epochs=3,
            save_dir='checkpoints',
            log_dir='logs',
            save_interval=10,
            resume='',
            pretrained=''
        )
        
        try:
            train_model()
            print("✓ 训练完成")
        except Exception as e:
            print(f"✗ 训练失败: {e}")
            return
    else:
        print("✓ 跳过训练步骤")
    
    print("\n=== 完成 ===")
    print(f"数据集已保存到: {args.output_dir}")
    print(f"模型检查点已保存到: checkpoints/")
    print(f"训练日志已保存到: logs/")
    
    print("\n下一步:")
    print("1. 检查训练日志: tensorboard --logdir logs")
    print("2. 测试模型: python examples/basic_detection.py")
    print("3. 评估模型: python evaluation/evaluate.py")


if __name__ == '__main__':
    main() 