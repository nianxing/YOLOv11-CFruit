#!/usr/bin/env python3
"""
油茶果检测数据准备和训练示例
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path

def create_sample_data():
    """创建示例数据用于演示"""
    print("创建示例数据...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"示例数据目录: {temp_dir}")
    
    # 创建示例图像和标注
    for i in range(5):
        # 创建简单的测试图像
        from PIL import Image
        import numpy as np
        
        # 创建随机图像
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = os.path.join(temp_dir, f'sample_{i:03d}.jpg')
        img.save(img_path)
        
        # 创建对应的JSON标注
        import json
        json_data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [
                {
                    "label": "cfruit",
                    "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
            ],
            "imagePath": f"sample_{i:03d}.jpg",
            "imageData": None,
            "imageHeight": 480,
            "imageWidth": 640
        }
        
        json_path = os.path.join(temp_dir, f'sample_{i:03d}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
    
    return temp_dir

def main():
    parser = argparse.ArgumentParser(description='油茶果检测数据准备和训练示例')
    parser.add_argument('--input-dir', type=str, 
                       help='包含图像和labelme JSON文件的输入目录')
    parser.add_argument('--output-dir', type=str, default='data/cfruit',
                       help='输出目录')
    parser.add_argument('--class-names', type=str, nargs='+', default=['cfruit'],
                       help='类别名称列表')
    parser.add_argument('--model-type', type=str, default='yolov8',
                       choices=['yolov8', 'yolov11'],
                       help='模型类型')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数（示例用较少轮数）')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='批次大小')
    parser.add_argument('--create-sample', action='store_true',
                       help='创建示例数据用于演示')
    
    args = parser.parse_args()
    
    print("=== 油茶果检测数据准备和训练示例 ===")
    
    # 如果没有提供输入目录，创建示例数据
    if not args.input_dir:
        if args.create_sample:
            args.input_dir = create_sample_data()
            print(f"使用示例数据: {args.input_dir}")
        else:
            print("请提供 --input-dir 参数或使用 --create-sample 创建示例数据")
            return
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 检查是否有图像和JSON文件
    image_files = []
    json_files = []
    
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        image_files.extend(Path(args.input_dir).glob(f'*{ext}'))
        image_files.extend(Path(args.input_dir).glob(f'*{ext.upper()}'))
    
    json_files = list(Path(args.input_dir).glob('*.json'))
    
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"找到 {len(json_files)} 个JSON文件")
    
    if len(image_files) == 0 or len(json_files) == 0:
        print("错误: 输入目录中缺少图像文件或JSON文件")
        return
    
    # 步骤1: 数据准备
    print("\n=== 步骤1: 数据准备 ===")
    
    try:
        # 导入数据准备模块
        sys.path.append('scripts')
        from prepare_data import main as prepare_data_main
        
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
        
        # 调用数据准备函数
        prepare_data_main()
        print("✓ 数据准备完成")
        
    except ImportError as e:
        print(f"✗ 导入数据准备模块失败: {e}")
        print("请确保已安装所有依赖包: pip install -r requirements.txt")
        return
    except Exception as e:
        print(f"✗ 数据准备失败: {e}")
        return
    
    # 步骤2: 检查数据集
    print("\n=== 步骤2: 检查数据集 ===")
    
    dataset_dir = Path(args.output_dir)
    for split in ['train', 'val', 'test']:
        img_dir = dataset_dir / split / 'images'
        label_dir = dataset_dir / split / 'labels'
        
        if img_dir.exists() and label_dir.exists():
            img_count = len(list(img_dir.glob('*')))
            label_count = len(list(label_dir.glob('*.txt')))
            print(f"{split} 集: {img_count} 图像, {label_count} 标签")
        else:
            print(f"警告: {split} 集目录不完整")
    
    # 步骤3: 训练配置
    print("\n=== 步骤3: 训练配置 ===")
    
    config_content = f"""
# 油茶果检测训练配置
model:
  type: {args.model_type}
  backbone:
    type: cspdarknet
    depth_multiple: 1.0
    width_multiple: 1.0
  neck:
    type: panet
    depth_multiple: 1.0
    width_multiple: 1.0
  head:
    type: anchor_free
    num_classes: {len(args.class_names)}
    depth_multiple: 1.0
    width_multiple: 1.0

training:
  epochs: {args.epochs}
  batch_size: {args.batch_size}
  img_size: 640
  device: auto
  workers: 2
  optimizer:
    type: adam
    lr: 0.001
    weight_decay: 0.0005
  scheduler:
    type: cosine
    warmup_epochs: 1
  loss:
    type: yolov8_loss
    box_weight: 0.05
    cls_weight: 0.5
    dfl_weight: 1.0

save:
  save_dir: checkpoints
  save_interval: 5
  log_dir: logs
"""
    
    config_path = f'configs/model/{args.model_type}_cfruit_example.yaml'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✓ 训练配置文件已创建: {config_path}")
    
    # 步骤4: 开始训练
    print("\n=== 步骤4: 开始训练 ===")
    
    try:
        # 导入训练模块
        sys.path.append('scripts')
        from train import main as train_main
        
        # 训练参数
        train_args = argparse.Namespace(
            config=config_path,
            data=f'configs/data/cfruit_{args.model_type}.yaml',
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=640,
            device='auto',
            workers=2,
            lr=0.001,
            weight_decay=0.0005,
            warmup_epochs=1,
            save_dir='checkpoints',
            log_dir='logs',
            save_interval=5,
            resume='',
            pretrained=''
        )
        
        # 调用训练函数
        train_main()
        print("✓ 训练完成")
        
    except ImportError as e:
        print(f"✗ 导入训练模块失败: {e}")
        print("请确保已安装所有依赖包: pip install -r requirements.txt")
        return
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        return
    
    # 清理示例数据
    if args.create_sample and args.input_dir.startswith(tempfile.gettempdir()):
        shutil.rmtree(args.input_dir)
        print(f"已清理示例数据: {args.input_dir}")
    
    print("\n=== 示例完成 ===")
    print("数据集已保存到:", args.output_dir)
    print("模型检查点已保存到: checkpoints/")
    print("训练日志已保存到: logs/")
    
    print("\n下一步:")
    print("1. 使用真实数据替换示例数据")
    print("2. 调整训练参数")
    print("3. 测试训练好的模型")

if __name__ == '__main__':
    main() 