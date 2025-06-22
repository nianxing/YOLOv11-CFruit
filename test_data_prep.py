#!/usr/bin/env python3
"""
测试数据准备功能
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

def create_test_data():
    """创建测试数据"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"创建测试数据目录: {temp_dir}")
    
    # 创建测试图像和标注
    for i in range(10):
        # 创建测试图像
        img = Image.new('RGB', (640, 480), color=(100, 150, 200))
        img_path = os.path.join(temp_dir, f'test_image_{i:03d}.jpg')
        img.save(img_path)
        
        # 创建对应的JSON标注文件
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
                },
                {
                    "label": "cfruit", 
                    "points": [[300, 300], [400, 300], [400, 400], [300, 400]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
            ],
            "imagePath": f"test_image_{i:03d}.jpg",
            "imageData": None,
            "imageHeight": 480,
            "imageWidth": 640
        }
        
        json_path = os.path.join(temp_dir, f'test_image_{i:03d}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
    
    return temp_dir

def test_data_preparation():
    """测试数据准备功能"""
    print("=== 测试数据准备功能 ===")
    
    # 创建测试数据
    test_data_dir = create_test_data()
    
    try:
        # 导入数据准备模块
        import sys
        sys.path.append('scripts')
        from prepare_data import LabelmeToYOLO, split_dataset, convert_labels
        
        print("✓ 成功导入数据准备模块")
        
        # 测试标签转换
        converter = LabelmeToYOLO(['cfruit'])
        json_path = os.path.join(test_data_dir, 'test_image_000.json')
        img_path = os.path.join(test_data_dir, 'test_image_000.jpg')
        
        # 创建输出目录
        output_dir = os.path.join(test_data_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # 测试转换
        converter.convert_annotation(json_path, img_path, output_dir)
        
        # 检查输出文件
        label_file = os.path.join(output_dir, 'test_image_000.txt')
        if os.path.exists(label_file):
            print("✓ 标签转换成功")
            with open(label_file, 'r') as f:
                content = f.read()
                print(f"转换后的标签内容: {content.strip()}")
        else:
            print("✗ 标签转换失败")
        
        # 测试数据集分割
        print("\n测试数据集分割...")
        split_output_dir = os.path.join(test_data_dir, 'split_output')
        split_dataset(test_data_dir, split_output_dir, 0.7, 0.2, 0.1)
        
        # 检查分割结果
        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(split_output_dir, split, 'images')
            if os.path.exists(img_dir):
                img_count = len(list(Path(img_dir).glob('*.jpg')))
                print(f"✓ {split} 集: {img_count} 个图像")
        
        print("✓ 数据集分割成功")
        
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
    finally:
        # 清理测试数据
        shutil.rmtree(test_data_dir)
        print(f"清理测试数据: {test_data_dir}")

def test_dataset_loading():
    """测试数据集加载功能"""
    print("\n=== 测试数据集加载功能 ===")
    
    try:
        # 导入数据集模块
        import sys
        sys.path.append('data')
        from dataset import CFruitDataset, get_dataset_stats
        
        print("✓ 成功导入数据集模块")
        
        # 创建测试数据
        test_data_dir = create_test_data()
        
        try:
            # 创建数据集
            dataset = CFruitDataset(
                img_dir=test_data_dir,
                label_dir=test_data_dir,
                img_size=640
            )
            
            print(f"✓ 数据集创建成功，包含 {len(dataset)} 个样本")
            
            # 测试数据加载
            sample = dataset[0]
            print(f"✓ 样本加载成功")
            print(f"  图像形状: {sample['image'].shape}")
            print(f"  标签数量: {len(sample['labels'])}")
            
            # 获取统计信息
            stats = get_dataset_stats(dataset)
            print(f"✓ 数据集统计信息获取成功")
            print(f"  总图像数: {stats['total_images']}")
            print(f"  总目标数: {stats['total_objects']}")
            
        finally:
            shutil.rmtree(test_data_dir)
            
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
    except Exception as e:
        print(f"✗ 测试失败: {e}")

def test_transforms():
    """测试数据变换功能"""
    print("\n=== 测试数据变换功能 ===")
    
    try:
        # 导入变换模块
        import sys
        sys.path.append('utils')
        from transforms import get_transforms, Resize, Normalize
        
        print("✓ 成功导入变换模块")
        
        # 创建测试图像
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bboxes = np.array([[0, 100, 100, 200, 200]])
        
        # 测试变换
        transform = get_transforms(img_size=640, is_training=True, fliplr=0.5)
        transformed_img, transformed_bboxes = transform(test_img, test_bboxes)
        
        print(f"✓ 数据变换成功")
        print(f"  原图像形状: {test_img.shape}")
        print(f"  变换后形状: {transformed_img.shape}")
        print(f"  原标签数量: {len(test_bboxes)}")
        print(f"  变换后标签数量: {len(transformed_bboxes)}")
        
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
    except Exception as e:
        print(f"✗ 测试失败: {e}")

def main():
    """主函数"""
    print("开始测试数据准备功能...")
    
    # 测试数据准备
    test_data_preparation()
    
    # 测试数据集加载
    test_dataset_loading()
    
    # 测试数据变换
    test_transforms()
    
    print("\n=== 测试完成 ===")
    print("如果所有测试都通过，您可以开始准备真实数据了！")
    print("\n使用示例:")
    print("python scripts/quick_train.py --input-dir /path/to/your/data --class-names cfruit")

if __name__ == '__main__':
    main() 