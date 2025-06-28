#!/usr/bin/env python3
"""
检查数据路径脚本
验证数据配置文件中的路径是否存在
"""

import os
import yaml
from pathlib import Path


def check_data_paths():
    """检查数据路径是否存在"""
    print("检查数据路径...")
    
    # 加载数据配置
    config_path = 'configs/data/cfruit.yaml'
    if not os.path.exists(config_path):
        print(f"✗ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 检查数据集路径
    dataset_config = data_config.get('dataset', {})
    
    paths_to_check = [
        ('训练图像', dataset_config.get('train', '')),
        ('训练标签', dataset_config.get('train_labels', '')),
        ('验证图像', dataset_config.get('val', '')),
        ('验证标签', dataset_config.get('val_labels', '')),
        ('测试图像', dataset_config.get('test', '')),
        ('测试标签', dataset_config.get('test_labels', ''))
    ]
    
    all_paths_exist = True
    
    for name, path in paths_to_check:
        if not path:
            print(f"✗ {name}路径未配置")
            all_paths_exist = False
            continue
            
        if os.path.exists(path):
            # 检查目录中的文件数量
            if os.path.isdir(path):
                files = list(Path(path).glob('*'))
                if name.endswith('图像'):
                    image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                    print(f"✓ {name}: {path} (找到 {len(image_files)} 个图像文件)")
                elif name.endswith('标签'):
                    label_files = [f for f in files if f.suffix.lower() == '.txt']
                    print(f"✓ {name}: {path} (找到 {len(label_files)} 个标签文件)")
                else:
                    print(f"✓ {name}: {path} (找到 {len(files)} 个文件)")
            else:
                print(f"✓ {name}: {path} (文件存在)")
        else:
            print(f"✗ {name}路径不存在: {path}")
            all_paths_exist = False
    
    return all_paths_exist


def create_sample_data():
    """创建示例数据目录结构"""
    print("\n创建示例数据目录结构...")
    
    sample_dirs = [
        'data/cfruit/train/images',
        'data/cfruit/train/labels',
        'data/cfruit/val/images',
        'data/cfruit/val/labels',
        'data/cfruit/test/images',
        'data/cfruit/test/labels'
    ]
    
    for dir_path in sample_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")
    
    # 创建示例图像文件
    import numpy as np
    from PIL import Image
    
    for split in ['train', 'val', 'test']:
        img_dir = f'data/cfruit/{split}/images'
        label_dir = f'data/cfruit/{split}/labels'
        
        # 创建示例图像
        for i in range(5):
            # 创建随机图像
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = os.path.join(img_dir, f'sample_{i}.jpg')
            Image.fromarray(img).save(img_path)
            
            # 创建示例标签
            label_path = os.path.join(label_dir, f'sample_{i}.txt')
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 0.3 0.3\n')  # 示例标签格式
    
    print("✓ 示例数据创建完成")


if __name__ == '__main__':
    print("数据路径检查")
    print("=" * 50)
    
    if check_data_paths():
        print("\n✓ 所有数据路径都存在！")
    else:
        print("\n✗ 部分数据路径不存在")
        print("\n是否创建示例数据目录结构？(y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            create_sample_data()
            print("\n现在可以运行训练脚本了！")
        else:
            print("\n请检查数据配置文件中的路径设置。") 