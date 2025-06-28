#!/usr/bin/env python3
"""
YOLOv8-CFruit 项目测试脚本
"""

import os
import sys
import yaml
import torch

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        from models.yolov11_cfruit import YOLOv11CFruit
        print("✓ YOLOv11CFruit 导入成功")
    except ImportError as e:
        print(f"✗ YOLOv11CFruit 导入失败: {e}")
        return False
    
    try:
        from models.backbone.cspdarknet import CSPDarknet
        print("✓ CSPDarknet 导入成功")
    except ImportError as e:
        print(f"✗ CSPDarknet 导入失败: {e}")
        return False
    
    try:
        from models.backbone.cbam import CBAM
        print("✓ CBAM 导入成功")
    except ImportError as e:
        print(f"✗ CBAM 导入失败: {e}")
        return False
    
    try:
        from models.neck.panet import PANetWithTransformer
        print("✓ PANetWithTransformer 导入成功")
    except ImportError as e:
        print(f"✗ PANetWithTransformer 导入失败: {e}")
        return False
    
    try:
        from models.head.anchor_free import AnchorFreeHead
        print("✓ AnchorFreeHead 导入成功")
    except ImportError as e:
        print(f"✗ AnchorFreeHead 导入失败: {e}")
        return False
    
    try:
        from utils.losses import YOLOv8Loss, EIoULoss
        print("✓ 损失函数导入成功")
    except ImportError as e:
        print(f"✗ 损失函数导入失败: {e}")
        return False
    
    return True


def test_configs():
    """测试配置文件"""
    print("\n测试配置文件...")
    
    config_files = [
        'configs/model/yolov8_cfruit.yaml',
        'configs/data/cfruit.yaml'
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✓ {config_file} 加载成功")
        except Exception as e:
            print(f"✗ {config_file} 加载失败: {e}")
            return False
    
    return True


def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        # 加载配置
        with open('configs/model/yolov8_cfruit.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建模型
        model = YOLOv11CFruit(config)
        print("✓ 模型创建成功")
        
        # 测试前向传播
        device = torch.device('cpu')
        model.to(device)
        
        # 创建测试输入
        batch_size = 2
        img_size = 640
        x = torch.randn(batch_size, 3, img_size, img_size)
        
        # 前向传播
        with torch.no_grad():
            cls_outputs, reg_outputs = model(x)
        
        print(f"✓ 前向传播成功")
        print(f"  输入形状: {x.shape}")
        print(f"  分类输出数量: {len(cls_outputs)}")
        print(f"  回归输出数量: {len(reg_outputs)}")
        
        # 获取模型信息
        model_info = model.get_model_info()
        print(f"✓ 模型信息获取成功")
        print(f"  总参数数量: {model_info['total_params']:,}")
        print(f"  可训练参数: {model_info['trainable_params']:,}")
        print(f"  模型大小: {model_info['model_size_mb']:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False


def test_loss_functions():
    """测试损失函数"""
    print("\n测试损失函数...")
    
    try:
        from utils.losses import YOLOv8Loss, EIoULoss, FocalLoss, DFLoss
        
        # 测试EIoU损失
        eiou_loss = EIoULoss()
        pred_boxes = torch.randn(10, 4)
        target_boxes = torch.randn(10, 4)
        loss = eiou_loss(pred_boxes, target_boxes)
        print(f"✓ EIoU损失计算成功: {loss.item():.4f}")
        
        # 测试Focal损失
        focal_loss = FocalLoss()
        pred_logits = torch.randn(10, 1)
        target_labels = torch.randint(0, 1, (10,))
        loss = focal_loss(pred_logits, target_labels)
        print(f"✓ Focal损失计算成功: {loss.item():.4f}")
        
        # 测试DFL损失
        dfl_loss = DFLoss(reg_max=16)
        pred_dist = torch.randn(10, 64)  # 4 * 16
        target_values = torch.randn(10, 4)
        loss = dfl_loss(pred_dist, target_values)
        print(f"✓ DFL损失计算成功: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        return False


def main():
    """主函数"""
    print("YOLOv8-CFruit 项目测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configs,
        test_model_creation,
        test_loss_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目结构正确。")
        return True
    else:
        print("❌ 部分测试失败，请检查项目结构。")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 