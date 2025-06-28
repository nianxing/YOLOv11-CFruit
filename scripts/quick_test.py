#!/usr/bin/env python3
"""
快速测试脚本 - 验证模型是否能正常运行
使用最小资源进行测试
"""

import os
import sys
import torch
import yaml
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov11_cfruit import YOLOv11CFruit
from utils.losses import YOLOv11Loss


def test_model():
    """测试模型是否能正常运行"""
    print("开始模型测试...")
    
    # 设置设备
    device = 'cpu'  # 使用CPU进行测试
    print(f"使用设备: {device}")
    
    # 创建简单配置
    config = {
        'model': {
            'backbone': {
                'cbam': True,
                'cbam_ratio': 16,
                'use_c2f': True
            },
            'neck': {
                'transformer_heads': 8,
                'transformer_dim': 256,
                'transformer_layers': 2
            },
            'head': {
                'num_classes': 1,
                'reg_max': 16
            }
        }
    }
    
    try:
        # 创建模型
        print("创建模型...")
        model = YOLOv11CFruit(config)
        model = model.to(device)
        model.eval()
        
        # 打印模型信息
        model_info = model.get_model_info()
        print(f"模型参数: {model_info['total_params']:,}")
        print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
        
        # 创建测试输入
        print("创建测试输入...")
        batch_size = 1
        img_size = 320  # 使用较小的图像尺寸
        x = torch.randn(batch_size, 3, img_size, img_size).to(device)
        
        # 前向传播
        print("执行前向传播...")
        with torch.no_grad():
            cls_outputs, reg_outputs = model(x)
        
        print(f"输入形状: {x.shape}")
        print(f"分类输出数量: {len(cls_outputs)}")
        print(f"回归输出数量: {len(reg_outputs)}")
        
        for i, (cls_out, reg_out) in enumerate(zip(cls_outputs, reg_outputs)):
            print(f"特征层 {i}: 分类输出形状 {cls_out.shape}, 回归输出形状 {reg_out.shape}")
        
        # 测试损失函数
        print("测试损失函数...")
        criterion = YOLOv11Loss(num_classes=1, reg_max=16)
        
        # 创建模拟标签
        targets = torch.zeros(batch_size, 1, 5)  # [B, N, 5] 格式
        targets[0, 0, 0] = 0  # class_id
        targets[0, 0, 1:5] = torch.tensor([100, 100, 200, 200])  # 边界框
        
        # 计算损失
        loss, loss_dict = criterion(cls_outputs, reg_outputs, targets)
        
        print(f"总损失: {loss.item():.6f}")
        print(f"损失详情: {loss_dict}")
        
        print("✓ 模型测试成功！")
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """测试内存使用情况"""
    print("\n测试内存使用...")
    
    import psutil
    import gc
    
    # 获取当前内存使用
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"测试前内存使用: {memory_before:.2f} MB")
    
    # 运行模型测试
    success = test_model()
    
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 获取测试后内存使用
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    print(f"测试后内存使用: {memory_after:.2f} MB")
    print(f"内存增加: {memory_after - memory_before:.2f} MB")
    
    return success


if __name__ == '__main__':
    print("YOLOv11-CFruit 快速测试")
    print("=" * 50)
    
    # 测试模型
    success = test_memory_usage()
    
    if success:
        print("\n✓ 所有测试通过！模型可以正常运行。")
        print("\n建议的训练参数:")
        print("- 批次大小: 1-2")
        print("- 工作进程数: 2-4")
        print("- 图像尺寸: 320-640")
        print("- 设备: CPU (如果GPU内存不足)")
    else:
        print("\n✗ 测试失败，请检查错误信息。") 