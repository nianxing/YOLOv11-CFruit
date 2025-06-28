#!/usr/bin/env python3
"""
模型评估脚本
评估训练结果并提供改进建议
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov11_cfruit import YOLOv11CFruit
from data.dataset import CFruitDataset, CFruitDataLoader
from utils.transforms import get_transforms


def load_training_logs(log_file):
    """加载训练日志"""
    train_losses = []
    val_losses = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if 'Train Loss:' in line and 'Val Loss:' in line:
                    parts = line.split('Train Loss:')[1].split(', Val Loss:')
                    train_loss = float(parts[0].strip())
                    val_loss = float(parts[1].strip())
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
    
    return train_losses, val_losses


def plot_training_curves(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_losses, 'r-', label='验证损失')
    plt.title('验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    # 计算损失差异
    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    plt.plot(epochs, loss_diff, 'g-', label='损失差异')
    plt.title('训练和验证损失差异')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {save_path}")


def analyze_training_results(train_losses, val_losses):
    """分析训练结果"""
    if not train_losses or not val_losses:
        print("没有找到训练日志数据")
        return
    
    print("\n=== 训练结果分析 ===")
    
    # 基本统计
    print(f"训练轮数: {len(train_losses)}")
    print(f"最终训练损失: {train_losses[-1]:.6f}")
    print(f"最终验证损失: {val_losses[-1]:.6f}")
    print(f"最佳验证损失: {min(val_losses):.6f} (第 {val_losses.index(min(val_losses)) + 1} 轮)")
    
    # 损失趋势分析
    train_trend = train_losses[-1] - train_losses[0]
    val_trend = val_losses[-1] - val_losses[0]
    
    print(f"\n损失趋势:")
    print(f"训练损失变化: {train_trend:.6f} ({'下降' if train_trend < 0 else '上升'})")
    print(f"验证损失变化: {val_trend:.6f} ({'下降' if val_trend < 0 else '上升'})")
    
    # 过拟合分析
    if len(train_losses) > 10:
        early_train_avg = np.mean(train_losses[:10])
        late_train_avg = np.mean(train_losses[-10:])
        early_val_avg = np.mean(val_losses[:10])
        late_val_avg = np.mean(val_losses[-10:])
        
        train_improvement = early_train_avg - late_train_avg
        val_improvement = early_val_avg - late_val_avg
        
        print(f"\n过拟合分析:")
        print(f"训练损失改进: {train_improvement:.6f}")
        print(f"验证损失改进: {val_improvement:.6f}")
        
        if train_improvement > 0 and val_improvement < 0:
            print("⚠️  可能存在过拟合现象")
        elif train_improvement > 0 and val_improvement > 0:
            print("✓ 训练效果良好")
        else:
            print("⚠️  训练效果不佳")
    
    # 建议
    print(f"\n=== 改进建议 ===")
    
    if val_losses[-1] > 0.1:
        print("1. 损失仍然较高，建议:")
        print("   - 增加训练轮数")
        print("   - 调整学习率")
        print("   - 检查数据质量")
        print("   - 改进损失函数")
    
    if abs(train_losses[-1] - val_losses[-1]) > 0.1:
        print("2. 训练和验证损失差异较大，建议:")
        print("   - 增加数据增强")
        print("   - 使用正则化技术")
        print("   - 减少模型复杂度")
        print("   - 使用早停机制")
    
    if val_losses[-1] < 0.01:
        print("3. 损失很低，可能存在问题:")
        print("   - 检查损失函数实现")
        print("   - 验证数据标签")
        print("   - 检查模型输出")


def test_model_inference(model_path, config_path, data_path, num_samples=5):
    """测试模型推理"""
    print(f"\n=== 模型推理测试 ===")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOv11CFruit(model_config)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ 模型已加载: {model_path}")
    else:
        print(f"✗ 模型文件不存在: {model_path}")
        return
    
    model = model.to(device)
    model.eval()
    
    # 创建测试数据集
    test_dataset = CFruitDataset(
        img_dir=data_config['dataset']['val'],
        label_dir=data_config['dataset']['val_labels'],
        transform=get_transforms(img_size=640, augment=False),
        img_size=640
    )
    
    # 测试推理
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            
            # 推理
            cls_outputs, reg_outputs = model(image)
            
            print(f"样本 {i+1}:")
            print(f"  输入形状: {image.shape}")
            print(f"  分类输出: {len(cls_outputs)} 个特征层")
            print(f"  回归输出: {len(reg_outputs)} 个特征层")
            
            for j, (cls_out, reg_out) in enumerate(zip(cls_outputs, reg_outputs)):
                print(f"    特征层 {j}: 分类 {cls_out.shape}, 回归 {reg_out.shape}")


def main():
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--log-file', type=str, default='logs/train_improved.log',
                       help='训练日志文件')
    parser.add_argument('--model-path', type=str, default='checkpoints/best.pt',
                       help='模型文件路径')
    parser.add_argument('--config', type=str, default='configs/model/yolov11_cfruit_improved.yaml',
                       help='模型配置文件')
    parser.add_argument('--data', type=str, default='configs/data/cfruit.yaml',
                       help='数据配置文件')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载训练日志
    train_losses, val_losses = load_training_logs(args.log_file)
    
    # 分析训练结果
    analyze_training_results(train_losses, val_losses)
    
    # 绘制训练曲线
    if train_losses and val_losses:
        plot_path = os.path.join(args.output_dir, 'training_curves.png')
        plot_training_curves(train_losses, val_losses, plot_path)
    
    # 测试模型推理
    test_model_inference(args.model_path, args.config, args.data)
    
    print(f"\n评估完成！结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main() 