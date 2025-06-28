#!/usr/bin/env python3
"""
训练结果可视化脚本
生成视频展示训练过程和结果
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import cv2
from PIL import Image, ImageDraw, ImageFont
import logging
from pathlib import Path
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov11_cfruit import YOLOv11CFruit
from data.dataset import CFruitDataset, CFruitDataLoader
from utils.transforms import get_transforms


def load_training_logs(log_file):
    """加载训练日志"""
    train_losses = []
    val_losses = []
    learning_rates = []
    timestamps = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if 'Train Loss:' in line and 'Val Loss:' in line:
                    # 解析训练和验证损失
                    parts = line.split('Train Loss:')[1].split(', Val Loss:')
                    train_loss = float(parts[0].strip())
                    val_loss = float(parts[1].strip())
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    
                    # 解析学习率
                    if 'LR=' in line:
                        lr_part = line.split('LR=')[1].split(',')[0]
                        lr = float(lr_part)
                        learning_rates.append(lr)
                    else:
                        learning_rates.append(0.001)  # 默认值
                    
                    # 解析时间戳
                    if 'Epoch' in line:
                        epoch_part = line.split('Epoch')[1].split('/')[0]
                        timestamps.append(int(epoch_part.strip()))
    
    return train_losses, val_losses, learning_rates, timestamps


def create_training_animation(train_losses, val_losses, learning_rates, output_path, fps=2):
    """创建训练曲线动画"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # 绘制损失曲线
        epochs = range(1, frame + 1)
        if frame > 0:
            ax1.plot(epochs, train_losses[:frame], 'b-', label='训练损失', linewidth=2)
            ax1.plot(epochs, val_losses[:frame], 'r-', label='验证损失', linewidth=2)
            ax1.set_title(f'训练进度 - 第 {frame} 轮', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 设置y轴范围
            if train_losses and val_losses:
                all_losses = train_losses[:frame] + val_losses[:frame]
                if all_losses:
                    ax1.set_ylim(0, max(all_losses) * 1.1)
        
        # 绘制学习率曲线
        if frame > 0 and learning_rates:
            ax2.plot(epochs, learning_rates[:frame], 'g-', label='学习率', linewidth=2)
            ax2.set_title('学习率变化', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        plt.tight_layout()
    
    # 创建动画
    frames = len(train_losses) if train_losses else 1
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                 interval=1000//fps, repeat=True)
    
    # 保存动画
    print(f"正在生成训练曲线动画...")
    anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close()
    
    print(f"训练曲线动画已保存到: {output_path}")


def create_model_prediction_video(model_path, config_path, data_path, output_path, 
                                num_samples=20, fps=1):
    """创建模型预测视频"""
    print(f"正在生成模型预测视频...")
    
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
    
    # 创建数据集
    dataset = CFruitDataset(
        img_dir=data_config['dataset']['val'],
        label_dir=data_config['dataset']['val_labels'],
        transform=get_transforms(img_size=640, augment=False),
        img_size=640
    )
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))
    
    # 生成预测视频
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            original_image = sample['image'].permute(1, 2, 0).numpy()
            
            # 推理
            cls_outputs, reg_outputs = model(image)
            
            # 创建可视化图像
            vis_image = create_prediction_visualization(
                original_image, cls_outputs, reg_outputs, sample.get('labels', [])
            )
            
            # 调整图像大小并转换为BGR格式
            vis_image = cv2.resize(vis_image, (1280, 720))
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            
            # 添加文本信息
            cv2.putText(vis_image, f'Sample {i+1}/{num_samples}', (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(vis_image, f'Model: YOLOv11-CFruit', (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 写入视频
            for _ in range(fps * 2):  # 每帧显示2秒
                out.write(vis_image)
    
    out.release()
    print(f"模型预测视频已保存到: {output_path}")


def create_prediction_visualization(image, cls_outputs, reg_outputs, labels):
    """创建预测可视化"""
    # 转换图像格式
    image = (image * 255).astype(np.uint8)
    
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('YOLOv11-CFruit 预测结果', fontsize=16, fontweight='bold')
    
    # 原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 分类预测热力图
    if cls_outputs:
        cls_heatmap = cls_outputs[0][0].cpu().numpy()  # 第一个特征层的分类预测
        axes[0, 1].imshow(cls_heatmap, cmap='hot')
        axes[0, 1].set_title('分类预测热力图')
        axes[0, 1].axis('off')
    
    # 回归预测可视化
    if reg_outputs:
        reg_heatmap = reg_outputs[0][0].cpu().numpy()  # 第一个特征层的回归预测
        axes[1, 0].imshow(reg_heatmap, cmap='viridis')
        axes[1, 0].set_title('回归预测热力图')
        axes[1, 0].axis('off')
    
    # 特征层信息
    axes[1, 1].text(0.1, 0.8, f'特征层数量: {len(cls_outputs)}', fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'分类输出形状: {cls_outputs[0].shape if cls_outputs else "N/A"}', fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'回归输出形状: {reg_outputs[0].shape if reg_outputs else "N/A"}', fontsize=12)
    axes[1, 1].text(0.1, 0.2, f'标签数量: {len(labels)}', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    # 保存图像
    plt.tight_layout()
    fig.canvas.draw()
    
    # 转换为numpy数组
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    return img_array


def create_training_summary_video(log_file, model_path, config_path, data_path, 
                                output_dir, fps=2):
    """创建训练总结视频"""
    print("正在生成训练总结视频...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载训练日志
    train_losses, val_losses, learning_rates, timestamps = load_training_logs(log_file)
    
    # 生成各部分视频
    training_curve_video = os.path.join(output_dir, 'training_curves.mp4')
    prediction_video = os.path.join(output_dir, 'model_predictions.mp4')
    summary_video = os.path.join(output_dir, 'training_summary.mp4')
    
    # 1. 训练曲线动画
    if train_losses and val_losses:
        create_training_animation(train_losses, val_losses, learning_rates, 
                                training_curve_video, fps)
    
    # 2. 模型预测视频
    create_model_prediction_video(model_path, config_path, data_path, 
                                prediction_video, num_samples=10, fps=fps)
    
    # 3. 创建总结视频
    create_summary_video(training_curve_video, prediction_video, summary_video, 
                        train_losses, val_losses, fps)
    
    print(f"训练总结视频已保存到: {summary_video}")


def create_summary_video(curve_video, pred_video, output_path, train_losses, val_losses, fps):
    """创建总结视频"""
    # 读取视频
    curve_cap = cv2.VideoCapture(curve_video)
    pred_cap = cv2.VideoCapture(pred_video)
    
    # 获取视频属性
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 创建总结帧
    summary_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加标题
    cv2.putText(summary_frame, 'YOLOv11-CFruit 训练总结', (width//2-200, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # 添加训练统计
    if train_losses and val_losses:
        stats_text = [
            f'训练轮数: {len(train_losses)}',
            f'最终训练损失: {train_losses[-1]:.6f}',
            f'最终验证损失: {val_losses[-1]:.6f}',
            f'最佳验证损失: {min(val_losses):.6f}',
            f'训练时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = 150 + i * 40
            cv2.putText(summary_frame, text, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 添加说明
    instructions = [
        '视频包含以下内容:',
        '1. 训练曲线动画 - 显示损失变化',
        '2. 模型预测结果 - 展示检测效果',
        '3. 训练统计信息 - 关键指标总结'
    ]
    
    for i, text in enumerate(instructions):
        y_pos = 400 + i * 30
        cv2.putText(summary_frame, text, (50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # 写入总结帧
    for _ in range(fps * 3):  # 显示3秒
        out.write(summary_frame)
    
    # 释放资源
    curve_cap.release()
    pred_cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(description='训练结果可视化')
    parser.add_argument('--log-file', type=str, default='logs/train_improved.log',
                       help='训练日志文件')
    parser.add_argument('--model-path', type=str, default='checkpoints/best.pt',
                       help='模型文件路径')
    parser.add_argument('--config', type=str, default='configs/model/yolov11_cfruit_improved.yaml',
                       help='模型配置文件')
    parser.add_argument('--data', type=str, default='configs/data/cfruit.yaml',
                       help='数据配置文件')
    parser.add_argument('--output-dir', type=str, default='visualization',
                       help='输出目录')
    parser.add_argument('--fps', type=int, default=2,
                       help='视频帧率')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['curves', 'predictions', 'summary', 'all'],
                       help='可视化模式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== YOLOv11-CFruit 训练结果可视化 ===")
    
    if args.mode in ['curves', 'all']:
        # 生成训练曲线动画
        train_losses, val_losses, learning_rates, _ = load_training_logs(args.log_file)
        if train_losses and val_losses:
            curve_video = os.path.join(args.output_dir, 'training_curves.mp4')
            create_training_animation(train_losses, val_losses, learning_rates, 
                                    curve_video, args.fps)
    
    if args.mode in ['predictions', 'all']:
        # 生成模型预测视频
        pred_video = os.path.join(args.output_dir, 'model_predictions.mp4')
        create_model_prediction_video(args.model_path, args.config, args.data, 
                                    pred_video, num_samples=15, fps=args.fps)
    
    if args.mode in ['summary', 'all']:
        # 生成训练总结视频
        create_training_summary_video(args.log_file, args.model_path, args.config, 
                                    args.data, args.output_dir, args.fps)
    
    print(f"\n可视化完成！结果保存在: {args.output_dir}")
    print("\n生成的文件:")
    print("- training_curves.mp4: 训练曲线动画")
    print("- model_predictions.mp4: 模型预测视频")
    print("- training_summary.mp4: 训练总结视频")


if __name__ == '__main__':
    main() 