#!/usr/bin/env python3
"""
快速可视化训练结果
生成视频展示训练过程和模型预测
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov11_cfruit import YOLOv11CFruit
from data.dataset import CFruitDataset
from utils.transforms import get_transforms


def load_training_logs(log_file):
    """加载训练日志"""
    train_losses = []
    val_losses = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Train Loss:' in line and 'Val Loss:' in line:
                    try:
                        parts = line.split('Train Loss:')[1].split(', Val Loss:')
                        train_loss = float(parts[0].strip())
                        val_loss = float(parts[1].strip())
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                    except:
                        continue
    
    return train_losses, val_losses


def create_loss_animation(train_losses, val_losses, output_path, fps=3):
    """创建损失曲线动画"""
    if not train_losses or not val_losses:
        print("没有找到训练日志数据")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def animate(frame):
        ax.clear()
        
        epochs = range(1, frame + 1)
        if frame > 0:
            ax.plot(epochs, train_losses[:frame], 'b-', label='训练损失', linewidth=3, marker='o')
            ax.plot(epochs, val_losses[:frame], 'r-', label='验证损失', linewidth=3, marker='s')
            
            ax.set_title(f'YOLOv11-CFruit 训练进度 - 第 {frame} 轮', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 设置y轴范围
            all_losses = train_losses[:frame] + val_losses[:frame]
            if all_losses:
                ax.set_ylim(0, max(all_losses) * 1.1)
            
            # 添加当前损失值标注
            if frame > 0:
                ax.text(0.02, 0.98, f'当前训练损失: {train_losses[frame-1]:.6f}', 
                       transform=ax.transAxes, fontsize=10, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
                ax.text(0.02, 0.92, f'当前验证损失: {val_losses[frame-1]:.6f}', 
                       transform=ax.transAxes, fontsize=10, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # 创建动画
    frames = len(train_losses)
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                 interval=1000//fps, repeat=True)
    
    print(f"正在生成训练曲线动画...")
    anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close()
    
    print(f"✓ 训练曲线动画已保存到: {output_path}")


def create_model_demo_video(model_path, config_path, data_path, output_path, num_samples=10, fps=2):
    """创建模型演示视频"""
    print(f"正在生成模型演示视频...")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"✗ 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(config_path):
        print(f"✗ 配置文件不存在: {config_path}")
        return
    
    if not os.path.exists(data_path):
        print(f"✗ 数据配置文件不存在: {data_path}")
        return
    
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return
    
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    try:
        model = YOLOv11CFruit(model_config)
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        print(f"✓ 模型已加载: {model_path}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 创建数据集
    try:
        dataset = CFruitDataset(
            img_dir=data_config['dataset']['val'],
            label_dir=data_config['dataset']['val_labels'],
            transform=get_transforms(img_size=640, augment=False),
            img_size=640
        )
        print(f"✓ 数据集已加载，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        return
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))
    
    # 生成演示视频
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            try:
                sample = dataset[i]
                image = sample['image'].unsqueeze(0).to(device)
                original_image = sample['image'].permute(1, 2, 0).numpy()
                
                # 推理
                cls_outputs, reg_outputs = model(image)
                
                # 创建可视化图像
                vis_image = create_demo_visualization(
                    original_image, cls_outputs, reg_outputs, i+1, num_samples
                )
                
                # 调整图像大小并转换为BGR格式
                vis_image = cv2.resize(vis_image, (1280, 720))
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                
                # 写入视频
                for _ in range(fps * 3):  # 每帧显示3秒
                    out.write(vis_image)
                
                print(f"处理样本 {i+1}/{num_samples}")
                
            except Exception as e:
                print(f"处理样本 {i+1} 时出错: {e}")
                continue
    
    out.release()
    print(f"✓ 模型演示视频已保存到: {output_path}")


def create_demo_visualization(image, cls_outputs, reg_outputs, sample_idx, total_samples):
    """创建演示可视化"""
    # 转换图像格式
    image = (image * 255).astype(np.uint8)
    
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'YOLOv11-CFruit 模型演示 - 样本 {sample_idx}/{total_samples}', 
                fontsize=16, fontweight='bold')
    
    # 原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('原始图像', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 分类预测热力图
    if cls_outputs and len(cls_outputs) > 0:
        try:
            cls_heatmap = cls_outputs[0][0].cpu().numpy()
            axes[0, 1].imshow(cls_heatmap, cmap='hot')
            axes[0, 1].set_title('分类预测热力图', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
        except:
            axes[0, 1].text(0.5, 0.5, '分类预测\n(数据格式错误)', 
                           ha='center', va='center', fontsize=12)
            axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, '分类预测\n(无数据)', 
                       ha='center', va='center', fontsize=12)
        axes[0, 1].axis('off')
    
    # 回归预测可视化
    if reg_outputs and len(reg_outputs) > 0:
        try:
            reg_heatmap = reg_outputs[0][0].cpu().numpy()
            axes[1, 0].imshow(reg_heatmap, cmap='viridis')
            axes[1, 0].set_title('回归预测热力图', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
        except:
            axes[1, 0].text(0.5, 0.5, '回归预测\n(数据格式错误)', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, '回归预测\n(无数据)', 
                       ha='center', va='center', fontsize=12)
        axes[1, 0].axis('off')
    
    # 模型信息
    info_text = [
        f'模型: YOLOv11-CFruit',
        f'特征层数量: {len(cls_outputs)}',
        f'分类输出: {cls_outputs[0].shape if cls_outputs else "N/A"}',
        f'回归输出: {reg_outputs[0].shape if reg_outputs else "N/A"}',
        f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    ]
    
    for i, text in enumerate(info_text):
        axes[1, 1].text(0.1, 0.8 - i * 0.15, text, fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
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


def main():
    parser = argparse.ArgumentParser(description='快速可视化训练结果')
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
                       choices=['loss', 'demo', 'all'],
                       help='可视化模式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== YOLOv11-CFruit 快速可视化 ===")
    print(f"输出目录: {args.output_dir}")
    print(f"视频帧率: {args.fps} FPS")
    
    if args.mode in ['loss', 'all']:
        # 生成训练损失动画
        train_losses, val_losses = load_training_logs(args.log_file)
        if train_losses and val_losses:
            loss_video = os.path.join(args.output_dir, 'training_loss.mp4')
            create_loss_animation(train_losses, val_losses, loss_video, args.fps)
        else:
            print("✗ 没有找到训练日志数据")
    
    if args.mode in ['demo', 'all']:
        # 生成模型演示视频
        demo_video = os.path.join(args.output_dir, 'model_demo.mp4')
        create_model_demo_video(args.model_path, args.config, args.data, 
                              demo_video, num_samples=8, fps=args.fps)
    
    print(f"\n✓ 可视化完成！")
    print(f"结果保存在: {args.output_dir}")
    
    if args.mode in ['loss', 'all'] and os.path.exists(os.path.join(args.output_dir, 'training_loss.mp4')):
        print("- training_loss.mp4: 训练损失曲线动画")
    
    if args.mode in ['demo', 'all'] and os.path.exists(os.path.join(args.output_dir, 'model_demo.mp4')):
        print("- model_demo.mp4: 模型演示视频")


if __name__ == '__main__':
    main() 