#!/usr/bin/env python3
"""
简化训练脚本
用于测试损失函数和基本训练流程
"""

import os
import sys
import argparse
import yaml
import torch
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov11_cfruit import YOLOv11CFruit
from data.dataset import CFruitDataset, CFruitDataLoader
from utils.transforms import get_transforms
from utils.simple_loss import SimpleYOLOLoss
from training.scheduler import CosineAnnealingWarmupRestarts


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11-CFruit 简化训练')
    parser.add_argument('--config', type=str, default='configs/model/yolov11_cfruit_improved.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--data', type=str, default='configs/data/cfruit.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='批次大小')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='数据加载器工作进程数')
    parser.add_argument('--img-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备 (auto/cpu/cuda)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='日志保存目录')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'simple_train.log')),
            logging.StreamHandler()
        ]
    )
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.data, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 设置设备
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.empty_cache()
            logging.info(f"CUDA可用，使用GPU: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            logging.info("CUDA不可用，使用CPU")
    else:
        device = args.device
    
    logging.info(f"使用设备: {device}")
    
    # 创建模型
    model = YOLOv11CFruit(model_config)
    model = model.to(device)
    
    # 打印模型信息
    model_info = model.get_model_info()
    logging.info(f"模型参数: {model_info['total_params']:,}")
    logging.info(f"模型大小: {model_info['model_size_mb']:.2f} MB")
    
    # 数据变换
    transforms = get_transforms(
        img_size=args.img_size,
        augment=True,
        **data_config.get('augmentation', {}).get('train', {})
    )
    
    # 创建数据集和数据加载器
    train_dataset = CFruitDataset(
        img_dir=data_config['dataset']['train'],
        label_dir=data_config['dataset']['train_labels'],
        transform=transforms,
        img_size=args.img_size
    )
    
    val_dataset = CFruitDataset(
        img_dir=data_config['dataset']['val'],
        label_dir=data_config['dataset']['val_labels'],
        transform=get_transforms(img_size=args.img_size, augment=False, **data_config.get('augmentation', {}).get('val', {})),
        img_size=args.img_size
    )
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == 'cuda',
        collate_fn=CFruitDataLoader.collate_fn,
        persistent_workers=False  # 简化版本不使用persistent_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == 'cuda',
        collate_fn=CFruitDataLoader.collate_fn,
        persistent_workers=False
    )
    
    logging.info(f"训练集大小: {len(train_dataset)}")
    logging.info(f"验证集大小: {len(val_dataset)}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"工作进程数: {args.num_workers}")
    
    # 损失函数
    criterion = SimpleYOLOLoss(
        num_classes=model_config['model']['head']['num_classes'],
        reg_max=model_config['model']['head']['reg_max']
    )
    
    # 优化器
    optimizer_config = model_config.get('training', {}).get('optimizer', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.get('lr', 0.0001),
        weight_decay=optimizer_config.get('weight_decay', 0.0005),
        betas=optimizer_config.get('betas', [0.9, 0.999])
    )
    
    # 学习率调度器
    scheduler_config = model_config.get('training', {}).get('scheduler', {})
    if scheduler_config.get('type', 'cosine') == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=len(train_loader) * args.epochs,
            cycle_mult=1.0,
            max_lr=optimizer_config.get('lr', 0.0001),
            min_lr=scheduler_config.get('min_lr', 0.000001),
            warmup_steps=len(train_loader) * scheduler_config.get('warmup_epochs', 5),
            gamma=1.0
        )
    else:
        scheduler = None
    
    # 开始训练
    logging.info("开始训练...")
    
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # 训练模式
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                images = batch['images'].to(device)
                targets = batch['labels'].to(device)
                
                # 前向传播
                optimizer.zero_grad()
                cls_outputs, reg_outputs = model(images)
                
                # 测试损失函数
                logging.info(f"Batch {batch_idx}: 测试损失函数...")
                loss, loss_dict = criterion(cls_outputs, reg_outputs, targets)
                logging.info(f"Batch {batch_idx}: 损失函数成功，总损失: {loss.item():.4f}")
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
                
                total_train_loss += loss.item()
                num_batches += 1
                
                # 每10个批次记录一次
                if batch_idx % 10 == 0:
                    avg_loss = total_train_loss / (batch_idx + 1)
                    logging.info(f"Batch {batch_idx}: 平均损失: {avg_loss:.4f}")
                
                # 只训练几个批次进行测试
                if batch_idx >= 5:
                    break
                    
            except Exception as e:
                logging.error(f"Batch {batch_idx} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # 验证
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    images = batch['images'].to(device)
                    targets = batch['labels'].to(device)
                    
                    cls_outputs, reg_outputs = model(images)
                    loss, loss_dict = criterion(cls_outputs, reg_outputs, targets)
                    
                    total_val_loss += loss.item()
                    val_batches += 1
                    
                    # 只验证几个批次
                    if batch_idx >= 2:
                        break
                        
                except Exception as e:
                    logging.error(f"验证批次 {batch_idx} 失败: {e}")
                    break
        
        if num_batches > 0:
            avg_train_loss = total_train_loss / num_batches
            avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0.0
            
            logging.info(f"Epoch {epoch + 1} 完成 - 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        
        # 保存模型
        if (epoch + 1) % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }
            checkpoint_path = os.path.join(args.save_dir, f'simple_epoch_{epoch + 1}.pt')
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"检查点已保存: {checkpoint_path}")
    
    logging.info("训练完成！")


if __name__ == '__main__':
    main() 