#!/usr/bin/env python3
"""
内存优化的训练脚本
减少批次大小和工作进程数以降低内存使用
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov11_cfruit import YOLOv11CFruit
from data.dataset import CFruitDataset, CFruitDataLoader
from utils.transforms import get_transforms
from utils.losses import YOLOv11Loss
from training.trainer import Trainer
from training.scheduler import CosineAnnealingWarmupRestarts


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11-CFruit 内存优化训练')
    parser.add_argument('--config', type=str, default='configs/model/yolov11_cfruit.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--data', type=str, default='configs/data/cfruit.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=1,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=1,  # 减少批次大小
                       help='批次大小')
    parser.add_argument('--num-workers', type=int, default=2,  # 减少工作进程
                       help='数据加载器工作进程数')
    parser.add_argument('--img-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='cpu',  # 使用CPU以减少GPU内存
                       help='训练设备 (cpu/cuda)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='日志保存目录')
    parser.add_argument('--resume', action='store_true',
                       help='是否从检查点恢复训练')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.data, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 设置设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        # 设置GPU内存分配策略
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        args.device = 'cpu'
    
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
    
    # 使用内存优化的数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,  # 减少工作进程
        pin_memory=False,  # 禁用pin_memory以减少内存使用
        collate_fn=CFruitDataLoader.collate_fn,
        persistent_workers=False  # 禁用持久化工作进程
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,  # 减少工作进程
        pin_memory=False,  # 禁用pin_memory
        collate_fn=CFruitDataLoader.collate_fn,
        persistent_workers=False
    )
    
    logging.info(f"训练集大小: {len(train_dataset)}")
    logging.info(f"验证集大小: {len(val_dataset)}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"工作进程数: {args.num_workers}")
    
    # 损失函数
    loss_weights = model_config.get('training', {}).get('loss_weights', {})
    criterion = YOLOv11Loss(
        num_classes=model_config['model']['head']['num_classes'],
        reg_max=model_config['model']['head']['reg_max'],
        loss_weights=loss_weights
    )
    
    # 优化器
    optimizer_config = model_config.get('training', {}).get('optimizer', {})
    if optimizer_config.get('type', 'adamw') == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.get('lr', 0.001),
            weight_decay=optimizer_config.get('weight_decay', 0.0005),
            betas=optimizer_config.get('betas', [0.9, 0.999])
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.get('lr', 0.001),
            weight_decay=optimizer_config.get('weight_decay', 0.0005)
        )
    
    # 学习率调度器
    scheduler_config = model_config.get('training', {}).get('scheduler', {})
    if scheduler_config.get('type', 'cosine') == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=len(train_loader) * args.epochs,
            cycle_mult=1.0,
            max_lr=optimizer_config.get('lr', 0.001),
            min_lr=scheduler_config.get('min_lr', 0.00001),
            warmup_steps=len(train_loader) * scheduler_config.get('warmup_epochs', 3),
            gamma=1.0
        )
    else:
        scheduler = None
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        save_interval=5  # 减少保存频率
    )
    
    # 恢复训练
    if args.resume:
        latest_checkpoint = os.path.join(args.save_dir, 'latest.pt')
        if os.path.exists(latest_checkpoint):
            trainer.load_checkpoint(latest_checkpoint)
            logging.info(f"从检查点恢复训练: {latest_checkpoint}")
    
    # 开始训练
    try:
        trainer.train(epochs=args.epochs)
        logging.info("训练完成！")
    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    main() 