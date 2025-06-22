#!/usr/bin/env python3
"""
YOLOv8-CFruit 训练脚本
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov8_cfruit import YOLOv8CFruit
from data.dataset import CFruitDataset
from utils.losses import YOLOv8Loss
from utils.transforms import get_transforms
from training.trainer import Trainer
from training.scheduler import CosineAnnealingWarmupRestarts


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv8-CFruit Training')
    
    # 配置文件
    parser.add_argument('--config', type=str, default='configs/model/yolov8_cfruit.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--data', type=str, default='configs/data/cfruit.yaml',
                       help='数据集配置文件路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0', help='训练设备')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器工作进程数')
    
    # 优化器参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='预热轮数')
    
    # 保存和日志
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='保存目录')
    parser.add_argument('--log-dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--save-interval', type=int, default=10, help='保存间隔')
    
    # 其他
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--pretrained', type=str, default='', help='预训练权重路径')
    
    return parser.parse_args()


def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, device):
    """创建模型"""
    model = YOLOv8CFruit(config)
    model = model.to(device)
    
    # 打印模型信息
    model_info = model.get_model_info()
    logging.info(f"模型信息: {model_info}")
    
    return model


def create_dataloaders(data_config, batch_size, img_size, workers):
    """创建数据加载器"""
    # 训练数据增强
    train_transforms = get_transforms(
        img_size=img_size,
        is_training=True,
        **data_config.get('augmentation', {}).get('train', {})
    )
    
    # 验证数据增强
    val_transforms = get_transforms(
        img_size=img_size,
        is_training=False,
        **data_config.get('augmentation', {}).get('val', {})
    )
    
    # 训练数据集
    train_dataset = CFruitDataset(
        data_config['dataset']['train'],
        data_config['dataset']['train_labels'],
        transform=train_transforms,
        img_size=img_size
    )
    
    # 验证数据集
    val_dataset = CFruitDataset(
        data_config['dataset']['val'],
        data_config['dataset']['val_labels'],
        transform=val_transforms,
        img_size=img_size
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    
    logging.info(f"训练集大小: {len(train_dataset)}")
    logging.info(f"验证集大小: {len(val_dataset)}")
    
    return train_loader, val_loader


def create_optimizer(model, config, args):
    """创建优化器"""
    optimizer_config = config['training']['optimizer']
    
    if optimizer_config['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif optimizer_config['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.937,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_config['type']}")
    
    return optimizer


def create_scheduler(optimizer, config, args, train_loader):
    """创建学习率调度器"""
    scheduler_config = config['training']['scheduler']
    
    if scheduler_config['type'] == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=len(train_loader) * args.epochs,
            cycle_mult=1.0,
            max_lr=args.lr,
            min_lr=args.lr * 0.01,
            warmup_steps=len(train_loader) * args.warmup_epochs,
            gamma=1.0
        )
    else:
        raise ValueError(f"不支持的学习率调度器类型: {scheduler_config['type']}")
    
    return scheduler


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 设置日志
    setup_logging(args.log_dir)
    
    # 加载配置
    model_config = load_config(args.config)
    data_config = load_config(args.data)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建模型
    model = create_model(model_config, device)
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        data_config, args.batch_size, args.img_size, args.workers
    )
    
    # 创建损失函数
    loss_weights = model_config['training']['loss_weights']
    criterion = YOLOv8Loss(
        num_classes=model_config['model']['head']['num_classes'],
        reg_max=model_config['model']['head']['reg_max'],
        loss_weights=loss_weights
    )
    
    # 创建优化器
    optimizer = create_optimizer(model, model_config, args)
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, model_config, args, train_loader)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logging.info(f"从检查点恢复训练: {args.resume}, 起始轮数: {start_epoch}")
    
    # 加载预训练权重
    if args.pretrained and not args.resume:
        trainer.load_pretrained(args.pretrained)
        logging.info(f"加载预训练权重: {args.pretrained}")
    
    # 开始训练
    logging.info("开始训练...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        start_epoch=start_epoch,
        save_interval=args.save_interval
    )
    
    logging.info("训练完成!")


if __name__ == '__main__':
    main() 