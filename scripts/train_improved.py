#!/usr/bin/env python3
"""
改进的训练脚本
包含更好的监控、早停和优化功能
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
import time
from collections import deque
from typing import Dict
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov11_cfruit import YOLOv11CFruit
from data.dataset import CFruitDataset, CFruitDataLoader
from utils.transforms import get_transforms
from utils.losses import YOLOv11Loss
from training.trainer import Trainer
from training.scheduler import CosineAnnealingWarmupRestarts


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class ImprovedTrainer(Trainer):
    """改进的训练器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stopping = None
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        self.use_amp = kwargs.get('use_amp', False)
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
    def set_early_stopping(self, patience, min_delta):
        """设置早停"""
        self.early_stopping = EarlyStopping(patience, min_delta)
        
    def train_epoch(self) -> Dict[str, float]:
        """改进的训练epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            targets = batch['labels'].to(self.device)
            
            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    cls_outputs, reg_outputs = self.model(images)
                    loss, loss_dict = self.criterion(cls_outputs, reg_outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
            else:
                cls_outputs, reg_outputs = self.model(images)
                loss, loss_dict = self.criterion(cls_outputs, reg_outputs, targets)
                loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # 更新统计
            total_loss += loss.item() * self.gradient_accumulation_steps
            avg_loss = total_loss / (batch_idx + 1)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                'Cls': f'{loss_dict["cls_loss"].item():.4f}',
                'Box': f'{loss_dict["box_loss"].item():.4f}',
                'DFL': f'{loss_dict["dfl_loss"].item():.4f}'
            })
            
            # 记录到TensorBoard
            if batch_idx % 10 == 0:
                step = self.epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Total_Loss', loss.item(), step)
                self.writer.add_scalar('Train/Cls_Loss', loss_dict['cls_loss'].item(), step)
                self.writer.add_scalar('Train/Box_Loss', loss_dict['box_loss'].item(), step)
                self.writer.add_scalar('Train/DFL_Loss', loss_dict['dfl_loss'].item(), step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]["lr"], step)
        
        return {'train_loss': total_loss / num_batches}
    
    def train(self, epochs: int, resume: bool = False):
        """改进的训练循环"""
        logging.info(f"开始训练，总轮数: {epochs}")
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['train_loss'])
            
            # 验证
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['val_loss'])
            
            epoch_time = time.time() - epoch_start_time
            
            # 记录日志
            logging.info(
                f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )
            
            # 早停检查
            if self.early_stopping and self.early_stopping(val_metrics['val_loss'], self.model):
                logging.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break
            
            # 保存检查点
            is_best = val_metrics['val_loss'] < min(self.val_losses[:-1]) if len(self.val_losses) > 1 else True
            self.save_checkpoint(is_best=is_best)
            
            # 定期保存
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(filename=f'epoch_{epoch + 1}.pt')
        
        logging.info("训练完成！")
        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11-CFruit 改进训练')
    parser.add_argument('--config', type=str, default='configs/model/yolov11_cfruit_improved.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--data', type=str, default='configs/data/cfruit.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--img-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备 (auto/cpu/cuda)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='日志保存目录')
    parser.add_argument('--resume', action='store_true',
                       help='是否从检查点恢复训练')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建保存目录（在设置日志之前）
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'train_improved.log')),
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
        try:
            if torch.cuda.is_available():
                # 测试CUDA是否真的可用
                test_tensor = torch.tensor([1.0]).cuda()
                device = 'cuda'
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                logging.info(f"CUDA可用，使用GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logging.info("CUDA不可用，使用CPU")
        except Exception as e:
            logging.warning(f"CUDA初始化失败: {e}")
            device = 'cpu'
            logging.info("回退到CPU训练")
    else:
        device = args.device
        if device == 'cuda':
            try:
                if not torch.cuda.is_available():
                    logging.warning("CUDA不可用，强制使用CPU")
                    device = 'cpu'
            except Exception as e:
                logging.warning(f"CUDA检查失败: {e}")
                device = 'cpu'
    
    logging.info(f"使用设备: {device}")
    
    # 根据设备调整批次大小
    if device == 'cpu':
        # CPU训练时减小批次大小
        original_batch_size = args.batch_size
        args.batch_size = min(args.batch_size, 2)
        if original_batch_size != args.batch_size:
            logging.info(f"CPU训练，调整批次大小: {original_batch_size} -> {args.batch_size}")
        
        # CPU训练时减少工作进程数
        original_num_workers = args.num_workers
        args.num_workers = min(args.num_workers, 2)
        if original_num_workers != args.num_workers:
            logging.info(f"CPU训练，调整工作进程数: {original_num_workers} -> {args.num_workers}")
    
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == 'cuda',
        collate_fn=CFruitDataLoader.collate_fn,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == 'cuda',
        collate_fn=CFruitDataLoader.collate_fn,
        persistent_workers=True
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
            lr=optimizer_config.get('lr', 0.0001),
            weight_decay=optimizer_config.get('weight_decay', 0.0005),
            betas=optimizer_config.get('betas', [0.9, 0.999])
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.get('lr', 0.0001),
            weight_decay=optimizer_config.get('weight_decay', 0.0005)
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
        save_interval=5
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