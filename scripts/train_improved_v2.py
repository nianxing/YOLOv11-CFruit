#!/usr/bin/env python3
"""
改进的YOLOv11-CFruit训练脚本
包含更好的学习率调度、数据增强、早停等策略
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov11_cfruit import YOLOv11CFruit
from data.dataset import CFruitDataset, CFruitDataLoader
from utils.transforms import get_transforms
from utils.losses import YOLOv11Loss


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
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


class ImprovedTrainer:
    """改进的训练器"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        if config['training']['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config['training']['scheduler_t0'],
                T_mult=config['training']['scheduler_t_mult'],
                eta_min=config['training']['min_lr']
            )
        elif config['training']['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=config['training']['min_lr']
            )
        else:
            self.scheduler = None
        
        # 损失函数
        self.criterion = YOLOv11Loss(
            num_classes=config['model']['num_classes'],
            reg_max=config['model'].get('reg_max', 16)
        )
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            min_delta=config['training']['early_stopping_min_delta']
        )
        
        # 训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # 梯度累积
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        accumulation_steps = self.gradient_accumulation_steps
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 混合精度训练
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    cls_outputs, reg_outputs = self.model(images)
                    loss, loss_dict = self.criterion(cls_outputs, reg_outputs, labels)
                    loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                cls_outputs, reg_outputs = self.model(images)
                loss, loss_dict = self.criterion(cls_outputs, reg_outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            
            # 梯度累积步数达到或最后一个batch时，执行step
            is_accum_step = (batch_idx + 1) % accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches
            if is_accum_step or is_last_batch:
                if self.config['training'].get('gradient_clip', 0) > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            if batch_idx % 10 == 0:
                logging.info(f'Epoch {self.epoch+1}, Batch {batch_idx}/{num_batches}, '
                           f'Loss: {loss.item() * accumulation_steps:.4f}, '
                           f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        cls_outputs, reg_outputs = self.model(images)
                        loss, loss_dict = self.criterion(cls_outputs, reg_outputs, labels)
                else:
                    cls_outputs, reg_outputs = self.model(images)
                    loss, loss_dict = self.criterion(cls_outputs, reg_outputs, labels)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, is_best=False, filename=None):
        """保存检查点"""
        if filename is None:
            filename = f'epoch_{self.epoch + 1}.pt'
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.config['training']['save_dir'], filename)
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"检查点已保存: {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.config['training']['save_dir'], 'best.pt')
            torch.save(checkpoint, best_path)
            logging.info(f"最佳模型已保存: {best_path}")
    
    def train(self, num_epochs):
        """训练模型"""
        logging.info(f"开始改进训练，总轮数: {num_epochs}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录日志
            logging.info(f'Epoch {epoch+1}/{num_epochs} - '
                        f'Train Loss: {train_loss:.4f}, '
                        f'Val Loss: {val_loss:.4f}, '
                        f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # 保存检查点
            self.save_checkpoint(is_best=is_best)
            
            # 早停检查
            if self.early_stopping(val_loss, self.model):
                logging.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        logging.info("训练完成！")


def main():
    parser = argparse.ArgumentParser(description='改进的YOLOv11-CFruit训练')
    parser.add_argument('--config', type=str, default='configs/model/yolov11_cfruit_improved.yaml',
                       help='配置文件路径')
    parser.add_argument('--data-config', type=str, default='configs/data/cfruit.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto, cpu, cuda)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='保存目录')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='日志目录')
    parser.add_argument('--resume', type=str, default='',
                       help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logging.info(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.log_dir, f'train_improved_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.data_config, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 更新配置
    model_config['training']['batch_size'] = args.batch_size
    model_config['training']['save_dir'] = args.save_dir
    
    # 创建模型
    model = YOLOv11CFruit(model_config)
    logging.info(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"模型大小: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024:.2f} MB")
    
    # 创建数据变换
    train_transform = get_transforms(
        img_size=model_config['model']['input_size'],
        augment=True,
        mixup=model_config['training']['mixup'],
        mosaic=model_config['training']['mosaic']
    )
    
    val_transform = get_transforms(
        img_size=model_config['model']['input_size'],
        augment=False
    )
    
    # 创建数据加载器
    train_loader = CFruitDataLoader.create_dataloader(
        img_dir=data_config['dataset']['train'],
        label_dir=data_config['dataset']['train_labels'],
        batch_size=args.batch_size,
        num_workers=4,
        transform=train_transform,
        img_size=model_config['model']['input_size'],
        shuffle=True
    )
    
    val_loader = CFruitDataLoader.create_dataloader(
        img_dir=data_config['dataset']['val'],
        label_dir=data_config['dataset']['val_labels'],
        batch_size=args.batch_size,
        num_workers=4,
        transform=val_transform,
        img_size=model_config['model']['input_size'],
        shuffle=False
    )
    
    logging.info(f"训练集大小: {len(train_loader.dataset)}")
    logging.info(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建训练器
    trainer = ImprovedTrainer(model, train_loader, val_loader, device, model_config)
    
    # 恢复训练
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        trainer.learning_rates = checkpoint.get('learning_rates', [])
        logging.info(f"从检查点恢复训练: {args.resume}")
    
    # 开始训练
    trainer.train(args.epochs)


if __name__ == '__main__':
    main() 