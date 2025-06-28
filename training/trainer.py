"""
YOLOv8-CFruit 训练器
"""

import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional


class Trainer:
    """YOLOv8-CFruit 训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device: str,
        save_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        save_interval: int = 10,
        **kwargs
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.save_interval = save_interval
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        
        # 训练状态
        self.epoch = 0
        self.best_map = 0.0
        self.train_losses = []
        self.val_losses = []
        
        logging.info(f"训练器初始化完成，设备: {device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            targets = batch['labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 更新统计
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 记录到TensorBoard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), 
                                     self.epoch * num_batches + batch_idx)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]["lr"], 
                                     self.epoch * num_batches + batch_idx)
        
        return {'train_loss': total_loss / num_batches}
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # 更新统计
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                # 更新进度条
                pbar.set_postfix({'Val Loss': f'{avg_loss:.4f}'})
        
        val_loss = total_loss / num_batches
        
        # 记录到TensorBoard
        self.writer.add_scalar('Val/Loss', val_loss, self.epoch)
        
        return {'val_loss': val_loss}
    
    def save_checkpoint(self, is_best: bool = False, filename: str = None):
        """保存检查点"""
        if filename is None:
            filename = f'epoch_{self.epoch + 1}.pt'
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_map': self.best_map,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"检查点已保存: {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            logging.info(f"最佳模型已保存: {best_path}")
        
        # 保存最新模型
        latest_path = os.path.join(self.save_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_map = checkpoint.get('best_map', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logging.info(f"检查点已加载: {checkpoint_path}")
        logging.info(f"从第 {self.epoch + 1} 轮开始训练")
    
    def train(self, epochs: int, resume: bool = False):
        """训练模型"""
        logging.info(f"开始训练，总轮数: {epochs}")
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['train_loss'])
            
            # 验证
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['val_loss'])
            
            # 记录日志
            logging.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )
            
            # 保存检查点
            is_best = val_metrics['val_loss'] < min(self.val_losses[:-1]) if len(self.val_losses) > 1 else True
            self.save_checkpoint(is_best=is_best)
            
            # 定期保存
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(filename=f'epoch_{epoch + 1}.pt')
        
        logging.info("训练完成！")
        self.writer.close()
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                images = batch['images'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        logging.info(f"评估结果 - 平均损失: {avg_loss:.4f}")
        
        return {'val_loss': avg_loss} 