"""
学习率调度器
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    余弦退火学习率调度器，支持预热和重启
    
    Args:
        optimizer: 优化器
        first_cycle_steps: 第一个周期的步数
        cycle_mult: 周期倍数
        max_lr: 最大学习率
        min_lr: 最小学习率
        warmup_steps: 预热步数
        gamma: 学习率衰减因子
        last_epoch: 上一个epoch
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        # 先设置所有属性，避免在父类初始化时调用get_lr()时出错
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        
        # 然后调用父类初始化
        super().__init__(optimizer, last_epoch)
        
        # 最后设置step_count
        self.step_count = self.last_epoch + 1
    
    def get_lr(self):
        """获取当前学习率"""
        if self.step_count <= self.warmup_steps:
            # 预热阶段
            return [(self.max_lr - base_lr) * self.step_count / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        
        # 余弦退火阶段
        if self.step_count - self.warmup_steps >= self.cur_cycle_steps:
            # 周期结束，重启
            self.cycle += 1
            self.step_count = self.warmup_steps
            self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        
        # 计算余弦退火
        cos_out = math.cos(math.pi * (self.step_count - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps))
        cos_out = (cos_out + 1) / 2  # 归一化到 [0, 1]
        
        return [base_lr + (self.max_lr - base_lr) * cos_out for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_count = self.last_epoch + 1
            self.last_epoch = epoch
        else:
            self.step_count = epoch
            self.last_epoch = epoch
        
        # 更新最大学习率
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        
        # 更新学习率
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr 