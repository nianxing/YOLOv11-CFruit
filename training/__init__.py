"""
Training module for YOLOv8-CFruit
"""

from .trainer import Trainer
from .scheduler import CosineAnnealingWarmupRestarts

__all__ = ['Trainer', 'CosineAnnealingWarmupRestarts'] 