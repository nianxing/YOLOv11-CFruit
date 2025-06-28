import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EIoULoss(nn.Module):
    """EIoU损失函数"""
    
    def __init__(self, reduction='mean'):
        super(EIoULoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: 预测边界框 [N, 4] (x1, y1, x2, y2)
            target: 真实边界框 [N, 4] (x1, y1, x2, y2)
            weight: 权重 [N]
        Returns:
            loss: EIoU损失
        """
        # 计算IoU
        iou = self._box_iou(pred, target)
        
        # 计算中心点距离
        pred_center = (pred[:, :2] + pred[:, 2:]) / 2
        target_center = (target[:, :2] + target[:, 2:]) / 2
        center_dist = torch.sum((pred_center - target_center) ** 2, dim=1)
        
        # 计算外接矩形
        pred_wh = pred[:, 2:] - pred[:, :2]
        target_wh = target[:, 2:] - target[:, :2]
        
        # 外接矩形的左上角和右下角
        enclose_left = torch.min(pred[:, 0], target[:, 0])
        enclose_top = torch.min(pred[:, 1], target[:, 1])
        enclose_right = torch.max(pred[:, 2], target[:, 2])
        enclose_bottom = torch.max(pred[:, 3], target[:, 3])
        
        enclose_wh = torch.stack([enclose_right - enclose_left, enclose_bottom - enclose_top], dim=1)
        enclose_dist = torch.sum(enclose_wh ** 2, dim=1)
        
        # 计算长宽比一致性
        pred_ratio = pred_wh[:, 0] / (pred_wh[:, 1] + 1e-6)
        target_ratio = target_wh[:, 0] / (target_wh[:, 1] + 1e-6)
        ratio_loss = torch.sum((pred_ratio - target_ratio) ** 2, dim=0)
        
        # EIoU损失
        eiou_loss = 1 - iou + center_dist / enclose_dist + ratio_loss
        
        if weight is not None:
            eiou_loss = eiou_loss * weight
            
        if self.reduction == 'mean':
            return eiou_loss.mean()
        elif self.reduction == 'sum':
            return eiou_loss.sum()
        else:
            return eiou_loss
    
    def _box_iou(self, boxes1, boxes2):
        """计算IoU"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        union = area1 + area2 - inter
        iou = inter / (union + 1e-6)
        
        return iou


class FocalLoss(nn.Module):
    """Focal Loss for 分类损失"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: 预测概率 [N, C]
            target: 真实标签 [N]
            weight: 权重 [N]
        Returns:
            loss: Focal Loss
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if weight is not None:
            focal_loss = focal_loss * weight
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DFLoss(nn.Module):
    """分布焦点损失"""
    
    def __init__(self, reg_max=16, reduction='mean'):
        super(DFLoss, self).__init__()
        self.reg_max = reg_max
        self.reduction = reduction
        
    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: 预测分布 [N, 4*reg_max]
            target: 真实值 [N, 4]
            weight: 权重 [N]
        Returns:
            loss: DFL损失
        """
        pred = pred.view(-1, 4, self.reg_max)
        target = target.view(-1, 4)
        
        # 计算目标分布
        target_dist = self._target_distribution(target)
        
        # 计算KL散度
        loss = F.kl_div(
            F.log_softmax(pred, dim=-1),
            target_dist,
            reduction='none'
        ).sum(dim=-1)
        
        if weight is not None:
            loss = loss * weight
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _target_distribution(self, target):
        """计算目标分布"""
        # target: [N, 4]
        target_dist = torch.zeros_like(target.unsqueeze(-1).expand(-1, -1, self.reg_max))
        
        # 计算每个值的分布
        for i in range(4):
            target_i = target[:, i]
            target_i = torch.clamp(target_i, 0, self.reg_max - 1)
            
            # 计算左右两个整数的权重
            left = torch.floor(target_i).long()
            right = torch.clamp(left + 1, 0, self.reg_max - 1)
            
            # 计算权重
            weight_right = target_i - left.float()
            weight_left = 1 - weight_right
            
            # 分配权重
            target_dist[:, i].scatter_add_(-1, left.unsqueeze(-1), weight_left)
            target_dist[:, i].scatter_add_(-1, right.unsqueeze(-1), weight_right)
            
        return target_dist


class YOLOv11Loss(nn.Module):
    """YOLOv11总损失函数 - 改进版本"""
    
    def __init__(self, num_classes=1, reg_max=16, loss_weights=None):
        super(YOLOv11Loss, self).__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # 默认损失权重 - YOLOv11优化
        if loss_weights is None:
            loss_weights = {
                'cls': 0.3,  # 降低分类损失权重
                'box': 8.0,  # 增加边界框损失权重
                'dfl': 1.5
            }
        self.loss_weights = loss_weights
        
        # 损失函数
        self.cls_loss = FocalLoss(alpha=1, gamma=2)
        self.box_loss = EIoULoss()
        self.dfl_loss = DFLoss(reg_max)
        
    def forward(self, cls_preds, reg_preds, targets):
        """
        Args:
            cls_preds: 分类预测列表 [P3, P4, P5, P6]
            reg_preds: 回归预测列表 [P3, P4, P5, P6]
            targets: 目标张量 [B, N, 5] (class_id, x1, y1, x2, y2)
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        # 初始化损失为tensor，确保梯度可以传播
        device = cls_preds[0].device
        total_cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_box_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_dfl_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        batch_size = targets.shape[0]
        num_valid_batches = 0
        
        # 处理每个批次
        for b in range(batch_size):
            batch_targets = targets[b]  # [N, 5]
            
            # 过滤掉填充的零标签
            valid_targets = batch_targets[batch_targets[:, 0] >= 0]  # 假设-1表示填充
            
            if len(valid_targets) == 0:
                continue
            
            num_valid_batches += 1
            
            # 处理每个特征层
            for i, (cls_pred, reg_pred) in enumerate(zip(cls_preds, reg_preds)):
                # 获取当前特征层的预测
                cls_pred_b = cls_pred[b]  # [num_classes, H, W]
                reg_pred_b = reg_pred[b]  # [4*reg_max, H, W]
                
                # 获取特征层的空间尺寸
                _, _, height, width = cls_pred_b.shape
                
                # 简化的标签分配 - 为每个目标分配最近的网格点
                if len(valid_targets) > 0:
                    # 计算网格坐标
                    grid_y, grid_x = torch.meshgrid(
                        torch.arange(height, device=device),
                        torch.arange(width, device=device),
                        indexing='ij'
                    )
                    
                    # 为每个目标创建标签
                    for target in valid_targets:
                        class_id = int(target[0])
                        x1, y1, x2, y2 = target[1:5]
                        
                        # 计算中心点
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # 找到最近的网格点
                        grid_x_norm = center_x / (cls_pred_b.shape[3] * 8 * (2 ** i))  # 8是基础步长
                        grid_y_norm = center_y / (cls_pred_b.shape[2] * 8 * (2 ** i))
                        
                        grid_x_idx = int(grid_x_norm * width)
                        grid_y_idx = int(grid_y_norm * height)
                        
                        # 确保索引在有效范围内
                        grid_x_idx = max(0, min(width - 1, grid_x_idx))
                        grid_y_idx = max(0, min(height - 1, grid_y_idx))
                        
                        # 创建分类标签
                        cls_target = torch.zeros(height, width, dtype=torch.long, device=device)
                        cls_target[grid_y_idx, grid_x_idx] = class_id
                        
                        # 分类损失
                        cls_loss = F.cross_entropy(
                            cls_pred_b.view(-1, self.num_classes),
                            cls_target.view(-1)
                        )
                        total_cls_loss = total_cls_loss + cls_loss
                        
                        # 回归损失 - 使用L1损失
                        # 计算目标回归值
                        target_reg = torch.zeros(4, device=device)
                        target_reg[0] = (center_x - grid_x_idx * 8 * (2 ** i)) / (8 * (2 ** i))  # 相对偏移
                        target_reg[1] = (center_y - grid_y_idx * 8 * (2 ** i)) / (8 * (2 ** i))
                        target_reg[2] = (x2 - x1) / (8 * (2 ** i))  # 宽度
                        target_reg[3] = (y2 - y1) / (8 * (2 ** i))  # 高度
                        
                        # 获取预测的回归值
                        pred_reg = reg_pred_b[:, grid_y_idx, grid_x_idx]  # [4*reg_max]
                        pred_reg = pred_reg.view(4, self.reg_max)
                        
                        # 回归损失
                        reg_loss = F.l1_loss(pred_reg.mean(dim=1), target_reg)
                        total_box_loss = total_box_loss + reg_loss
                        
                        # DFL损失
                        dfl_loss = F.l1_loss(pred_reg, target_reg.unsqueeze(1).expand(-1, self.reg_max))
                        total_dfl_loss = total_dfl_loss + dfl_loss
        
        # 如果没有有效批次，返回零损失tensor
        if num_valid_batches == 0:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss_dict = {
                'total_loss': total_loss,
                'cls_loss': total_cls_loss,
                'box_loss': total_box_loss,
                'dfl_loss': total_dfl_loss
            }
            return total_loss, loss_dict
        
        # 计算总损失
        total_loss = (
            self.loss_weights['cls'] * total_cls_loss +
            self.loss_weights['box'] * total_box_loss +
            self.loss_weights['dfl'] * total_dfl_loss
        )
        
        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': total_cls_loss,
            'box_loss': total_box_loss,
            'dfl_loss': total_dfl_loss
        }
        
        return total_loss, loss_dict


# 保持向后兼容性
class YOLOv8Loss(YOLOv11Loss):
    """向后兼容的YOLOv8损失函数"""
    
    def __init__(self, num_classes=1, reg_max=16, loss_weights=None):
        if loss_weights is None:
            loss_weights = {
                'cls': 0.5,
                'box': 7.5,
                'dfl': 1.5
            }
        super(YOLOv8Loss, self).__init__(num_classes, reg_max, loss_weights) 