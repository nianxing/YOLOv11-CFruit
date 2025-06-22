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
            cls_preds: 分类预测列表 [P3, P4, P5]
            reg_preds: 回归预测列表 [P3, P4, P5]
            targets: 目标列表 [P3, P4, P5]
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        total_cls_loss = 0
        total_box_loss = 0
        total_dfl_loss = 0
        
        for i, (cls_pred, reg_pred, target) in enumerate(zip(cls_preds, reg_preds, targets)):
            if len(target) == 0:
                continue
                
            # 解析目标
            target_cls = target[:, 0].long()
            target_box = target[:, 1:5]
            target_reg = target[:, 5:9]
            
            # 分类损失
            cls_loss = self.cls_loss(cls_pred, target_cls)
            total_cls_loss += cls_loss
            
            # 边界框损失
            box_loss = self.box_loss(reg_pred, target_box)
            total_box_loss += box_loss
            
            # DFL损失
            dfl_loss = self.dfl_loss(reg_pred, target_reg)
            total_dfl_loss += dfl_loss
        
        # 计算总损失 - YOLOv11优化
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