import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleYOLOLoss(nn.Module):
    """简化的YOLO损失函数，用于测试"""
    
    def __init__(self, num_classes=1, reg_max=16):
        super(SimpleYOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        
    def forward(self, cls_preds, reg_preds, targets):
        """
        Args:
            cls_preds: 分类预测列表 [P3, P4, P5, P6] - 每个元素是 [B, num_classes, H, W]
            reg_preds: 回归预测列表 [P3, P4, P5, P6] - 每个元素是 [B, 4*reg_max, H, W]
            targets: 目标张量 [B, N, 5] (class_id, x1, y1, x2, y2)
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        device = cls_preds[0].device
        
        # 简化的损失计算
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        box_loss = torch.tensor(0.0, device=device, requires_grad=True)
        dfl_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        batch_size = targets.shape[0]
        num_valid_batches = 0
        
        for b in range(batch_size):
            batch_targets = targets[b]
            valid_targets = batch_targets[batch_targets[:, 0] >= 0]
            
            if len(valid_targets) == 0:
                continue
                
            num_valid_batches += 1
            
            # 处理每个特征层
            for i, (cls_pred, reg_pred) in enumerate(zip(cls_preds, reg_preds)):
                cls_pred_b = cls_pred[b]  # [num_classes, H, W]
                reg_pred_b = reg_pred[b]  # [4*reg_max, H, W]
                
                # 获取特征层的空间尺寸
                _, height, width = cls_pred_b.shape
                
                # 简化的损失计算
                if len(valid_targets) > 0:
                    for target in valid_targets:
                        class_id = int(target[0])
                        x1, y1, x2, y2 = target[1:5]
                        
                        # 计算中心点
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # 找到最近的网格点
                        grid_x_norm = center_x / (cls_pred_b.shape[2] * 8 * (2 ** i))
                        grid_y_norm = center_y / (cls_pred_b.shape[1] * 8 * (2 ** i))
                        
                        grid_x_idx = int(grid_x_norm * width)
                        grid_y_idx = int(grid_y_norm * height)
                        
                        # 确保索引在有效范围内
                        grid_x_idx = max(0, min(width - 1, grid_x_idx))
                        grid_y_idx = max(0, min(height - 1, grid_y_idx))
                        
                        # 分类损失
                        cls_target = torch.zeros(height, width, dtype=torch.long, device=device)
                        cls_target[grid_y_idx, grid_x_idx] = class_id
                        
                        cls_loss_batch = F.cross_entropy(
                            cls_pred_b.view(-1, self.num_classes),
                            cls_target.view(-1)
                        )
                        cls_loss = cls_loss + cls_loss_batch
                        
                        # 回归损失
                        target_reg = torch.zeros(4, device=device)
                        target_reg[0] = (center_x - grid_x_idx * 8 * (2 ** i)) / (8 * (2 ** i))
                        target_reg[1] = (center_y - grid_y_idx * 8 * (2 ** i)) / (8 * (2 ** i))
                        target_reg[2] = (x2 - x1) / (8 * (2 ** i))
                        target_reg[3] = (y2 - y1) / (8 * (2 ** i))
                        
                        pred_reg = reg_pred_b[:, grid_y_idx, grid_x_idx]
                        pred_reg = pred_reg.view(4, self.reg_max)
                        
                        reg_loss_batch = F.l1_loss(pred_reg.mean(dim=1), target_reg)
                        box_loss = box_loss + reg_loss_batch
                        
                        dfl_loss_batch = F.l1_loss(pred_reg, target_reg.unsqueeze(1).expand(-1, self.reg_max))
                        dfl_loss = dfl_loss + dfl_loss_batch
        
        # 如果没有有效批次，返回零损失
        if num_valid_batches == 0:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss_dict = {
                'total_loss': total_loss,
                'cls_loss': cls_loss,
                'box_loss': box_loss,
                'dfl_loss': dfl_loss
            }
            return total_loss, loss_dict
        
        # 计算总损失
        total_loss = 0.3 * cls_loss + 8.0 * box_loss + 1.5 * dfl_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'dfl_loss': dfl_loss
        }
        
        return total_loss, loss_dict 