import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBNSiLU(nn.Module):
    """标准卷积块：Conv + BN + SiLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 padding=0, groups=1, bias=False):
        super(ConvBNSiLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DFL(nn.Module):
    """分布焦点损失模块"""
    
    def __init__(self, c1=16):
        super(DFL, self).__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        
    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class AnchorFreeHead(nn.Module):
    """无锚点检测头部"""
    
    def __init__(self, in_channels_list, num_classes=1, reg_max=16, stride_list=[8, 16, 32]):
        super(AnchorFreeHead, self).__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.stride_list = stride_list
        
        # 分类分支
        self.cls_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        
        # 回归分支
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        for i, in_channels in enumerate(in_channels_list):
            # 分类分支
            cls_conv = nn.Sequential(
                ConvBNSiLU(in_channels, in_channels, 3, 1, 1),
                ConvBNSiLU(in_channels, in_channels, 3, 1, 1)
            )
            self.cls_convs.append(cls_conv)
            
            cls_pred = nn.Conv2d(in_channels, num_classes, 1)
            self.cls_preds.append(cls_pred)
            
            # 回归分支
            reg_conv = nn.Sequential(
                ConvBNSiLU(in_channels, in_channels, 3, 1, 1),
                ConvBNSiLU(in_channels, in_channels, 3, 1, 1)
            )
            self.reg_convs.append(reg_conv)
            
            reg_pred = nn.Conv2d(in_channels, 4 * reg_max, 1)
            self.reg_preds.append(reg_pred)
        
        # DFL模块
        self.dfl = DFL(reg_max)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, features):
        """
        Args:
            features: 颈部网络输出的多尺度特征 [P3, P4, P5]
        Returns:
            cls_outputs: 分类预测 [B, num_classes, H, W]
            reg_outputs: 回归预测 [B, 4, H, W]
        """
        cls_outputs = []
        reg_outputs = []
        
        for i, feature in enumerate(features):
            # 分类分支
            cls_feat = self.cls_convs[i](feature)
            cls_output = self.cls_preds[i](cls_feat)
            cls_outputs.append(cls_output)
            
            # 回归分支
            reg_feat = self.reg_convs[i](feature)
            reg_output = self.reg_preds[i](reg_feat)
            reg_outputs.append(reg_output)
        
        return cls_outputs, reg_outputs
    
    def get_bboxes(self, cls_outputs, reg_outputs, img_shape, conf_thresh=0.25, nms_thresh=0.45):
        """
        从网络输出解码边界框
        Args:
            cls_outputs: 分类预测列表
            reg_outputs: 回归预测列表
            img_shape: 输入图像形状 (H, W)
            conf_thresh: 置信度阈值
            nms_thresh: NMS阈值
        Returns:
            bboxes: 检测到的边界框 [N, 6] (x1, y1, x2, y2, conf, cls)
        """
        device = cls_outputs[0].device
        bboxes = []
        
        for i, (cls_output, reg_output) in enumerate(zip(cls_outputs, reg_outputs)):
            stride = self.stride_list[i]
            
            # 获取网格坐标
            batch_size, _, height, width = cls_output.shape
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing='ij'
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
            
            # 解码分类预测
            cls_scores = torch.sigmoid(cls_output)  # [B, num_classes, H, W]
            
            # 解码回归预测
            reg_output = reg_output.view(batch_size, 4, self.reg_max, height, width)
            reg_output = self.dfl(reg_output)  # [B, 4, H, W]
            
            # 转换为边界框坐标
            pred_bboxes = self._decode_bboxes(reg_output, grid_xy, stride)
            
            # 过滤低置信度预测
            for b in range(batch_size):
                scores = cls_scores[b].view(self.num_classes, -1).t()  # [H*W, num_classes]
                bboxes_b = pred_bboxes[b].view(4, -1).t()  # [H*W, 4]
                
                # 应用置信度阈值
                keep = scores.max(dim=1)[0] > conf_thresh
                if keep.sum() == 0:
                    continue
                    
                scores = scores[keep]
                bboxes_b = bboxes_b[keep]
                
                # 执行NMS
                keep_indices = self._nms(bboxes_b, scores.max(dim=1)[0], nms_thresh)
                
                if len(keep_indices) > 0:
                    bboxes_b = bboxes_b[keep_indices]
                    scores = scores[keep_indices]
                    
                    # 添加批次信息
                    batch_bboxes = torch.cat([
                        bboxes_b,
                        scores.max(dim=1, keepdim=True)[0],
                        scores.argmax(dim=1, keepdim=True).float()
                    ], dim=1)
                    
                    bboxes.append(batch_bboxes)
        
        if len(bboxes) > 0:
            bboxes = torch.cat(bboxes, dim=0)
        else:
            bboxes = torch.empty((0, 6), device=device)
            
        return bboxes
    
    def _decode_bboxes(self, reg_output, grid_xy, stride):
        """解码回归输出为边界框坐标"""
        # reg_output: [B, 4, H, W]
        # grid_xy: [H, W, 2]
        
        batch_size, _, height, width = reg_output.shape
        
        # 计算中心点坐标
        center_x = (grid_xy[..., 0] + 0.5) * stride
        center_y = (grid_xy[..., 1] + 0.5) * stride
        
        # 解码边界框
        x1 = center_x - reg_output[:, 0] * stride
        y1 = center_y - reg_output[:, 1] * stride
        x2 = center_x + reg_output[:, 2] * stride
        y2 = center_y + reg_output[:, 3] * stride
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _nms(self, bboxes, scores, thresh):
        """非极大值抑制"""
        if len(bboxes) == 0:
            return []
            
        # 计算IoU矩阵
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            i = order[0]
            keep.append(i)
            
            # 计算IoU
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= thresh).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
            
        return keep 