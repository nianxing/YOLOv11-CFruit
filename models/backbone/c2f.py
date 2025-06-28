import torch
import torch.nn as nn
from .cbam import CBAM


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


class Bottleneck(nn.Module):
    """改进的瓶颈块"""
    
    def __init__(self, in_channels, out_channels, shortcut=True, 
                 expansion=0.5, cbam=True, cbam_ratio=16):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBNSiLU(hidden_channels, out_channels, 3, 1, 1)
        self.use_add = shortcut and in_channels == out_channels
        
        if cbam:
            self.cbam = CBAM(out_channels, cbam_ratio)
        else:
            self.cbam = nn.Identity()
        
    def forward(self, x):
        return self.cbam(x + self.cv2(self.cv1(x))) if self.use_add else self.cbam(self.cv2(self.cv1(x)))


class C2f(nn.Module):
    """YOLOv11的C2f模块 - 更高效的CSP模块"""
    
    def __init__(self, in_channels, out_channels, num_blocks=1, 
                 shortcut=True, expansion=0.5, cbam=True, cbam_ratio=16):
        super(C2f, self).__init__()
        hidden_channels = int(out_channels * expansion)
        
        # 输入投影
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        
        # 主干路径 - 使用更高效的实现
        self.m = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, cbam, cbam_ratio)
            for _ in range(num_blocks)
        ])
        
        # 输出投影
        self.cv3 = ConvBNSiLU(hidden_channels, hidden_channels, 1, 1)
        self.cv4 = ConvBNSiLU(hidden_channels, hidden_channels, 1, 1)
        
        # 最终输出 - 修正为3 * hidden_channels（因为连接了3个分支）
        self.cv5 = ConvBNSiLU(3 * hidden_channels, out_channels, 1, 1)
        
    def forward(self, x):
        # 分支1：直接路径
        y1 = self.cv2(x)
        
        # 分支2：主干路径
        y2 = self.cv1(x)
        y2 = self.m(y2)
        y2 = self.cv3(y2)
        
        # 分支3：残差路径
        y3 = self.cv4(x)
        
        # 融合所有路径
        return self.cv5(torch.cat([y1, y2, y3], dim=1))


class C2fWithAttention(nn.Module):
    """带注意力机制的C2f模块"""
    
    def __init__(self, in_channels, out_channels, num_blocks=1, 
                 shortcut=True, expansion=0.5, cbam=True, cbam_ratio=16):
        super(C2fWithAttention, self).__init__()
        
        self.c2f = C2f(in_channels, out_channels, num_blocks, shortcut, expansion, cbam, cbam_ratio)
        
        # 额外的注意力机制
        if cbam:
            self.attention = CBAM(out_channels, cbam_ratio)
        else:
            self.attention = nn.Identity()
        
    def forward(self, x):
        x = self.c2f(x)
        x = self.attention(x)
        return x 