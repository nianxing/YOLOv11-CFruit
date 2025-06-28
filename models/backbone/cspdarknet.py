import torch
import torch.nn as nn
from .cbam import CBAM, CBAMBlock
from .c2f import C2f, C2fWithAttention


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


class CSPBlock(nn.Module):
    """CSP (Cross Stage Partial) 块 - 兼容性保留"""
    
    def __init__(self, in_channels, out_channels, num_blocks=1, 
                 shortcut=True, expansion=0.5, cbam=True, cbam_ratio=16):
        super(CSPBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.conv3 = ConvBNSiLU(hidden_channels, hidden_channels, 1, 1)
        self.conv4 = ConvBNSiLU(hidden_channels, hidden_channels, 1, 1)
        
        # 主干路径
        self.m = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, cbam, cbam_ratio)
            for _ in range(num_blocks)
        ])
        
        self.conv5 = ConvBNSiLU(hidden_channels, hidden_channels, 1, 1)
        self.conv6 = ConvBNSiLU(hidden_channels, hidden_channels, 1, 1)
        
        # 输出卷积
        self.conv7 = ConvBNSiLU(2 * hidden_channels, out_channels, 1, 1)
        
    def forward(self, x):
        x1 = self.conv3(self.conv1(x))
        x2 = self.conv4(self.conv2(x))
        x1 = self.conv5(self.m(x1))
        x = torch.cat((x1, x2), dim=1)
        return self.conv7(x)


class Bottleneck(nn.Module):
    """标准瓶颈块"""
    
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


class CSPDarknetV11(nn.Module):
    """YOLOv11版本的CSPDarknet主干网络"""
    
    def __init__(self, base_channels=64, base_depth=3, cbam=True, cbam_ratio=16, use_c2f=True):
        super(CSPDarknetV11, self).__init__()
        
        self.use_c2f = use_c2f
        
        # 初始卷积层
        self.conv1 = ConvBNSiLU(3, base_channels, 3, 2, 1)
        self.conv2 = ConvBNSiLU(base_channels, base_channels * 2, 3, 2, 1)
        
        # 第一阶段 - 使用C2f或CSP
        if use_c2f:
            self.stage1 = C2fWithAttention(
                base_channels * 2, base_channels * 2, 
                base_depth, True, 1.0, cbam, cbam_ratio
            )
        else:
            self.stage1 = CSPBlock(
                base_channels * 2, base_channels * 2, 
                base_depth, True, 1.0, cbam, cbam_ratio
            )
        self.conv3 = ConvBNSiLU(base_channels * 2, base_channels * 4, 3, 2, 1)
        
        # 第二阶段
        if use_c2f:
            self.stage2 = C2fWithAttention(
                base_channels * 4, base_channels * 4, 
                base_depth * 2, True, 1.0, cbam, cbam_ratio
            )
        else:
            self.stage2 = CSPBlock(
                base_channels * 4, base_channels * 4, 
                base_depth * 2, True, 1.0, cbam, cbam_ratio
            )
        self.conv4 = ConvBNSiLU(base_channels * 4, base_channels * 8, 3, 2, 1)
        
        # 第三阶段
        if use_c2f:
            self.stage3 = C2fWithAttention(
                base_channels * 8, base_channels * 8, 
                base_depth * 3, True, 1.0, cbam, cbam_ratio
            )
        else:
            self.stage3 = CSPBlock(
                base_channels * 8, base_channels * 8, 
                base_depth * 3, True, 1.0, cbam, cbam_ratio
            )
        self.conv5 = ConvBNSiLU(base_channels * 8, base_channels * 16, 3, 2, 1)
        
        # 第四阶段
        if use_c2f:
            self.stage4 = C2fWithAttention(
                base_channels * 16, base_channels * 16, 
                base_depth, False, 1.0, cbam, cbam_ratio
            )
        else:
            self.stage4 = CSPBlock(
                base_channels * 16, base_channels * 16, 
                base_depth, False, 1.0, cbam, cbam_ratio
            )
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 16, 1000)
        
    def forward(self, x):
        # 特征提取
        x1 = self.conv1(x)      # 1/2
        x2 = self.conv2(x1)     # 1/4
        x3 = self.stage1(x2)    # 1/4
        x4 = self.conv3(x3)     # 1/8
        x5 = self.stage2(x4)    # 1/8
        x6 = self.conv4(x5)     # 1/16
        x7 = self.stage3(x6)    # 1/16
        x8 = self.conv5(x7)     # 1/32
        x9 = self.stage4(x8)    # 1/32
        
        # 返回多尺度特征用于目标检测
        return [x3, x5, x7, x9]


# 保持向后兼容性
class CSPDarknet(CSPDarknetV11):
    """向后兼容的CSPDarknet类"""
    
    def __init__(self, base_channels=64, base_depth=3, cbam=True, cbam_ratio=16):
        super(CSPDarknet, self).__init__(base_channels, base_depth, cbam, cbam_ratio, use_c2f=False) 