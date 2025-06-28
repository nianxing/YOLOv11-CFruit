import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import FeatureTransformer


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


class SPPF(nn.Module):
    """空间金字塔池化快速版本"""
    
    def __init__(self, in_channels, out_channels, k=5):
        super(SPPF, self).__init__()
        c_ = in_channels // 2
        self.cv1 = ConvBNSiLU(in_channels, c_, 1, 1)
        self.cv2 = ConvBNSiLU(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class PANet(nn.Module):
    """PANet（路径聚合网络）颈部网络"""
    
    def __init__(self, in_channels_list, out_channels_list, transformer=True, 
                 transformer_heads=8, transformer_dim=256, transformer_layers=2):
        super(PANet, self).__init__()
        
        self.transformer = transformer
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        
        # 自底向上路径（FPN）
        self.fpn_convs = nn.ModuleList()
        self.fpn_upsamples = nn.ModuleList()
        
        for i in range(len(in_channels_list) - 1):
            # 特征融合卷积
            self.fpn_convs.append(
                ConvBNSiLU(in_channels_list[i], out_channels_list[i], 1, 1)
            )
            # 上采样
            self.fpn_upsamples.append(
                nn.Upsample(scale_factor=2, mode='nearest')
            )
        
        # 自顶向下路径（PAN）
        self.pan_convs = nn.ModuleList()
        self.pan_downsamples = nn.ModuleList()
        
        for i in range(len(in_channels_list) - 1):
            # 特征融合卷积 - 输入通道数是out_channels_list[i] * 2（FPN融合后）
            self.pan_convs.append(
                ConvBNSiLU(out_channels_list[i] * 2, out_channels_list[i], 3, 1, 1)
            )
            # 下采样 - 输入通道数是out_channels_list[i] * 2（FPN融合后）
            self.pan_downsamples.append(
                ConvBNSiLU(out_channels_list[i] * 2, out_channels_list[i], 3, 2, 1)
            )
        
        # SPPF模块
        self.sppf = SPPF(in_channels_list[-1], out_channels_list[-1])
        
        # Transformer模块
        if transformer:
            self.transformers = nn.ModuleList([
                FeatureTransformer(
                    out_channels_list[i] * 4,  # 由于FPN和PAN的特征融合，通道数翻4倍
                    transformer_dim, 
                    transformer_heads, 
                    transformer_layers,
                    transformer_dim * 4
                )
                for i in range(len(out_channels_list))
            ])
        else:
            self.transformers = nn.ModuleList([
                nn.Identity() for _ in range(len(out_channels_list))
            ])
        
        # 输出通道调整层 - 将特征融合后的通道数调整回期望的输出通道数
        self.output_convs = nn.ModuleList([
            ConvBNSiLU(out_channels_list[i] * 4, out_channels_list[i], 1, 1)
            for i in range(len(out_channels_list))
        ])
        
    def forward(self, features):
        """
        Args:
            features: 主干网络输出的多尺度特征 [P3, P4, P5, P6]
        Returns:
            enhanced_features: 增强后的多尺度特征
        """
        # 自底向上路径（FPN）
        fpn_features = []
        for i in range(len(features) - 1, -1, -1):
            if i == len(features) - 1:
                # 最深层特征
                x = self.sppf(features[i])
            else:
                # 上采样并融合
                upsampled = self.fpn_upsamples[i](fpn_features[-1])
                x = torch.cat([self.fpn_convs[i](features[i]), upsampled], dim=1)
            
            fpn_features.append(x)
        
        # 反转顺序
        fpn_features = fpn_features[::-1]
        
        # 自顶向下路径（PAN）
        pan_features = []
        for i in range(len(fpn_features)):
            if i == 0:
                # 最浅层特征
                x = fpn_features[i]
            else:
                # 下采样并融合
                downsampled = self.pan_downsamples[i-1](pan_features[-1])
                x = torch.cat([fpn_features[i], downsampled], dim=1)
            
            # 应用Transformer（如果启用）
            x = self.transformers[i](x)
            
            # 调整输出通道数
            x = self.output_convs[i](x)
            
            pan_features.append(x)
        
        return pan_features


class PANetWithTransformer(nn.Module):
    """集成Transformer的PANet"""
    
    def __init__(self, in_channels_list, out_channels_list, 
                 transformer_heads=8, transformer_dim=256, transformer_layers=2):
        super(PANetWithTransformer, self).__init__()
        
        self.panet = PANet(
            in_channels_list, out_channels_list, 
            True, transformer_heads, transformer_dim, transformer_layers
        )
        
    def forward(self, features):
        return self.panet(features) 