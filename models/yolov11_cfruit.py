import torch
import torch.nn as nn
from .backbone.cspdarknet import CSPDarknetV11
from .neck.panet import PANetWithTransformer
from .head.anchor_free import AnchorFreeHead


class YOLOv11CFruit(nn.Module):
    """YOLOv11-CFruit 主模型"""
    
    def __init__(self, config):
        super(YOLOv11CFruit, self).__init__()
        
        # 解析配置
        backbone_config = config['model']['backbone']
        neck_config = config['model']['neck']
        head_config = config['model']['head']
        
        # 主干网络 - 使用YOLOv11的C2f模块
        self.backbone = CSPDarknetV11(
            base_channels=64,
            base_depth=3,
            cbam=backbone_config.get('cbam', True),
            cbam_ratio=backbone_config.get('cbam_ratio', 16),
            use_c2f=backbone_config.get('use_c2f', True)  # YOLOv11特性
        )
        
        # 特征通道数
        backbone_channels = [128, 256, 512, 1024]  # P3, P4, P5, P6
        neck_channels = [256, 256, 256]  # 统一颈部特征通道数
        
        # 颈部网络 - 改进的PANet
        self.neck = PANetWithTransformer(
            in_channels_list=backbone_channels,
            out_channels_list=neck_channels,
            transformer_heads=neck_config.get('transformer_heads', 8),
            transformer_dim=neck_config.get('transformer_dim', 256),
            transformer_layers=neck_config.get('transformer_layers', 2)
        )
        
        # 检测头部 - 改进的无锚点检测
        self.head = AnchorFreeHead(
            in_channels_list=neck_channels,
            num_classes=head_config.get('num_classes', 1),
            reg_max=head_config.get('reg_max', 16),
            stride_list=[8, 16, 32]
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重 - YOLOv11改进的初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # YOLOv11使用改进的权重初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            cls_outputs: 分类预测列表
            reg_outputs: 回归预测列表
        """
        # 主干网络特征提取
        backbone_features = self.backbone(x)
        
        # 颈部网络特征融合
        neck_features = self.neck(backbone_features)
        
        # 检测头部预测
        cls_outputs, reg_outputs = self.head(neck_features)
        
        return cls_outputs, reg_outputs
    
    def inference(self, x, conf_thresh=0.25, nms_thresh=0.45):
        """
        推理模式 - YOLOv11改进的推理
        Args:
            x: 输入图像 [B, 3, H, W]
            conf_thresh: 置信度阈值
            nms_thresh: NMS阈值
        Returns:
            bboxes: 检测结果 [N, 6] (x1, y1, x2, y2, conf, cls)
        """
        self.eval()
        with torch.no_grad():
            cls_outputs, reg_outputs = self.forward(x)
            bboxes = self.head.get_bboxes(
                cls_outputs, reg_outputs, 
                x.shape[2:], conf_thresh, nms_thresh
            )
        return bboxes
    
    @classmethod
    def from_pretrained(cls, checkpoint_path, config=None):
        """
        从预训练权重加载模型
        Args:
            checkpoint_path: 权重文件路径
            config: 模型配置
        Returns:
            model: 加载权重的模型
        """
        if config is None:
            # 默认配置
            config = {
                'model': {
                    'backbone': {
                        'cbam': True,
                        'cbam_ratio': 16,
                        'use_c2f': True  # YOLOv11特性
                    },
                    'neck': {
                        'transformer_heads': 8,
                        'transformer_dim': 256,
                        'transformer_layers': 2
                    },
                    'head': {
                        'num_classes': 1,
                        'reg_max': 16
                    }
                }
            }
        
        model = cls(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
            
        return model
    
    def save_checkpoint(self, save_path, optimizer=None, epoch=None, metrics=None):
        """
        保存模型检查点
        Args:
            save_path: 保存路径
            optimizer: 优化器状态
            epoch: 当前轮数
            metrics: 评估指标
        """
        checkpoint = {
            'model': self.state_dict(),
            'config': {
                'model': {
                    'backbone': {
                        'cbam': True,
                        'cbam_ratio': 16,
                        'use_c2f': True
                    },
                    'neck': {
                        'transformer_heads': 8,
                        'transformer_dim': 256,
                        'transformer_layers': 2
                    },
                    'head': {
                        'num_classes': 1,
                        'reg_max': 16
                    }
                }
            }
        }
        
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, save_path)
        
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'backbone': 'CSPDarknetV11 with C2f and CBAM',
            'neck': 'PANet with Transformer',
            'head': 'Anchor-Free Detection Head',
            'version': 'YOLOv11'
        } 