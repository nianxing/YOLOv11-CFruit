import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # 线性变换
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 输出投影
        output = self.w_o(attention_output)
        
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output, _ = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        x = self.positional_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x


class FeatureTransformer(nn.Module):
    """特征Transformer模块，用于处理2D特征图"""
    
    def __init__(self, in_channels, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(FeatureTransformer, self).__init__()
        
        self.in_channels = in_channels
        self.d_model = d_model
        
        # 特征投影
        self.feature_projection = nn.Conv2d(in_channels, d_model, 1)
        
        # Transformer编码器
        self.transformer = TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout)
        
        # 输出投影
        self.output_projection = nn.Conv2d(d_model, in_channels, 1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 投影到d_model维度
        x = self.feature_projection(x)
        
        # 重塑为序列形式 (batch_size, seq_len, d_model)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
        
        # Transformer处理
        x = self.transformer(x)
        
        # 重塑回2D特征图
        x = x.transpose(1, 2).view(batch_size, self.d_model, height, width)
        
        # 输出投影
        x = self.output_projection(x)
        
        return x 