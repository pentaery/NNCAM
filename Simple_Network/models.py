"""
Neural network models for climate prediction
包含各种神经网络架构的定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleImprovedNet(nn.Module):
    """简单改进版网络 - 更简单的架构 + 更好的正则化"""
    
    def __init__(self, input_dim: int = 305, output_dim: int = 242, 
                 dropout1: float = 0.2, dropout2: float = 0.3):
        super(SimpleImprovedNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Xavier初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.leaky_relu(self.fc1(x)))
        x = self.dropout2(self.leaky_relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class ResidualNet(nn.Module):
    """残差网络 - 残差连接 + 更深层次"""
    
    def __init__(self, input_dim: int = 305, output_dim: int = 242, 
                 hidden_dim: int = 256, dropout_rate: float = 0.2):
        super(ResidualNet, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 残差块
        self.residual_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.residual_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        
        # 残差连接
        residual1 = self.residual_block1(x)
        x = self.relu(x + residual1)
        
        residual2 = self.residual_block2(x)
        x = self.relu(x + residual2)
        
        x = self.output_layer(x)
        return x


class AttentionNet(nn.Module):
    """注意力网络 - 使用自注意力机制"""
    
    def __init__(self, input_dim: int = 305, output_dim: int = 242, 
                 embed_dim: int = 512, num_heads: int = 8, dropout_rate: float = 0.1):
        super(AttentionNet, self).__init__()
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout_rate
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征提取
        features = self.feature_extractor(x)
        
        # 为注意力机制添加序列维度
        features = features.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # 自注意力
        attn_output, _ = self.attention(features, features, features)
        attn_output = self.layer_norm(features + attn_output)  # 残差连接
        
        # 移除序列维度
        attn_output = attn_output.squeeze(1)  # (batch_size, embed_dim)
        
        # 输出
        output = self.output_layers(attn_output)
        return output


class DeepResidualNet(nn.Module):
    """深度残差网络 - 更多残差块的深层网络"""
    
    def __init__(self, input_dim: int = 305, output_dim: int = 242, 
                 hidden_dim: int = 256, num_residual_blocks: int = 4,
                 dropout_rate: float = 0.2):
        super(DeepResidualNet, self).__init__()
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.BatchNorm1d(hidden_dim)
        
        # 多个残差块
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
            self.residual_blocks.append(block)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入投影
        x = self.relu(self.input_norm(self.input_projection(x)))
        
        # 通过残差块
        for residual_block in self.residual_blocks:
            residual = residual_block(x)
            x = self.relu(x + residual)
        
        # 输出
        x = self.output_layer(x)
        return x


def create_model(model_type: str, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
    """
    模型工厂函数，根据类型创建相应的模型
    
    Args:
        model_type: 模型类型 ('simple', 'residual', 'attention', 'deep_residual')
        input_dim: 输入维度
        output_dim: 输出维度
        **kwargs: 其他模型参数
    
    Returns:
        创建的模型实例
    """
    model_registry = {
        'simple': SimpleImprovedNet,
        'residual': ResidualNet,
        'attention': AttentionNet,
        'deep_residual': DeepResidualNet
    }
    
    if model_type not in model_registry:
        available_types = ', '.join(model_registry.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available types: {available_types}")
    
    model_class = model_registry[model_type]
    return model_class(input_dim=input_dim, output_dim=output_dim, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """计算模型的参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> dict:
    """获取模型信息"""
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }
    
    return info