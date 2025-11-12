import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG


class ClimateNet(nn.Module):
    """
    气候数据神经网络模型
    - 对3D输入数据(10×30)进行多通道2D卷积
    - 展平后与2D输入数据(4)和坐标数据(3)拼接
    - 通过MLP处理
    - 输出3D数据(10×30)和2D数据(7)
    """
    def __init__(self, 
                 input_3d_channels=10,
                 input_3d_height=30,
                 input_2d_features=4,
                 input_coord_features=3,
                 output_3d_channels=10,
                 output_3d_height=30,
                 output_2d_features=7,
                 conv_channels=[32, 64, 128],
                 mlp_hidden_dims=[512, 256, 512]):
        """
        初始化网络
        
        Args:
            input_3d_channels: 3D输入通道数 (默认10)
            input_3d_height: 3D输入高度 (默认30)
            input_2d_features: 2D输入特征数 (默认4)
            input_coord_features: 坐标特征数 (默认3: time, lat, lon)
            output_3d_channels: 3D输出通道数 (默认10)
            output_3d_height: 3D输出高度 (默认30)
            output_2d_features: 2D输出特征数 (默认7)
            conv_channels: 卷积层通道数列表
            mlp_hidden_dims: MLP隐藏层维度列表
        """
        super(ClimateNet, self).__init__()
        
        self.input_3d_channels = input_3d_channels
        self.input_3d_height = input_3d_height
        self.input_2d_features = input_2d_features
        self.input_coord_features = input_coord_features
        self.output_3d_channels = output_3d_channels
        self.output_3d_height = output_3d_height
        self.output_2d_features = output_2d_features
        
        # 2D卷积网络处理3D数据 (10×30 -> 视为10通道的30×1图像)
        # 输入: (batch, 10, 30, 1)
        conv_layers = []
        in_channels = input_3d_channels
        
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ])
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*conv_layers)
        
        # 计算卷积后的特征维度
        # 卷积不改变空间维度 (padding保持), 输出: (batch, conv_channels[-1], 30, 1)
        conv_output_size = conv_channels[-1] * input_3d_height * 1
        
        # 拼接后的特征维度
        mlp_input_dim = conv_output_size + input_2d_features + input_coord_features
        
        # MLP网络
        mlp_layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # 输出层
        total_output_dim = output_3d_channels * output_3d_height + output_2d_features
        self.output_layer = nn.Linear(prev_dim, total_output_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, 307)
                - 前300维: 3D数据 (10变量 × 30层)
                - 301-304维: 2D数据 (4变量)
                - 305-307维: 坐标数据 (time, lat, lon)
        
        Returns:
            output: 输出张量 (batch, 307)
                - 前300维: 3D数据 (10变量 × 30层)
                - 301-307维: 2D数据 (7变量)
        """
        batch_size = x.size(0)
        
        # 分离3D数据、2D数据和坐标数据
        x_3d = x[:, :300]  # (batch, 300)
        x_2d = x[:, 300:304]  # (batch, 4)
        x_coord = x[:, 304:307]  # (batch, 3)
        
        # 重塑3D数据为 (batch, channels, height, width)
        # 300维 -> (10, 30) -> (batch, 10, 30, 1)
        x_3d = x_3d.view(batch_size, self.input_3d_channels, self.input_3d_height, 1)
        
        # 通过卷积网络
        conv_out = self.conv_net(x_3d)  # (batch, conv_channels[-1], 30, 1)
        
        # 展平卷积输出
        conv_out = conv_out.view(batch_size, -1)  # (batch, conv_channels[-1] * 30)
        
        # 拼接所有特征
        combined = torch.cat([conv_out, x_2d, x_coord], dim=1)  # (batch, mlp_input_dim)
        
        # 通过MLP
        mlp_out = self.mlp(combined)  # (batch, mlp_hidden_dims[-1])
        
        # 输出层
        output = self.output_layer(mlp_out)  # (batch, 307)
        
        return output


def create_model(device='cuda'):
    """
    创建模型实例
    
    Args:
        device: 设备 ('cuda' 或 'cpu')
    
    Returns:
        model: 模型实例
    """
    model = ClimateNet(**MODEL_CONFIG)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("Model Architecture:")
    print("="*60)
    print(model)
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*60)
    
    return model


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_model(device)
    
    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, 307).to(device)
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\nModel test passed! ✓")
