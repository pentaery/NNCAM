import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from config import MODEL_CONFIG, TRAIN_CONFIG, N_LEVELS


class ClimateNet(nn.Module):
    """
    气候预测神经网络
    
    输入:
        - inputs_3d: (batch, 10, 30) - 10个变量 × 30层
        - inputs_2d: (batch, 7) - 7个特征
    
    输出:
        - outputs_3d: (batch, 10, 30) - 10个变量 × 30层
        - outputs_2d: (batch, 7) - 7个特征
    
    架构:
        1. 3D输入 → 2D卷积 → 展平
        2. 展平的3D特征 + 2D输入 → MLP
        3. MLP输出 → 分支到3D和2D输出
    """
    
    def __init__(self, 
                 input_3d_channels=10, 
                 input_3d_height=30,
                 input_2d_features=7,
                 output_3d_channels=10,
                 output_3d_height=30,
                 output_2d_features=7,
                 conv_channels=[32, 64, 128],
                 mlp_hidden_dims=[512, 256, 512]):
        """
        Args:
            input_3d_channels: 3D输入的通道数 (10个变量)
            input_3d_height: 3D输入的高度 (30层)
            input_2d_features: 2D输入的特征数 (7个)
            output_3d_channels: 3D输出的通道数 (10个变量)
            output_3d_height: 3D输出的高度 (30层)
            output_2d_features: 2D输出的特征数 (7个)
            conv_channels: 卷积层的通道数列表
            mlp_hidden_dims: MLP隐藏层维度列表
        """
        super(ClimateNet, self).__init__()
        
        self.input_3d_channels = input_3d_channels
        self.input_3d_height = input_3d_height
        self.input_2d_features = input_2d_features
        self.output_3d_channels = output_3d_channels
        self.output_3d_height = output_3d_height
        self.output_2d_features = output_2d_features
        
        # ==================== 3D输入处理：2D卷积 ====================
        # 将 (batch, 10, 30) 视为 (batch, 10, 30, 1) 进行2D卷积
        conv_layers = []
        in_channels = input_3d_channels
        current_height = input_3d_height
        
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*conv_layers)
        
        # 计算卷积后的特征维度
        # 输入: (batch, 10, 30, 1) → 输出: (batch, conv_channels[-1], 30, 1)
        conv_output_size = conv_channels[-1] * current_height * 1
        
        # ==================== MLP处理组合特征 ====================
        # 组合特征: 展平的3D特征 + 2D输入
        combined_input_size = conv_output_size + input_2d_features
        
        mlp_layers = []
        in_dim = combined_input_size
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # ==================== 输出分支 ====================
        # 3D输出分支
        self.output_3d_branch = nn.Sequential(
            nn.Linear(mlp_hidden_dims[-1], 256),
            nn.ReLU(),
            nn.Linear(256, output_3d_channels * output_3d_height)
        )
        
        # 2D输出分支
        self.output_2d_branch = nn.Sequential(
            nn.Linear(mlp_hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, output_2d_features)
        )
    
    def forward(self, inputs_3d, inputs_2d):
        """
        前向传播
        
        Args:
            inputs_3d: (batch, 10, 30)
            inputs_2d: (batch, 7)
        
        Returns:
            outputs_3d: (batch, 10, 30)
            outputs_2d: (batch, 7)
        """
        batch_size = inputs_3d.shape[0]
        
        # 1. 处理3D输入: (batch, 10, 30) → (batch, 10, 30, 1)
        x_3d = inputs_3d.unsqueeze(-1)  # (batch, 10, 30, 1)
        
        # 2. 2D卷积
        x_3d = self.conv_net(x_3d)  # (batch, conv_channels[-1], 30, 1)
        
        # 3. 展平卷积输出
        x_3d_flat = x_3d.view(batch_size, -1)  # (batch, conv_output_size)
        
        # 4. 组合3D和2D特征
        x_combined = torch.cat([x_3d_flat, inputs_2d], dim=1)  # (batch, combined_size)
        
        # 5. MLP处理
        x_mlp = self.mlp(x_combined)  # (batch, mlp_hidden_dims[-1])
        
        # 6. 输出分支
        # 3D输出
        outputs_3d = self.output_3d_branch(x_mlp)  # (batch, 10*30)
        outputs_3d = outputs_3d.view(batch_size, self.output_3d_channels, self.output_3d_height)
        
        # 2D输出
        outputs_2d = self.output_2d_branch(x_mlp)  # (batch, 7)
        
        return outputs_3d, outputs_2d


class Trainer:
    """训练器类"""
    
    def __init__(self, model, train_loader, test_loader, device='cuda', lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # 损失函数 (MSE)
        self.criterion = nn.MSELoss()
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'train_loss_3d': [],
            'train_loss_2d': [],
            'test_loss': [],
            'test_loss_3d': [],
            'test_loss_2d': [],
        }
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_loss_3d = 0
        total_loss_2d = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, outputs in pbar:
            # 移动到设备
            inputs_3d = inputs['inputs_3d'].to(self.device)
            inputs_2d = inputs['inputs_2d'].to(self.device)
            targets_3d = outputs['outputs_3d'].to(self.device)
            targets_2d = outputs['outputs_2d'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            pred_3d, pred_2d = self.model(inputs_3d, inputs_2d)
            
            # 计算损失
            loss_3d = self.criterion(pred_3d, targets_3d)
            loss_2d = self.criterion(pred_2d, targets_2d)
            loss = loss_3d + loss_2d
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            total_loss_3d += loss_3d.item()
            total_loss_2d += loss_2d.item()
            n_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'loss_3d': f'{loss_3d.item():.4f}',
                'loss_2d': f'{loss_2d.item():.4f}'
            })
        
        avg_loss = total_loss / n_batches
        avg_loss_3d = total_loss_3d / n_batches
        avg_loss_2d = total_loss_2d / n_batches
        
        return avg_loss, avg_loss_3d, avg_loss_2d
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_loss_3d = 0
        total_loss_2d = 0
        n_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluating')
            for inputs, outputs in pbar:
                # 移动到设备
                inputs_3d = inputs['inputs_3d'].to(self.device)
                inputs_2d = inputs['inputs_2d'].to(self.device)
                targets_3d = outputs['outputs_3d'].to(self.device)
                targets_2d = outputs['outputs_2d'].to(self.device)
                
                # 前向传播
                pred_3d, pred_2d = self.model(inputs_3d, inputs_2d)
                
                # 计算损失
                loss_3d = self.criterion(pred_3d, targets_3d)
                loss_2d = self.criterion(pred_2d, targets_2d)
                loss = loss_3d + loss_2d
                
                # 记录损失
                total_loss += loss.item()
                total_loss_3d += loss_3d.item()
                total_loss_2d += loss_2d.item()
                n_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'loss_3d': f'{loss_3d.item():.4f}',
                    'loss_2d': f'{loss_2d.item():.4f}'
                })
        
        avg_loss = total_loss / n_batches
        avg_loss_3d = total_loss_3d / n_batches
        avg_loss_2d = total_loss_2d / n_batches
        
        return avg_loss, avg_loss_3d, avg_loss_2d
    
    def train(self, num_epochs=5, save_dir='./checkpoints'):
        """训练模型"""
        os.makedirs(save_dir, exist_ok=True)
        best_test_loss = float('inf')
        
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # 训练
            train_loss, train_loss_3d, train_loss_2d = self.train_epoch()
            
            # 评估
            test_loss, test_loss_3d, test_loss_2d = self.evaluate()
            
            # 更新学习率
            self.scheduler.step(test_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_loss_3d'].append(train_loss_3d)
            self.history['train_loss_2d'].append(train_loss_2d)
            self.history['test_loss'].append(test_loss)
            self.history['test_loss_3d'].append(test_loss_3d)
            self.history['test_loss_2d'].append(test_loss_2d)
            
            # 打印统计信息
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.6f} (3D: {train_loss_3d:.6f}, 2D: {train_loss_2d:.6f})")
            print(f"  Test Loss:  {test_loss:.6f} (3D: {test_loss_3d:.6f}, 2D: {test_loss_2d:.6f})")
            
            # 保存最佳模型
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_loss': test_loss,
                    'history': self.history
                }, checkpoint_path)
                print(f"  ✓ Best model saved to {checkpoint_path}")
            
            # 保存最新模型
            checkpoint_path = os.path.join(save_dir, 'latest_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'test_loss': test_loss,
                'history': self.history
            }, checkpoint_path)
        
        print("\n" + "="*60)
        print("Training Completed!")
        print(f"Best Test Loss: {best_test_loss:.6f}")
        print("="*60 + "\n")
        
        # 绘制训练曲线
        self.plot_history(save_dir)
    
    def plot_history(self, save_dir):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 总损失
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['test_loss'], 'r-', label='Test Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 3D损失
        axes[1].plot(epochs, self.history['train_loss_3d'], 'b-', label='Train Loss 3D')
        axes[1].plot(epochs, self.history['test_loss_3d'], 'r-', label='Test Loss 3D')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('3D Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # 2D损失
        axes[2].plot(epochs, self.history['train_loss_2d'], 'b-', label='Train Loss 2D')
        axes[2].plot(epochs, self.history['test_loss_2d'], 'r-', label='Test Loss 2D')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('2D Loss')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Training history plot saved to {plot_path}")
        plt.close()


def main():
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 创建数据加载器 (batch_size=32)
    print("\nLoading data...")
    train_loader, test_loader = get_dataloaders(batch_size=32, num_workers=0)
    
    # 创建模型
    print("\nCreating model...")
    model = ClimateNet(
        input_3d_channels=10,
        input_3d_height=30,
        input_2d_features=7,
        output_3d_channels=10,
        output_3d_height=30,
        output_2d_features=7,
        conv_channels=[32, 64, 128],
        mlp_hidden_dims=[512, 256, 512]
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        lr=0.001
    )
    
    # 训练模型 (5个epoch)
    trainer.train(num_epochs=5, save_dir='./checkpoints')


if __name__ == '__main__':
    main()
