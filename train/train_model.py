import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from train import create_dataloaders, data_files
from model import create_model


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch数
    
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 使用tqdm创建进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', ncols=100)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # 将数据移到设备上
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 计算损失（展平后的MSE）
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 累积损失
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device, epoch):
    """
    验证模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch数
    
    Returns:
        avg_loss: 平均损失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # 使用tqdm创建进度条
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]  ', ncols=100)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # 将数据移到设备上
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 累积损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def plot_losses(train_losses, val_losses, save_path='training_curve.png'):
    """
    绘制训练和验证损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=5)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nLoss curve saved to {save_path}")
    plt.close()


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_path):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        train_loss: 训练损失
        val_loss: 验证损失
        save_path: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
    
    Returns:
        start_epoch: 起始epoch
        train_losses: 训练损失历史
        val_losses: 验证损失历史
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {start_epoch}")
    
    return start_epoch


def main():
    """主训练函数"""
    
    # ==================== 配置参数 ====================
    config = {
        'batch_size': 128,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './checkpoints',
        'resume': None,  # 设置为检查点路径以继续训练
    }
    
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # ==================== 准备数据 ====================
    print("Preparing data loaders...")
    stats_path = os.path.join(os.path.dirname(__file__), "training_stats.npy")
    
    train_loader, val_loader = create_dataloaders(
        file_list=data_files,
        stats_path=stats_path,
        batch_size=config['batch_size'],
        train_ratio=0.8,
        num_workers=config['num_workers'],
        shuffle=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # ==================== 创建模型 ====================
    device = torch.device(config['device'])
    print(f"\nUsing device: {device}")
    
    model = create_model(device)
    
    # ==================== 损失函数和优化器 ====================
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # ==================== 恢复训练（可选） ====================
    start_epoch = 1
    train_losses = []
    val_losses = []
    
    if config['resume'] and os.path.exists(config['resume']):
        start_epoch = load_checkpoint(model, optimizer, config['resume'])
    
    # ==================== 训练循环 ====================
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")
    
    for epoch in range(start_epoch, config['num_epochs'] + 1):
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device, epoch)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印epoch摘要
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{config['num_epochs']} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, best_model_path)
            print(f"  ✓ New best model saved! (Val Loss: {val_loss:.6f})")
        
        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path)
        
        # 绘制损失曲线
        if epoch % 5 == 0 or epoch == config['num_epochs']:
            curve_path = os.path.join(config['save_dir'], 'training_curve.png')
            plot_losses(train_losses, val_losses, curve_path)
        
        print("-" * 60)
    
    # ==================== 训练完成 ====================
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Model saved in: {config['save_dir']}")
    
    # 保存最终模型
    final_model_path = os.path.join(config['save_dir'], 'final_model.pth')
    save_checkpoint(model, optimizer, config['num_epochs'], train_losses[-1], val_losses[-1], final_model_path)
    
    # 保存训练历史
    history_path = os.path.join(config['save_dir'], 'training_history.npz')
    np.savez(history_path, 
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses))
    print(f"Training history saved to {history_path}")
    
    # 绘制最终损失曲线
    final_curve_path = os.path.join(config['save_dir'], 'final_training_curve.png')
    plot_losses(train_losses, val_losses, final_curve_path)


if __name__ == "__main__":
    main()
