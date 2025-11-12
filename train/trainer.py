"""
训练脚本 - 使用训练集和测试集
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import *
from dataset import create_dataloaders
from model import create_model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch:2d} [Train]', ncols=100)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device, epoch, mode='Test'):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(test_loader, desc=f'Epoch {epoch:2d} [{mode:5s}]', ncols=100)
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(test_loader)


def plot_losses(train_losses, test_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=5)
    plt.plot(epochs, test_losses, 'r-s', label='Test Loss', linewidth=2, markersize=5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training and Test Loss Curves', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_checkpoint(model, optimizer, epoch, train_loss, test_loss, filepath):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, filepath)


def main():
    """主训练函数"""
    print_config()
    
    # 创建保存目录
    os.makedirs(TRAIN_CONFIG['save_dir'], exist_ok=True)
    
    # 准备数据
    print("\nPreparing data loaders...")
    train_loader, test_loader = create_dataloaders(
        batch_size=TRAIN_CONFIG['batch_size'],
        num_workers=TRAIN_CONFIG['num_workers']
    )
    
    # 创建模型
    device = torch.device(TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model = create_model(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    best_test_loss = float('inf')
    train_losses, test_losses = [], []
    
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")
    
    for epoch in range(1, TRAIN_CONFIG['num_epochs'] + 1):
        # 训练和测试
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_loss = evaluate(model, test_loader, criterion, device, epoch)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # 调整学习率
        scheduler.step(test_loss)
        
        # 打印摘要
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{TRAIN_CONFIG['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Test Loss:  {test_loss:.6f}")
        print(f"  LR: {lr:.2e}")
        
        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_path = os.path.join(TRAIN_CONFIG['save_dir'], 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, test_loss, save_path)
            print(f"  ✓ Best model saved! (Test Loss: {test_loss:.6f})")
        
        # 定期保存
        if epoch % 10 == 0:
            save_path = os.path.join(TRAIN_CONFIG['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, test_loss, save_path)
        
        # 绘制曲线
        if epoch % 5 == 0:
            plot_path = os.path.join(TRAIN_CONFIG['save_dir'], 'training_curve.png')
            plot_losses(train_losses, test_losses, plot_path)
        
        print("-" * 60)
    
    # 保存最终模型和历史
    final_path = os.path.join(TRAIN_CONFIG['save_dir'], 'final_model.pth')
    save_checkpoint(model, optimizer, TRAIN_CONFIG['num_epochs'], 
                   train_losses[-1], test_losses[-1], final_path)
    
    history_path = os.path.join(TRAIN_CONFIG['save_dir'], 'training_history.npz')
    np.savez(history_path, train_losses=train_losses, test_losses=test_losses)
    
    plot_path = os.path.join(TRAIN_CONFIG['save_dir'], 'final_training_curve.png')
    plot_losses(train_losses, test_losses, plot_path)
    
    print("\n" + "="*60)
    print("Training Completed!")
    print(f"Best Test Loss: {best_test_loss:.6f}")
    print(f"Models saved in: {TRAIN_CONFIG['save_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
