"""
快速测试训练流程 - 只训练2个epoch验证代码正确性
"""
import os
import torch
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from train import create_dataloaders, data_files
from model import create_model
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

print("="*60)
print("Quick Training Test (2 epochs)")
print("="*60)

# 配置
batch_size = 64  # 小批次用于快速测试
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# 准备数据（只用前2个文件快速测试）
stats_path = os.path.join(os.path.dirname(__file__), "training_stats.npy")
test_files = data_files[:2]  # 只用2个文件

print(f"\nUsing {len(test_files)} data files for testing")

train_loader, val_loader = create_dataloaders(
    file_list=test_files,
    stats_path=stats_path,
    batch_size=batch_size,
    train_ratio=0.8,
    num_workers=2,
    shuffle=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# 创建模型
model = create_model(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练2个epoch
for epoch in range(1, 3):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/2")
    print('='*60)
    
    # 训练
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', ncols=100)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # 只训练前10个batch用于快速测试
        if batch_idx >= 9:
            break
    
    avg_train_loss = train_loss / min(10, len(train_loader))
    
    # 验证
    model.eval()
    val_loss = 0.0
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]  ', ncols=100)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            # 只验证前5个batch
            if batch_idx >= 4:
                break
    
    avg_val_loss = val_loss / min(5, len(val_loader))
    
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {avg_train_loss:.6f}")
    print(f"  Val Loss:   {avg_val_loss:.6f}")

print("\n" + "="*60)
print("✓ Training test completed successfully!")
print("="*60)
print("\nThe training pipeline is working correctly.")
print("You can now run the full training with:")
print("  python train/train_model.py")
