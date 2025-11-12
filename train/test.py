"""
快速测试 - 验证数据加载和训练流程
"""
import torch
from config import *
from dataset import create_dataloaders
from model import create_model

print("="*60)
print("Quick Test")
print("="*60)

# 测试配置
print_config()

# 测试数据加载
print("\n1. Testing data loading...")
train_loader, test_loader = create_dataloaders(batch_size=64, num_workers=2)
print(f"✓ Train batches: {len(train_loader)}")
print(f"✓ Test batches: {len(test_loader)}")

# 测试批次
print("\n2. Testing batch loading...")
for inputs, outputs in train_loader:
    print(f"✓ Input shape: {inputs.shape}")
    print(f"✓ Output shape: {outputs.shape}")
    break

# 测试模型
print("\n3. Testing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(device)
print(f"✓ Model created on {device}")

# 测试前向传播
print("\n4. Testing forward pass...")
inputs = inputs.to(device)
with torch.no_grad():
    outputs_pred = model(inputs)
print(f"✓ Prediction shape: {outputs_pred.shape}")

# 测试损失计算
print("\n5. Testing loss calculation...")
import torch.nn as nn
criterion = nn.MSELoss()
outputs = outputs.to(device)
loss = criterion(outputs_pred, outputs)
print(f"✓ Loss: {loss.item():.6f}")

print("\n" + "="*60)
print("✓ All tests passed!")
print("="*60)
print("\nYou can now start training:")
print("  python train/trainer.py")
