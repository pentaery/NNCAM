"""
快速测试（使用少量数据）- 不需要完整的 training_mask.npy
"""
import numpy as np
import torch
import xarray as xr
from config import *

print("="*60)
print("Quick System Test (without full preprocessing)")
print("="*60)

# 1. 测试配置加载
print("\n1. Testing config loading...")
print_config()
print("✓ Config loaded")

# 2. 测试模型创建
print("\n2. Testing model creation...")
from model import create_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(device)
print(f"✓ Model created on {device}")

# 3. 测试模型前向传播
print("\n3. Testing forward pass...")
batch_size = 4
n_inputs = len(INPUTS_3D) * N_LEVELS + len(INPUTS_2D) + len(INPUTS_COORDS)
test_input = torch.randn(batch_size, n_inputs).to(device)
with torch.no_grad():
    test_output = model(test_input)
print(f"✓ Input shape: {test_input.shape}")
print(f"✓ Output shape: {test_output.shape}")

# 4. 测试损失计算
print("\n4. Testing loss calculation...")
import torch.nn as nn
criterion = nn.MSELoss()
test_target = torch.randn_like(test_output)
loss = criterion(test_output, test_target)
print(f"✓ Loss: {loss.item():.6f}")

# 5. 测试反向传播
print("\n5. Testing backward pass...")
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("✓ Backward pass successful")

# 6. 检查文件是否存在
print("\n6. Checking required files...")
stats_file = get_stats_path()
mask_file = get_train_mask_path()

if os.path.exists(stats_file):
    print(f"✓ Found: {stats_file}")
    stats = np.load(stats_file, allow_pickle=True).item()
    print(f"  Stats keys: {list(stats.keys())}")
else:
    print(f"✗ Missing: {stats_file}")
    print("  → Run: python train/preprocessing.py")

if os.path.exists(mask_file):
    print(f"✓ Found: {mask_file}")
    mask = np.load(mask_file)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Train samples: {mask.sum()}")
    print(f"  Test samples: {(~mask).sum()}")
else:
    print(f"✗ Missing: {mask_file}")
    print("  → Run: python train/preprocessing.py")

print("\n" + "="*60)
print("✓ Core system test passed!")
print("="*60)

if os.path.exists(stats_file) and os.path.exists(mask_file):
    print("\n✅ System ready for training!")
    print("   Run: python train/trainer.py")
else:
    print("\n⚠️  Need to generate stats and mask first!")
    print("   Run: python train/preprocessing.py")
    print("   (This will take some time)")
