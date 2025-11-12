# 气候神经网络 - 完整训练系统

## 📁 文件结构

```
train/
├── train.py              # DataLoader和数据预处理
├── model.py              # 神经网络模型定义 (ClimateNet)
├── train_model.py        # 完整训练脚本 (主程序)
├── test_training.py      # 快速测试脚本
├── preprocessing.py      # 数据预处理和统计计算
├── training_stats.npy    # 数据标准化统计信息
├── start_training.sh     # 训练启动脚本
└── TRAINING_README.md    # 详细使用文档
```

## 🎯 快速开始

### 1. 快速测试（推荐先运行）
```bash
cd /home/ET/yjzhou/projects/NNCAM
python train/test_training.py
```
这会运行2个epoch的快速测试，验证整个训练流程。

### 2. 开始完整训练
```bash
cd /home/ET/yjzhou/projects/NNCAM
python train/train_model.py
```

## 🧠 网络架构详解

### 输入数据 (307维)
```
┌─────────────────────────────────────┐
│ 3D输入: 300维                       │
│ - 10个变量 × 30层                   │
│ - U, V, T, Q, CLDLIQ, CLDICE,      │
│   PMID, DPRES, Z3, HEIGHT          │
├─────────────────────────────────────┤
│ 2D输入: 4维                         │
│ - TAUX, TAUY, SHFLX, LHFLX         │
├─────────────────────────────────────┤
│ 坐标: 3维                           │
│ - time, lat, lon                   │
└─────────────────────────────────────┘
```

### 网络处理流程
```
1️⃣ 3D数据处理 (2D卷积):
   (300,) → reshape → (10, 30, 1)
   ↓
   Conv2D: 10→32→64→128 通道
   (每层: Conv + BatchNorm + ReLU + Dropout)
   ↓
   flatten → (3840,)

2️⃣ 特征融合:
   CNN特征(3840) + 2D数据(4) + 坐标(3) = (3847,)

3️⃣ MLP处理:
   (3847) → 512 → 256 → 512
   (每层: Linear + BatchNorm + ReLU + Dropout)

4️⃣ 输出:
   (512) → 307
```

### 输出数据 (307维)
```
┌─────────────────────────────────────┐
│ 3D输出: 300维                       │
│ - 10个变量 × 30层                   │
│ - SPDQ, SPDQC, SPDQI, SPNC, SPNI,  │
│   SPDT, CLOUD, CLOUDTOP, QRL, QRS  │
├─────────────────────────────────────┤
│ 2D输出: 7维                         │
│ - PRECC, PRECSC, FSNT, FSDS,       │
│   FSNS, FLNS, FLNT                 │
└─────────────────────────────────────┘
```

## ⚙️ 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 批次大小 | 128 | 可根据GPU内存调整 |
| 训练轮数 | 50 | 可修改 |
| 学习率 | 0.001 | Adam优化器 |
| 权重衰减 | 1e-5 | L2正则化 |
| 损失函数 | MSE | 均方误差 |
| 学习率调度 | ReduceLROnPlateau | patience=5, factor=0.5 |
| 数据加载线程 | 4 | 可根据CPU调整 |

## 📊 训练监控

训练过程使用 **tqdm** 实时显示:
- ✅ 实时进度条
- ✅ 当前损失值
- ✅ 预计剩余时间
- ✅ 训练/验证状态

示例输出:
```
Epoch 1 [Train]: 100%|████████| 745558/745558 [45:23<00:00, loss=0.123456]
Epoch 1 [Val]:   100%|████████| 186390/186390 [10:12<00:00, loss=0.098765]

Epoch 1/50 Summary:
  Train Loss: 0.123456
  Val Loss:   0.098765
  Learning Rate: 1.00e-03
  ✓ New best model saved! (Val Loss: 0.098765)
```

## 💾 输出文件

训练会自动生成:
```
checkpoints/
├── best_model.pth              # 🏆 验证损失最低的模型
├── final_model.pth             # 📝 最终训练完成的模型
├── checkpoint_epoch_10.pth     # 💾 定期检查点 (每10轮)
├── checkpoint_epoch_20.pth
├── ...
├── training_curve.png          # 📈 训练曲线图
├── final_training_curve.png    # 📈 最终训练曲线图
└── training_history.npz        # 📊 训练历史数据
```

## 🔧 自定义训练

修改 `train_model.py` 中的 config:
```python
config = {
    'batch_size': 128,          # ⬇️ GPU内存不足时减小
    'num_epochs': 50,           # 📈 增加训练时间
    'learning_rate': 0.001,     # 🎯 调整学习速度
    'weight_decay': 1e-5,       # 🛡️ 正则化强度
    'num_workers': 4,           # 🚀 数据加载速度
    'device': 'cuda',           # 🎮 'cuda' 或 'cpu'
    'save_dir': './checkpoints',# 💾 保存位置
    'resume': None,             # 🔄 继续训练的检查点
}
```

## 🚀 从检查点恢复训练

如果训练中断，可以继续训练:
```python
config = {
    ...
    'resume': './checkpoints/checkpoint_epoch_20.pth',
    ...
}
```

## 📈 模型信息

- **总参数**: 2,425,491 个 (~9.3 MB)
- **可训练参数**: 2,425,491 个
- **架构**: CNN + MLP 混合网络

## 💽 数据规模

- **总样本**: 119,439,360
- **训练集**: 95,551,488 (80%)
- **验证集**: 23,887,872 (20%)
- **每epoch批次**: ~745,558 (batch_size=128)

## ⏱️ 预计训练时间

| 硬件 | 每epoch | 50 epochs |
|------|---------|-----------|
| RTX 3090/A100 | 30-60分钟 | 25-50小时 |
| RTX 2080Ti | 1-2小时 | 50-100小时 |
| CPU | ❌ 不推荐 | 太慢 |

## ⚠️ 注意事项

1. ✅ 确保GPU内存 ≥ 11GB (或减小batch_size)
2. ✅ 定期检查训练曲线判断是否过拟合
3. ✅ 可随时中断训练，使用resume参数恢复
4. ✅ 最佳模型会自动保存
5. ✅ 学习率会根据验证损失自动调整

## 🎓 训练技巧

### 如果训练太慢:
```python
config = {
    'batch_size': 256,      # 增大批次
    'num_workers': 8,       # 增加数据加载线程
}
```

### 如果GPU内存不足:
```python
config = {
    'batch_size': 64,       # 减小批次
}
```

### 如果过拟合:
```python
config = {
    'weight_decay': 1e-4,   # 增强正则化
}
# 或修改 model.py 中的 dropout 比例
```

## 📞 故障排查

### 问题: FileNotFoundError
```bash
# 确保在正确目录运行
cd /home/ET/yjzhou/projects/NNCAM
python train/train_model.py
```

### 问题: CUDA out of memory
```python
# 减小batch_size
config = {'batch_size': 64}  # 或更小
```

### 问题: 训练速度慢
```python
# 检查数据加载
config = {'num_workers': 8}  # 增加线程数
```

## ✅ 测试结果

测试已通过 ✓
- 数据加载正常
- 模型前向传播正常
- 损失计算正常
- 梯度反向传播正常
- 模型保存/加载正常

## 🎉 开始训练

现在一切就绪，可以开始训练了：
```bash
cd /home/ET/yjzhou/projects/NNCAM
python train/train_model.py
```

祝训练顺利！🚀
