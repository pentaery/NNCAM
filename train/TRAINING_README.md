# 气候神经网络训练说明

## 网络架构

### 模型设计 (ClimateNet)

```
输入 (307维):
├── 3D数据: 10变量 × 30层 = 300维
├── 2D数据: 4变量 = 4维  (TAUX, TAUY, SHFLX, LHFLX)
└── 坐标: 3维 (time, lat, lon)

网络结构:
1. 2D卷积模块 (处理3D数据):
   - 输入重塑: (batch, 300) → (batch, 10, 30, 1)
   - Conv2D: 10 → 32 → 64 → 128 通道
   - 每层: Conv + BatchNorm + ReLU + Dropout
   - 输出: (batch, 128, 30, 1) → 展平为 (batch, 3840)

2. 特征拼接:
   - 卷积特征: 3840维
   - 2D数据: 4维
   - 坐标数据: 3维
   - 合计: 3847维

3. MLP模块:
   - 全连接: 3847 → 512 → 256 → 512
   - 每层: Linear + BatchNorm + ReLU + Dropout

4. 输出层:
   - 全连接: 512 → 307
   
输出 (307维):
├── 3D数据: 10变量 × 30层 = 300维
└── 2D数据: 7变量 = 7维
```

## 训练配置

- **批次大小**: 128
- **训练轮数**: 50 epochs
- **学习率**: 0.001 (Adam优化器)
- **权重衰减**: 1e-5
- **学习率调度**: ReduceLROnPlateau (patience=5, factor=0.5)
- **损失函数**: MSE Loss (展平后的均方误差)

## 文件说明

- `train.py`: DataLoader和数据预处理
- `model.py`: 神经网络模型定义
- `train_model.py`: 训练主程序
- `training_stats.npy`: 数据标准化统计信息
- `start_training.sh`: 训练启动脚本

## 使用方法

### 方式1: 直接运行Python脚本

```bash
cd /home/ET/yjzhou/projects/NNCAM
/home/ET/yjzhou/HPCSoft/miniconda3/envs/mytorch/bin/python train/train_model.py
```

### 方式2: 使用启动脚本

```bash
cd /home/ET/yjzhou/projects/NNCAM/train
chmod +x start_training.sh
./start_training.sh
```

### 方式3: 从检查点恢复训练

修改 `train_model.py` 中的配置:
```python
config = {
    ...
    'resume': './checkpoints/checkpoint_epoch_20.pth',  # 指定检查点路径
    ...
}
```

## 训练输出

训练过程中会生成以下文件:

```
checkpoints/
├── best_model.pth              # 验证损失最低的模型
├── final_model.pth             # 最终训练完成的模型
├── checkpoint_epoch_10.pth     # 每10轮保存的检查点
├── checkpoint_epoch_20.pth
├── ...
├── training_curve.png          # 训练曲线图
├── final_training_curve.png    # 最终训练曲线图
└── training_history.npz        # 训练历史数据
```

## 训练监控

训练过程使用 tqdm 显示进度条，实时显示:
- 当前 epoch 和 batch 进度
- 实时损失值
- 训练/验证阶段
- 预计剩余时间

示例输出:
```
Epoch 1 [Train]: 100%|████████| 745/745 [02:15<00:00, loss=0.123456]
Epoch 1 [Val]:   100%|████████| 187/187 [00:30<00:00, loss=0.098765]

Epoch 1/50 Summary:
  Train Loss: 0.123456
  Val Loss:   0.098765
  Learning Rate: 1.00e-03
  ✓ New best model saved! (Val Loss: 0.098765)
```

## 模型参数

- **总参数**: 2,425,491 个
- **可训练参数**: 2,425,491 个

## 数据规模

- **总样本**: 119,439,360
- **训练样本**: 95,551,488 (80%)
- **验证样本**: 23,887,872 (20%)
- **每epoch训练批次**: 745,558 (batch_size=128)
- **每epoch验证批次**: 186,390 (batch_size=128)

## 预计训练时间

根据硬件配置不同，每个epoch大约需要:
- GPU (RTX 3090/A100): 30-60分钟
- GPU (GTX 1080Ti): 1-2小时
- CPU: 不推荐 (太慢)

50 epochs 总计约: 25-50 小时 (GPU)

## 自定义训练

修改 `train_model.py` 中的 config 字典:

```python
config = {
    'batch_size': 128,          # 根据GPU内存调整
    'num_epochs': 50,           # 训练轮数
    'learning_rate': 0.001,     # 初始学习率
    'weight_decay': 1e-5,       # L2正则化
    'num_workers': 4,           # 数据加载线程数
    'device': 'cuda',           # 'cuda' 或 'cpu'
    'save_dir': './checkpoints', # 模型保存目录
    'resume': None,             # 恢复训练的检查点路径
}
```

## 注意事项

1. 确保有足够的GPU内存 (建议 ≥ 11GB)
2. 如果内存不足，减小 batch_size
3. 训练会自动保存最佳模型和定期检查点
4. 可随时中断训练，使用 resume 参数恢复
5. 训练曲线会定期更新并保存为图片
