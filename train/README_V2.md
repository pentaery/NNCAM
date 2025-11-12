# 气候神经网络训练系统 - 重构版

## 📁 文件结构（精简版）

```
train/
├── config.py              # 统一配置文件
├── preprocessing.py       # 数据预处理（生成stats和mask）
├── dataset.py            # 数据加载器（支持训练/测试集）
├── model.py              # 神经网络模型
├── trainer.py            # 训练脚本
├── test.py               # 快速测试脚本
├── training_stats.npy    # 训练集的mean和std（需要先运行preprocessing.py）
└── training_mask.npy     # 训练/测试集划分掩码（需要先运行preprocessing.py）
```

## 🔄 完整工作流程

### 第一步：生成统计信息和掩码（只需运行一次）

```bash
cd /home/ET/yjzhou/projects/NNCAM
python train/preprocessing.py
```

这会生成两个文件：
- `training_stats.npy` - 训练集的 mean 和 std（用于标准化）
- `training_mask.npy` - 布尔数组，True=训练集(80%)，False=测试集(20%)

### 第二步：测试系统

```bash
python train/test.py
```

验证：
1. 数据加载正常
2. 训练/测试集分离正常
3. 模型创建正常
4. 前向传播正常
5. 损失计算正常

### 第三步：开始训练

```bash
python train/trainer.py
```

## 🎯 关键改进

### 1. **统一配置管理（config.py）**
所有配置集中在一个文件：
```python
# 数据文件
DATA_FILES = [...]

# 变量
INPUTS_3D = ['U', 'V', ...]
INPUTS_2D = ['TAUX', 'TAUY', ...]
OUTPUTS_3D = ['SPDQ', 'SPDQC', ...]
OUTPUTS_2D = ['PRECC', 'PRECSC', ...]

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 128,
    'num_epochs': 50,
    ...
}

# 模型配置
MODEL_CONFIG = {
    'input_3d_channels': 10,
    ...
}
```

### 2. **训练/测试集分离（dataset.py）**
使用 preprocessing.py 生成的训练掩码：
- `is_train=True` → 使用 `training_mask == True` 的样本（80%）
- `is_train=False` → 使用 `training_mask == False` 的样本（20%）
- **两者都使用训练集的 mean 和 std 进行标准化**

```python
# 创建数据加载器
train_loader, test_loader = create_dataloaders(
    batch_size=128,
    num_workers=4
)
```

### 3. **精简的训练流程（trainer.py）**
```python
for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch(model, train_loader, ...)
    test_loss = evaluate(model, test_loader, ...)
    
    # 保存最佳模型
    if test_loss < best_test_loss:
        save_checkpoint(...)
```

## 📊 数据流程

```
原始数据 (NetCDF files)
    ↓
preprocessing.py
    ↓
├── training_stats.npy (训练集的 mean & std)
└── training_mask.npy (80% True, 20% False)
    ↓
dataset.py
    ├── TrainDataset (mask==True) → 使用 stats 标准化
    └── TestDataset (mask==False) → 使用 stats 标准化
    ↓
trainer.py
    ├── Train on TrainDataset
    └── Evaluate on TestDataset
```

## 🔧 修改配置

### 调整训练参数
编辑 `config.py`:
```python
TRAIN_CONFIG = {
    'batch_size': 256,      # 增大批次
    'num_epochs': 100,      # 更多轮次
    'learning_rate': 0.0005, # 调整学习率
    ...
}
```

### 调整模型结构
编辑 `config.py`:
```python
MODEL_CONFIG = {
    'conv_channels': [64, 128, 256],  # 更深的卷积网络
    'mlp_hidden_dims': [1024, 512, 1024],  # 更大的MLP
    ...
}
```

## ⚠️ 重要说明

1. **标准化方式**：
   - ✅ 训练集：使用训练集的 mean 和 std
   - ✅ 测试集：**也使用训练集的 mean 和 std**（避免数据泄露）

2. **数据划分**：
   - preprocessing.py 中的 `create_training_mask()` 已经做了 80/20 划分
   - 训练和测试样本在时空上随机分布，不重叠

3. **第一次运行**：
   - 必须先运行 `preprocessing.py` 生成 stats 和 mask
   - 这一步比较耗时（可能需要几小时）
   - 生成后可以反复使用，不需要重新运行

## 📈 训练输出

```
checkpoints/
├── best_model.pth              # 测试集损失最低的模型
├── checkpoint_epoch_10.pth     # 每10轮的检查点
├── final_model.pth             # 最终模型
├── training_curve.png          # 训练曲线
└── training_history.npz        # 历史数据
```

## 🚀 快速开始

```bash
# 1. 如果还没有 training_stats.npy 和 training_mask.npy
cd /home/ET/yjzhou/projects/NNCAM
python train/preprocessing.py

# 2. 测试系统
python train/test.py

# 3. 开始训练
python train/trainer.py
```

## 💡 代码特点

- ✅ **精简**：核心代码组织清晰，易于理解和修改
- ✅ **统一配置**：所有参数集中管理
- ✅ **正确划分**：训练/测试集不重叠，都用训练集统计量标准化
- ✅ **tqdm可视化**：实时显示训练进度
- ✅ **自动保存**：最佳模型和定期检查点

## 🔍 故障排查

### 问题：FileNotFoundError: training_mask.npy
**解决**：先运行 `python train/preprocessing.py`

### 问题：CUDA out of memory
**解决**：在 `config.py` 中减小 `batch_size`

### 问题：训练太慢
**解决**：
1. 增大 `batch_size`（如果GPU内存足够）
2. 增大 `num_workers`（数据加载线程）

## 📊 与旧版本的差异

| 方面 | 旧版本 | 新版本 |
|------|--------|--------|
| 配置管理 | 分散在各个文件 | 统一在config.py |
| 数据划分 | 随机split | 使用预生成的mask |
| 测试集标准化 | 未明确 | 明确使用训练集统计量 |
| 代码组织 | train.py, train_model.py等 | dataset.py, trainer.py等 |
| 文件数量 | 7-8个 | 6个核心文件 |

## ✅ 优势

1. **更科学**：测试集使用训练集的统计量标准化，避免数据泄露
2. **更清晰**：训练/测试集通过mask明确划分，可复现
3. **更精简**：代码组织更合理，易于维护
4. **更灵活**：统一配置，修改参数更方便
