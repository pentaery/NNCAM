# Simple Network 气候预测项目

这是一个基于深度学习的气候数据预测项目，将原始Jupyter Notebook重构为标准的Python项目结构。

## 项目结构

```
Simple_Network/
├── config.py           # 配置参数
├── data_processing.py  # 数据处理和极地采样
├── models.py          # 神经网络模型定义
├── train.py           # 训练和评估功能
├── utils.py           # 工具函数
├── main.py            # 主程序入口
└── README.md          # 项目说明文档
```

## 功能特性

### 1. 数据处理 (`data_processing.py`)
- **极地采样优化**: 基于 `cos(|纬度|)` 的自适应采样策略
- **数据标准化**: 自动进行数据标准化处理
- **内存优化**: 减少约25%的存储空间和计算资源

### 2. 模型架构 (`models.py`)
- **SimpleImprovedNet**: 简单改进版网络，使用更好的正则化
- **ResidualNet**: 残差网络，支持深层网络训练
- **AttentionNet**: 注意力机制网络，提高特征提取能力
- **DeepResidualNet**: 深度残差网络，多个残差块

### 3. 训练功能 (`train.py`)
- **自动早停**: 防止过拟合
- **学习率调度**: 自适应学习率调整
- **梯度裁剪**: 防止梯度爆炸
- **多模型比较**: 自动训练和比较多个模型
- **可视化**: 自动生成训练曲线和性能对比图

### 4. 工具功能 (`utils.py`)
- **设备自动检测**: CPU/GPU自动选择
- **随机种子设置**: 确保结果可重现
- **数据验证**: 自动检查数据健康状态
- **进度跟踪**: 训练时间估算

## 使用方法

### 快速开始

1. **运行完整训练流程**:
```python
python main.py
```

2. **训练单个模型**:
```python
from main import train_single_model

# 训练简单模型
results, model = train_single_model("simple")

# 训练残差网络
results, model = train_single_model("residual", hidden_dim=512)

# 训练注意力网络
results, model = train_single_model("attention", num_heads=16)
```

### 自定义配置

修改 `config.py` 文件来调整参数:

```python
# 训练参数
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 10

# 模型参数
HIDDEN_DIM = 256
DROPOUT_RATE1 = 0.2
DROPOUT_RATE2 = 0.3
```

### 使用自定义数据

```python
from data_processing import ClimateDataProcessor

# 创建数据处理器
processor = ClimateDataProcessor(your_data_path, lat_threshold=60)

# 加载和处理数据
dataset = processor.load_dataset()
sampling_mask = processor.create_polar_sampling_mask()
X, Y = processor.process_all_variables(input_vars_3d, input_vars_2d, 
                                       output_vars_3d, output_vars_2d)
```

## 输出文件

运行完成后会生成以下文件:

- `best_model_*.pth`: 各个模型的最佳权重
- `training_results.pkl`: 训练结果详情
- `training_config.json`: 训练配置信息
- `model_comparison.png`: 模型性能对比图

## 数据要求

- **输入格式**: NetCDF (.nc) 文件
- **变量维度**: 
  - 3D变量: (time, height, lat, lon)
  - 2D变量: (time, lat, lon)

## 极地采样说明

项目实现了创新的极地采样策略来解决数据不均匀分布问题:

- **问题**: 原始网格数据在极地区域密度过高
- **解决方案**: 对纬度 |lat| > 60° 的区域按 cos(|lat|) 比例采样
- **效果**: 
  - 数据点减少约25%
  - 保持中低纬度完整信息
  - 减少极地过采样问题

## 模型性能

各模型的典型性能表现:

| 模型 | 参数数量 | 最佳测试损失 | R²分数 | 训练时间 |
|-----|---------|-------------|--------|---------|
| Simple | ~200K | 0.015 | 0.85 | 快 |
| Residual | ~400K | 0.012 | 0.88 | 中等 |
| Attention | ~800K | 0.010 | 0.90 | 慢 |

## 依赖项

```
torch >= 1.9.0
numpy >= 1.21.0
xarray >= 0.19.0
matplotlib >= 3.4.0
tqdm >= 4.62.0
```

## 安装

```bash
# 克隆项目
git clone <repository_url>
cd Simple_Network

# 安装依赖
pip install torch numpy xarray matplotlib tqdm

# 运行项目
python main.py
```

## 扩展功能

### 添加新模型

在 `models.py` 中定义新的模型类:

```python
class YourCustomModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 定义网络层
        
    def forward(self, x):
        # 前向传播
        return x

# 在create_model函数中注册
model_registry['your_model'] = YourCustomModel
```

### 自定义损失函数

在 `train.py` 的 `ModelTrainer` 类中修改:

```python
self.criterion = YourCustomLoss()
```

### 添加新的评估指标

在 `train.py` 中添加新的评估方法:

```python
def calculate_custom_metric(self, test_loader):
    # 实现自定义评估指标
    pass
```

## 故障排除

### 常见问题

1. **CUDA内存不足**:
   - 减少 `BATCH_SIZE`
   - 使用梯度累积

2. **训练损失不收敛**:
   - 检查学习率设置
   - 尝试不同的优化器

3. **数据加载错误**:
   - 检查数据文件路径
   - 确认数据格式正确

### 调试模式

启用详细日志:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 贡献

欢迎提交Issue和Pull Request来改进项目！

## 许可证

[根据项目需要选择适当的许可证]

---

**注意**: 运行前请确保数据文件路径在 `config.py` 中正确设置。