#!/bin/bash

# 气候神经网络训练启动脚本

echo "=========================================="
echo "Climate Neural Network Training"
echo "=========================================="
echo ""

# 检查Python环境
PYTHON_CMD="/home/ET/yjzhou/HPCSoft/miniconda3/envs/mytorch/bin/python"

if [ ! -f "$PYTHON_CMD" ]; then
    echo "Error: Python interpreter not found at $PYTHON_CMD"
    exit 1
fi

# 进入项目目录
cd /home/ET/yjzhou/projects/NNCAM

echo "Python: $PYTHON_CMD"
echo "Working directory: $(pwd)"
echo ""

# 检查必要文件
if [ ! -f "train/training_stats.npy" ]; then
    echo "Error: training_stats.npy not found!"
    exit 1
fi

if [ ! -f "train/train_model.py" ]; then
    echo "Error: train_model.py not found!"
    exit 1
fi

echo "All files checked ✓"
echo ""
echo "Starting training..."
echo "=========================================="
echo ""

# 开始训练
$PYTHON_CMD train/train_model.py

echo ""
echo "=========================================="
echo "Training finished!"
echo "=========================================="
