"""
Configuration file for Simple Network project
包含所有项目的配置参数
"""

from pathlib import Path

# Data configuration
DATA_PATH = Path("/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-13-00800.nc")

# Variable definitions
INPUTS_VARIABLE1 = ['U', 'V', 'T', 'Q', 'CLDLIQ', 'CLDICE', 'PMID', 'DPRES', 'Z3', 'HEIGHT']
INPUTS_VARIABLE2 = ['TAUX', 'TAUY', 'SHFLX', 'LHFLX']
OUTPUT_VARIABLE1 = ['SPDQ', 'SPDQC', 'SPDQI', 'SPNC', 'SPNI', 'SPDT', 'CLOUD', 'CLOUDTOP']
OUTPUT_VARIABLE2 = ['PRECC', 'PRECSC']

# Sampling configuration
LAT_THRESHOLD = 60  # 纬度阈值，用于极地采样

# Training configuration
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 10
WEIGHT_DECAY = 1e-4

# Data split configuration
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2

# Model configuration
INPUT_DIM = 305  # Will be calculated based on actual data
OUTPUT_DIM = 242  # Will be calculated based on actual data
HIDDEN_DIM = 256

# Device configuration
DEVICE = 'cuda'  # Will be auto-detected in main script

# Paths
MODEL_SAVE_PATH = 'best_model.pth'
RESULTS_SAVE_PATH = 'training_results.pkl'

# Regularization
DROPOUT_RATE1 = 0.2
DROPOUT_RATE2 = 0.3
GRADIENT_CLIP_VALUE = 1.0

# Loss function
HUBER_DELTA = 1.0

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.7
LR_SCHEDULER_PATIENCE = 5
MIN_LR = 1e-6