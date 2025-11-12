import os

# ==================== 数据配置 ====================
DATA_FILES = [
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-13-00800.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-13-22400.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-13-44000.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-13-65600.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-14-00800.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-14-22400.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-14-44000.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-14-65600.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-15-00800.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-15-22400.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-15-44000.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-15-65600.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-16-00800.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-16-22400.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-16-44000.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-16-65600.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-17-00800.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-17-22400.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-17-44000.nc",
    "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-17-65600.nc"
]

# 变量配置
INPUTS_3D = ['U', 'V', 'T', 'Q', 'CLDLIQ', 'CLDICE', 'PMID', 'DPRES', 'Z3', 'HEIGHT']
INPUTS_2D = ['TAUX', 'TAUY', 'SHFLX', 'LHFLX']
INPUTS_COORDS = ['time', 'lat', 'lon']
OUTPUTS_3D = ['SPDQ', 'SPDQC', 'SPDQI', 'SPNC', 'SPNI', 'SPDT', 'CLOUD', 'CLOUDTOP', 'QRL', 'QRS']
OUTPUTS_2D = ['PRECC', 'PRECSC', 'FSNT', 'FSDS', 'FSNS', 'FLNS', 'FLNT']

# 数据维度
N_LEVELS = 30  # 垂直层数

# 文件路径
STATS_FILE = 'training_stats.npy'
TRAIN_MASK_FILE = 'training_mask.npy'

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    'batch_size': 128,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'num_workers': 4,
    'device': 'cuda',
    'save_dir': './checkpoints',
    'resume': None,
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    'input_3d_channels': len(INPUTS_3D),
    'input_3d_height': N_LEVELS,
    'input_2d_features': len(INPUTS_2D),
    'input_coord_features': len(INPUTS_COORDS),
    'output_3d_channels': len(OUTPUTS_3D),
    'output_3d_height': N_LEVELS,
    'output_2d_features': len(OUTPUTS_2D),
    'conv_channels': [32, 64, 128],
    'mlp_hidden_dims': [512, 256, 512],
}

# ==================== 辅助函数 ====================
def get_stats_path():
    """获取统计文件的绝对路径"""
    return os.path.join(os.path.dirname(__file__), STATS_FILE)

def get_train_mask_path():
    """获取训练掩码文件的绝对路径"""
    return os.path.join(os.path.dirname(__file__), TRAIN_MASK_FILE)

def print_config():
    """打印配置信息"""
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"\nData Files: {len(DATA_FILES)}")
    print(f"\nInput Variables:")
    print(f"  3D: {len(INPUTS_3D)} variables × {N_LEVELS} levels = {len(INPUTS_3D) * N_LEVELS} features")
    print(f"  2D: {len(INPUTS_2D)} variables")
    print(f"  Coords: {len(INPUTS_COORDS)} variables")
    print(f"  Total: {len(INPUTS_3D) * N_LEVELS + len(INPUTS_2D) + len(INPUTS_COORDS)} features")
    print(f"\nOutput Variables:")
    print(f"  3D: {len(OUTPUTS_3D)} variables × {N_LEVELS} levels = {len(OUTPUTS_3D) * N_LEVELS} features")
    print(f"  2D: {len(OUTPUTS_2D)} variables")
    print(f"  Total: {len(OUTPUTS_3D) * N_LEVELS + len(OUTPUTS_2D)} features")
    print(f"\nTraining Config:")
    for key, value in TRAIN_CONFIG.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
