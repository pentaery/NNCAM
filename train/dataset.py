import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from config import (
    DATA_FILES, INPUTS_3D, INPUTS_2D, OUTPUTS_3D, OUTPUTS_2D, 
    N_LEVELS, TRAIN_CONFIG
)


class ClimateDataset(Dataset):
    """
    气候数据集类 - 以文件为单位逐个加载，避免内存溢出
    
    根据 training_mask.npy 区分训练集和测试集
    使用 training_stats.npy 进行归一化
    每次只加载一个文件到内存
    """
    
    def __init__(self, is_train=True, stats_path=None, mask_path=None):
        """
        Args:
            is_train: True 为训练集，False 为测试集
            stats_path: 统计信息文件路径
            mask_path: 训练掩码文件路径
        """
        super().__init__()
        
        self.is_train = is_train
        
        # 获取文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if stats_path is None:
            stats_path = os.path.join(script_dir, 'training_stats.npy')
        if mask_path is None:
            mask_path = os.path.join(script_dir, 'training_mask.npy')
        
        print(f"\n{'='*60}")
        print(f"Loading {'Training' if is_train else 'Test'} Dataset")
        print(f"{'='*60}")
        
        # 加载统计信息和掩码
        print("Loading statistics and mask...")
        self.stats = np.load(stats_path, allow_pickle=True).item()
        self.training_mask = np.load(mask_path)
        
        # 根据 is_train 选择掩码
        if is_train:
            self.mask = self.training_mask
        else:
            self.mask = ~self.training_mask
        
        print(f"Total samples: {self.mask.sum()}")
        
        # 验证文件
        print("\nValidating data files...")
        self.valid_files = []
        for i, file_path in enumerate(DATA_FILES):
            try:
                test_ds = xr.open_dataset(file_path)
                test_ds.close()
                self.valid_files.append(file_path)
                print(f"  ✓ File {i+1}/{len(DATA_FILES)}")
            except Exception as e:
                print(f"  ✗ File {i+1}/{len(DATA_FILES)} - SKIPPED")
        
        if len(self.valid_files) == 0:
            raise ValueError("No valid files found!")
        
        # 读取第一个文件获取坐标信息（不加载数据）
        print("\nReading coordinate information from first file...")
        with xr.open_dataset(self.valid_files[0]) as ds_temp:
            self.lats = ds_temp.lat.values
            self.lons = ds_temp.lon.values
            
            # 获取时间信息并转换为数值
            time_values = ds_temp.time.values
            if hasattr(time_values[0], 'year'):  # cftime 对象
                first_time = (time_values[0].year * 365.25 * 24 * 3600 + 
                             time_values[0].month * 30 * 24 * 3600 + 
                             time_values[0].day * 24 * 3600 + 
                             time_values[0].hour * 3600 + 
                             time_values[0].minute * 60 + 
                             time_values[0].second)
                self.first_time = first_time
                self.is_cftime = True
            else:  # datetime64 对象
                first_time = np.array(time_values[0], dtype='datetime64[ns]')
                self.first_time = first_time
                self.is_cftime = False
        
        # 构建样本索引：为每个样本记录 (文件索引, 文件内时间索引, lat索引, lon索引)
        print("\nBuilding sample index mapping...")
        
        # 读取单个文件的时间步数（所有文件维度相同，均为27步）
        with xr.open_dataset(self.valid_files[0]) as ds_temp:
            self.time_steps_per_file = len(ds_temp.time)
        
        print(f"Time steps per file: {self.time_steps_per_file}")
        print(f"Number of files: {len(self.valid_files)}")
        
        # 构建样本列表
        self.sample_list = []
        
        for t_idx in range(self.mask.shape[0]):
            for lat_idx in range(self.mask.shape[1]):
                for lon_idx in range(self.mask.shape[2]):
                    if self.mask[t_idx, lat_idx, lon_idx]:
                        # 计算该时间索引对应的文件和文件内索引
                        file_idx = t_idx // self.time_steps_per_file
                        local_t_idx = t_idx % self.time_steps_per_file
                        
                        # 确保文件索引有效
                        if file_idx < len(self.valid_files):
                            self.sample_list.append((file_idx, local_t_idx, lat_idx, lon_idx, t_idx))
        
        print(f"Total samples indexed: {len(self.sample_list)}")
        
        # 缓存当前加载的文件
        self.current_file_idx = -1
        self.current_ds = None
        self.current_time_numeric = None
        
        print(f"\n{'='*60}")
        print(f"Dataset initialized: {len(self)} samples")
        print(f"Files will be loaded on-demand during iteration")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.sample_list)
    
    def _load_file(self, file_idx):
        """
        加载指定文件（如果尚未加载或不是当前文件）
        
        Args:
            file_idx: 文件索引
        """
        if file_idx != self.current_file_idx:
            # 关闭之前的文件
            if self.current_ds is not None:
                self.current_ds.close()
            
            # 加载新文件
            self.current_file_idx = file_idx
            self.current_ds = xr.open_dataset(self.valid_files[file_idx])
            
            # 处理该文件的时间坐标
            time_values = self.current_ds.time.values
            if self.is_cftime:
                time_numeric = np.array([
                    (t.year * 365.25 * 24 * 3600 + 
                     t.month * 30 * 24 * 3600 + 
                     t.day * 24 * 3600 + 
                     t.hour * 3600 + 
                     t.minute * 60 + 
                     t.second) for t in time_values
                ])
                self.current_time_numeric = time_numeric - self.first_time
            else:
                time_values_arr = np.array(time_values, dtype='datetime64[ns]')
                self.current_time_numeric = (time_values_arr - self.first_time).astype('timedelta64[s]').astype(float)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            inputs: 字典，包含 'inputs_3d' 和 'inputs_2d'
            outputs: 字典，包含 'outputs_3d' 和 'outputs_2d'
        """
        # 获取样本信息 (文件索引, 文件内时间索引, lat索引, lon索引, 全局时间索引)
        file_idx, local_t_idx, lat_idx, lon_idx, global_t_idx = self.sample_list[idx]
        
        # 按需加载文件
        self._load_file(file_idx)
        
        # ==================== 处理输入 ====================
        
        # 1. 处理 3D 输入变量 (10 个变量 × 30 层)
        inputs_3d_list = []
        for var_name in INPUTS_3D:
            if var_name in self.current_ds:
                # HEIGHT 使用 ilev，其他使用 lev
                if var_name == 'HEIGHT':
                    var_data = self.current_ds[var_name].isel(
                        time=local_t_idx, 
                        ilev=slice(0, N_LEVELS),
                        lat=lat_idx, 
                        lon=lon_idx
                    ).values
                else:
                    var_data = self.current_ds[var_name].isel(
                        time=local_t_idx, 
                        lev=slice(0, N_LEVELS),
                        lat=lat_idx, 
                        lon=lon_idx
                    ).values
                
                # 归一化（每层独立归一化）
                mean = self.stats['inputs_3D'][var_name]['mean']
                std = self.stats['inputs_3D'][var_name]['std']
                var_data = (var_data - mean) / (std + 1e-8)
                
                inputs_3d_list.append(var_data)
        
        # 转换为 numpy 数组: shape (10, 30)
        inputs_3d = np.stack(inputs_3d_list, axis=0)  # (n_vars, n_levels)
        
        # 2. 处理 2D 输入变量 (4 个变量 + 3 个坐标)
        inputs_2d_list = []
        
        # 2.1 处理物理变量 (TAUX, TAUY, SHFLX, LHFLX)
        for var_name in INPUTS_2D:
            if var_name in self.current_ds:
                var_data = self.current_ds[var_name].isel(
                    time=local_t_idx,
                    lat=lat_idx,
                    lon=lon_idx
                ).values
                
                # 归一化
                mean = self.stats['inputs_2D'][var_name]['mean']
                std = self.stats['inputs_2D'][var_name]['std']
                var_data = (var_data - mean) / (std + 1e-8)
                
                inputs_2d_list.append(float(var_data))
        
        # 2.2 添加坐标信息 (time, lat, lon)
        # time - 使用当前文件的时间信息
        time_val = self.current_time_numeric[local_t_idx]
        time_mean = self.stats['inputs_2D']['time']['mean']
        time_std = self.stats['inputs_2D']['time']['std']
        time_normalized = (time_val - time_mean) / (time_std + 1e-8)
        inputs_2d_list.append(float(time_normalized))
        
        # lat
        lat_val = self.lats[lat_idx]
        lat_mean = self.stats['inputs_2D']['lat']['mean']
        lat_std = self.stats['inputs_2D']['lat']['std']
        lat_normalized = (lat_val - lat_mean) / (lat_std + 1e-8)
        inputs_2d_list.append(float(lat_normalized))
        
        # lon
        lon_val = self.lons[lon_idx]
        lon_mean = self.stats['inputs_2D']['lon']['mean']
        lon_std = self.stats['inputs_2D']['lon']['std']
        lon_normalized = (lon_val - lon_mean) / (lon_std + 1e-8)
        inputs_2d_list.append(float(lon_normalized))
        
        # 转换为 numpy 数组: shape (7,)
        inputs_2d = np.array(inputs_2d_list, dtype=np.float32)
        
        # ==================== 处理输出 ====================
        
        # 1. 处理 3D 输出变量 (10 个变量 × 30 层)
        outputs_3d_list = []
        for var_name in OUTPUTS_3D:
            if var_name in self.current_ds:
                var_data = self.current_ds[var_name].isel(
                    time=local_t_idx,
                    lev=slice(0, N_LEVELS),
                    lat=lat_idx,
                    lon=lon_idx
                ).values
                
                # 归一化（每层独立归一化）
                mean = self.stats['outputs_3D'][var_name]['mean']
                std = self.stats['outputs_3D'][var_name]['std']
                var_data = (var_data - mean) / (std + 1e-8)
                
                outputs_3d_list.append(var_data)
        
        # 转换为 numpy 数组: shape (10, 30)
        outputs_3d = np.stack(outputs_3d_list, axis=0)
        
        # 2. 处理 2D 输出变量 (7 个变量)
        outputs_2d_list = []
        for var_name in OUTPUTS_2D:
            if var_name in self.current_ds:
                var_data = self.current_ds[var_name].isel(
                    time=local_t_idx,
                    lat=lat_idx,
                    lon=lon_idx
                ).values
                
                # 归一化
                mean = self.stats['outputs_2D'][var_name]['mean']
                std = self.stats['outputs_2D'][var_name]['std']
                var_data = (var_data - mean) / (std + 1e-8)
                
                outputs_2d_list.append(float(var_data))
        
        # 转换为 numpy 数组: shape (7,)
        outputs_2d = np.array(outputs_2d_list, dtype=np.float32)
        
        # 转换为 PyTorch 张量
        inputs = {
            'inputs_3d': torch.from_numpy(inputs_3d).float(),  # (10, 30)
            'inputs_2d': torch.from_numpy(inputs_2d).float()   # (7,)
        }
        
        outputs = {
            'outputs_3d': torch.from_numpy(outputs_3d).float(),  # (10, 30)
            'outputs_2d': torch.from_numpy(outputs_2d).float()   # (7,)
        }
        
        return inputs, outputs
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'current_ds') and self.current_ds is not None:
            self.current_ds.close()


def get_dataloaders(batch_size=None, num_workers=None, stats_path=None, mask_path=None):
    """
    创建训练集和测试集的 DataLoader
    
    Args:
        batch_size: 批量大小，默认从 config 读取
        num_workers: 工作进程数，默认从 config 读取
        stats_path: 统计信息文件路径
        mask_path: 训练掩码文件路径
    
    Returns:
        train_loader: 训练集 DataLoader
        test_loader: 测试集 DataLoader
    """
    if batch_size is None:
        batch_size = TRAIN_CONFIG['batch_size']
    if num_workers is None:
        num_workers = TRAIN_CONFIG['num_workers']
    
    print("\n" + "="*60)
    print("Creating DataLoaders")
    print("="*60)
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    
    # 创建训练集和测试集
    train_dataset = ClimateDataset(is_train=True, stats_path=stats_path, mask_path=mask_path)
    test_dataset = ClimateDataset(is_train=False, stats_path=stats_path, mask_path=mask_path)
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("="*60 + "\n")
    
    return train_loader, test_loader


def print_data_shapes():
    """打印数据维度信息"""
    print("\n" + "="*60)
    print("Data Shapes Summary")
    print("="*60)
    print("\nInputs:")
    print(f"  3D: ({len(INPUTS_3D)}, {N_LEVELS}) - {len(INPUTS_3D)} variables × {N_LEVELS} levels")
    print(f"      Variables: {INPUTS_3D}")
    print(f"  2D: (7,) - {len(INPUTS_2D)} physical variables + 3 coordinates (time, lat, lon)")
    print(f"      Physical: {INPUTS_2D}")
    print(f"      Coordinates: ['time', 'lat', 'lon']")
    
    print("\nOutputs:")
    print(f"  3D: ({len(OUTPUTS_3D)}, {N_LEVELS}) - {len(OUTPUTS_3D)} variables × {N_LEVELS} levels")
    print(f"      Variables: {OUTPUTS_3D}")
    print(f"  2D: ({len(OUTPUTS_2D)},) - {len(OUTPUTS_2D)} variables")
    print(f"      Variables: {OUTPUTS_2D}")
    print("="*60 + "\n")


if __name__ == '__main__':
    """
    测试数据加载
    """
    print_data_shapes()
    
    # 创建 DataLoader (使用较小的 batch_size 和 num_workers=0 避免多进程问题)
    train_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    
    # 测试加载一个 batch
    print("\nTesting data loading...")
    for inputs, outputs in train_loader:
        print("\nBatch loaded successfully!")
        print(f"Inputs 3D shape: {inputs['inputs_3d'].shape}")  # (batch_size, 10, 30)
        print(f"Inputs 2D shape: {inputs['inputs_2d'].shape}")  # (batch_size, 7)
        print(f"Outputs 3D shape: {outputs['outputs_3d'].shape}")  # (batch_size, 10, 30)
        print(f"Outputs 2D shape: {outputs['outputs_2d'].shape}")  # (batch_size, 7)
        
        # 检查数值范围（归一化后应该接近标准正态分布）
        print(f"\nInputs 3D - mean: {inputs['inputs_3d'].mean():.4f}, std: {inputs['inputs_3d'].std():.4f}")
        print(f"Inputs 2D - mean: {inputs['inputs_2d'].mean():.4f}, std: {inputs['inputs_2d'].std():.4f}")
        print(f"Outputs 3D - mean: {outputs['outputs_3d'].mean():.4f}, std: {outputs['outputs_3d'].std():.4f}")
        print(f"Outputs 2D - mean: {outputs['outputs_2d'].mean():.4f}, std: {outputs['outputs_2d'].std():.4f}")
        break
    
    print("\n" + "="*60)
    print("Dataset test completed successfully!")
    print("Files are loaded on-demand, only one file in memory at a time")
    print("="*60)
