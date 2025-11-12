"""
数据加载模块 - 使用训练掩码区分训练集和测试集
"""
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

from config import *


class ClimateDataset(Dataset):
    """气候数据集 - 支持训练集和测试集"""
    
    def __init__(self, file_list, stats_path, mask_path, is_train=True):
        """
        初始化数据集
        
        Args:
            file_list: NetCDF文件路径列表
            stats_path: 统计信息文件路径（训练集的mean和std）
            mask_path: 训练掩码文件路径
            is_train: True=训练集, False=测试集
        """
        self.file_list = file_list
        self.is_train = is_train
        
        # 加载统计信息（训练集的）
        self.stats = np.load(stats_path, allow_pickle=True).item()
        
        # 加载训练掩码
        training_mask = np.load(mask_path)
        # 根据is_train选择对应的掩码
        self.mask = training_mask if is_train else ~training_mask
        
        # 构建索引
        self.indices = []
        self._build_index()
        
    def _build_index(self):
        """构建数据索引"""
        print(f"\nBuilding {'training' if self.is_train else 'test'} dataset index...")
        
        for file_idx, file_path in enumerate(self.file_list):
            try:
                with xr.open_dataset(file_path) as ds:
                    n_times = len(ds['time'])
                    n_lats = len(ds['lat'])
                    n_lons = len(ds['lon'])
                    
                    # 为该文件构建索引
                    for t in range(n_times):
                        for lat in range(n_lats):
                            for lon in range(n_lons):
                                # 只添加掩码为True的样本
                                if self.mask[t, lat, lon]:
                                    self.indices.append({
                                        'file_idx': file_idx,
                                        't': t, 'lat': lat, 'lon': lon
                                    })
                
                print(f"  File {file_idx+1}/{len(self.file_list)}: {len(self.indices)} samples")
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
        
        print(f"Total samples: {len(self.indices)}")
    
    def _normalize(self, data, var_name, var_type):
        """标准化数据（使用训练集的mean和std）"""
        if var_type in self.stats and var_name in self.stats[var_type]:
            mean = self.stats[var_type][var_name]['mean']
            std = self.stats[var_type][var_name]['std']
            std = np.where(std == 0, 1, std)
            return (data - mean) / std
        return data
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        info = self.indices[idx]
        file_path = self.file_list[info['file_idx']]
        t, lat, lon = info['t'], info['lat'], info['lon']
        
        try:
            with xr.open_dataset(file_path) as ds:
                inputs_list = []
                outputs_list = []
                
                # 读取3D输入
                for var in INPUTS_3D:
                    if var in ds:
                        if var == 'HEIGHT':
                            data = ds[var].isel(time=t, lat=lat, lon=lon, ilev=slice(0, N_LEVELS)).values
                        else:
                            data = ds[var].isel(time=t, lat=lat, lon=lon).values
                        inputs_list.append(self._normalize(data, var, 'inputs_3D'))
                
                # 读取2D输入
                for var in INPUTS_2D:
                    if var in ds:
                        data = ds[var].isel(time=t, lat=lat, lon=lon).values
                        inputs_list.append(np.array([self._normalize(data, var, 'inputs_2D')]))
                
                # 读取坐标
                time_val = ds['time'].values[t]
                if hasattr(time_val, 'year'):
                    time_val = (time_val.year * 365.25 * 24 * 3600 + 
                               time_val.month * 30 * 24 * 3600 + 
                               time_val.day * 24 * 3600 + 
                               time_val.hour * 3600 + 
                               time_val.minute * 60 + 
                               time_val.second)
                else:
                    time_val = float(time_val)
                
                inputs_list.append(np.array([self._normalize(time_val, 'time', 'inputs_2D')]))
                inputs_list.append(np.array([self._normalize(ds['lat'].values[lat], 'lat', 'inputs_2D')]))
                inputs_list.append(np.array([self._normalize(ds['lon'].values[lon], 'lon', 'inputs_2D')]))
                
                # 读取3D输出
                for var in OUTPUTS_3D:
                    if var in ds:
                        data = ds[var].isel(time=t, lat=lat, lon=lon).values
                        outputs_list.append(self._normalize(data, var, 'outputs_3D'))
                
                # 读取2D输出
                for var in OUTPUTS_2D:
                    if var in ds:
                        data = ds[var].isel(time=t, lat=lat, lon=lon).values
                        outputs_list.append(np.array([self._normalize(data, var, 'outputs_2D')]))
                
                # 拼接并转换为张量
                inputs = torch.from_numpy(np.concatenate(inputs_list)).float()
                outputs = torch.from_numpy(np.concatenate(outputs_list)).float()
                
                return inputs, outputs
                
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            n_in = len(INPUTS_3D) * N_LEVELS + len(INPUTS_2D) + len(INPUTS_COORDS)
            n_out = len(OUTPUTS_3D) * N_LEVELS + len(OUTPUTS_2D)
            return torch.zeros(n_in), torch.zeros(n_out)


def create_dataloaders(batch_size=128, num_workers=4):
    """
    创建训练和测试DataLoader
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, test_loader
    """
    stats_path = get_stats_path()
    mask_path = get_train_mask_path()
    
    # 创建训练集
    train_dataset = ClimateDataset(
        file_list=DATA_FILES,
        stats_path=stats_path,
        mask_path=mask_path,
        is_train=True
    )
    
    # 创建测试集
    test_dataset = ClimateDataset(
        file_list=DATA_FILES,
        stats_path=stats_path,
        mask_path=mask_path,
        is_train=False
    )
    
    # 创建DataLoader
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
    
    return train_loader, test_loader


if __name__ == "__main__":
    """测试数据加载"""
    print_config()
    
    train_loader, test_loader = create_dataloaders(batch_size=64, num_workers=2)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 测试加载一个批次
    print("\nTesting data loading...")
    for inputs, outputs in train_loader:
        print(f"Input shape: {inputs.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
        print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
        break
    
    print("\n✓ Data loading test passed!")
