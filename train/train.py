import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
import os


# 数据文件列表
data_files = ["/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-13-00800.nc",
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
            "/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-17-65600.nc"]

inputs_variable3D = ['U', 'V', 'T', 'Q', 'CLDLIQ', 'CLDICE', 'PMID', 'DPRES', 'Z3', 'HEIGHT']
inputs_variable2D = ['TAUX', 'TAUY', 'SHFLX', 'LHFLX']
inputs_coordinates = ['time', 'lat', 'lon']  # 坐标变量也需要归一化
output_variable3D = ['SPDQ', 'SPDQC', 'SPDQI', 'SPNC', 'SPNI', 'SPDT', 'CLOUD', 'CLOUDTOP', 'QRL', 'QRS']
output_variable2D = ['PRECC', 'PRECSC', 'FSNT', 'FSDS', 'FSNS', 'FLNS', 'FLNT']

class ClimateDataset(Dataset):
    """
    自定义Dataset类用于读取气候数据
    支持批量读取NetCDF文件并使用预计算的统计信息进行标准化
    """
    def __init__(self, file_list, stats_path, 
                 inputs_3d_vars, inputs_2d_vars, inputs_coord_vars,
                 outputs_3d_vars, outputs_2d_vars,
                 transform=None):
        """
        初始化Dataset
        
        Args:
            file_list: NetCDF文件路径列表
            stats_path: training_stats.npy文件路径
            inputs_3d_vars: 3D输入变量名列表
            inputs_2d_vars: 2D输入变量名列表
            inputs_coord_vars: 坐标变量名列表 (time, lat, lon)
            outputs_3d_vars: 3D输出变量名列表
            outputs_2d_vars: 2D输出变量名列表
            transform: 可选的额外数据转换函数
        """
        self.file_list = file_list
        self.inputs_3d_vars = inputs_3d_vars
        self.inputs_2d_vars = inputs_2d_vars
        self.inputs_coord_vars = inputs_coord_vars
        self.outputs_3d_vars = outputs_3d_vars
        self.outputs_2d_vars = outputs_2d_vars
        self.transform = transform
        
        # 加载统计信息
        self.stats = np.load(stats_path, allow_pickle=True).item()
        
        # 构建文件索引：记录每个样本属于哪个文件及其在文件中的位置
        self.file_indices = []
        self._build_index()
        
    def _build_index(self):
        """
        构建索引：统计所有文件中的样本数量
        每个文件可能有不同的时间步数
        """
        print("Building dataset index...")
        for file_idx, file_path in enumerate(self.file_list):
            try:
                # 只读取文件维度信息，不加载实际数据
                with xr.open_dataset(file_path) as ds:
                    n_times = len(ds['time'])
                    n_lats = len(ds['lat'])
                    n_lons = len(ds['lon'])
                    
                    # 为这个文件的每个时间步、纬度、经度组合创建索引
                    for t in range(n_times):
                        for lat in range(n_lats):
                            for lon in range(n_lons):
                                self.file_indices.append({
                                    'file_idx': file_idx,
                                    'time_idx': t,
                                    'lat_idx': lat,
                                    'lon_idx': lon
                                })
                print(f"File {file_idx+1}/{len(self.file_list)}: {n_times} times x {n_lats} lats x {n_lons} lons")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
        print(f"Total samples: {len(self.file_indices)}")
    
    def _normalize_variable(self, data, var_name, var_type):
        """
        使用预计算的统计信息标准化变量
        
        Args:
            data: 原始数据 (numpy array)
            var_name: 变量名
            var_type: 变量类型 ('inputs_2D', 'inputs_3D', 'outputs_2D', 'outputs_3D')
        
        Returns:
            标准化后的数据
        """
        if var_type in self.stats and var_name in self.stats[var_type]:
            mean = self.stats[var_type][var_name]['mean']
            std = self.stats[var_type][var_name]['std']
            
            # 避免除以零
            std = np.where(std == 0, 1, std)
            
            # 标准化: (x - mean) / std
            normalized = (data - mean) / std
            return normalized
        else:
            print(f"Warning: No stats found for {var_name} in {var_type}, returning original data")
            return data
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            inputs: 标准化后的输入数据 (包含2D和3D变量)
            outputs: 标准化后的输出数据 (包含2D和3D变量)
        """
        # 获取索引信息
        index_info = self.file_indices[idx]
        file_idx = index_info['file_idx']
        time_idx = index_info['time_idx']
        lat_idx = index_info['lat_idx']
        lon_idx = index_info['lon_idx']
        
        file_path = self.file_list[file_idx]
        
        try:
            # 读取数据（只读取需要的单个样本）
            with xr.open_dataset(file_path) as ds:
                inputs_list = []
                outputs_list = []
                
                # 读取3D输入变量 (time, lat, lon, lev)
                for var in self.inputs_3d_vars:
                    if var in ds:
                        # 提取特定时间、纬度、经度的所有垂直层数据
                        if var == 'HEIGHT':
                            # HEIGHT 使用 ilev 维度，取前30层
                            data = ds[var].isel(time=time_idx, lat=lat_idx, lon=lon_idx, ilev=slice(0, 30)).values
                        else:
                            # 其他变量使用 lev 维度
                            data = ds[var].isel(time=time_idx, lat=lat_idx, lon=lon_idx).values
                        # 标准化
                        data_norm = self._normalize_variable(data, var, 'inputs_3D')
                        inputs_list.append(data_norm)
                
                # 读取2D输入变量 (time, lat, lon)
                for var in self.inputs_2d_vars:
                    if var in ds:
                        data = ds[var].isel(time=time_idx, lat=lat_idx, lon=lon_idx).values
                        # 标准化
                        data_norm = self._normalize_variable(data, var, 'inputs_2D')
                        # 转换为1D数组以便拼接
                        inputs_list.append(np.array([data_norm]))
                
                # 读取坐标变量 (time, lat, lon) - 参考preprocessing.py的处理方式
                for coord_var in self.inputs_coord_vars:
                    if coord_var == 'time':
                        # 读取time坐标值
                        time_value = ds['time'].values[time_idx]
                        # 处理cftime对象，转换为数值
                        if hasattr(time_value, 'year'):  # cftime对象
                            time_numeric = (time_value.year * 365.25 * 24 * 3600 + 
                                          time_value.month * 30 * 24 * 3600 + 
                                          time_value.day * 24 * 3600 + 
                                          time_value.hour * 3600 + 
                                          time_value.minute * 60 + 
                                          time_value.second)
                        else:  # datetime64对象
                            time_numeric = float(time_value)
                        # 标准化
                        data_norm = self._normalize_variable(time_numeric, 'time', 'inputs_2D')
                        inputs_list.append(np.array([data_norm]))
                    
                    elif coord_var == 'lat':
                        # 读取lat坐标值
                        lat_value = ds['lat'].values[lat_idx]
                        # 标准化
                        data_norm = self._normalize_variable(lat_value, 'lat', 'inputs_2D')
                        inputs_list.append(np.array([data_norm]))
                    
                    elif coord_var == 'lon':
                        # 读取lon坐标值
                        lon_value = ds['lon'].values[lon_idx]
                        # 标准化
                        data_norm = self._normalize_variable(lon_value, 'lon', 'inputs_2D')
                        inputs_list.append(np.array([data_norm]))
                
                # 读取3D输出变量
                for var in self.outputs_3d_vars:
                    if var in ds:
                        data = ds[var].isel(time=time_idx, lat=lat_idx, lon=lon_idx).values
                        # 标准化
                        data_norm = self._normalize_variable(data, var, 'outputs_3D')
                        outputs_list.append(data_norm)
                
                # 读取2D输出变量
                for var in self.outputs_2d_vars:
                    if var in ds:
                        data = ds[var].isel(time=time_idx, lat=lat_idx, lon=lon_idx).values
                        # 标准化
                        data_norm = self._normalize_variable(data, var, 'outputs_2D')
                        outputs_list.append(np.array([data_norm]))
                
                # 拼接所有输入和输出
                inputs = np.concatenate(inputs_list)
                outputs = np.concatenate(outputs_list)
                
                # 转换为PyTorch张量
                inputs = torch.from_numpy(inputs).float()
                outputs = torch.from_numpy(outputs).float()
                
                if self.transform:
                    inputs, outputs = self.transform(inputs, outputs)
                
                return inputs, outputs
                
        except Exception as e:
            print(f"Error loading sample {idx} from file {file_path}: {e}")
            # 返回零张量作为fallback
            n_inputs = len(self.inputs_3d_vars) * 30 + len(self.inputs_2d_vars) + len(self.inputs_coord_vars)
            n_outputs = len(self.outputs_3d_vars) * 30 + len(self.outputs_2d_vars)
            return torch.zeros(n_inputs), torch.zeros(n_outputs)


def create_dataloaders(file_list, stats_path, batch_size=32, 
                       train_ratio=0.8, num_workers=4, shuffle=True):
    """
    创建训练和验证DataLoader
    
    Args:
        file_list: 数据文件列表
        stats_path: 统计信息文件路径
        batch_size: 批次大小
        train_ratio: 训练集比例
        num_workers: 数据加载工作进程数
        shuffle: 是否打乱数据
        
    Returns:
        train_loader, val_loader: 训练和验证DataLoader
    """
    # 创建完整数据集
    full_dataset = ClimateDataset(
        file_list=file_list,
        stats_path=stats_path,
        inputs_3d_vars=inputs_variable3D,
        inputs_2d_vars=inputs_variable2D,
        inputs_coord_vars=inputs_coordinates,
        outputs_3d_vars=output_variable3D,
        outputs_2d_vars=output_variable2D
    )
    
    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU传输
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# 使用示例
if __name__ == "__main__":
    # 设置路径（使用绝对路径或相对于脚本的路径）
    stats_path = os.path.join(os.path.dirname(__file__), "training_stats.npy")
    
    # 创建DataLoader
    train_loader, val_loader = create_dataloaders(
        file_list=data_files,
        stats_path=stats_path,
        batch_size=64,
        train_ratio=0.8,
        num_workers=4,
        shuffle=True
    )
    
    # 测试加载一个批次
    print("\n" + "="*50)
    print("Testing data loading...")
    print("\nInput feature breakdown:")
    print(f"  3D variables: {len(inputs_variable3D)} vars × 30 levels = {len(inputs_variable3D) * 30} features")
    print(f"  2D variables: {len(inputs_variable2D)} vars = {len(inputs_variable2D)} features")
    print(f"  Coordinates: {len(inputs_coordinates)} vars (time, lat, lon) = {len(inputs_coordinates)} features")
    print(f"  Total input features: {len(inputs_variable3D) * 30 + len(inputs_variable2D) + len(inputs_coordinates)}")
    print(f"\nOutput feature breakdown:")
    print(f"  3D variables: {len(output_variable3D)} vars × 30 levels = {len(output_variable3D) * 30} features")
    print(f"  2D variables: {len(output_variable2D)} vars = {len(output_variable2D)} features")
    print(f"  Total output features: {len(output_variable3D) * 30 + len(output_variable2D)}")
    
    for batch_idx, (inputs, outputs) in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
        print(f"  Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
        
        # 只测试第一个批次
        if batch_idx == 0:
            break
    
    print("\nDataLoader created successfully!")


