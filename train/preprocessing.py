import math
from pathlib import Path
import xarray as xr
import numpy as np
import dask


data = ["/home/ET/mnwong/ML/data/Qobs10_SPCAMM.000.cam.h1.0001-02-13-00800.nc",
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
output_variable3D = ['SPDQ', 'SPDQC', 'SPDQI', 'SPNC', 'SPNI', 'SPDT', 'CLOUD', 'CLOUDTOP', 'QRL', 'QRS']
output_variable2D = ['PRECC', 'PRECSC', 'FSNT', 'FSDS', 'FSNS', 'FLNS', 'FLNT']


def create_spatial_mask(lats, lons, lat_threshold=60):
    """
    创建空间采样掩码（仅lat, lon维度）
    对于纬度绝对值大于lat_threshold的区域，按cos(|lat|)比例进行经度采样
    
    Args:
        lats: 纬度数组
        lons: 经度数组
        lat_threshold: 纬度阈值，默认60度
    
    Returns:
        spatial_mask: 布尔数组，形状为(lat, lon)
    """
    # 创建空间掩码，形状为(lat, lon)
    spatial_mask = np.ones((len(lats), len(lons)), dtype=bool)
    
    # 应用极地采样策略
    for i, lat in enumerate(lats):
        abs_lat = abs(lat)
        if abs_lat > lat_threshold:
            # 计算采样比例
            sampling_ratio = math.cos(math.radians(abs_lat))
            
            # 计算需要采样的经度点数
            n_lons_to_sample = max(1, int(len(lons) * sampling_ratio))
            
            # 均匀采样经度索引
            lon_indices = np.linspace(0, len(lons)-1, n_lons_to_sample, dtype=int)
            
            # 创建该纬度的掩码（只保留采样的经度点）
            lat_mask = np.zeros(len(lons), dtype=bool)
            lat_mask[lon_indices] = True
            spatial_mask[i, :] = lat_mask
    
    return spatial_mask


def create_training_mask(n_times, spatial_mask, train_ratio=0.8, random_seed=42):
    """
    基于空间掩码创建训练掩码，在时空三维上随机采样
    
    Args:
        n_times: 时间步数
        spatial_mask: 空间掩码，形状为(lat, lon)
        train_ratio: 训练集比例，默认0.8(80%)
        random_seed: 随机种子，默认42
    
    Returns:
        training_mask: 布尔数组，形状为(time, lat, lon)
    """
    np.random.seed(random_seed)
    
    # 将空间掩码扩展到时间维度
    full_mask = np.broadcast_to(spatial_mask[np.newaxis, :, :], (n_times, *spatial_mask.shape)).copy()
    
    # 在三个维度上随机采样train_ratio比例的数据点
    total_points = full_mask.sum()
    n_train_points = int(total_points * train_ratio)
    
    # 获取所有True的位置索引
    true_indices = np.argwhere(full_mask)
    
    # 随机打乱索引
    np.random.shuffle(true_indices)
    
    # 选择前train_ratio比例的点作为训练集
    train_indices = true_indices[:n_train_points]
    
    # 创建最终掩码
    training_mask = np.zeros((n_times, *spatial_mask.shape), dtype=bool)
    training_mask[train_indices[:, 0], train_indices[:, 1], train_indices[:, 2]] = True
    
    return training_mask


# 先验证哪些文件可以正常打开
print("\nValidating data files...")
valid_files = []
for i, file_path in enumerate(data):
    try:
        # 尝试打开文件
        test_ds = xr.open_dataset(file_path)
        test_ds.close()
        valid_files.append(file_path)
        print(f"  ✓ File {i+1}/{len(data)}: {Path(file_path).name}")
    except Exception as e:
        print(f"  ✗ File {i+1}/{len(data)}: {Path(file_path).name} - SKIPPED (Error: {str(e)[:100]})")

print(f"\nValid files: {len(valid_files)}/{len(data)}")

if len(valid_files) == 0:
    raise ValueError("No valid files found!")

# 使用 open_mfdataset 批量加载所有有效文件（延迟加载，大幅提速）
print("\nLoading valid data files with open_mfdataset (lazy loading)...")
ds_full = xr.open_mfdataset(
    valid_files, 
    combine='by_coords',
    parallel=True,  # 并行加载
    chunks={'time': 10}  # 使用 dask 分块处理
)
print(f"Dataset loaded (lazy): time={len(ds_full.time)}, lat={len(ds_full.lat)}, lon={len(ds_full.lon)}")

# 获取坐标信息（只需要读取维度大小，非常快）
lats = ds_full.lat.values
lons = ds_full.lon.values
n_times = len(ds_full.time)

# 创建空间掩码（只依赖lat/lon，与时间无关）
print("\nCreating spatial mask...")
spatial_mask = create_spatial_mask(lats, lons, lat_threshold=60)
print(f"Spatial mask shape: {spatial_mask.shape}, Spatial samples: {spatial_mask.sum()} / {spatial_mask.size}")

# 创建训练掩码
print("\nCreating training mask...")
training_mask = create_training_mask(n_times, spatial_mask, train_ratio=0.8, random_seed=42)
print(f"Training mask shape: {training_mask.shape}")
print(f"Total training samples: {training_mask.sum()} / {training_mask.size} ({100*training_mask.sum()/training_mask.size:.2f}%)")

# 统计字典
stats = {
    'inputs_2D': {},
    'inputs_3D': {},
    'outputs_2D': {},
    'outputs_3D': {}
}

# 处理2D输入变量
print("\nProcessing 2D input variables...")
for var_name in inputs_variable2D:
    if var_name in ds_full:
        print(f"  Processing {var_name}...")
        # 只加载需要的数据（延迟加载）
        var_data = ds_full[var_name].values  # shape: (time, lat, lon)
        
        # 使用训练掩码提取训练数据
        train_data = var_data[training_mask]
        
        # 计算均值和标准差
        mean = float(np.mean(train_data))
        std = float(np.std(train_data))
        
        stats['inputs_2D'][var_name] = {
            'mean': mean,
            'std': std
        }
        print(f"    Mean: {mean:.6e}, Std: {std:.6e}")
    else:
        print(f"  Warning: {var_name} not found in dataset")

# 处理time, lat, lon坐标变量
print("\nProcessing coordinate variables (time, lat, lon)...")

# 处理time
print(f"  Processing time...")
time_values = ds_full.time.values
# 处理cftime对象，将其转换为从第一个时间点开始的秒数
if hasattr(time_values[0], 'year'):  # 检查是否是cftime对象
    # 将cftime对象转换为从第一个时间点开始的总秒数
    time_numeric = np.array([(t.year * 365.25 * 24 * 3600 + 
                              t.month * 30 * 24 * 3600 + 
                              t.day * 24 * 3600 + 
                              t.hour * 3600 + 
                              t.minute * 60 + 
                              t.second) for t in time_values])
    # 归一化：减去第一个时间点的值
    time_numeric = time_numeric - time_numeric[0]
else:
    # 处理标准的datetime64对象
    time_values = np.array(time_values, dtype='datetime64[ns]')
    time_numeric = (time_values - time_values[0]).astype('timedelta64[s]').astype(float)
    
# 创建与training_mask相同形状的time数组
time_grid = np.broadcast_to(time_numeric[:, np.newaxis, np.newaxis], training_mask.shape)
train_time = time_grid[training_mask]
mean_time = float(np.mean(train_time))
std_time = float(np.std(train_time))
stats['inputs_2D']['time'] = {
    'mean': mean_time,
    'std': std_time
}
print(f"    Mean: {mean_time:.6e}, Std: {std_time:.6e}")

# 处理lat
print(f"  Processing lat...")
lat_values = ds_full.lat.values
# 创建与training_mask相同形状的lat数组
lat_grid = np.broadcast_to(lat_values[np.newaxis, :, np.newaxis], training_mask.shape)
train_lat = lat_grid[training_mask]
mean_lat = float(np.mean(train_lat))
std_lat = float(np.std(train_lat))
stats['inputs_2D']['lat'] = {
    'mean': mean_lat,
    'std': std_lat
}
print(f"    Mean: {mean_lat:.6e}, Std: {std_lat:.6e}")

# 处理lon
print(f"  Processing lon...")
lon_values = ds_full.lon.values
# 创建与training_mask相同形状的lon数组
lon_grid = np.broadcast_to(lon_values[np.newaxis, np.newaxis, :], training_mask.shape)
train_lon = lon_grid[training_mask]
mean_lon = float(np.mean(train_lon))
std_lon = float(np.std(train_lon))
stats['inputs_2D']['lon'] = {
    'mean': mean_lon,
    'std': std_lon
}
print(f"    Mean: {mean_lon:.6e}, Std: {std_lon:.6e}")

# 处理2D输出变量
print("\nProcessing 2D output variables...")
for var_name in output_variable2D:
    if var_name in ds_full:
        print(f"  Processing {var_name}...")
        var_data = ds_full[var_name].values  # shape: (time, lat, lon)
        
        # 使用训练掩码提取训练数据
        train_data = var_data[training_mask]
        
        # 计算均值和标准差
        mean = float(np.mean(train_data))
        std = float(np.std(train_data))
        
        stats['outputs_2D'][var_name] = {
            'mean': mean,
            'std': std
        }
        print(f"    Mean: {mean:.6e}, Std: {std:.6e}")
    else:
        print(f"  Warning: {var_name} not found in dataset")

# 处理3D输入变量（前30层）
print("\nProcessing 3D input variables (first 30 levels)...")
for var_name in inputs_variable3D:
    if var_name in ds_full:
        print(f"  Processing {var_name}...")
        # HEIGHT 使用 ilev 维度（31层），其他变量使用 lev 维度（30层）
        if var_name == 'HEIGHT':
            # HEIGHT 有31个 ilev，取前30个
            var_data = ds_full[var_name].isel(ilev=slice(0, 30)).values  # shape: (time, 30, lat, lon)
        else:
            # 其他变量有30个 lev
            var_data = ds_full[var_name].isel(lev=slice(0, 30)).values  # shape: (time, 30, lat, lon)
        
        # 初始化该变量的统计信息
        means_list = []
        stds_list = []
        
        # 对每一层分别计算统计信息
        for lev in range(30):
            # 提取该层的数据
            lev_data = var_data[:, lev, :, :]
            
            # 使用训练掩码提取训练数据
            train_data = lev_data[training_mask]
            
            # 计算均值和标准差
            means_list.append(float(np.mean(train_data)))
            stds_list.append(float(np.std(train_data)))
        
        # 转换为numpy数组
        stats['inputs_3D'][var_name] = {
            'mean': np.array(means_list),
            'std': np.array(stds_list)
        }
        
        print(f"    Mean range: [{stats['inputs_3D'][var_name]['mean'].min():.6e}, {stats['inputs_3D'][var_name]['mean'].max():.6e}]")
        print(f"    Std range: [{stats['inputs_3D'][var_name]['std'].min():.6e}, {stats['inputs_3D'][var_name]['std'].max():.6e}]")
    else:
        print(f"  Warning: {var_name} not found in dataset")

# 处理3D输出变量（前30层）
print("\nProcessing 3D output variables (first 30 levels)...")
for var_name in output_variable3D:
    if var_name in ds_full:
        print(f"  Processing {var_name}...")
        # 只加载前30层数据
        var_data = ds_full[var_name].isel(lev=slice(0, 30)).values  # shape: (time, 30, lat, lon)
        
        # 初始化该变量的统计信息
        means_list = []
        stds_list = []
        
        # 对每一层分别计算统计信息
        for lev in range(30):
            # 提取该层的数据
            lev_data = var_data[:, lev, :, :]
            
            # 使用训练掩码提取训练数据
            train_data = lev_data[training_mask]
            
            # 计算均值和标准差
            means_list.append(float(np.mean(train_data)))
            stds_list.append(float(np.std(train_data)))
        
        # 转换为numpy数组
        stats['outputs_3D'][var_name] = {
            'mean': np.array(means_list),
            'std': np.array(stds_list)
        }
        
        print(f"    Mean range: [{stats['outputs_3D'][var_name]['mean'].min():.6e}, {stats['outputs_3D'][var_name]['mean'].max():.6e}]")
        print(f"    Std range: [{stats['outputs_3D'][var_name]['std'].min():.6e}, {stats['outputs_3D'][var_name]['std'].max():.6e}]")
    else:
        print(f"  Warning: {var_name} not found in dataset")

# 保存统计信息（使用 allow_pickle=True 保存字典）
print("\nSaving statistics...")
np.save('training_stats.npy', stats, allow_pickle=True)
print("Statistics saved to 'training_stats.npy'")

# 保存训练掩码（用于区分训练集和测试集）
print("\nSaving training mask...")
np.save('training_mask.npy', training_mask)
print("Training mask saved to 'training_mask.npy'")
print(f"  Training samples (True): {training_mask.sum()}")
print(f"  Test samples (False): {(~training_mask).sum()}")

print("\nDone!")
