"""
Data processing utilities for climate data
包含数据加载、预处理和极地采样功能
"""

import math
import numpy as np
import torch
import xarray as xr
from pathlib import Path
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClimateDataProcessor:
    """气候数据处理器，负责数据加载、采样和预处理"""
    
    def __init__(self, data_path: Path, lat_threshold: float = 60):
        """
        初始化数据处理器
        
        Args:
            data_path: 数据文件路径
            lat_threshold: 纬度阈值，用于极地采样
        """
        self.data_path = data_path
        self.lat_threshold = lat_threshold
        self.dataset = None
        self.sampling_mask = None
        
    def load_dataset(self) -> xr.Dataset:
        """加载数据集"""
        try:
            with xr.open_dataset(self.data_path) as ds:
                self.dataset = ds.load()  # Load data into memory
            logger.info(f"Dataset loaded successfully from {self.data_path}")
            logger.info(f"Data variables: {list(self.dataset.data_vars)}")
            logger.info(f"Coordinates: {list(self.dataset.coords)}")
            return self.dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
            
    def create_polar_sampling_mask(self) -> np.ndarray:
        """
        创建极地采样掩码
        对于纬度绝对值大于lat_threshold的区域，按cos(|lat|)比例进行经度采样
        
        Returns:
            sampling_mask: 布尔数组，形状为(lat, lon)
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        lats = self.dataset.lat.values
        lons = self.dataset.lon.values
        
        # 创建采样掩码
        sampling_mask = np.ones((len(lats), len(lons)), dtype=bool)
        
        for i, lat in enumerate(lats):
            abs_lat = abs(lat)
            if abs_lat > self.lat_threshold:
                # 计算采样比例
                sampling_ratio = math.cos(math.radians(abs_lat))
                
                # 计算需要采样的经度点数
                n_lons_to_sample = max(1, int(len(lons) * sampling_ratio))
                
                # 均匀采样经度索引
                lon_indices = np.linspace(0, len(lons)-1, n_lons_to_sample, dtype=int)
                
                # 创建该纬度的掩码（只保留采样的经度点）
                lat_mask = np.zeros(len(lons), dtype=bool)
                lat_mask[lon_indices] = True
                sampling_mask[i, :] = lat_mask
                
                logger.info(f"纬度 {lat:.1f}°: 采样比例 {sampling_ratio:.3f}, "
                           f"采样 {n_lons_to_sample}/{len(lons)} 个经度点")
        
        self.sampling_mask = sampling_mask
        logger.info(f"采样掩码形状: {sampling_mask.shape}")
        logger.info(f"总的有效采样点: {sampling_mask.sum()}/{sampling_mask.size}")
        logger.info(f"采样比例: {sampling_mask.sum()/sampling_mask.size:.3f}")
        
        return sampling_mask
    
    def process_variable_with_polar_sampling(self, var_name: str, variable_type: str = '3D') -> torch.Tensor:
        """
        处理变量并应用极地采样
        
        Args:
            var_name: 变量名
            variable_type: '3D' 表示有高度维度，'2D' 表示只有lat/lon
        
        Returns:
            reshaped_tensor: 处理后的张量
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        if self.sampling_mask is None:
            raise ValueError("Sampling mask not created. Call create_polar_sampling_mask() first.")
            
        var_data = np.array(self.dataset[var_name])
        
        if variable_type == '3D':
            # 形状: (time, height, lat, lon)
            # 应用采样掩码到lat和lon维度
            sampled_data = []
            for t in range(var_data.shape[0]):
                for h in range(var_data.shape[1]):
                    # 获取当前时间和高度的数据切片
                    data_slice = var_data[t, h, :, :]
                    # 应用采样掩码
                    sampled_points = data_slice[self.sampling_mask]
                    sampled_data.append(sampled_points)
            
            # 重塑为 (time*height*sampled_points,) 然后再重塑
            sampled_data = np.array(sampled_data)
            # 转换为 (time*sampled_points, height)
            reshaped = sampled_data.reshape(var_data.shape[0], -1, var_data.shape[1]).transpose(0, 2, 1).reshape(-1, var_data.shape[1])
            
        else:  # 2D variables
            # 形状: (time, lat, lon)
            sampled_data = []
            for t in range(var_data.shape[0]):
                data_slice = var_data[t, :, :]
                sampled_points = data_slice[self.sampling_mask]
                sampled_data.append(sampled_points)
            
            # 重塑为 (time*sampled_points, 1)
            reshaped = np.array(sampled_data).reshape(-1, 1)
        
        # 转换为tensor并标准化
        reshaped_tensor = torch.tensor(reshaped).float()
        mean = reshaped_tensor.mean(dim=0, keepdim=True)
        std = reshaped_tensor.std(dim=0, keepdim=True)
        reshaped_tensor = (reshaped_tensor - mean) / (std + 1e-6)
        
        return reshaped_tensor
    
    def process_all_variables(self, inputs_3d: List[str], inputs_2d: List[str], 
                             outputs_3d: List[str], outputs_2d: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理所有输入和输出变量
        
        Args:
            inputs_3d: 3D输入变量列表
            inputs_2d: 2D输入变量列表
            outputs_3d: 3D输出变量列表
            outputs_2d: 2D输出变量列表
        
        Returns:
            X_sampled: 输入数据张量
            Y_sampled: 输出数据张量
        """
        logger.info("开始处理输入变量...")
        tensors_to_concat = []

        # 处理3D输入变量 (有高度维度)
        for var in inputs_3d:
            logger.info(f"处理3D输入变量: {var}")
            tensor = self.process_variable_with_polar_sampling(var, '3D')
            tensors_to_concat.append(tensor)
            logger.info(f"  - 形状: {tensor.shape}")

        # 处理2D输入变量 (没有高度维度)  
        for var in inputs_2d:
            logger.info(f"处理2D输入变量: {var}")
            tensor = self.process_variable_with_polar_sampling(var, '2D')
            tensors_to_concat.append(tensor)
            logger.info(f"  - 形状: {tensor.shape}")

        X_sampled = torch.cat(tensors_to_concat, dim=1)
        logger.info(f"输入数据最终形状: {X_sampled.shape}")

        # 处理输出变量
        logger.info("开始处理输出变量...")
        tensors_to_concat_y = []

        # 处理3D输出变量
        for var in outputs_3d:
            logger.info(f"处理3D输出变量: {var}")
            tensor = self.process_variable_with_polar_sampling(var, '3D')
            tensors_to_concat_y.append(tensor)
            logger.info(f"  - 形状: {tensor.shape}")

        # 处理2D输出变量
        for var in outputs_2d:
            logger.info(f"处理2D输出变量: {var}")
            tensor = self.process_variable_with_polar_sampling(var, '2D')
            tensors_to_concat_y.append(tensor)
            logger.info(f"  - 形状: {tensor.shape}")

        Y_sampled = torch.cat(tensors_to_concat_y, dim=1)
        logger.info(f"输出数据最终形状: {Y_sampled.shape}")

        # 数据统计信息
        logger.info("采样前后数据点数比较:")
        original_points = 27 * 384 * 576  # 时间 × 纬度 × 经度
        logger.info(f"原始数据点数: {original_points:,}")
        logger.info(f"采样后数据点数: {X_sampled.shape[0]:,}")
        logger.info(f"数据压缩比: {X_sampled.shape[0] / original_points:.3f}")

        return X_sampled, Y_sampled
    
    def get_data_info(self) -> dict:
        """获取数据集信息"""
        if self.dataset is None:
            return {}
            
        info = {
            'lat_range': (self.dataset.lat.min().values, self.dataset.lat.max().values),
            'lon_range': (self.dataset.lon.min().values, self.dataset.lon.max().values),
            'lat_shape': self.dataset.lat.shape,
            'lon_shape': self.dataset.lon.shape,
            'data_vars': list(self.dataset.data_vars),
            'coords': list(self.dataset.coords)
        }
        
        if self.sampling_mask is not None:
            info['sampling_stats'] = {
                'mask_shape': self.sampling_mask.shape,
                'total_points': self.sampling_mask.sum(),
                'total_possible': self.sampling_mask.size,
                'sampling_ratio': self.sampling_mask.sum() / self.sampling_mask.size
            }
            
        return info