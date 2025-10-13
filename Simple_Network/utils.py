"""
Utility functions for the Simple Network project
包含通用工具函数
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """
    设置计算设备，优先使用GPU
    
    Returns:
        torch.device: 可用的计算设备
    """
    try:
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("使用GPU加速")
            # 清空GPU缓存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')
            logger.info("使用CPU")
    except Exception as e:
        logger.warning(f"设置设备时出现问题: {e}，使用CPU")
        device = torch.device('cpu')
    
    return device


def get_system_info() -> Dict[str, Any]:
    """
    获取系统信息
    
    Returns:
        Dict: 系统信息字典
    """
    info = {
        'torch_available': True,
        'cuda_available': False
    }
    
    # 获取PyTorch版本
    try:
        if hasattr(torch, '__version__'):
            info['torch_version'] = torch.__version__
    except:
        info['torch_version'] = 'unknown'
    
    # 检查CUDA可用性
    try:
        if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available'):
            info['cuda_available'] = torch.cuda.is_available()
            if info['cuda_available'] and hasattr(torch.cuda, 'device_count'):
                info['gpu_count'] = torch.cuda.device_count()
    except:
        info['cuda_available'] = False
    
    return info


def calculate_memory_usage(X: torch.Tensor, Y: torch.Tensor) -> Dict[str, float]:
    """
    计算数据的内存占用
    
    Args:
        X: 输入数据张量
        Y: 输出数据张量
        
    Returns:
        Dict: 内存使用统计
    """
    x_memory_bytes = X.element_size() * X.nelement()
    y_memory_bytes = Y.element_size() * Y.nelement()
    total_memory_bytes = x_memory_bytes + y_memory_bytes
    
    memory_stats = {
        'input_memory_mb': x_memory_bytes / (1024**2),
        'output_memory_mb': y_memory_bytes / (1024**2),
        'total_memory_mb': total_memory_bytes / (1024**2),
        'total_memory_gb': total_memory_bytes / (1024**3),
        'input_shape': tuple(X.shape),
        'output_shape': tuple(Y.shape),
        'input_dtype': str(X.dtype),
        'output_dtype': str(Y.dtype)
    }
    
    return memory_stats


def set_seed(seed: int = 42):
    """
    设置随机种子以确保结果可重现
    
    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 尝试设置CUDA种子
    try:
        if hasattr(torch, 'cuda'):
            if hasattr(torch.cuda, 'manual_seed'):
                torch.cuda.manual_seed(seed)
            if hasattr(torch.cuda, 'manual_seed_all'):
                torch.cuda.manual_seed_all(seed)
    except:
        pass
    
    # 尝试设置CUDNN确定性
    try:
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass
    
    logger.info(f"随机种子设置为: {seed}")


def save_config(config_dict: Dict[str, Any], filepath: str):
    """
    保存配置到JSON文件
    
    Args:
        config_dict: 配置字典
        filepath: 保存路径
    """
    try:
        # 转换Path对象为字符串
        serializable_config = {}
        for key, value in config_dict.items():
            if isinstance(value, Path):
                serializable_config[key] = str(value)
            else:
                serializable_config[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已保存到: {filepath}")
    except Exception as e:
        logger.error(f"保存配置时出错: {e}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    从JSON文件加载配置
    
    Args:
        filepath: 配置文件路径
        
    Returns:
        Dict: 配置字典
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"配置已从 {filepath} 加载")
        return config
    except Exception as e:
        logger.error(f"加载配置时出错: {e}")
        return {}


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}小时{minutes}分{secs:.1f}秒"


def format_number(number: float, unit: str = "", precision: int = 2) -> str:
    """
    格式化数字显示，自动选择合适的单位
    
    Args:
        number: 数字
        unit: 基本单位
        precision: 小数精度
        
    Returns:
        str: 格式化的数字字符串
    """
    if number >= 1e9:
        return f"{number/1e9:.{precision}f}B{unit}"
    elif number >= 1e6:
        return f"{number/1e6:.{precision}f}M{unit}"
    elif number >= 1e3:
        return f"{number/1e3:.{precision}f}K{unit}"
    else:
        return f"{number:.{precision}f}{unit}"


def print_system_summary():
    """打印系统信息摘要"""
    info = get_system_info()
    
    print("="*60)
    print("系统信息摘要")
    print("="*60)
    print(f"PyTorch版本: {info.get('torch_version', 'unknown')}")
    print(f"CUDA可用: {info['cuda_available']}")
    
    if info['cuda_available']:
        gpu_count = info.get('gpu_count', 'unknown')
        print(f"GPU数量: {gpu_count}")
    
    print("="*60)


def create_directory_if_not_exists(path: str):
    """如果目录不存在则创建"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_model_summary(model: nn.Module) -> str:
    """获取模型摘要信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
模型摘要:
- 模型类型: {model.__class__.__name__}
- 总参数数: {format_number(total_params)}
- 可训练参数: {format_number(trainable_params)}
- 模型大小: {total_params * 4 / (1024*1024):.2f} MB (float32)
    """
    
    return summary


def validate_data_shapes(X: torch.Tensor, Y: torch.Tensor, 
                        expected_input_dim: Optional[int] = None,
                        expected_output_dim: Optional[int] = None):
    """
    验证数据形状
    
    Args:
        X: 输入数据
        Y: 输出数据
        expected_input_dim: 期望的输入维度
        expected_output_dim: 期望的输出维度
    """
    logger.info("验证数据形状...")
    
    if len(X.shape) != 2:
        raise ValueError(f"输入数据应该是2D张量，实际形状: {X.shape}")
    
    if len(Y.shape) != 2:
        raise ValueError(f"输出数据应该是2D张量，实际形状: {Y.shape}")
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"输入和输出的样本数不匹配: {X.shape[0]} vs {Y.shape[0]}")
    
    if expected_input_dim is not None and X.shape[1] != expected_input_dim:
        logger.warning(f"输入维度不匹配期望值: 实际={X.shape[1]}, 期望={expected_input_dim}")
    
    if expected_output_dim is not None and Y.shape[1] != expected_output_dim:
        logger.warning(f"输出维度不匹配期望值: 实际={Y.shape[1]}, 期望={expected_output_dim}")
    
    logger.info(f"数据形状验证通过: X={X.shape}, Y={Y.shape}")


def check_tensor_health(tensor: torch.Tensor, name: str = "tensor"):
    """
    检查张量的健康状态（NaN, Inf等）
    
    Args:
        tensor: 要检查的张量
        name: 张量名称
    """
    if torch.isnan(tensor).any():
        logger.warning(f"{name} 包含 NaN 值")
        
    if torch.isinf(tensor).any():
        logger.warning(f"{name} 包含 Inf 值")
        
    if (tensor == 0).all():
        logger.warning(f"{name} 全部为零")
        
    # 打印统计信息
    logger.info(f"{name} 统计: min={tensor.min():.6f}, max={tensor.max():.6f}, "
               f"mean={tensor.mean():.6f}, std={tensor.std():.6f}")


class ProgressTracker:
    """训练进度跟踪器"""
    
    def __init__(self):
        self.start_time = None
        self.epoch_times = []
        
    def start_training(self):
        """开始训练计时"""
        import time
        self.start_time = time.time()
        
    def end_epoch(self):
        """结束一个epoch"""
        import time
        if self.start_time is not None:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            return epoch_time
        return 0
        
    def get_eta(self, current_epoch: int, total_epochs: int) -> str:
        """估算剩余时间"""
        if len(self.epoch_times) == 0:
            return "未知"
            
        avg_epoch_time = np.mean(self.epoch_times[-5:])  # 使用最近5个epoch的平均时间
        remaining_epochs = total_epochs - current_epoch - 1
        eta_seconds = avg_epoch_time * remaining_epochs
        
        return format_time(eta_seconds)
        
    def get_total_time(self) -> str:
        """获取总训练时间"""
        if len(self.epoch_times) == 0:
            return "0秒"
        
        total_seconds = sum(self.epoch_times)
        return format_time(total_seconds)