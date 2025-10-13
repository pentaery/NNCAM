"""
Training and evaluation utilities
包含训练循环、评估函数和相关工具
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from typing import Tuple, List, Dict, Optional
import numpy as np
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器类，封装训练和评估逻辑"""
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 huber_delta: float = 1.0):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            huber_delta: Huber损失的delta参数
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.HuberLoss(delta=huber_delta)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5, 
            verbose=True, min_lr=1e-6
        )
        
        # 训练历史
        self.train_losses = []
        self.test_losses = []
        self.best_test_loss = float('inf')
        
    def train_epoch(self, train_loader: DataLoader, gradient_clip_value: float = 1.0) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_X, batch_Y in train_pbar:
            batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_Y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_value)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.6f}', 
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / num_batches
    
    def evaluate(self, test_loader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(test_loader)
        
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_Y)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def calculate_r2_score(self, test_loader: DataLoader) -> float:
        """计算R²分数"""
        self.model.eval()
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                outputs = self.model(batch_X)
                y_true_list.append(batch_Y.cpu())
                y_pred_list.append(outputs.cpu())
        
        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)
        
        # 计算R²分数
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2.item()
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader,
              num_epochs: int = 50, patience: int = 10, 
              model_save_path: str = 'best_model.pth',
              gradient_clip_value: float = 1.0) -> Dict:
        """
        完整的训练流程
        
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            num_epochs: 最大训练轮数
            patience: 早停耐心值
            model_save_path: 模型保存路径
            gradient_clip_value: 梯度裁剪值
            
        Returns:
            训练结果字典
        """
        logger.info(f"开始训练，最大轮数: {num_epochs}, 早停耐心值: {patience}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, gradient_clip_value)
            self.train_losses.append(train_loss)
            
            # 评估
            test_loss = self.evaluate(test_loader)
            self.test_losses.append(test_loss)
            
            # 学习率调度
            self.scheduler.step(test_loss)
            
            # 早停检查
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), model_save_path)
                logger.info(f"保存新的最佳模型，测试损失: {test_loss:.6f}")
            else:
                patience_counter += 1
            
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                       f'Train Loss: {train_loss:.6f}, '
                       f'Test Loss: {test_loss:.6f}, '
                       f'Best: {self.best_test_loss:.6f}')
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(model_save_path))
        
        # 计算最终指标
        final_r2 = self.calculate_r2_score(test_loader)
        
        results = {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'best_test_loss': self.best_test_loss,
            'final_r2_score': final_r2,
            'num_epochs_trained': len(self.train_losses),
            'model_save_path': model_save_path
        }
        
        logger.info(f"训练完成！最佳测试损失: {self.best_test_loss:.6f}, R²分数: {final_r2:.6f}")
        
        return results


def create_data_loaders(X: torch.Tensor, Y: torch.Tensor, 
                       batch_size: int = 256, train_ratio: float = 0.8,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器
    
    Args:
        X: 输入数据
        Y: 输出数据
        batch_size: 批次大小
        train_ratio: 训练数据比例
        num_workers: 数据加载工作进程数
        
    Returns:
        (train_loader, test_loader): 训练和测试数据加载器
    """
    # 转换为float32以提高GPU效率
    X = X.float()
    Y = Y.float()
    
    # 创建数据集
    dataset = TensorDataset(X, Y)
    
    # 划分训练和测试集
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    logger.info(f"数据集统计:")
    logger.info(f"总样本数: {total_size:,}")
    logger.info(f"训练样本数: {train_size:,}")
    logger.info(f"测试样本数: {test_size:,}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"测试批次数: {len(test_loader)}")
    
    return train_loader, test_loader


def compare_models(models_results: Dict[str, Dict], save_path: Optional[str] = None):
    """
    比较多个模型的性能并可视化
    
    Args:
        models_results: 模型结果字典，格式为 {model_name: results}
        save_path: 图片保存路径（可选）
    """
    plt.figure(figsize=(15, 10))
    
    # 1. 损失曲线对比 - 训练损失
    plt.subplot(2, 3, 1)
    for name, results in models_results.items():
        plt.plot(results['train_losses'], label=f'{name} (Train)', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 损失曲线对比 - 测试损失
    plt.subplot(2, 3, 2)
    for name, results in models_results.items():
        plt.plot(results['test_losses'], label=f'{name} (Test)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 最佳测试损失对比
    plt.subplot(2, 3, 3)
    names = list(models_results.keys())
    best_losses = [models_results[name]['best_test_loss'] for name in names]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(names)]
    
    bars = plt.bar(names, best_losses, color=colors)
    plt.ylabel('Best Test Loss')
    plt.title('Best Test Loss Comparison')
    plt.xticks(rotation=45)
    
    for bar, loss in zip(bars, best_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{loss:.4f}', ha='center', va='bottom')
    
    # 4. R²分数对比
    plt.subplot(2, 3, 4)
    r2_scores = [models_results[name]['final_r2_score'] for name in names]
    
    bars = plt.bar(names, r2_scores, color=colors)
    plt.ylabel('R² Score')
    plt.title('R² Score Comparison')
    plt.xticks(rotation=45)
    
    for bar, r2 in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r2:.4f}', ha='center', va='bottom')
    
    # 5. 训练轮数对比
    plt.subplot(2, 3, 5)
    epochs_trained = [models_results[name]['num_epochs_trained'] for name in names]
    
    bars = plt.bar(names, epochs_trained, color=colors)
    plt.ylabel('Epochs Trained')
    plt.title('Training Epochs Comparison')
    plt.xticks(rotation=45)
    
    for bar, epochs in zip(bars, epochs_trained):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{epochs}', ha='center', va='bottom')
    
    # 6. 训练改善对比
    plt.subplot(2, 3, 6)
    improvements = []
    for name in names:
        results = models_results[name]
        if len(results['test_losses']) > 1:
            initial_loss = results['test_losses'][0]
            final_loss = results['best_test_loss']
            improvement = (initial_loss - final_loss) / initial_loss * 100
            improvements.append(improvement)
        else:
            improvements.append(0)
    
    bars = plt.bar(names, improvements, color=colors)
    plt.ylabel('Loss Improvement (%)')
    plt.title('Training Improvement Comparison')
    plt.xticks(rotation=45)
    
    for bar, imp in zip(bars, improvements):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{imp:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")
    
    plt.show()
    
    # 打印详细对比结果
    print("\n" + "="*80)
    print("详细性能对比")
    print("="*80)
    print(f"{'模型名称':<20} {'最佳测试损失':<15} {'R²分数':<10} {'训练轮数':<10} {'损失改善':<10}")
    print("-"*80)
    
    for name in names:
        results = models_results[name]
        if len(results['test_losses']) > 1:
            improvement = (results['test_losses'][0] - results['best_test_loss']) / results['test_losses'][0] * 100
        else:
            improvement = 0
        
        print(f"{name:<20} {results['best_test_loss']:<15.6f} {results['final_r2_score']:<10.6f} "
              f"{results['num_epochs_trained']:<10} {improvement:<10.1f}%")
    
    # 推荐最佳模型
    best_model_name = min(models_results.keys(), key=lambda x: models_results[x]['best_test_loss'])
    print(f"\n推荐模型: {best_model_name}")
    print(f"理由: 在测试集上获得了最低的损失值 ({models_results[best_model_name]['best_test_loss']:.6f})")
    
    best_r2 = models_results[best_model_name]['final_r2_score']
    if best_r2 > 0.8:
        print("该模型表现优秀，R²分数 > 0.8")
    elif best_r2 > 0.6:
        print("该模型表现良好，R²分数 > 0.6")
    else:
        print("模型性能仍有改进空间")


def save_results(results: Dict, filename: str = 'training_results.pkl'):
    """保存训练结果到文件"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"结果已保存到: {filename}")
    except Exception as e:
        logger.error(f"保存结果时出错: {e}")


def load_results(filename: str = 'training_results.pkl') -> Dict:
    """从文件加载训练结果"""
    try:
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"结果已从 {filename} 加载")
        return results
    except Exception as e:
        logger.error(f"加载结果时出错: {e}")
        return {}