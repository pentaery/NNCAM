"""
Main script for Simple Network climate prediction
主程序文件，整合所有功能进行气候预测模型的训练和评估
"""

import logging
from pathlib import Path
import torch
from typing import Dict, Any

# 导入项目模块
from config import *
from data_processing import ClimateDataProcessor
from models import create_model, get_model_info
from train import ModelTrainer, create_data_loaders, compare_models, save_results
from utils import (
    setup_device, set_seed, print_system_summary, 
    validate_data_shapes, check_tensor_health, 
    calculate_memory_usage, save_config
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    
    print("="*80)
    print("Simple Network 气候预测项目")
    print("="*80)
    
    # 1. 系统设置
    logger.info("初始化系统设置...")
    print_system_summary()
    
    # 设置随机种子
    set_seed(42)
    
    # 设置计算设备
    device = setup_device()
    
    # 2. 数据处理
    logger.info("开始数据处理...")
    
    # 创建数据处理器
    processor = ClimateDataProcessor(DATA_PATH, LAT_THRESHOLD)
    
    # 加载数据集
    dataset = processor.load_dataset()
    
    # 创建极地采样掩码
    sampling_mask = processor.create_polar_sampling_mask()
    
    # 处理所有变量
    X_sampled, Y_sampled = processor.process_all_variables(
        INPUTS_VARIABLE1, INPUTS_VARIABLE2,
        OUTPUT_VARIABLE1, OUTPUT_VARIABLE2
    )
    
    # 验证数据形状
    validate_data_shapes(X_sampled, Y_sampled)
    
    # 检查张量健康状态
    check_tensor_health(X_sampled, "输入数据")
    check_tensor_health(Y_sampled, "输出数据")
    
    # 计算内存使用
    memory_stats = calculate_memory_usage(X_sampled, Y_sampled)
    logger.info(f"数据内存占用: {memory_stats['total_memory_gb']:.2f} GB")
    
    # 更新配置中的维度信息
    global INPUT_DIM, OUTPUT_DIM
    INPUT_DIM = X_sampled.shape[1]
    OUTPUT_DIM = Y_sampled.shape[1]
    
    logger.info(f"输入维度: {INPUT_DIM}, 输出维度: {OUTPUT_DIM}")
    
    # 3. 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, test_loader = create_data_loaders(
        X_sampled, Y_sampled, 
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO
    )
    
    # 4. 训练多个模型
    logger.info("开始训练多个模型...")
    
    models_to_test = [
        ("Simple Improved", "simple", {}),
        ("Residual Network", "residual", {"hidden_dim": HIDDEN_DIM}),
        ("Attention Network", "attention", {"dropout_rate": DROPOUT_RATE1})
    ]
    
    results = {}
    
    for name, model_type, kwargs in models_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"训练模型: {name}")
        logger.info(f"{'='*60}")
        
        # 创建模型
        model = create_model(model_type, INPUT_DIM, OUTPUT_DIM, **kwargs)
        
        # 打印模型信息
        model_info = get_model_info(model)
        logger.info(f"参数数量: {model_info['total_parameters']:,}")
        
        # 创建训练器
        trainer = ModelTrainer(
            model=model,
            device=device,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            huber_delta=HUBER_DELTA
        )
        
        # 训练模型
        model_save_path = f"best_model_{model_type}.pth"
        training_results = trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            model_save_path=model_save_path,
            gradient_clip_value=GRADIENT_CLIP_VALUE
        )
        
        # 保存结果
        results[name] = {
            **training_results,
            'model_info': model_info,
            'model_type': model_type
        }
        
        logger.info(f"模型 {name} 训练完成")
        logger.info(f"最佳测试损失: {training_results['best_test_loss']:.6f}")
        logger.info(f"R²分数: {training_results['final_r2_score']:.6f}")
    
    # 5. 比较模型结果
    logger.info("\n开始比较模型结果...")
    compare_models(results, save_path="model_comparison.png")
    
    # 6. 保存所有结果
    logger.info("保存训练结果...")
    save_results(results, RESULTS_SAVE_PATH)
    
    # 保存配置
    config_to_save = {
        'data_path': DATA_PATH,
        'input_variables_3d': INPUTS_VARIABLE1,
        'input_variables_2d': INPUTS_VARIABLE2,
        'output_variables_3d': OUTPUT_VARIABLE1,
        'output_variables_2d': OUTPUT_VARIABLE2,
        'lat_threshold': LAT_THRESHOLD,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'patience': PATIENCE,
        'train_ratio': TRAIN_RATIO,
        'input_dim': INPUT_DIM,
        'output_dim': OUTPUT_DIM,
        'device': str(device)
    }
    
    save_config(config_to_save, 'training_config.json')
    
    # 7. 输出最终总结
    print("\n" + "="*80)
    print("训练完成总结")
    print("="*80)
    
    # 找到最佳模型
    best_model_name = min(results.keys(), key=lambda x: results[x]['best_test_loss'])
    best_result = results[best_model_name]
    
    print(f"最佳模型: {best_model_name}")
    print(f"最佳测试损失: {best_result['best_test_loss']:.6f}")
    print(f"R²分数: {best_result['final_r2_score']:.6f}")
    print(f"训练轮数: {best_result['num_epochs_trained']}")
    print(f"参数数量: {best_result['model_info']['total_parameters']:,}")
    
    print(f"\n数据统计:")
    print(f"原始数据压缩比: {X_sampled.shape[0] / (27 * 384 * 576):.3f}")
    print(f"节省内存: {(1 - X_sampled.shape[0] / (27 * 384 * 576)) * 100:.1f}%")
    
    print(f"\n文件保存:")
    print(f"- 最佳模型: best_model_{best_result['model_type']}.pth")
    print(f"- 训练结果: {RESULTS_SAVE_PATH}")
    print(f"- 配置文件: training_config.json")
    print(f"- 比较图表: model_comparison.png")
    
    print("="*80)
    logger.info("程序执行完成！")


def train_single_model(model_type: str = "simple", **kwargs):
    """
    训练单个模型的便捷函数
    
    Args:
        model_type: 模型类型
        **kwargs: 其他参数
    """
    # 基本设置
    set_seed(42)
    device = setup_device()
    
    # 数据处理
    processor = ClimateDataProcessor(DATA_PATH, LAT_THRESHOLD)
    dataset = processor.load_dataset()
    sampling_mask = processor.create_polar_sampling_mask()
    
    X_sampled, Y_sampled = processor.process_all_variables(
        INPUTS_VARIABLE1, INPUTS_VARIABLE2,
        OUTPUT_VARIABLE1, OUTPUT_VARIABLE2
    )
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        X_sampled, Y_sampled, batch_size=BATCH_SIZE
    )
    
    # 创建模型
    input_dim = X_sampled.shape[1]
    output_dim = Y_sampled.shape[1]
    model = create_model(model_type, input_dim, output_dim, **kwargs)
    
    # 训练
    trainer = ModelTrainer(model, device)
    results = trainer.train(train_loader, test_loader, NUM_EPOCHS, PATIENCE)
    
    return results, model


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise