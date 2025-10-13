#!/usr/bin/env python3
"""
快速测试脚本 - 验证项目结构和基本功能
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")
    
    try:
        import config
        print("✓ config 模块导入成功")
    except Exception as e:
        print(f"✗ config 模块导入失败: {e}")
        return False
    
    try:
        from data_processing import ClimateDataProcessor
        print("✓ data_processing 模块导入成功")
    except Exception as e:
        print(f"✗ data_processing 模块导入失败: {e}")
        return False
    
    try:
        from models import create_model, get_model_info
        print("✓ models 模块导入成功")
    except Exception as e:
        print(f"✗ models 模块导入失败: {e}")
        return False
    
    try:
        from train import ModelTrainer, create_data_loaders
        print("✓ train 模块导入成功")
    except Exception as e:
        print(f"✗ train 模块导入失败: {e}")
        return False
    
    try:
        from utils import setup_device, set_seed
        print("✓ utils 模块导入成功")
    except Exception as e:
        print(f"✗ utils 模块导入失败: {e}")
        return False
    
    return True


def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        from models import create_model
        
        # 测试不同类型的模型
        models_to_test = [
            ("simple", {}),
            ("residual", {"hidden_dim": 256}),
            ("attention", {"embed_dim": 256, "num_heads": 4})
        ]
        
        for model_type, kwargs in models_to_test:
            model = create_model(model_type, input_dim=100, output_dim=50, **kwargs)
            print(f"✓ {model_type} 模型创建成功")
        
        return True
    
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        traceback.print_exc()
        return False


def test_device_setup():
    """测试设备设置"""
    print("\n测试设备设置...")
    
    try:
        from utils import setup_device, print_system_summary
        
        device = setup_device()
        print(f"✓ 设备设置成功: {device}")
        
        print_system_summary()
        print("✓ 系统信息显示成功")
        
        return True
    
    except Exception as e:
        print(f"✗ 设备设置失败: {e}")
        traceback.print_exc()
        return False


def test_config_access():
    """测试配置访问"""
    print("\n测试配置访问...")
    
    try:
        import config
        
        # 检查关键配置项
        required_configs = [
            'INPUTS_VARIABLE1', 'INPUTS_VARIABLE2',
            'OUTPUT_VARIABLE1', 'OUTPUT_VARIABLE2',
            'BATCH_SIZE', 'LEARNING_RATE', 'NUM_EPOCHS'
        ]
        
        for config_name in required_configs:
            if hasattr(config, config_name):
                value = getattr(config, config_name)
                print(f"✓ {config_name}: {value}")
            else:
                print(f"✗ 缺少配置项: {config_name}")
                return False
        
        return True
    
    except Exception as e:
        print(f"✗ 配置访问失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("Simple Network 项目结构测试")
    print("="*60)
    
    # 检查当前工作目录
    current_dir = Path.cwd()
    print(f"当前工作目录: {current_dir}")
    
    # 检查必要文件
    required_files = [
        'config.py', 'data_processing.py', 'models.py', 
        'train.py', 'utils.py', 'main.py', 'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 不存在")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n错误: 缺少文件 {missing_files}")
        return False
    
    # 运行各项测试
    tests = [
        test_imports,
        test_config_access,
        test_device_setup,
        test_model_creation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试 {test.__name__} 出现异常: {e}")
            traceback.print_exc()
            results.append(False)
    
    # 总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    
    if all(results):
        print("✓ 所有测试通过！项目结构正确。")
        print("\n可以运行以下命令开始训练:")
        print("python main.py")
    else:
        print("✗ 部分测试失败，请检查项目设置。")
        return False
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试过程中出现未处理的异常: {e}")
        traceback.print_exc()
        sys.exit(1)