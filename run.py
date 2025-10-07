"""
CIFAR-10神经网络项目主运行脚本
"""

import os
import sys
import subprocess

def check_dependencies():
    """检查是否安装了必需的依赖包"""
    try:
        import numpy
        import matplotlib
        print("✓ 依赖包已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def check_dataset():
    """检查CIFAR-10数据集是否可用"""
    dataset_path = "./cifar-10-python"
    if os.path.exists(dataset_path):
        print("✓ 找到CIFAR-10数据集")
        return True
    else:
        print("❌ 未找到CIFAR-10数据集")
        print("请运行: python download_cifar10.py")
        return False

# 移除了测试和演示功能，只保留基本训练功能

def run_full_training():
    """运行完整训练"""
    print("\n" + "="*50)
    print("正在运行完整训练")
    print("="*50)
    
    try:
        subprocess.run([sys.executable, "cifar10_neural_network.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ 训练失败")
        return False

def main():
    """主函数"""
    print("CIFAR-10神经网络项目")
    print("="*50)
    
    # 检查依赖包
    if not check_dependencies():
        return
    
    # 检查数据集
    if not check_dataset():
        return
    
    # 直接运行完整训练，无需测试/演示菜单
    run_full_training()

if __name__ == "__main__":
    main()

