"""
CIFAR-10数据集下载脚本
"""

import urllib.request
import tarfile
import os

def download_cifar10():
    """下载并解压CIFAR-10数据集"""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    
    print("正在下载CIFAR-10数据集...")
    urllib.request.urlretrieve(url, filename)
    
    print("正在解压数据集...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()
    
    # 清理临时文件
    os.remove(filename)
    
    print("CIFAR-10数据集下载和解压成功！")
    print("数据现在可以在'./cifar-10-batches-py/'目录中找到")

if __name__ == "__main__":
    download_cifar10()

