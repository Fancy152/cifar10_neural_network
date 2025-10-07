# CIFAR-10 全连接神经网络分类器 / CIFAR-10 Fully Connected Neural Network Classifier

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Neural Network](https://img.shields.io/badge/Neural%20Network-From%20Scratch-red.svg)](https://github.com)

---

## 中文 / Chinese

一个从头开始实现的全连接神经网络，用于CIFAR-10图像分类。本项目实现了神经网络的所有核心组件，没有使用TensorFlow或PyTorch等高级框架，适合作为机器学习的简要学习资料。

### 功能特性

#### 核心实现 (30分)
- ✅ **数据加载**: CIFAR-10数据集加载和预处理
- ✅ **神经网络架构**: 可配置的全连接层架构
- ✅ **前向传播**: 多种激活函数的自定义实现
- ✅ **反向传播**: 梯度计算和参数更新
- ✅ **训练循环**: 完整的训练流程和验证
- ✅ **模型评估**: 性能指标和测试

#### 附加功能 (10分)
- ✅ **多种损失函数**: 交叉熵、均方误差
- ✅ **正则化方法**: L1、L2正则化、Dropout
- ✅ **优化算法**: SGD、Momentum、Adam
- ✅ **权重初始化**: Xavier、He、随机初始化
- ✅ **激活函数**: ReLU、Sigmoid、Tanh

### 项目结构

```
cifar10_neural_network/
├── cifar10_neural_network.py    # 主要神经网络实现
├── download_cifar10.py          # 数据集下载脚本
├── run.py                       # 主运行脚本
├── requirements.txt             # Python依赖
├── .gitignore                   # Git忽略文件
└── README.md                   # 本文档
```

---

## English

A fully connected neural network implemented from scratch for CIFAR-10 image classification. This project implements all core components of neural networks without using high-level frameworks like TensorFlow or PyTorch.

### Features

#### Core Implementation (30 points)
- ✅ **Data Loading**: CIFAR-10 dataset loading and preprocessing
- ✅ **Neural Network Architecture**: Configurable fully connected layer architecture
- ✅ **Forward Propagation**: Custom implementation of various activation functions
- ✅ **Backward Propagation**: Gradient computation and parameter updates
- ✅ **Training Loop**: Complete training process and validation
- ✅ **Model Evaluation**: Performance metrics and testing

#### Additional Features (10 points)
- ✅ **Multiple Loss Functions**: Cross-entropy, Mean Squared Error
- ✅ **Regularization Methods**: L1, L2 regularization, Dropout
- ✅ **Optimization Algorithms**: SGD, Momentum, Adam
- ✅ **Weight Initialization**: Xavier, He, Random initialization
- ✅ **Activation Functions**: ReLU, Sigmoid, Tanh

### Project Structure

```
cifar10_neural_network/
├── cifar10_neural_network.py    # Main neural network implementation
├── download_cifar10.py          # Dataset download script
├── run.py                       # Main execution script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                   # This document
```