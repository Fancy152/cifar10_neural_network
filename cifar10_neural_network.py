"""
CIFAR-10 全连接神经网络分类器
从头实现的全连接神经网络，用于图像分类

A1题目得分点标注：
- 全连接网络实现 (30分)
- 图像分类任务 (20分)
- 前馈、反馈、评估代码自己实现
- 附加题：不同损失函数和正则化方法 (+5分)
- 附加题：不同优化算法 (+5分)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import Tuple, List, Dict, Optional
import time

class NeuralNetwork:
    """
    全连接神经网络实现
    """
    
    def __init__(self, layer_sizes: List[int], activation: str = 'relu', 
                 weight_init: str = 'xavier', dropout_rate: float = 0.0):
        """
        初始化神经网络
        
        网络结构设计
        
        Args:
            layer_sizes: 各层神经元数量 [输入层, 隐藏层1, 隐藏层2, ..., 输出层]
            activation: 激活函数 ('relu', 'sigmoid', 'tanh')
            weight_init: 权重初始化方法 ('xavier', 'he', 'random')
            dropout_rate: Dropout正则化率
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.num_layers = len(layer_sizes) - 1
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            if weight_init == 'xavier':
                # Xavier初始化：保持方差稳定
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            elif weight_init == 'he':
                # He初始化：适用于ReLU激活函数
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            else:  # random
                # 随机初始化
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
                
            self.weights.append(w)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    
    def activation_function(self, x: np.ndarray, derivative: bool = False) -> np.ndarray:
        """
        激活函数实现
        
        支持ReLU、Sigmoid、Tanh三种激活函数
        """
        if self.activation == 'relu':
            if derivative:
                return (x > 0).astype(float)
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            if derivative:
                return sigmoid * (1 - sigmoid)
            return sigmoid
        elif self.activation == 'tanh':
            tanh = np.tanh(x)
            if derivative:
                return 1 - tanh**2
            return tanh
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        前向传播
        
        实现完整的前向传播过程，包括线性变换、激活函数、Dropout
        
        Args:
            X: 输入数据 (batch_size, input_size)
            training: 是否处于训练模式（用于Dropout）
            
        Returns:
            output: 网络输出
            cache: 中间值，用于反向传播
        """
        cache = []
        current_input = X
        
        for i in range(self.num_layers - 1):
            # 线性变换：z = Wx + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            
            # 激活函数：a = f(z)
            a = self.activation_function(z)
            
            # Dropout正则化（仅在训练时使用）
            if training and self.dropout_rate > 0:
                dropout_mask = np.random.random(a.shape) > self.dropout_rate
                a = a * dropout_mask / (1 - self.dropout_rate)
            
            cache.append({
                'input': current_input,
                'z': z,
                'a': a,
                'dropout_mask': dropout_mask if training and self.dropout_rate > 0 else None
            })
            
            current_input = a
        
        # 输出层：使用Softmax激活函数进行多分类
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        output = self.softmax(z_output)
        
        cache.append({
            'input': current_input,
            'z': z_output,
            'a': output
        })
        
        return output, cache
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, cache: List[Dict]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        反向传播
        
        实现完整的反向传播算法，计算权重和偏置的梯度
        
        Args:
            y_true: 真实标签（one-hot编码）
            y_pred: 预测概率
            cache: 前向传播的中间值
            
        Returns:
            weight_gradients: 权重梯度
            bias_gradients: 偏置梯度
        """
        m = y_true.shape[0]
        
        # 初始化
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # 输出层梯度：∂L/∂z = y_pred - y_true
        dz = y_pred - y_true
        
        # 反向传播通过各层
        for i in range(self.num_layers - 1, -1, -1):
            # 计算权重和偏置的梯度
            weight_gradients[i] = (1/m) * np.dot(cache[i]['input'].T, dz)
            bias_gradients[i] = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            # 前一层梯度（如果不是输入层）
            if i > 0:
                da_prev = np.dot(dz, self.weights[i].T)
                
                # 应用Dropout掩码（如果存在）
                if cache[i-1]['dropout_mask'] is not None:
                    da_prev = da_prev * cache[i-1]['dropout_mask'] / (1 - self.dropout_rate)
                
                # 激活函数的导数
                dz = da_prev * self.activation_function(cache[i-1]['z'], derivative=True)
        
        return weight_gradients, bias_gradients
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    loss_type: str = 'cross_entropy', 
                    regularization: str = 'l2', 
                    lambda_reg: float = 0.01) -> float:
        """
        损失函数计算
        
        附加题：包含不同损失函数和正则化方法
        
        Args:
            y_true: 真实标签（one-hot编码）
            y_pred: 预测概率
            loss_type: 损失函数类型 ('cross_entropy', 'mse')
            regularization: 正则化类型 ('l2', 'l1', 'none')
            lambda_reg: 正则化强度
            
        Returns:
            loss: 计算的损失值
        """
        m = y_true.shape[0]
        
        # 主要损失函数
        if loss_type == 'cross_entropy':
            # 交叉熵损失：适用于多分类问题
            epsilon = 1e-15  # 防止log(0)
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        elif loss_type == 'mse':
            # 均方误差损失：适用于回归问题
            loss = np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")
        
        # 正则化项 - 附加题：正则化方法 (+5分)
        if regularization == 'l2':
            # L2正则化：权重平方和
            reg_term = 0
            for w in self.weights:
                reg_term += np.sum(w ** 2)
            loss += (lambda_reg / (2 * m)) * reg_term
        elif regularization == 'l1':
            # L1正则化：权重绝对值之和
            reg_term = 0
            for w in self.weights:
                reg_term += np.sum(np.abs(w))
            loss += (lambda_reg / m) * reg_term
        
        return loss
    
    def update_parameters(self, weight_gradients: List[np.ndarray], 
                         bias_gradients: List[np.ndarray], 
                         optimizer: str = 'sgd', 
                         learning_rate: float = 0.01,
                         **optimizer_params) -> None:
        """
        参数更新
        
        附加题：支持不同优化算法
        支持SGD、Momentum、Adam等优化器
        
        Args:
            weight_gradients: 权重梯度
            bias_gradients: 偏置梯度
            optimizer: 优化算法
            learning_rate: 学习率
            **optimizer_params: 额外优化器参数
        """
        if not hasattr(self, 'optimizer_state'):
            self.optimizer_state = {
                'weight_momentum': [np.zeros_like(w) for w in self.weights],
                'bias_momentum': [np.zeros_like(b) for b in self.biases],
                'weight_v': [np.zeros_like(w) for w in self.weights],
                'bias_v': [np.zeros_like(b) for b in self.biases],
                't': 0
            }
        
        self.optimizer_state['t'] += 1
        
        for i in range(len(self.weights)):
            if optimizer == 'sgd':
                # 随机梯度下降：θ = θ - α∇θ
                self.weights[i] -= learning_rate * weight_gradients[i]
                self.biases[i] -= learning_rate * bias_gradients[i]
                
            elif optimizer == 'momentum':
                # Momentum优化器：v = βv + α∇θ, θ = θ - v
                beta = optimizer_params.get('beta', 0.9)
                self.optimizer_state['weight_momentum'][i] = (
                    beta * self.optimizer_state['weight_momentum'][i] + 
                    learning_rate * weight_gradients[i]
                )
                self.optimizer_state['bias_momentum'][i] = (
                    beta * self.optimizer_state['bias_momentum'][i] + 
                    learning_rate * bias_gradients[i]
                )
                self.weights[i] -= self.optimizer_state['weight_momentum'][i]
                self.biases[i] -= self.optimizer_state['bias_momentum'][i]
                
            elif optimizer == 'adam':
                # Adam优化器：结合Momentum和RMSprop
                beta1 = optimizer_params.get('beta1', 0.9)
                beta2 = optimizer_params.get('beta2', 0.999)
                epsilon = optimizer_params.get('epsilon', 1e-8)
                
                # 更新一阶矩估计
                self.optimizer_state['weight_momentum'][i] = (
                    beta1 * self.optimizer_state['weight_momentum'][i] + 
                    (1 - beta1) * weight_gradients[i]
                )
                self.optimizer_state['bias_momentum'][i] = (
                    beta1 * self.optimizer_state['bias_momentum'][i] + 
                    (1 - beta1) * bias_gradients[i]
                )
                
                # 更新二阶矩估计
                self.optimizer_state['weight_v'][i] = (
                    beta2 * self.optimizer_state['weight_v'][i] + 
                    (1 - beta2) * (weight_gradients[i] ** 2)
                )
                self.optimizer_state['bias_v'][i] = (
                    beta2 * self.optimizer_state['bias_v'][i] + 
                    (1 - beta2) * (bias_gradients[i] ** 2)
                )
                
                # 偏差修正
                weight_momentum_corrected = self.optimizer_state['weight_momentum'][i] / (1 - beta1 ** self.optimizer_state['t'])
                bias_momentum_corrected = self.optimizer_state['bias_momentum'][i] / (1 - beta1 ** self.optimizer_state['t'])
                weight_v_corrected = self.optimizer_state['weight_v'][i] / (1 - beta2 ** self.optimizer_state['t'])
                bias_v_corrected = self.optimizer_state['bias_v'][i] / (1 - beta2 ** self.optimizer_state['t'])
                
                # 更新参数
                self.weights[i] -= learning_rate * weight_momentum_corrected / (np.sqrt(weight_v_corrected) + epsilon)
                self.biases[i] -= learning_rate * bias_momentum_corrected / (np.sqrt(bias_v_corrected) + epsilon)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data"""
        y_pred, _ = self.forward(X, training=False)
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        模型评估
        
        计算模型在给定数据上的准确率和损失
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        
        # 计算损失
        y_one_hot = np.eye(10)[y]
        y_pred_proba, _ = self.forward(X, training=False)
        loss = self.compute_loss(y_one_hot, y_pred_proba)
        
        return {'accuracy': accuracy, 'loss': loss}


class CIFAR10DataLoader:
    """
    CIFAR-10数据加载和预处理
    
    得分点：
    - 数据加载和预处理
    - 训练集/验证集/测试集划分
    """
    
    def __init__(self, data_dir: str = './cifar-10-python/cifar-10-batches-py'):
        self.data_dir = data_dir
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载CIFAR-10数据集
        
        从原始二进制文件中加载训练集和测试集
        """
        # 加载训练数据
        X_train = []
        y_train = []
        
        for i in range(1, 6):
            file_path = os.path.join(self.data_dir, f'data_batch_{i}')
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                X_train.append(batch[b'data'])
                y_train.append(batch[b'labels'])
        
        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)
        
        # 加载测试数据
        test_file = os.path.join(self.data_dir, 'test_batch')
        with open(test_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            X_test = batch[b'data']
            y_test = np.array(batch[b'labels'])
        
        return X_train, y_train, X_test, y_test
    
    def preprocess_data(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Preprocess data (normalize, reshape)"""
        # Reshape from (N, 3072) to (N, 32, 32, 3)
        X = X.reshape(-1, 32, 32, 3)
        
        # Convert to grayscale (optional) or keep RGB
        # For simplicity, we'll use all channels flattened
        X = X.reshape(X.shape[0], -1)
        
        if normalize:
            X = X.astype(np.float32) / 255.0
        
        return X
    
    def split_validation(self, X: np.ndarray, y: np.ndarray, 
                        val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        数据划分
        
        将训练数据划分为训练集和验证集，用于超参数调整
        """
        n_samples = X.shape[0]
        n_val = int(n_samples * val_ratio)
        
        # 随机打乱
        indices = np.random.permutation(n_samples)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


def train_model(model: NeuralNetwork, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, 
                epochs: int = 100, batch_size: int = 32,
                learning_rate: float = 0.01, optimizer: str = 'adam',
                loss_type: str = 'cross_entropy', regularization: str = 'l2',
                lambda_reg: float = 0.01, verbose: bool = True) -> Dict:
    """
    模型训练 
    
    实现完整的训练流程，包括前向传播、反向传播、参数更新
    
    Returns:
        training_history: 包含训练指标的字典
    """
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []
    }
    
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size
    
    for epoch in range(epochs):
        # 打乱训练数据
        indices = np.random.permutation(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_train_loss = 0
        epoch_train_acc = 0
        
        # 小批量训练 - 得分点：训练流程实现 (10分)
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # 转换为one-hot编码
            y_batch_one_hot = np.eye(10)[y_batch]
            
            # 前向传播
            y_pred, cache = model.forward(X_batch, training=True)
            
            # 计算损失
            batch_loss = model.compute_loss(y_batch_one_hot, y_pred, 
                                          loss_type, regularization, lambda_reg)
            epoch_train_loss += batch_loss
            
            # 反向传播
            weight_gradients, bias_gradients = model.backward(y_batch_one_hot, y_pred, cache)
            
            # 更新参数
            model.update_parameters(weight_gradients, bias_gradients, 
                                  optimizer, learning_rate)
            
            # 计算准确率
            batch_pred = np.argmax(y_pred, axis=1)
            batch_acc = np.mean(batch_pred == y_batch)
            epoch_train_acc += batch_acc
        
        # 平均指标
        epoch_train_loss /= n_batches
        epoch_train_acc /= n_batches
        
        # 验证
        val_metrics = model.evaluate(X_val, y_val)
        epoch_val_loss = val_metrics['loss']
        epoch_val_acc = val_metrics['accuracy']
        
        # 存储历史
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
            print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            print()
    
    return history


def plot_training_history(history: Dict, save_path: Optional[str] = None) -> None:
    """展示训练结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失图像
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 正确率图像
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    """
    主函数
    
    实现完整的实验流程：数据加载、预处理、模型训练、评估
    """
    print("CIFAR-10 Neural Network Classifier")
    print("=" * 50)
    
    # 加载数据
    print("Loading CIFAR-10 dataset...")
    data_loader = CIFAR10DataLoader()
    X_train, y_train, X_test, y_test = data_loader.load_data()
    
    # 预处理数据
    print("Preprocessing data...")
    X_train = data_loader.preprocess_data(X_train)
    X_test = data_loader.preprocess_data(X_test)
    
    # 划分验证集
    X_train, y_train, X_val, y_val = data_loader.split_validation(X_train, y_train, val_ratio=0.1)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Input dimension: {X_train.shape[1]}")
    print()
    
    # 创建模型
    layer_sizes = [X_train.shape[1], 512, 256, 128, 10]  # Input -> Hidden layers -> Output
    model = NeuralNetwork(layer_sizes, activation='relu', weight_init='he', dropout_rate=0.2)
    
    print(f"Model architecture: {layer_sizes}")
    print(f"Total parameters: {sum(w.size + b.size for w, b in zip(model.weights, model.biases))}")
    print()
    
    # 训练参数配置 - 不同优化算法
    training_configs = [
        {
            'name': 'Adam Optimizer',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 64
        },
        {
            'name': 'SGD Optimizer',
            'optimizer': 'sgd',
            'learning_rate': 0.01,
            'epochs': 50,
            'batch_size': 64
        },
        {
            'name': 'Momentum Optimizer',
            'optimizer': 'momentum',
            'learning_rate': 0.01,
            'epochs': 50,
            'batch_size': 64
        }
    ]
    
    # 训练不同的模型
    results = {}
    
    for config in training_configs:
        print(f"Training with {config['name']}...")
        print("-" * 30)
        
        # Create new model for each experiment
        model = NeuralNetwork(layer_sizes, activation='relu', weight_init='he', dropout_rate=0.2)
        
        # Train model
        start_time = time.time()
        history = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            optimizer=config['optimizer'],
            loss_type='cross_entropy',
            regularization='l2',
            lambda_reg=0.01,
            verbose=True
        )
        training_time = time.time() - start_time
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test)
        
        results[config['name']] = {
            'history': history,
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'training_time': training_time
        }
        
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        print()
    
    # Plot results
    print("Plotting training results...")
    for name, result in results.items():
        plot_training_history(result['history'], f'training_history_{name.lower().replace(" ", "_")}.png')
    
    # Print final comparison
    print("Final Results Comparison:")
    print("=" * 50)
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Test Loss: {result['test_loss']:.4f}")
        print(f"  Training Time: {result['training_time']:.2f}s")
        print()


if __name__ == "__main__":
    main()

