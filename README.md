# Fishbook - 《深度学习入门：基于Python的理论与实现》

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-required-green.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-Educational-orange.svg)](LICENSE)

*从零开始学习深度学习 - 使用 Python 和 NumPy 构建神经网络*

[English](#english-version) | [中文](#chinese-version)

</div>

---
### 项目简介

本仓库是《深度学习入门：基于Python的理论与实现》一书的配套代码实现。通过纯 Python 和 NumPy，从零开始构建神经网络和深度学习算法，帮助读者深入理解深度学习的核心原理。

**核心特点**：
- 🔧 **从零实现**：不依赖深度学习框架，使用 NumPy 手工实现所有组件
- 📖 **循序渐进**：按章节组织，从感知机到卷积神经网络，逐步深入
- 🎯 **实战导向**：每章包含可运行的示例代码和 MNIST 数据集实践
- 🧮 **数学直观**：将抽象的数学概念转化为可执行的 Python 代码

### 环境要求

**必需依赖**：
- Python 3.x (推荐 3.6 或更高版本)
- NumPy >= 1.15.0
- Matplotlib >= 2.0.0
- Pillow (PIL) >= 5.0.0

### 快速开始

#### 1. 克隆仓库

```bash
git clone https://github.com/MengQxuan/fishbook.git
cd fishbook
```

#### 2. 鱼书配套源代码 + MNIST 数据集

```bash
通过网盘分享的文件：MNIST等2个文件
链接: https://pan.baidu.com/s/1THFM60LqHEJtftWiJuZmwA?pwd=n3t5 提取码: n3t5 
--来自百度网盘超级会员v3的分享
```


### 📂 项目结构详解

```plaintext
fishbook/
├── chapter1/                   # 第1章：Python 入门
│   └── hello.py               # Python 基础：变量、类型、类、NumPy 广播
│
├── chapter2/                   # 第2章：感知机
│   ├── and_gate.py            # AND 逻辑门实现
│   ├── or_gate.py             # OR 逻辑门实现
│   ├── nand_gate.py           # NAND 逻辑门实现
│   └── xor_gate.py            # XOR 逻辑门（多层感知机）
│
├── chapter3/                   # 第3章：神经网络
│   ├── sigmoid.py             # Sigmoid 激活函数及可视化
│   ├── relu.py                # ReLU 激活函数
│   ├── step_function.py       # 阶跃函数
│   ├── forward.py             # 前向传播示例
│   ├── mnist_show.py          # 显示 MNIST 图像
│   ├── neuralnet_mnist.py     # 神经网络推理（单样本）
│   ├── neuralnet_mnist_batch.py  # 批量推理（提升性能）
│   └── sample_weight.pkl      # 预训练权重文件
│
├── chapter4/                   # 第4章：神经网络的学习
│   ├── gradient_1d.py         # 一维函数的梯度（数值微分）
│   ├── gradient_2d.py         # 二维函数的梯度可视化
│   ├── gradient_method.py     # 梯度下降法
│   ├── gradient_simplenet.py  # 简单网络的梯度计算
│   ├── two_layer_net.py       # 两层神经网络类
│   └── train_neuralnet.py     # 完整训练流程
│
├── chapter5/                   # 第5章：误差反向传播法
│   ├── buy_apple.py           # 计算图示例（简单）
│   ├── buy_apple_orange.py    # 计算图示例（复杂）
│   ├── layer_naive.py         # 乘法层和加法层的简单实现
│   ├── gradient_check.py      # 梯度检验（对比数值梯度和反向传播）
│   ├── two_layer_net.py       # 使用层的两层神经网络
│   └── train_neuralnet.py     # 使用反向传播的训练
│
├── chapter6/                   # 第6章：与学习相关的技巧
│   ├── optimizer_compare_naive.py    # 优化器比较（简单函数）
│   ├── optimizer_compare_mnist.py    # 优化器在 MNIST 上的比较
│   ├── weight_init_activation_histogram.py  # 权重初始化对激活值分布的影响
│   ├── weight_init_compare.py        # 权重初始化方法比较
│   ├── batch_norm_test.py            # Batch Normalization 效果测试
│   ├── overfit_weight_decay.py       # 权重衰减（Weight Decay）
│   ├── overfit_dropout.py            # Dropout 正则化
│   └── hyperparameter_optimization.py  # 超参数优化
│
├── chapter7/                   # 第7章：卷积神经网络
│   ├── simple_convnet.py      # 简单卷积神经网络实现
│   ├── train_convnet.py       # 训练卷积神经网络
│   ├── visualize_filter.py    # 可视化卷积滤波器
│   └── params.pkl             # 预训练的卷积网络权重
│
├── chapter8/                   # 第8章：深度学习
│   ├── deep_convnet.py        # 深度卷积神经网络（99%+ 准确率）
│   ├── train_deepnet.py       # 训练深度网络
│   └── deep_convnet_params.pkl  # 深度网络预训练权重
│
├── common/                     # 公共模块
│   ├── functions.py           # 激活函数、损失函数
│   ├── gradient.py            # 数值梯度计算
│   ├── layers.py              # 层的实现（Affine、Softmax、Conv、Pooling）
│   ├── multi_layer_net.py     # 多层神经网络
│   ├── multi_layer_net_extend.py  # 扩展的多层网络（支持 Dropout、BN）
│   ├── optimizer.py           # 优化器（SGD、Momentum、AdaGrad、Adam）
│   ├── trainer.py             # 训练器（封装训练循环）
│   └── util.py                # 辅助函数
│
└── dataset/                    # 数据集模块
    ├── mnist.py               # MNIST 数据加载器
    └── mnist.pkl              # MNIST 数据集
```

### 📖 章节详细说明

#### Chapter 1: Python 入门
介绍 Python 编程基础，包括变量、数据类型、类、NumPy 的基本使用和广播机制。

**主要内容**：
- Python 基本语法
- 类与对象
- NumPy 数组操作和广播


---

#### Chapter 2: 感知机
实现最简单的神经网络——感知机，并通过组合实现逻辑门。

**主要内容**：
- 感知机的基本原理
- 实现 AND、OR、NAND 门
- 多层感知机实现 XOR 门

**学习要点**：
- 理解线性分类器的局限性
- 了解多层感知机的表达能力

---

#### Chapter 3: 神经网络
深入神经网络的核心概念，包括激活函数、前向传播和 MNIST 数据集的推理。

**主要内容**：
- 激活函数（Sigmoid、ReLU、Step Function）
- 三层神经网络的实现
- MNIST 手写数字识别
- 批处理优化

**关键文件**：
- `sigmoid.py` / `relu.py`: 激活函数的实现和可视化
- `neuralnet_mnist.py`: 使用预训练权重进行推理
- `neuralnet_mnist_batch.py`: 批量处理提升推理速度

---

#### Chapter 4: 神经网络的学习
学习如何通过梯度下降法训练神经网络。

**主要内容**：
- 损失函数（均方误差、交叉熵）
- 数值微分和梯度计算
- 梯度下降法
- mini-batch 学习

**关键文件**：
- `gradient_1d.py` / `gradient_2d.py`: 梯度可视化
- `gradient_method.py`: 梯度下降法演示
- `two_layer_net.py`: 两层神经网络类
- `train_neuralnet.py`: 完整的训练流程


**训练输出**：
- 逐个 epoch 显示训练/测试准确率
- 损失函数值的变化曲线

---

#### Chapter 5: 误差反向传播法
使用计算图和链式法则实现高效的反向传播算法。

**主要内容**：
- 计算图的前向传播和反向传播
- 各层的反向传播实现（Affine、ReLU、Softmax）
- 梯度检验
- 使用层构建神经网络

**关键文件**：
- `buy_apple.py`: 计算图基础示例
- `layer_naive.py`: 层的简单实现
- `gradient_check.py`: 验证反向传播的正确性
- `train_neuralnet.py`: 使用反向传播训练


**学习要点**：
- 理解计算图和链式法则
- 掌握反向传播的高效性（比数值微分快 100+ 倍）

---

#### Chapter 6: 与学习相关的技巧
深度学习中的高级技巧，包括优化器、权重初始化、正则化和超参数调优。

**主要内容**：
- **优化器**：SGD、Momentum、AdaGrad、RMSprop、Adam
- **权重初始化**：Xavier 初始化、He 初始化
- **Batch Normalization**：加速训练、提高泛化
- **正则化**：Weight Decay、Dropout
- **超参数优化**：验证集、网格搜索

**关键文件**：
- `optimizer_compare_mnist.py`: 对比各优化器在 MNIST 上的表现
- `weight_init_compare.py`: 权重初始化方法对比
- `batch_norm_test.py`: BN 的效果
- `overfit_dropout.py`: Dropout 防止过拟合
- `hyperparameter_optimization.py`: 超参数搜索


**学习要点**：
- Adam 通常是最佳选择
- 合理的权重初始化至关重要
- Dropout 和 BN 可有效防止过拟合

---

#### Chapter 7: 卷积神经网络 (CNN)
实现用于图像识别的卷积神经网络。

**主要内容**：
- 卷积层（Convolution）的实现
- 池化层（Pooling）的实现
- CNN 的整体结构
- 卷积滤波器的可视化

**网络结构**：
```
Input → Conv → ReLU → Pooling → Affine → ReLU → Affine → Softmax
```

**关键文件**：
- `simple_convnet.py`: 简单 CNN 实现
- `train_convnet.py`: 训练 CNN（准确率 ~98%）
- `visualize_filter.py`: 可视化学到的滤波器


---

#### Chapter 8: 深度学习
构建更深的网络，探索深度学习的威力。

**主要内容**：
- 深度卷积神经网络
- 网络加深的技巧
- 数据增强
- 迁移学习的基础

**网络结构**：
```
Conv-ReLU-Conv-ReLU-Pool ×3 → Affine-ReLU-Dropout-Affine-Dropout-Softmax
```

**关键文件**：
- `deep_convnet.py`: 深度 CNN（99%+ 准确率）
- `train_deepnet.py`: 训练深度网络


---

### 🧰 公共模块详解

#### `common/functions.py`
核心数学函数实现。

**激活函数**：
- `sigmoid(x)`: Sigmoid 函数，输出范围 (0, 1)
- `relu(x)`: ReLU 函数，`max(0, x)`
- `softmax(x)`: Softmax 函数，用于多分类输出层
- `step_function(x)`: 阶跃函数

**损失函数**：
- `mean_squared_error(y, t)`: 均方误差（回归）
- `cross_entropy_error(y, t)`: 交叉熵误差（分类）

---

#### `common/layers.py`
神经网络层的实现，支持前向传播和反向传播。

**基础层**：
- `Relu`: ReLU 激活层
- `Sigmoid`: Sigmoid 激活层
- `Affine`: 全连接层（矩阵乘法 + 偏置）
- `SoftmaxWithLoss`: Softmax 和交叉熵损失的组合

**CNN 层**：
- `Convolution`: 卷积层
- `Pooling`: 池化层

**正则化层**：
- `Dropout`: Dropout 层
- `BatchNormalization`: Batch Normalization 层

---

#### `common/optimizer.py`
多种优化算法的实现。

**可用优化器**：
1. **SGD (Stochastic Gradient Descent)**
   ```python
   optimizer = SGD(lr=0.01)
   ```

2. **Momentum**
   ```python
   optimizer = Momentum(lr=0.01, momentum=0.9)
   ```

3. **AdaGrad**
   ```python
   optimizer = AdaGrad(lr=0.01)
   ```

4. **RMSprop**
   ```python
   optimizer = RMSprop(lr=0.01, decay_rate=0.99)
   ```

5. **Adam** (推荐)
   ```python
   optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)
   ```

**使用方法**：
```python
from common.optimizer import Adam

optimizer = Adam(lr=0.001)

# 训练循环
for epoch in range(epochs):
    # 计算梯度
    grads = network.gradient(x_batch, t_batch)
    
    # 更新参数
    optimizer.update(network.params, grads)
```

---

#### `common/trainer.py`
封装训练循环，简化训练代码。

**主要功能**：
- 自动化训练循环
- 记录训练/测试准确率
- 绘制学习曲线

**示例**：
```python
from common.trainer import Trainer
from common.optimizer import Adam

network = YourNetwork()
optimizer = Adam()

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer=optimizer, optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 绘制学习曲线
trainer.plot()
```

---

#### `dataset/mnist.py`
MNIST 数据集加载器，支持自动下载和预处理。

**功能特性**：
- 自动从镜像源下载数据集（阿里云镜像）
- 图像归一化（0-255 → 0.0-1.0）
- One-hot 编码标签
- 支持展平或保持图像形状

**API 说明**：
```python
from dataset.mnist import load_mnist

# 加载数据集
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True,      # 归一化到 [0.0, 1.0]
    flatten=True,        # 展平为一维数组 (784,)
    one_hot_label=False  # 标签为整数 (0-9)
)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)   # (10000, 784)
print(t_test.shape)   # (10000,)
```

**数据集规模**：
- 训练集：60,000 张图像
- 测试集：10,000 张图像
- 图像尺寸：28×28 灰度图

---
### 💡 代码示例

#### 从头训练一个神经网络
```python
import numpy as np
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import Adam

# 加载数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 创建网络
network = MultiLayerNet(input_size=784, 
                        hidden_size_list=[100, 100],
                        output_size=10)

# 创建优化器
optimizer = Adam(lr=0.001)

# 训练参数
epochs = 20
batch_size = 100
train_size = x_train.shape[0]
iter_per_epoch = max(train_size / batch_size, 1)

for epoch in range(epochs):
    # 每个 epoch
    for i in range(int(iter_per_epoch)):
        # mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 计算梯度
        grads = network.gradient(x_batch, t_batch)
        
        # 更新参数
        optimizer.update(network.params, grads)
    
    # 评估
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    print(f"Epoch {epoch+1}: Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}")
```

#### 使用 Trainer 简化训练
```python
from common.trainer import Trainer

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001})
trainer.train()
trainer.plot()  # 绘制学习曲线
```

---


### 📚 参考资料

**相关书籍**：
- 《深度学习入门：基于Python的理论与实现》（原书）
- 《Deep Learning》 by Ian Goodfellow
- 《Neural Networks and Deep Learning》 by Michael Nielsen

**在线资源**：
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Neural Networks and Deep Learning (免费在线书籍)](http://neuralnetworksanddeeplearning.com/)

**推荐后续学习**：
- PyTorch 官方教程
- TensorFlow 官方教程
- Fast.ai 深度学习课程

---

### 🙏 致谢

感谢《深度学习入门：基于Python的理论与实现》一书的作者，为深度学习初学者提供了如此优秀的学习材料。

---
<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！**

**⭐ Star this project if you find it helpful!**

</div>