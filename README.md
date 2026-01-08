# Fishbook - 《深度学习入门：基于Python的理论与实现》

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-required-green.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-Educational-orange.svg)](LICENSE)

*从零开始学习深度学习 - 使用 Python 和 NumPy 构建神经网络*

[English](#english-version) | [中文](#chinese-version)

</div>

---

## <a name="chinese-version"></a>📚 中文版本

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

#### 2. 安装依赖

```bash
# 使用 pip 安装
pip install numpy matplotlib pillow

# 或使用 pip 一次性安装所有依赖
pip install -r requirements.txt  # 如果有 requirements.txt
```

#### 3. 下载 MNIST 数据集

首次运行时，MNIST 数据集将自动从镜像源下载。如需手动下载：

```bash
cd dataset
python mnist.py
```

#### 4. 运行示例

```bash
# 测试感知机
python chapter2/and_gate.py

# 运行 MNIST 神经网络推理
python chapter3/neuralnet_mnist.py

# 训练神经网络
python chapter4/train_neuralnet.py
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
    └── mnist.pkl              # MNIST 数据集（首次运行自动下载）
```

### 📖 章节详细说明

#### Chapter 1: Python 入门
介绍 Python 编程基础，包括变量、数据类型、类、NumPy 的基本使用和广播机制。

**主要内容**：
- Python 基本语法
- 类与对象
- NumPy 数组操作和广播

**运行示例**：
```bash
python chapter1/hello.py
```

---

#### Chapter 2: 感知机
实现最简单的神经网络——感知机，并通过组合实现逻辑门。

**主要内容**：
- 感知机的基本原理
- 实现 AND、OR、NAND 门
- 多层感知机实现 XOR 门

**运行示例**：
```bash
python chapter2/and_gate.py     # 输出: AND 门真值表
python chapter2/xor_gate.py     # 多层感知机实现 XOR
```

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

**运行示例**：
```bash
python chapter3/mnist_show.py          # 显示 MNIST 图像
python chapter3/neuralnet_mnist.py     # 神经网络推理
python chapter3/neuralnet_mnist_batch.py  # 批量推理
```

**预期结果**：
- 识别准确率约 93.52%

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

**运行示例**：
```bash
python chapter4/gradient_2d.py      # 可视化二维梯度
python chapter4/train_neuralnet.py  # 训练神经网络
```

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

**运行示例**：
```bash
python chapter5/buy_apple_orange.py   # 计算图示例
python chapter5/gradient_check.py     # 梯度检验
python chapter5/train_neuralnet.py    # 高效训练
```

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

**运行示例**：
```bash
python chapter6/optimizer_compare_mnist.py    # 优化器对比
python chapter6/batch_norm_test.py            # BN 效果测试
python chapter6/overfit_dropout.py            # Dropout 演示
```

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

**运行示例**：
```bash
python chapter7/train_convnet.py       # 训练 CNN
python chapter7/visualize_filter.py    # 可视化滤波器
```

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

**运行示例**：
```bash
python chapter8/train_deepnet.py       # 训练深度网络（耗时较长）
```

**预期结果**：
- MNIST 测试集准确率 > 99%

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

**示例**：
```python
from common.functions import sigmoid, relu, softmax, cross_entropy_error
import numpy as np

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))  # [0.26894142 0.73105858 0.88079708]
print(relu(x))     # [0. 1. 2.]

# 计算损失
y = np.array([0.1, 0.8, 0.1])  # 预测概率
t = np.array([0, 1, 0])        # 真实标签
loss = cross_entropy_error(y, t)
```

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

**示例**：
```python
from common.layers import Affine, Relu, SoftmaxWithLoss
import numpy as np

# 构建网络
affine1 = Affine(W1, b1)
relu1 = Relu()
affine2 = Affine(W2, b2)
last_layer = SoftmaxWithLoss()

# 前向传播
x = np.random.randn(100, 784)  # 100 个样本
out = affine1.forward(x)
out = relu1.forward(out)
out = affine2.forward(out)
loss = last_layer.forward(out, t)

# 反向传播
dout = 1
dout = last_layer.backward(dout)
dout = affine2.backward(dout)
dout = relu1.backward(dout)
dout = affine1.backward(dout)
```

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

### 🎓 学习路径建议

**初学者路径**（按顺序学习）：
1. **Chapter 1-2**：Python 基础和感知机（1-2 天）
2. **Chapter 3**：神经网络基础和 MNIST（2-3 天）
3. **Chapter 4**：梯度下降和训练（3-4 天）
4. **Chapter 5**：反向传播算法（4-5 天）
5. **Chapter 6**：优化技巧（3-4 天）
6. **Chapter 7**：卷积神经网络（4-5 天）
7. **Chapter 8**：深度网络（3-4 天）

**总计学习时间**：约 20-30 天（每天 2-3 小时）

**进阶者路径**：
- 如果熟悉 Python 和 NumPy，可跳过 Chapter 1
- 如果了解神经网络基础，可从 Chapter 4 开始
- 重点关注 Chapter 5（反向传播）和 Chapter 6（优化技巧）

**实践建议**：
1. 每章代码都要亲自运行和调试
2. 尝试修改超参数观察效果
3. 可视化中间结果（激活值、梯度、损失曲线）
4. 在其他数据集上测试（如 Fashion-MNIST）

---

### 🔧 常见问题解答

#### Q1: MNIST 数据集下载失败怎么办？
**A**: 项目已配置阿里云镜像源，通常不会失败。如遇网络问题：
```python
# 手动下载
cd dataset
python mnist.py
```

#### Q2: 训练太慢怎么办？
**A**: 
- 减少训练样本：`x_train = x_train[:10000]`
- 减少 epoch 数量
- 使用更强的优化器（Adam）
- 增大 batch size

#### Q3: 如何在自己的数据集上使用？
**A**: 
```python
# 参考 dataset/mnist.py 实现数据加载器
# 确保数据格式为：(N, 784) 或 (N, 1, 28, 28)
# 标签格式为：(N,) 或 (N, 10)

from common.multi_layer_net import MultiLayerNet

network = MultiLayerNet(input_size=your_input_size, 
                        hidden_size_list=[100, 100],
                        output_size=your_num_classes)
```

#### Q4: 出现 "MemoryError" 怎么办？
**A**: 减少 batch size 或减少隐藏层神经元数量。

#### Q5: 如何保存训练好的模型？
**A**: 
```python
import pickle

# 保存模型
with open('my_model.pkl', 'wb') as f:
    pickle.dump(network.params, f)

# 加载模型
with open('my_model.pkl', 'rb') as f:
    params = pickle.load(f)
    network.params = params
```

#### Q6: 可以用 GPU 加速吗？
**A**: 本项目为教育目的使用纯 NumPy 实现，不支持 GPU。如需 GPU 加速，建议使用 TensorFlow 或 PyTorch。

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

### 🤝 贡献指南

欢迎对本项目做出贡献！

**贡献方式**：
1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

**建议的贡献方向**：
- 修复 bug
- 改进代码注释和文档
- 添加更多示例
- 性能优化
- 添加新的优化器或层实现
- 添加单元测试

---

### 📄 许可证

本项目仅供教育和学习使用。

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

## <a name="english-version"></a>📚 English Version

### Project Introduction

This repository contains the implementation code for the book *"Deep Learning from Scratch: Python-Based Theory and Implementation"*. It builds neural networks and deep learning algorithms from scratch using pure Python and NumPy, helping readers deeply understand the core principles of deep learning.

**Key Features**:
- 🔧 **Built from Scratch**: All components implemented using NumPy without deep learning frameworks
- 📖 **Step-by-Step Learning**: Organized by chapters, from perceptrons to CNNs
- 🎯 **Hands-On Practice**: Runnable example code and MNIST dataset practice in each chapter
- 🧮 **Mathematical Intuition**: Translating abstract mathematical concepts into executable Python code

### Requirements

**Required Dependencies**:
- Python 3.x (3.6+ recommended)
- NumPy >= 1.15.0
- Matplotlib >= 2.0.0
- Pillow (PIL) >= 5.0.0

### Quick Start

#### 1. Clone the Repository

```bash
git clone https://github.com/MengQxuan/fishbook.git
cd fishbook
```

#### 2. Install Dependencies

```bash
pip install numpy matplotlib pillow
```

#### 3. Download MNIST Dataset

The MNIST dataset will be automatically downloaded on first run from a mirror source.

#### 4. Run Examples

```bash
# Test perceptron
python chapter2/and_gate.py

# Run MNIST neural network inference
python chapter3/neuralnet_mnist.py

# Train neural network
python chapter4/train_neuralnet.py
```

### 📂 Project Structure

```plaintext
fishbook/
├── chapter1/          # Chapter 1: Python Basics
├── chapter2/          # Chapter 2: Perceptron
├── chapter3/          # Chapter 3: Neural Networks
├── chapter4/          # Chapter 4: Training Neural Networks
├── chapter5/          # Chapter 5: Backpropagation
├── chapter6/          # Chapter 6: Training Techniques
├── chapter7/          # Chapter 7: Convolutional Neural Networks
├── chapter8/          # Chapter 8: Deep Learning
├── common/            # Common Modules
│   ├── functions.py   # Activation and loss functions
│   ├── layers.py      # Layer implementations
│   ├── optimizer.py   # Optimizers (SGD, Adam, etc.)
│   ├── trainer.py     # Training utilities
│   └── util.py        # Helper functions
└── dataset/           # Dataset Module
    ├── mnist.py       # MNIST data loader
    └── mnist.pkl      # MNIST dataset (auto-downloaded)
```

### 📖 Chapter Overview

- **Chapter 1**: Python basics including NumPy arrays and broadcasting
- **Chapter 2**: Perceptron and logic gates (AND, OR, NAND, XOR)
- **Chapter 3**: Neural network fundamentals, activation functions, MNIST inference
- **Chapter 4**: Training neural networks with gradient descent
- **Chapter 5**: Backpropagation algorithm and computational graphs
- **Chapter 6**: Optimization techniques (Adam, Dropout, Batch Normalization)
- **Chapter 7**: Convolutional Neural Networks (CNNs)
- **Chapter 8**: Deep learning and advanced networks (99%+ accuracy on MNIST)

### 🧰 Common Modules

#### `common/functions.py`
Core mathematical functions including:
- Activation functions: `sigmoid()`, `relu()`, `softmax()`
- Loss functions: `mean_squared_error()`, `cross_entropy_error()`

#### `common/layers.py`
Neural network layer implementations:
- Basic layers: `Relu`, `Sigmoid`, `Affine`, `SoftmaxWithLoss`
- CNN layers: `Convolution`, `Pooling`
- Regularization: `Dropout`, `BatchNormalization`

#### `common/optimizer.py`
Optimization algorithms:
- `SGD`: Stochastic Gradient Descent
- `Momentum`: SGD with momentum
- `AdaGrad`: Adaptive learning rates
- `RMSprop`: Root Mean Square Propagation
- `Adam`: Adaptive Moment Estimation (recommended)

#### `dataset/mnist.py`
MNIST dataset loader with automatic download and preprocessing.

### 💡 Code Example

```python
import numpy as np
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import Adam

# Load data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Create network
network = MultiLayerNet(input_size=784, 
                        hidden_size_list=[100, 100],
                        output_size=10)

# Create optimizer
optimizer = Adam(lr=0.001)

# Training loop
epochs = 20
batch_size = 100

for epoch in range(epochs):
    # Mini-batch training
    batch_mask = np.random.choice(len(x_train), batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # Calculate gradients
    grads = network.gradient(x_batch, t_batch)
    
    # Update parameters
    optimizer.update(network.params, grads)
    
    # Evaluate
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    print(f"Epoch {epoch+1}: Train={train_acc:.4f}, Test={test_acc:.4f}")
```

### 🎓 Learning Path

**Beginner Path** (follow in order):
1. Chapters 1-2: Python basics and perceptron (1-2 days)
2. Chapter 3: Neural network basics (2-3 days)
3. Chapter 4: Training with gradient descent (3-4 days)
4. Chapter 5: Backpropagation (4-5 days)
5. Chapter 6: Optimization techniques (3-4 days)
6. Chapter 7: CNNs (4-5 days)
7. Chapter 8: Deep networks (3-4 days)

**Total Learning Time**: ~20-30 days (2-3 hours per day)

### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### 📄 License

This project is for educational purposes only.

### 📚 References

- *Deep Learning from Scratch* (Original Book)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！**

**⭐ Star this project if you find it helpful!**

</div>