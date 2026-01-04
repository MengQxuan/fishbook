# Fishbook - 《深度学习入门：基于Python的理论与实现》

本仓库包含神经网络和深度学习概念的 Python 实现，按章节组织。代码展示了深度学习的基本概念，从基础的 Python 编程到使用反向传播实现神经网络。

## 项目概述

本项目使用 Python 和 NumPy 从零实现神经网络和深度学习算法。内容涵盖了感知机、激活函数、梯度下降和反向传播等基本概念，并通过 MNIST 数据集提供了实际示例。

## 项目结构

```plaintext
fishbook/
├── chapter1/          # Python 基础和基本概念
├── chapter2/          # 逻辑门 (AND, OR, NAND, XOR)
├── chapter3/          # 神经网络基础和 MNIST
├── chapter4/          # 梯度下降方法
├── chapter5/          # 反向传播和计算图
├── common/            # 公共工具和模块
│   ├── functions.py   # 激活函数和损失函数
│   ├── layers.py      # 层的实现
│   ├── optimizer.py   # 优化算法
│   ├── trainer.py     # 训练工具
│   └── util.py        # 辅助函数
└── dataset/           # 数据集加载和预处理
    ├── mnist.py       # MNIST 数据集加载器
    └── mnist.pkl      # MNIST 数据集
```

## 环境要求

- Python 3.x
- NumPy
- Matplotlib
- PIL (Pillow)

## 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/MengQxuan/fishbook.git
   cd fishbook
   ```

2. 安装所需依赖：
   ```bash
   pip install numpy matplotlib pillow
   ```

## 使用方法

### Chapter 1: Python 基础

介绍 Python 编程概念，包括变量、类型、类和 NumPy 广播。

```bash
python chapter1/hello.py
```

### Chapter 2: 感知机与逻辑门

使用感知机实现基本逻辑门。

```bash
python chapter2/and_gate.py
python chapter2/or_gate.py
python chapter2/nand_gate.py
python chapter2/xor_gate.py
```

### Chapter 3: 神经网络

使用 MNIST 数据集实现神经网络。

- `sigmoid.py` - Sigmoid 激活函数
- `relu.py` - ReLU 激活函数
- `step_function.py` - 阶跃函数
- `neuralnet_mnist.py` - MNIST 数据集上的神经网络推理
- `neuralnet_mnist_batch.py` - MNIST 的批量处理

```bash
python chapter3/neuralnet_mnist.py
```

### Chapter 4: 训练神经网络

梯度下降和训练算法。

- `gradient_1d.py` - 一维梯度
- `gradient_2d.py` - 二维梯度
- `gradient_method.py` - 梯度下降方法
- `train_neuralnet.py` - 完整训练示例

```bash
python chapter4/train_neuralnet.py
```

### Chapter 5: 反向传播

实现反向传播和计算图。

- `buy_apple.py` - 简单计算图示例
- `buy_apple_orange.py` - 扩展计算图
- `layer_naive.py` - 基本层实现
- `train_neuralnet.py` - 使用反向传播进行训练

```bash
python chapter5/train_neuralnet.py
```

## 功能特点

- **从零实现**：所有神经网络组件均使用 NumPy 从零实现。
- **教育重点**：代码结构清晰，适合学习，遵循渐进式学习路径。
- **支持 MNIST 数据集**：包含处理 MNIST 手写数字数据集的工具。
- **多种激活函数**：Sigmoid、ReLU、Softmax 和 Step 函数。
- **优化算法**：支持 SGD、Momentum、AdaGrad 和 Adam 优化器。
- **层抽象**：模块化的层实现，用于构建神经网络。
- **梯度检查**：提供验证梯度计算的工具。

## 公共模块

### functions.py

包含激活函数、损失函数及其梯度：

- `sigmoid()`, `relu()`, `softmax()`
- `cross_entropy_error()`, `mean_squared_error()`

### layers.py

神经网络的层实现：

- `Relu`, `Sigmoid`, `Affine`, `SoftmaxWithLoss`

### optimizer.py

优化算法：

- `SGD`, `Momentum`, `AdaGrad`, `Adam`

### trainer.py

管理训练循环的工具。

## 贡献

欢迎贡献代码！请随时提交 Pull Request。

## 许可证

本项目仅供教育用途，按原样提供。