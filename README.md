# Fishbook - Deep Learning from Scratch

This repository contains Python implementations of neural networks and deep learning concepts, organized by chapters. The code demonstrates fundamental concepts in deep learning, from basic Python programming to implementing neural networks with backpropagation.

## Project Overview

This project implements neural networks and deep learning algorithms from scratch using Python and NumPy. It covers fundamental concepts including perceptrons, activation functions, gradient descent, and backpropagation, with practical examples using the MNIST dataset.

## Project Structure

```plaintext
fishbook/
├── chapter1/          # Python basics and fundamentals
├── chapter2/          # Logic gates (AND, OR, NAND, XOR)
├── chapter3/          # Neural network basics and MNIST
├── chapter4/          # Gradient descent methods
├── chapter5/          # Backpropagation and computational graphs
├── common/            # Shared utilities and modules
│   ├── functions.py   # Activation functions and loss functions
│   ├── layers.py      # Layer implementations
│   ├── optimizer.py   # Optimization algorithms
│   ├── trainer.py     # Training utilities
│   └── util.py        # Helper functions
└── dataset/           # Dataset loading and preprocessing
    ├── mnist.py       # MNIST dataset loader
    └── mnist.pkl      # MNIST dataset
```

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- PIL (Pillow)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MengQxuan/fishbook.git
   cd fishbook
   ```

2. Install the required dependencies:
   ```bash
   pip install numpy matplotlib pillow
   ```

## Usage

### Chapter 1: Python Basics

Introduction to Python programming concepts including variables, types, classes, and NumPy broadcasting.

```bash
python chapter1/hello.py
```

### Chapter 2: Perceptrons and Logic Gates

Implementation of basic logic gates using perceptrons.

```bash
python chapter2/and_gate.py
python chapter2/or_gate.py
python chapter2/nand_gate.py
python chapter2/xor_gate.py
```

### Chapter 3: Neural Networks

Neural network implementation with the MNIST dataset.

- `sigmoid.py` - Sigmoid activation function
- `relu.py` - ReLU activation function
- `step_function.py` - Step function
- `neuralnet_mnist.py` - Neural network inference on MNIST
- `neuralnet_mnist_batch.py` - Batch processing for MNIST

```bash
python chapter3/neuralnet_mnist.py
```

### Chapter 4: Training Neural Networks

Gradient descent and training algorithms.

- `gradient_1d.py` - One-dimensional gradient
- `gradient_2d.py` - Two-dimensional gradient
- `gradient_method.py` - Gradient descent method
- `train_neuralnet.py` - Complete training example

```bash
python chapter4/train_neuralnet.py
```

### Chapter 5: Backpropagation

Implementation of backpropagation and computational graphs.

- `buy_apple.py` - Simple computational graph example
- `buy_apple_orange.py` - Extended computational graph
- `layer_naive.py` - Basic layer implementation
- `train_neuralnet.py` - Training with backpropagation

```bash
python chapter5/train_neuralnet.py
```

## Features

- **From Scratch Implementation**: All neural network components are implemented from scratch using NumPy.
- **Educational Focus**: Code is structured to be clear and educational, following a progressive learning path.
- **MNIST Support**: Includes utilities for working with the MNIST handwritten digit dataset.
- **Multiple Activation Functions**: Sigmoid, ReLU, Softmax, and Step functions.
- **Optimization Algorithms**: SGD, Momentum, AdaGrad, Adam optimizers.
- **Layer Abstractions**: Modular layer implementations for building neural networks.
- **Gradient Checking**: Tools for verifying gradient calculations.

## Common Modules

### functions.py

Contains activation functions, loss functions, and their gradients:

- `sigmoid()`, `relu()`, `softmax()`
- `cross_entropy_error()`, `mean_squared_error()`

### layers.py

Layer implementations for neural networks:

- `Relu`, `Sigmoid`, `Affine`, `SoftmaxWithLoss`

### optimizer.py

Optimization algorithms:

- `SGD`, `Momentum`, `AdaGrad`, `Adam`

### trainer.py

Training utilities for managing the training loop.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is provided as-is for educational purposes.

## Acknowledgments

This project is inspired by deep learning educational materials and implements fundamental concepts in neural networks and deep learning from scratch.