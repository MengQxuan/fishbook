# coding: utf-8
import sys, os
#sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

# 计算数值梯度
grad_numerical = network.numerical_gradient(x_batch, t_batch)
# 计算反向传播梯度
grad_backprop = network.gradient(x_batch, t_batch)

# 比较数值梯度和反向传播梯度的差异
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))