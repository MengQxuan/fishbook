# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNet:
    """全连接的多层神经网络

    网络结构（hidden_layer_num = len(hidden_size_list)）：
        Affine1 -> Activation1 -> Affine2 -> Activation2 -> ... -> Affine(last) -> SoftmaxWithLoss

    注意：
    - self.layers 不包含最后的 SoftmaxWithLoss（它单独放在 self.last_layer）
    - 支持 weight decay（L2 正则），会影响 loss 和 dW

    Parameters
    ----------
    input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay（L2范数）的强度
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda # L2 正则系数
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 生成层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        # 依次添加：Affine -> Activation（隐藏层）
        for idx in range(1, self.hidden_layer_num+1):
            # Affine 层保存 W、b 的引用，用于 forward/backward
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                    self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        # 最后一层（输出层）只有 Affine，不加激活
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss() # # 最后的损失层：softmax + cross entropy

    def __init_weight(self, weight_init_std):
        """设定权重的初始值

        权重矩阵维度：
            W1: (input_size, hidden_size_list[0])
            W2: (hidden_size_list[0], hidden_size_list[1])
            ...
            W_last: (hidden_size_list[-1], output_size)

        偏置向量维度：
            b1: (hidden_size_list[0],)
            ...
            b_last: (output_size,)

        Parameters
        ----------
        weight_init_std : 指定权重的标准差（e.g. 0.01）
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        # 把所有层的输入/输出尺寸串起来，方便循环初始化
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        # 根据激活函数推荐初始化策略
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值

            # W: 高斯随机初始化 * scale
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            # b: 初始化为 0
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        """仅做前向传播，输出网络最后一层 Affine 的结果（scores/logits）"""
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """求损失函数

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        损失函数的值
        """
        # forward：得到 logits
        y = self.predict(x)

        # weight decay：0.5 * lambda * sum(W^2)
        # 0.5 的系数是为了求导时更简洁：d/dW = lambda * W
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):# +2：包含输出层
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        # 数据损失（softmax + cross entropy） + 正则损失
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # 若 t 是 one-hot，则转成类别索引
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """求梯度（数值微分）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """求梯度（误差反向传播法）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        # 调用 loss 会触发 predict + last_layer.forward，并在各层缓存反向传播所需中间变量
        self.loss(x, t)

        # backward
        dout = 1 # 对 loss 的导数起点：dL/dL = 1
        dout = self.last_layer.backward(dout) # 先从 SoftmaxWithLoss 反传到最后一层 Affine 的输出

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            # dW = Affine 层反传得到的 dW + L2 正则项的梯度（lambda * W）
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            # db = Affine 层反传得到的 db
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
