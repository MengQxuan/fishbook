import numpy as np

class SGD:
    """随机梯度下降法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        """
        初始化学习率
        :param lr: 学习率 (默认值: 0.01)
        """
        self.lr = lr
        
    def update(self, params, grads):
        """
        更新参数
        :param params: 模型参数 (字典)
        :param grads: 参数的梯度 (字典)
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]  # 参数更新


class Momentum:
    """Momentum 随机梯度下降法
    Momentum 在 SGD 的基础上引入了动量的概念，模拟物理中的惯性。
    通过累积之前的梯度，减少震荡并加速收敛。
    """

    def __init__(self, lr=0.01, momentum=0.9):
        """
        初始化学习率和动量因子
        :param lr: 学习率 (默认值: 0.01)
        :param momentum: 动量因子 (默认值: 0.9)
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None  # 初始化速度为 None
        
    def update(self, params, grads):
        """
        更新参数
        :param params: 模型参数 (字典)
        :param grads: 参数的梯度 (字典)
        """
        if self.v is None:
            self.v = {key: np.zeros_like(val) for key, val in params.items()}  # 初始化速度
        
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]  # 更新速度
            params[key] += self.v[key]  # 更新参数


class Nesterov:
    """Nesterov 加速梯度法
    NAG 是 Momentum 的改进版本，它在计算梯度时提前考虑了动量的影响，从而更准确地更新参数。
    """

    def __init__(self, lr=0.01, momentum=0.9):
        """
        初始化学习率和动量因子
        :param lr: 学习率 (默认值: 0.01)
        :param momentum: 动量因子 (默认值: 0.9)
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        """
        更新参数
        :param params: 模型参数 (字典)
        :param grads: 参数的梯度 (字典)
        """
        if self.v is None:
            self.v = {key: np.zeros_like(val) for key, val in params.items()}  # 初始化速度
        
        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:
    """AdaGrad 自适应梯度算法
    AdaGrad 自适应地调整每个参数的学习率。对于更新频繁的参数，学习率会减小；对于更新较少的参数，学习率会增大。
    """

    def __init__(self, lr=0.01):
        """
        初始化学习率
        :param lr: 学习率 (默认值: 0.01)
        """
        self.lr = lr
        self.h = None  # 累积梯度平方和
        
    def update(self, params, grads):
        """
        更新参数
        :param params: 模型参数 (字典)
        :param grads: 参数的梯度 (字典)
        """
        if self.h is None:
            self.h = {key: np.zeros_like(val) for key, val in params.items()}  # 初始化累积梯度平方和
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]  # 累积梯度平方
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)  # 更新参数


class RMSprop:
    """RMSprop 优化算法
    RMSprop 是 AdaGrad 的改进版本，通过引入指数加权平均来平滑梯度的平方值，避免学习率过快下降。
    """

    def __init__(self, lr=0.01, decay_rate=0.99):
        """
        初始化学习率和衰减率
        :param lr: 学习率 (默认值: 0.01)
        :param decay_rate: 衰减率 (默认值: 0.99)
        """
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        """
        更新参数
        :param params: 模型参数 (字典)
        :param grads: 参数的梯度 (字典)
        """
        if self.h is None:
            self.h = {key: np.zeros_like(val) for key, val in params.items()}  # 初始化累积梯度平方和
        
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]  # 平滑梯度平方
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)  # 更新参数


class Adam:
    """Adam 优化算法
    Adam 结合了 Momentum 和 RMSprop 的优点，既考虑了梯度的一阶动量（均值），也考虑了二阶动量（方差），是目前最常用的优化算法之一。
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """
        初始化学习率和超参数
        :param lr: 学习率 (默认值: 0.001)
        :param beta1: 一阶动量衰减率 (默认值: 0.9)
        :param beta2: 二阶动量衰减率 (默认值: 0.999)
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None  # 一阶动量
        self.v = None  # 二阶动量
        
    def update(self, params, grads):
        """
        更新参数
        :param params: 模型参数 (字典)
        :param grads: 参数的梯度 (字典)
        """
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)  # 偏差校正
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])  # 更新一阶动量
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])  # 更新二阶动量
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)  # 更新参数