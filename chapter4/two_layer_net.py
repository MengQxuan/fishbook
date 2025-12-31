# 2层神经网络
import sys, os
#sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:
    # 初始化神经网络参数 参数依次表示输入层的神经元数、隐藏层的神经元数、输出层的神经元数、权重的标准差
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01): 
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 第一层的权重
        self.params['b1'] = np.zeros(hidden_size) # 第一层的偏置
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # 第二层的权重
        self.params['b2'] = np.zeros(output_size) # 第二层的偏置

    # 进行预测推理
    def predict(self, x): # 参数X是图像数据
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # 计算损失函数
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    # 计算准确率
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # 使用数值微分方法计算梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # 第1层权重的梯度
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) # 第1层偏置的梯度
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) # 第2层权重的梯度
        grads['b2'] = numerical_gradient(loss_W, self.params['b2']) # 第2层偏置的梯度
        
        return grads
        
    # 使用误差反向传播法计算梯度
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads