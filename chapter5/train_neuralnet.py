import sys, os
#sys.path.append(os.pardir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 初始化两层神经网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 设置训练迭代次数、训练集大小、批量大小和学习率
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 初始化训练损失列表、训练准确率列表和测试准确率列表
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 计算每轮迭代中包含的批量数量
iter_per_epoch = max(train_size / batch_size, 1)

# 开始训练循环
for i in range(iters_num):
    # 随机选择批量数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 计算梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 根据梯度更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 计算损失并添加到列表中
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 每轮迭代结束后，计算并打印训练和测试准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)