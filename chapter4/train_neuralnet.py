# mini-batch学习
import sys, os
#sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 初始化一个两层神经网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  # 适当设定循环的次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = [] # 训练损失列表
train_acc_list = [] # 训练准确率列表
test_acc_list = [] # 测试准确率列表

# 每个epoch的迭代次数
iter_per_epoch = max(train_size / batch_size, 1)

# 循环进行训练
for i in range(iters_num):
    # 随机选择mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 计算梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 计算mini-batch的损失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 每个epoch结束时，计算训练和测试的准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制训练和测试准确率的变化图
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()