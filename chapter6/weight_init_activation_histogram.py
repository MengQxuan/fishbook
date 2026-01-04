import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Sigmoid 激活函数：将输入压缩到 (0, 1)。输入过大/过小会饱和。"""
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    """ReLU 激活函数：max(0, x)。负值被截断为 0。"""
    return np.maximum(0, x)


def tanh(x):
    """tanh 激活函数：将输入压缩到 (-1, 1)。也会发生饱和。"""
    return np.tanh(x)


# 生成输入数据：1000 条样本，每条 100 维，标准正态分布
input_data = np.random.randn(1000, 100)

# 每层神经元数量（这里隐藏层都设为 100）
node_num = 100

# 隐藏层层数
hidden_layer_size = 5

# 用字典保存每一层的激活值输出（方便后面画图）
activations = {}

# x 表示当前层的输入，初始为输入数据
x = input_data

# 逐层做前向传播
for i in range(hidden_layer_size):
    # 从第 2 层开始，输入改为上一层的输出（激活值）
    if i != 0:
        x = activations[i - 1]

    # 权重初始化（可切换不同策略观察激活值分布差异）
    # 1) 标准正态 * 1：方差较大，sigmoid 容易饱和到 0/1
    # w = np.random.randn(node_num, node_num) * 1

    # 2) 很小的初始权重：输出会集中在 0.5 附近（对 sigmoid）
    # w = np.random.randn(node_num, node_num) * 0.01

    # 3) Xavier 初始化（适合 sigmoid/tanh 等）：保持信号方差更稳定
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

    # 4) He 初始化（适合 ReLU）：保持 ReLU 网络的方差更稳定
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    # 线性变换：a 是激活函数之前的“预激活”值
    a = np.dot(x, w)

    # 选择激活函数（切换以观察不同非线性带来的分布变化）
    # z = sigmoid(a)
    z = ReLU(a)
    # z = tanh(a)

    # 保存当前层的输出激活值
    activations[i] = z

# ===== 绘制各层激活值分布的直方图 =====
for i, a in activations.items():
    # 1 行 N 列的子图布局，第 i+1 个子图
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")

    # 除第一个子图外，隐藏 y 轴刻度以减少视觉干扰
    if i != 0:
        plt.yticks([], [])

    # 下面两行可用于固定显示范围（按需打开）
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)

    # 画直方图：
    # a.flatten()：把 (样本数, 节点数) 拉平成 1D
    # 30：bins 数量
    # range=(0,1)：适合 sigmoid 输出范围。若用 tanh/ReLU 需相应修改。
    plt.hist(a.flatten(), 30, range=(0, 1))

# 显示绘图窗口
plt.show()