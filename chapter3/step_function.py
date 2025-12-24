# 阶跃函数
"""
    阶跃函数：将输入数组中的元素转换为二进制值（0或1）
    参数:
        x: 输入数组或标量，可以是数值类型或数组类型
    返回值:
        numpy.ndarray: 与输入形状相同的数组，其中每个元素根据原值是否大于0被转换为1（大于0）或0（小于等于0）
"""
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=int)

X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 指定图中绘制的y轴的范围
plt.show()
