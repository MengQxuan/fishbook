# coding: utf-8
import numpy as np


def smooth_curve(x):
    """用于使损失函数的图形变圆滑

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """打乱数据集

    Parameters
    ----------
    x : 训练数据
    t : 监督数据

    Returns
    -------
    x, t : 打乱的训练数据和监督数据
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将输入的 4D 张量按卷积/池化的滑窗规则展开为 2D 矩阵（im2col）。

    目的
    ----
    把“每个输出位置对应的一块局部区域（receptive field）”摊平成一行，
    从而将卷积/池化运算改写为矩阵乘法或批量点积，提高 numpy 实现的效率。

    参数
    ----
    input_data : ndarray, shape = (N, C, H, W)
        N: 样本数（batch size）
        C: 通道数
        H: 输入高
        W: 输入宽
    filter_h : int
        滤波器（窗口）的高 FH
    filter_w : int
        滤波器（窗口）的宽 FW
    stride : int
        步幅 S
    pad : int
        填充 P（四周补 0 的像素数）

    返回
    ----
    col : ndarray, shape = (N*out_h*out_w, C*filter_h*filter_w)
        每一行是一个输出位置对应的窗口展开后的向量。
        行数：所有样本、所有输出位置的总数
        列数：一个窗口的元素总数（C*FH*FW）
    """
    # 输入维度
    N, C, H, W = input_data.shape

    # 卷积/池化输出空间尺寸（整除保证是整数）
    # out_h = (H + 2P - FH) / S + 1
    # out_w = (W + 2P - FW) / S + 1
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 对输入做 padding：只在 H/W 维补 0，N/C 维不变
    # pad 后 img shape = (N, C, H+2P, W+2P)
    img = np.pad(
        input_data,
        [(0, 0), (0, 0), (pad, pad), (pad, pad)],
        mode='constant'
    )

    # 先用 6D 数组暂存“窗口采样结果”
    # col[n, c, y, x, oh, ow] 表示：
    # 第 n 个样本、第 c 个通道，
    # 窗口内部偏移 (y, x)，输出位置 (oh, ow) 对应的那个输入像素
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 遍历窗口内部坐标 (y, x)，用切片一次性取出所有输出位置 (oh, ow) 的值
    for y in range(filter_h):
        # y:y_max:stride 会产生 out_h 个采样点
        # 右边界用 y + stride*out_h 才能覆盖到最后一个点（切片右边界不包含）
        y_max = y + stride * out_h

        for x in range(filter_w):
            x_max = x + stride * out_w

            # img[:, :, y:y_max:stride, x:x_max:stride] 的 shape = (N, C, out_h, out_w)
            # 直接填入 col 在该 (y, x) 偏移处对应的所有窗口位置
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 将 6D 的 col 调整维度顺序并 reshape 为 2D：
    # (N, C, FH, FW, out_h, out_w)
    #   -> transpose -> (N, out_h, out_w, C, FH, FW)
    #   -> reshape   -> (N*out_h*out_w, C*FH*FW)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]