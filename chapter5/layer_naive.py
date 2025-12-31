# 乘法层的实现
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        # 前向传播计算乘法
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout):
        # 反向传播计算梯度
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        # 前向传播计算加法
        out = x + y

        return out

    def backward(self, dout):
        # 反向传播计算梯度，加法层的梯度为1
        dx = dout * 1
        dy = dout * 1

        return dx, dy