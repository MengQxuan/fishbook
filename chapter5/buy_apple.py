from layer_naive import *

apple = 100
apple_num = 2
tax = 1.1

# 乘法层类，用于实现前向传播和反向传播的乘法操作
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
# 计算苹果的总价（单价乘以数量）
apple_price = mul_apple_layer.forward(apple, apple_num)
# 计算包含税后的最终价格
price = mul_tax_layer.forward(apple_price, tax)

# backward
# 反向传播计算最终价格对苹果总价的梯度
dapple_price, dtax = mul_tax_layer.backward(price)
# 反向传播计算苹果总价对苹果单价和数量的梯度
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)