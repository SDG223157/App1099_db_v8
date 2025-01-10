import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x):
    return x**3 - 3*x**2 + 2

# 定义x的取值范围
x = np.linspace(-1, 4, 400)
y = f(x)

# 计算导数
f_prime = 3*x**2 - 6*x
f_double_prime = 6*x - 6

# 绘制函数图像
plt.figure(figsize=(8,6))
plt.plot(x, y, label='$f(x) = x^3 - 3x^2 + 2$', color='blue')

# 标出临界点
critical_points = [0, 2]
for cp in critical_points:
    plt.plot(cp, f(cp), 'ro')  # 红色圆点标出
    plt.text(cp, f(cp)+1, f'({cp}, {f(cp)})', horizontalalignment='center')

# 添加标题和标签
plt.title('函数 $f(x) = x^3 - 3x^2 + 2$ 的图像')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)  # x轴
plt.axvline(0, color='black', linewidth=0.5)  # y轴

# 显示图像
plt.show()
