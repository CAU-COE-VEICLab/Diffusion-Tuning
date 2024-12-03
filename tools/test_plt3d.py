# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from scipy.optimize import curve_fit


# def non_convex_function(x, y):
#     # 定义一个高度非凸的函数形式
#     return (1 - x) * np.exp(-(x**2) - (y + 1)**2) - 10 * (x / 5 - x**3 - y**5) * np.exp(-x**2 - y**2) - 1/3 * np.exp(-(x + 1)**2 - y**2)


# def fit_function(p, q):
#     # 模拟优化路径上的点
#     return p * np.cos(q), p * np.sin(q)


# # 创建网格
# X = np.linspace(-3, 3, 100)
# Y = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(X, Y)

# # 计算Z值
# Z = non_convex_function(X, Y)

# # 创建图形
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # 绘制曲面
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # 添加色标
# fig.colorbar(surf, shrink=0.5, aspect=5)

# # 模拟优化路径
# p = np.linspace(0, 1, 100)
# q = np.linspace(0, 2*np.pi, 100)
# x_path, y_path = fit_function(p, q)
# z_path = non_convex_function(x_path, y_path)
# ax.plot(x_path, y_path, z_path, color='k', lw=2)

# # 设置坐标轴标签
# ax.set_xlabel('Head Weight Space')
# ax.set_ylabel('Backbone Weight Space')
# ax.set_zlabel('Downstream Performance')

# # 调整视角
# ax.view_init(elev=30., azim=-60)

# # 显示图形
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

# 创建数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)

# 高度非凸的函数，包含多个局部最大值
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)
z = np.sin(r) * np.exp(-0.1*r) + r**2 * np.cos(theta) * np.exp(-0.1*r**2)

# 确保最小值为0
z -= z.min()

# 创建图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=0, antialiased=False)

# 添加色标
fig.colorbar(surf, shrink=0.5, aspect=5)

# 设置坐标轴标签
ax.set_xlabel('Head Weight Space')
ax.set_ylabel('Backbone Weight Space')
ax.set_zlabel('Downstream Performance')

# 显示图形
plt.show()

