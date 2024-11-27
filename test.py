import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 常数定义
g = 9.81  # 重力加速度 (m/s^2)
l1 = 1.0  # 第一个摆锤的长度 (m)
l2 = 1.0  # 第二个摆锤的长度 (m)
m1 = 1.0  # 第一个摆锤的质量 (kg)
m2 = 1.0  # 第二个摆锤的质量 (kg)

# 运动方程
def equations(t, y):
    # y[0] = θ1, y[1] = θ2, y[2] = ω1, y[3] = ω2
    θ1, θ2, ω1, ω2 = y

    # 中间变量
    delta = θ2 - θ1
    denominator1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) ** 2
    denominator2 = (l2 / l1) * denominator1

    # 角加速度公式（基于拉格朗日方程）
    ω1_dot = (m2 * g * np.sin(θ2) * np.cos(delta) - m2 * np.sin(delta) * (l2 * ω2 ** 2 + l1 * ω1 ** 2 * np.cos(delta)) -
              (m1 + m2) * g * np.sin(θ1)) / denominator1
    ω2_dot = (m2 * l1 * ω1 ** 2 * np.sin(delta) - m2 * g * np.sin(θ2) + m2 * np.cos(delta) * (l2 * ω2 ** 2 + l1 * ω1 ** 2 * np.cos(delta)) -
              (m1 + m2) * g * np.sin(θ1) * np.cos(delta)) / denominator2

    return [ω1, ω2, ω1_dot, ω2_dot]

# 初始条件：角度和角速度
θ1_0 = np.pi / 2  # 初始角度（弧度）
θ2_0 = np.pi / 2  # 初始角度（弧度）
ω1_0 = 0.0        # 初始角速度
ω2_0 = 0.0        # 初始角速度

# 初始状态向量
y0 = [θ1_0, θ2_0, ω1_0, ω2_0]

# 时间区间
t_span = (0, 20)  # 时间范围
t_eval = np.linspace(t_span[0], t_span[1], 500)  # 时间步长

# 求解运动方程
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval, method='RK45')

# 提取解
θ1, θ2 = sol.y[0], sol.y[1]

# 计算摆锤的 x 和 y 坐标
x1 = l1 * np.sin(θ1)
y1 = -l1 * np.cos(θ1)

x2 = x1 + l2 * np.sin(θ2)
y2 = y1 - l2 * np.cos(θ2)

# 动态绘图
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# 初始摆锤的线条
line1, = ax.plot([], [], 'ro-', lw=2)  # 第一个摆锤
line2, = ax.plot([], [], 'bo-', lw=2)  # 第二个摆锤
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

# 更新函数
def update(num):
    line1.set_data([0, x1[num]], [0, y1[num]])
    line2.set_data([x1[num], x2[num]], [y1[num], y2[num]])
    time_text.set_text(f't = {sol.t[num]:.2f} s')
    return line1, line2, time_text

# 动画设置
from matplotlib.animation import FuncAnimation

ani = FuncAnimation(fig, update, frames=len(sol.t), interval=50, blit=True)

plt.show()
