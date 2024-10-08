import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def sigmoid(z):
    y = 1 / (1 + np.exp(-z))
    return y

def formula3_27(X: np.ndarray, y: np.ndarray, B: np.ndarray) -> float:
    """
    计算逻辑回归的代价函数。

    该函数实现了逻辑回归的代价函数，用于评估模型参数B的预测效果。
    代价函数的计算基于所有样本的预测值和真实值。

    Args:
        X: 特征矩阵，形状为 (m, d)，其中 m 是样本数量，d 是特征数量。
        y: 标签向量，形状为 (m, 1)，包含每个样本的二分类标签（0或1）。
        B: 参数向量，形状为 (d, 1)，包含逻辑回归模型的权重和截距。

    Returns:
        LB: 标量，表示整个训练集的代价函数值。
    
    Notes
        - 代价函数公式为: L(B) = - SUM[y*Z + log(1 + exp(Z))]. 
        - 函数假定X的最后一列已包含了截距项，即无需手动添加。
        - 函数内部会对B向量进行reshape操作，以确保其形状与X矩阵兼容。
        - 函数使用了np.log(1 + np.exp(Z))来提高数值稳定性，避免直接计算np.log(exp(Z))可能导致的溢出问题。
        - 如果B的形状不正确，函数将尝试reshape，但这可能导致错误，调用者应确保B的形状是正确的。
    """

    # 样本点为 m 个样本，为 X 的行数
    m = X.shape[0]
    # 最后一列拼接一列 1
    X_hat = np.c_[X, np.ones((m, 1))]
    B = B.reshape(-1, 1)
    y = y.reshape(-1, 1)
    Z = np.dot(X_hat, B)

    LB = -y * Z + np.log(1 + np.exp(Z)) 
    return LB.sum()

def initialize_beta(n):
    """
    初始化 Beta 向量，采用随机正态分布

    Args:
        n: 属性数量
    
    Returns:
        beta: 返回初始化后的权重向量
    """

    beta = np.random.randn(n + 1, 1) * 0.5 + 1
    return beta

if __name__ == '__main__':

    data = pd.read_csv(r'3线性模型\3.3\watermelon3_0_Ch.csv').values
    
    is_good = data[:, 9] == '是'
    is_bad = data[:, 9] == '否'

    X = data[:, 7:9].astype(float)
    y = data[:, 9]

    y[y == '是'] = 1
    y[y == '否'] = 0
    y = y.astype(int)

    plt.scatter(data[:, 7][is_good], data[:, 8][is_good], c='k', marker='o')
    plt.scatter(data[:, 7][is_bad], data[:, 8][is_bad], c='r', marker='x')

    plt.xlabel('密度')
    plt.ylabel('含糖量')

    # # 可视化模型结果
    # beta = logistic_model(X, y, print_cost=True, method='gradDesc', learning_rate=0.3, num_iterations=1000)
    # w1, w2, intercept = beta
    # x1 = np.linspace(0, 1)
    # y1 = -(w1 * x1 + intercept) / w2

    # ax1, = plt.plot(x1, y1, label=r'my_logistic_gradDesc')

    # lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)  # 注意sklearn的逻辑回归中，C越大表示正则化程度越低。
    # lr.fit(X, y)

    # lr_beta = np.c_[lr.coef_, lr.intercept_]
    # print(J_cost(X, y, lr_beta))

    # # 可视化sklearn LogisticRegression 模型结果
    # w1_sk, w2_sk = lr.coef_[0, :]

    # x2 = np.linspace(0, 1)
    # y2 = -(w1_sk * x2 + lr.intercept_) / w2

    # ax2, = plt.plot(x2, y2, label=r'sklearn_logistic')

    # plt.legend(loc='upper right')
    # plt.show()


