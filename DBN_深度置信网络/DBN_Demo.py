import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))
        self.v_bias = nn.Parameter(torch.zeros(visible_units))

    def forward(self, v):
        # 定义前向传播
        # 省略其他代码...
        i=0

class DBN(nn.Module):
    def __init__(self, layers):
        super(DBN, self).__init__()
        self.rbms = nn.ModuleList([RBM(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

    def forward(self, v):
        h = v
        for rbm in self.rbms:
            h = rbm(h)
        return h

# 定义DBN的层大小
layers = [784, 500, 200, 100]

# 创建DBN模型
dbn = DBN(layers)

