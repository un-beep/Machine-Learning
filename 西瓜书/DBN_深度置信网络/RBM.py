import torch
import torchvision
import matplotlib.pyplot as plt
 
# 加载 MNIST 数据集
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
 
# 定义 RBM 模型
class RBM(torch.nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(n_hidden, n_visible))
        self.bv = torch.nn.Parameter(torch.randn(n_visible))
        self.bh = torch.nn.Parameter(torch.randn(n_hidden))
 
    def forward(self, x):
        h = torch.sigmoid(torch.matmul(x, self.W.t()) + self.bh)
        v = torch.sigmoid(torch.matmul(h, self.W) + self.bv)
        return v
 
# 训练 RBM 模型
rbm = RBM(784, 128)
optimizer = torch.optim.Adam(rbm.parameters(), lr=0.01)
for epoch in range(10):
    for x, _ in trainloader:
        x = x.view(-1, 784)
        v1 = x
        h1 = torch.sigmoid(torch.matmul(v1, rbm.W.t()) + rbm.bh)
        v2 = torch.sigmoid(torch.matmul(h1, rbm.W) + rbm.bv)
        h2 = torch.sigmoid(torch.matmul(v2, rbm.W.t()) + rbm.bh)
        loss = torch.mean(torch.sum((v1 - v2) ** 2, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
# 绘制 RBM 模型学习到的特征
plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(rbm.W[i].detach().view(28, 28), cmap='gray')
    plt.axis('off')
plt.show()