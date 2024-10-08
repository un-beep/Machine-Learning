import numpy as np
import os
a = np.array([[1,2,3]])
b = np.array([[4,5,6]])
c = a.T * b.T + np.log(1 + np.exp(b.T))

beta = np.random.randn(5 + 1, 1)
print(beta)