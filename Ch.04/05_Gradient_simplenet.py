# %%

import numpy as np
from gradient import numerical_gradient
from functions import softmax, CEE



class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 랜덤으로 0 - 1 표준벚유분포 난수 생성 (2x3 난수)
        # print(self.W)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x , t):
        z = self.predict(x)
        y = softmax(z)
        loss = CEE(y, t)

        return loss
    

net = simpleNet()
print(net.W)

# %%
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

# %%
np.argmax(p)

# %%
t = np.array([0, 0, 2])  # 정답 레이블
net.loss(x, t)

# %%

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)

