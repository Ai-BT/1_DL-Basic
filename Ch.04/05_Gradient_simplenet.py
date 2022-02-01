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
print('randn 값 = ',net.W) # randn(2,3)

# %%
x = np.array([0.6, 0.9]) # (1,2) * (2,3)
p = net.predict(x) # = (1,3)
print('p = ',p)

# %%
np.argmax(p) # 가장 큰 값 인덱스 출력

# %%
# 0 일때, 오차
t = np.array([1, 0, 0])  # 정답 레이블
a = net.loss(x, t)
print('0 일때, 오차 = ',a)

# 1 일때, 오차
t = np.array([0, 1, 0])  # 정답 레이블
b = net.loss(x, t)
print('1 일때, 오차 = ',b)

# 2 일때, 오차
t = np.array([0, 0, 2])  # 정답 레이블
c = net.loss(x, t)
print('2 일때, 오차 = ',c)

# CEE 
# 오차를 극대화시켜서 정답을 더 가깝게 맞춤


# %%

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)

# p.135
# randn(2,3) 값을 넣은것에 기울기를 구할 수 있다.
# 결과값은 뜻은 w11 양의 값이면 손실함수가 커진다는 것이다.
# 반대로 음의 값이면 손실함수가 작아진다는 의미이다.
# 손실함수를 줄이기 위해 - 를 곱해서 반대 방향으로 바꿔줘야 한다.
# 또 어떠한 값이 다른 값보다 더 크게 기여한다는 것을 알 수 있다.


# %%

# 

