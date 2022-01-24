
# %%
from ctypes.wintypes import WIN32_FIND_DATAA
from tkinter import W
import numpy as np

# 3층 신경망 구현
# p.86 그림 3-18 참고

# 시그모이드
def sigmoid(x):
    return 1 / (1 + np.exp(x))


# 결과 리턴 함수
def result(x):
    return x

# 1층 1번째 원소에서 2층 1번째 원소로 가는 함수
X = np.array([1.0, 2.0]) # 1 X 2
W1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]) # 2 X 3
b1 = np.array([0.1, 0.2, 0.3]) # 1 X 3

A1 = np.dot(X, W1) + b1 # (1 X 3) + (1 X 3)
Z1 = sigmoid(A1) # (1 X 3)
# print(Z1)


# 2층 1번째 원소에서 3층 1번째 원소로 가는 함수
W2 = np.array([[0.1, 0.4], [0.2, 0.3], [0.3, 0.5]]) # 3 x 2
b2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + b2 # (1 X 2) + (1 X 2)
Z2 = sigmoid(A2) # (1 X 2)
# print(Z2)


# 3층 1번째 원소에서 y층 1번째 원소로 가는 함수
W3 = np.array([[0.1, 0.4], [0.2, 0.3]]) # 2 x 2
b3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + b3 # (1 X 2) + (1 X 2)
Y = result(A3)
print(A3)




# %%

# 구현 정리

# network 함수
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

# forward 함수
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] 
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = result(a3)

    return y


# 결과 출력
network = init_network()
x = np.array([0.1, 0.5])
y = forward(network, x)
print(y)


# %%
