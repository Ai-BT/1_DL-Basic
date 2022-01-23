# 활성화 함수 

# %%

import numpy as np
import matplotlib.pylab as plt

# 문제 1. 계단함수 구현

# 일반 실수만 받는 함수
def step_func(x):
    if x > 0:
        return 1
    else:
        return 0

# 넘파일로 변환
# bool 을 사용해서 true 1 , false 0
def step_func_np(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([5.0,-1.0,3.0])
y = step_func_np(x)

print(y)



# %%

# 문제 2. 계단함수 그래프

def step_func_plt(x):
    return np.array(x > 0, dtype=np.int)

# arange() -> -5 에서 5 까지 0.1 간격으로 넘파이 배열 생성
x = np.arange(-5.0, 5.0, 0.1) 
y = step_func_plt(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()


# %%

# 문제 3. 시그모이드 그래프

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

a = np.array([-1.0, 1.0, 2.0])
sigmoid(a)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1 , 1.1) # y 축 범위
plt.show()


# 계단함수는 0과 1 중 하나의 값만 돌려주는 반면,
# 시그모이드 함수는 부드러운 곡선을 그리면서 값을 반환
# 하지만 둘 다 입력이 작을 때의 출력은 0 에 가깝고. 커지면 1 에 가깝다
# 그리고 둘 다 비선형 함수

# 비선형 함수 = 직선 1개로만 그릴 수 없는 함수

# %%

# 문제 4. ReLU 함수
# 0을 넘으면 그 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 함수

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 1)
y = relu(x)

plt.plot(x,y)
plt.ylim(-1, 6)
plt.show()

# %%
