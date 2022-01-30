# %%
import numpy as np

# Cross Entropy Error

def CEE(y, t):
    # 0을 입력하면 마이너스 무한대로 계산이 안된다. 
    # 그래서 0 이 절대 되지 않게 작은 값을 더해준다. (delta)
    delta = 1e-7 
    return -np.sum(t * np.log(y+delta))

# 정답은 '2'
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# '2'일 확률이 가장 높다고 추정함
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
a = CEE(np.array(y), np.array(t)) # 0.0975
print(a)

# '7'일 확률이 가장 높다고 추정함
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
b = CEE(np.array(y), np.array(t)) # 0.5975
print(b)

# CEE 는 오차를 극대화 시켜서
# 정답을 맞추는 것.
# MSE, CEE 그래프 보면 이해 가능


# %%
