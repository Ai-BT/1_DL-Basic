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
