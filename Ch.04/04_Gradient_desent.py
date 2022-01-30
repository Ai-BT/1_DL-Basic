# %%

import numpy as np
import matplotlib.pylab as plt
# 경사법으로 x^2 + y^2 최소값 구하기

# 기울기
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    
    return grad

# 경사하강법
# lr = learning rate 학습률
# 갱신하는 양 즉, 매개변수 값을 얼마나 갱신하느냐를 정하는 것
# 극소값, 최소값을 찾기 위해 내려가는데 얼마나 내려가면서 찾을것인지 정하는 값이 학습률 (lr
def gradient_descent(f, init_x, lr = 0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    
    return x


def function_2(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])
result_1 = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(result_1)

# 너무 크면 큰 값으로 발산
init_x = np.array([-3.0, 4.0])
result_2 = gradient_descent(function_2, init_x=init_x, lr=10, step_num=100)
print(result_2)

# 너무 작으면 갱신을 하지 않고 끝나버림
init_x = np.array([-3.0, 4.0])
result_3 = gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
print(result_3)



# %%

# 그래프 p.132
init_x = np.array([-3.0, 4.0])
x = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=20)

plt.plot([-5, 5], [0,0], '--b')
plt.plot([0, 0], [-5, 5], '--b')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.show

# %%
