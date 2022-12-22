
# %%
from cv2 import exp
import numpy as np

# 회귀(숫자를 출력)에는 항등 함수
# 분류(성별처럼 분류)에는 소프트 맥스 함수 

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a) # 지수함수
print(exp_a)

sum_exp_a = np.sum(exp_a) # 지수함수의 합
print(sum_exp_a)

y = exp_a / sum_exp_a
print('y = ', y)


# %%

# 소프트맥스
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# %%

# 위의 소프트 맥스는 오버플로 문제가 있다.
# 지수함수의 값이 크만 무한대를 출력하는 문제가 있다. 그래서 개선이 필요하다.

a = np.array([1010, 1000, 900])
y = np.exp(a) / np.sum(np.exp(a))
print(y) # [nan, nan, nan] not a number

# 최대값을 사용하여 오버플로 방지
c = np.max(a)
print( a - c ) 

z = np.exp(a-c) / np.sum(np.exp(a-c))
print(z)


# %%

# 오버플로 방지 soft max
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버플로 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# soft max 함수 특징
# 출력의 총합이 1 이 된다.

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))

# 이러한 성질 때문에 출력 된것을 확률로 해석 가능하다.


# %%
