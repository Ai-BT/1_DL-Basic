

# Chapter 2. 퍼셉트론

# %%

# 문제 1. 간단한 퍼셉트론 구현 (AND 게이트 구현)

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta :
        return 1

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))


# %%
import numpy as np
# numpy 는 배열을 만들어주는 메소드

# 문제 2. 가중치와 편향을 도입한 'AND 게이트' 구현
# theta 값을 b 로 치환해서 넘겨줌

# 둘다 1 이면, 1 출력
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    # print(tmp)
    if tmp <= 0 :
        return 0
    elif tmp > 0 :
        return 1

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))


# %%

# 문제 3. NAND 와 OR 게이트를 구현

# AND의 반대
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0 :
        return 0
    elif tmp > 0 :
        return 1

print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))

print('------------------------')

# 입력 신호 중 하나 이상이 1 이면, 1 출력
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x*w) + b
    # print(tmp)
    if tmp <= 0 :
        return 0
    elif tmp > 0 :
        return 1


print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))






# %%

# 문제 4. XOR 구현

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))



# %%
