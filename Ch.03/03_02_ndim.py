# 다차원 배열

# %%
import numpy as np

# 문제 1. 1차원 배열
A = np.array([1,2,3,4])
print( A )

print( np.ndim(A) ) # 몇 차원?
print( A.shape ) 
print( A.shape[0] ) # 몇개의 원소?



# %%

# 문제 2. 2차원 배열
B = np.array([[1,2], [3,4], [5,6]])
print( B )

print( np.ndim(B) ) # 몇 차원?
print( B.shape ) # 3행, 2열
print( B.shape[0] ) # shape[3,2]의 0 번째 , 1 번째


# %%

# 문제 3. 신경망의 내적
# X(2,) * W(2,3) = Y

X = np.array([1,2])
print(X.shape)

W = np.array([[1,2,3],[4,5,6]])
print(W.shape)

Y = np.dot(X, W)
print(Y)


