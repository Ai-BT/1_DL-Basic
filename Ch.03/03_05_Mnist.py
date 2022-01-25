
# %%
from copyreg import pickle
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle

# flatten 는 이미지를 1차원 배열로 만들지 유무 (True 는 1*28*28=784 1차원 배열 이고, false는 3차원 배열로 1*28*28)
# normalize 는 정규화 설정 (True 0.0 - 1.0 사이, False는 0-255)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape) # 훈련 이미지
print(t_train.shape) # 훈련 레이블
print(x_test.shape) # 시험 이미지
print(t_test.shape) # 시험 레이블


# %%

# 데이터 하나 show
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)


# %%

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# C:\Users\the35\Documents\Z. etc\Basic_DL\Ch.03\dataset\sample_weight.pkl
def init_network():
    with open("C:/Users/the35/Documents/Z. etc/Basic_DL/Ch.03/dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

# 시그모이드
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 소프트맥스
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버플로 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] 
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y

x, t = get_data() # x = (10000, 784), t = (10000,)

network = init_network()

accuracy_cnt = 0

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print('정확도: ' + str(float(accuracy_cnt)/len(x)))



# %%


batch_size = 100 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size): # 100장씩 묶어서 꺼낸다
    x_batch = x[i: i+batch_size] # 
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print('정확도: ' + str(float(accuracy_cnt)/len(x)))


# %%

x = np.array([
    [0.1, 0.8, 0.1], 
    [0.3, 0.1, 0.6], 
    [0.2, 0.5, 0.3], 
    [0.8, 0.1, 0.1]
    ])

y1 = np.sum(x, axis=0)
print('y1 = ', y1) # 
y2 = np.sum(x, axis=1)
print(y2.shape) # axis는 1번째 차원축으로 sum 을 해라
print('y2 = ', y2)

y = np.argmax(x, axis=1) # axis는 1번째 차원축으로 argmax(가장 큰 인덱스 값) 을 해라
print('y = ',y)


# %%
