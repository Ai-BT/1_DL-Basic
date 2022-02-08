# CNN
# Convolution을 통한 Filtering과 Max pooling을 반복하여, 정말 중요한 특징들을 정제한 후, Classification을 하는것
# 어떤 이미지 하나가 (1024 * 860 * 3) 3차원 RGB를 갖는다고 할때, 모두 곱하면 2,580,480 데이터를 가진다.
# 이 수많은 데이터를 학습하는데는 무리가 있어서, 필터와 풀링을 통해서 중요한 데이터만 남겨서 압축한다고 생각하면 된다.
# https://limitsinx.tistory.com/140

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

t = torch.randint(10, size=(20,16,50,100))
# print(t)
# print(t.shape)
# print(t.ndim)


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    # self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1)
    # self.conv1 = nn.Conv2d(3,16,5)
    self.conv1 = nn.Conv2d(16, 33, 3, stride=2)
    # self.fc1 = nn.Linear(10 * 12 * 12, 50)
    # self.fc2 = nn.Linear(50, 10)
  
  def forward(self, x):
    print("1. 연산 전", x.size())

    x = F.relu(self.conv1(x))
    print("2. conv1 연산 후", x.size())

    # x = F.relu(self.conv2(x))
    # print("3. conv2 연산 후",x.size())

    # x = x.view(-1, 10 * 12 * 12)
    # print("4. 차원 감소 후", x.size())

    # x = F.relu(self.fc1(x))
    # print("5. fc1 연산 후", x.size())

    # x = self.fc2(x)
    # print("6. fc2 연산 후", x.size())
    return x

cnn = CNN()
output = cnn(torch.randn(20, 16, 50, 100)) # Input Size: (10, 1, 20, 20) 4차원


# %%

# in_channels: 입력 채널 수을 뜻합니다. 흑백 이미지일 경우 1, RGB 값을 가진 이미지일 경우 3 을 가진 경우가 많습니다.
# out_channels: 출력 채널 수을 뜻합니다.
# kernel_size (filter): 커널 사이즈를 뜻합니다. int 혹은 tuple이 올 수 있습니다.
# stride: stride 사이즈를 뜻합니다. int 혹은 tuple이 올 수 있습니다. 기본 값은 1입니다.
# padding: padding 사이즈를 뜻합니다. int 혹은 tuple이 올 수 있습니다. 기본 값은 0입니다.
# padding_mode: padding mode를 설정할 수 있습니다. 기본 값은 'zeros' 입니다. 아직 zero padding만 지원 합니다.
# dilation: 커널 사이 간격 사이즈를 조절 합니다. 해당 링크를 확인 하세요.
# groups: 입력 층의 그룹 수을 설정하여 입력의 채널 수를 그룹 수에 맞게 분류 합니다. 그 다음, 출력의 채널 수를 그룹 수에 맞게 분리하여, 입력 그룹과 출력 그룹의 짝을 지은 다음 해당 그룹 안에서만 연산이 이루어지게 합니다.
# bias: bias 값을 설정 할 지, 말지를 결정합니다. 기본 값은 True 입니다.


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=16, out_channels=33, kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1)
    self.fc1 = nn.Linear(10 * 12 * 12, 50)
    self.fc2 = nn.Linear(50, 10)
  
  def forward(self, x):
    print("1. 연산 전", x.size())

    x = F.relu(self.conv1(x))
    print("2. conv1 연산 후", x.size())

    x = F.relu(self.conv2(x))
    print("3. conv2 연산 후",x.size())

    x = x.view(-1, 10 * 12 * 12)
    print("4. 차원 감소 후", x.size())

    x = F.relu(self.fc1(x))
    print("5. fc1 연산 후", x.size())

    x = self.fc2(x)
    print("6. fc2 연산 후", x.size())
    return x

cnn = CNN()
output = cnn(torch.randn(10, 1, 20, 20))  # Input Size: (10, 1, 20, 20) 4차원
# %%

# 1. numpy 1차원

t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)

print('차원 = ', t.ndim)
print('shape = ', t.shape) # (7,) = (1,7) 즉, 1x7 크기를 가지는 벡터



# %%

# 2. numpy 2차원
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)

print('차원 = ', t.ndim)
print('shape = ', t.shape) # (7,) = (1,7) 즉, 1x7 크기를 가지는 벡터

# %%

# 3. torch 1차원

t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

print('차원 = ',t.dim())  # rank. 즉, 차원
print('shape = ',t.shape)  # shape
print('shape = ',t.size()) # shape

# %%

# 4. torch 2차원
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
print(t)
print('차원 = ',t.dim())  # rank. 즉, 차원
print('shape = ',t.shape)  # shape
print('shape = ',t.size()) # shape

# %%

t = torch.randn(10, 1, 20, 20)
print(t)
print('차원 = ',t.dim())  # rank. 즉, 차원
print('shape = ',t.shape)  # shape
print('shape = ',t.size()) # shape

# %%

# 평균이 0이고 표준편차가 1인 가우시안 정규분포를 이용해 생성
t = torch.randn(5)
print(t)


# %%
