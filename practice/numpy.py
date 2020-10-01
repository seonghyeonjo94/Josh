import numpy as np

# ndarray 데이터 생성
x = np.array([1, 2, 3, 4])
x

y = np.array([[2, 3, 4], [1, 2, 5]])
y

# np.arange([start,] stop[, step,])
np.arange(10)
np.arange(1, 10)
np.arange(1, 10, 2)
np.arange(5, 101, 5)

# np.ones(shape), np.zeros(shape)
np.ones((4, 5)) # 튜플을 이용해 명시
np.ones((2, 3, 4))

np.zeros((2, 3))
np.zeros((2, 3, 8))
np.zeros((2, 3, 4, 4))

# np.empty(shape)
# np.full(shape, fill_value)
np.empty((3, 3))

np.full((3, 4), 7)
np.full((3, 4, 2), 7)

# np.eye(N, k=0) 단위 행렬 생성
np.eye(5)

np.eye(3, k=1)

# np.linspace(start, stop, num=50, endpoint=True, retstep=False, axis=0)
np.linspace(1, 10, 3)
np.linspace(1, 10, 4)
np.linspace(1, 10, 5)

# np.reshape(a, newshape)
x = np.arange(1, 16)
x

x.shape

x.reshape(3, 5)
x.reshape(5, 3)
x.reshape(5, 3, 1)

# np.random.rand - 0, 1 사이의 분포로 랜덤한 ndarray 생성
np.random.rand(2, 3)
np.random.rand(5)
np.random.rand(2, 5, 3)

# np.random.randn - 정규분포로 샘플링된 랜덤 ndarray 생성
np.random.randn(3, 4)
np.random.randn(3, 4, 2)
np.random.randn(5)

# np.random.randint(low, hight=None, size=None) - 특정 정수 사이에서 랜덤하게 샘플링
np.random.randint(1, 100, size=(3, 5))
np.random.randint(1, 100, size=(3, 4, 2))
np.random.randint(1, 100, size=(5,))

# np.random.seed(seed=None) - 랜덤한 값을 동일하게 다시 생성하고자 할 때 사용
np.random.seed(100)
np.random.randn(3, 4)

# np.random.choice(a, size=None, replace=True)
''' 
주어진 1차원 ndarray로 부터 랜덤으로 샘플링
정수가 주어진 경우, np.arange(해당숫자)로 간주
'''
np.random.choice(100, size=(3, 4))

x = np.array([1, 2, 3, 1.5, 2.6, 4.9])

np.random.choice(x, size=(2, 2))
np.random.choice(x, size=(2, 2), replace=False)

# 확률분포에 따른 ndarray 생성
# np.random.uniform(low=0.0, high=1.0, size=None)
# np.random.normal(loc=0.0, scale=1.0, size=None) - loc = mean, scale = std
np.random.uniform(1.0, 3.0, size=(4, 5))

np.random.normal(size = (3, 4)) # np.random.randn(3, 4)와 같은 함수

# 1차원 벡터 인덱싱
x = np.arange(10)
x

x[0]
x[9]
x[-1]

x[3] = 100
x

# 2차원 행렬 인덱싱
x = np.arange(10).reshape(2, 5)
x

x[0]
x[1]
x[0, 2]
x[1, 2]

# 3차원 텐서 인덱싱
x = np.arange(36).reshape(3, 4, 3)
x

x[0]
x[1, 2]
x[1, 2, 1]

# 1차원 벡터 슬라이싱
x = np.arange(10)
x

x[1:7]
x[1:]
x[:]


# 2차원 행렬 슬라이싱
x = np.arange(10).reshape(2, 5)
x

x[:, 1:4]
x[:, 0:2]
x[0, :2] # 벡터 (인덱싱은 차원의 갯수를 하나 줄여줌)
x[:1, :2] # 행렬

# 3차원 텐서 슬라이싱
x = np.arange(54).reshape(2, 9, 3)
x

x[:1, :2, :]
x[0, :2, :] # 2차원 행렬

# np.ravel(a, order='C') - 다차원배열을 1차원으로 변경 'C': row 우선 변경, 'F' - col 우선 변경
# np.flatten(a, order='C') - 다차원배열을 1차원으로 변경 'C': row 우선 변경, 'F' - col 우선 변경
# ravel과는 다르게 copy를 생성하여 변경함
x = np.arange(15).reshape(3, 5)
x

x.ravel()
np.ravel(x)
np.ravel(x, order='F')

# np.reshape(a, newshape, order='C')
x= np.arange(36)

print(x)
print(x.shape)
print(x.ndim)

x.reshape(6, 6)
x.reshape(6, -1)
x.reshape(-1, 6)

y = x.reshape(6, 6)

print(y.shape)
print(y.ndim)

k = x.reshape(3, 3, 4) # k = x.reshape(3, 3, -1)도 같음

print(k)
print(k.shape)
print(k.ndim)

x = np.arange(15).reshape(3, 5)
y = np.random.rand(15).reshape(3, 5)
print(x)
print(y)

# 연산 함수 (add, subtract, multiply, divide)
np.add(x, y)
np.subtract(x, y)
np.multiply(x, y)
np.divide(x, y)

x + y
x - y
x * y
x / y

# 통계 함수
print(y)
np.mean(y)
y.mean()
np.max(y)
np.argmax(y)
np.var(y), np.median(y), np.std(y)

# 집계 함수
y

np.sum(y)
sum(y)

np.sum(y, axis=0)

np.cumsum(y)

# any, all 함수
'''
any: 특정 조건을 만족하는 것이 하나라도 있으면 Trun, 아니면 False
all: 모든 원소가 특정 조건을 만족한다면 True, 아니면 False
'''
z = np.random.randn(10)
z

z > 0

np.any(z > 0)

np.all(z != 0)

# np.where(condition, [x, y])
'''
조건에 따라 선별적으로 값을 선택 가능
사용 예 ) 음수인 경우는 0, 나머지는 그대로 값을 쓰는 경우
'''
z = np.random.randn(10)
print(z)

np.where(z > 0 , z, 0)

# axis
x = np.arange(15).reshape(3, 5)
x

np.sum(x)
np.sum(x, axis=0)
np.sum(x, axis=1)

y = np.arange(36).reshape(3, 4, 3)
y

np.sum(y)
np.sum(y, axis=0)
np.sum(y, axis=1)
np.sum(y, axis=2)
np.sum(y, axis=-1)

print(y)

np.sum(y, axis=(0, 1))
np.sum(y, axis=(0, 2))

# Boolean indexing으로 조건에 맞는 데이터 선택
x = np.random.randint(1, 100, size=10)
print(x)

even_mask = x % 2 == 0
print(even_mask)

x[even_mask]
x[x % 2 == 0]

x[x > 30]

# 다중조건 - 파이썬 논리 연산자 and, or, not 키워드 사용 불가
x % 2 == 0
x < 30

x[(x % 2 == 0) & (x < 30)]
x[(x % 2 == 0) | (x < 30)]

# linalg 서브모듈 - 선형대수 연산
# np.linalg.inv - 역행렬
x = np.random.rand(3, 3)
print(x)

np.linalg.inv(x)

x @ np.linalg.inv(x) # 행렬곱은 @ 사용
np.matmul(x, np.linalg.inv(x)) # 동일 함수

# np.linalg.solve
'''
Ax = B 형태의 선형대수식 솔루션을 제공
ex)
x + y = 25
2x + 4y = 64
'''
A = np.array([[1, 1], [2, 4]])
B = np.array([25, 64])

x = np.linalg.solve(A, B)
print(x)

np.allclose(A@x, B)