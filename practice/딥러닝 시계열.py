########### Chap2 딥러닝 시작 ###########
##### 1에서 10까지 예측 모델 구하기
import numpy as np
# 데이터 생성
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(1, input_dim=1, activation='relu'))
model.compile(loss='mean_squared_error',optimizer='adam',
              metrics=['accuracy'])
model.fit(x, y, epochs= 500, batch_size=1)
loss, acc = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)


##### 101에서 110까지 예측 모델 구하기
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim =1 , activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data = (x_test, y_test))
loss, acc = model.evaluate(x_test, y_test, batch_size =1)
print("loss : ", loss)
print("acc : ", acc)

output = model.predict(x_test)
print("결과물 : \n", output)



########### Chap3 회귀 모델 ###########
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim =1 , activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data = (x_test, y_test))
loss, mse = model.evaluate(x_test, y_test, batch_size =1)

print("loss : ", loss)
print("mse : ", mse)
y_predict = model.predict(x_test)
print("결과물 : \n", y_predict)



########### Chap4 회귀 모델의 판별식 ###########
##### 회귀 모델의 판별식
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim =1 , activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data = (x_test, y_test))
loss, mse = model.evaluate(x_test, y_test, batch_size =1)

print("loss : ", loss)
print("mse : ", mse)
y_predict = model.predict(x_test)
print("결과물 : \n", y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)


##### 회귀 모델 추가 코딩
### 1) Validation data 추가
#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_val = np.array([101,102,103,104,105])
y_val = np.array([101,102,103,104,105])

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim = 1, activation ='relu'))
model.add(Dense(5, input_shape = (1, ), activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))
# model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=1,
validation_data=(x_val, y_val))

#4. 평가 예측
mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

### 2)데이터 분리
#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]
y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim = 1, activation ='relu'))
model.add(Dense(5, input_shape = (1, ), activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))
# model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_data=(x_val, y_val))
# epochs를 1000 에서 300 으로 줄였습니다.

#4. 평가 예측
mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

### 3) train_test_split
#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))
# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
       x, y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5, shuffle=False
)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim = 1, activation ='relu'))
model.add(Dense(5, input_shape = (1, ), activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=300, batch_size=1,
validation_data=(x_val, y_val))

#4. 평가 예측
mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)


##### 함수형 모델
### 1) 1:1
#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
x, y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
x_test, y_test, random_state=66, test_size=0.5, shuffle=False
)

#2. 모델 구성
from keras.models import Sequential, Model # Model을 추가해준다.
from keras.layers import Dense, Input # Input 레이어를 추가해준다.

# model = Sequential()
input1 = Input(shape=(1,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs= output1)
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x, y, epochs=100, batch_size=3)
model.fit(x_train, y_train, epochs=100, batch_size=1,
validation_data=(x_val, y_val))

#4. 평가 예측
mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

### 2) 다:다
#1. 데이터
import numpy as np
x = np.array([range(100), range(301,401)])
y = np.array([range(100), range(301,401)])

print(x.shape)
print(y.shape)

x = np.transpose(x)
y = np.transpose(y)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
x, y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
x_test, y_test, random_state=66, test_size=0.5, shuffle=False
)
#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
# model.add(Dense(5, input_dim = 2, activation ='relu'))
model.add(Dense(5, input_shape = (2, ), activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=300, batch_size=1,

validation_data=(x_val, y_val))
#4. 평가 예측
mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_test)
# print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

### 3) 다:1
#1. 데이터
import numpy as np

x = np.array([range(100), range(301,401)])
y = np.array(range(201,301))
x = np.transpose(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
x, y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
x_test, y_test, random_state=66, test_size=0.5, shuffle=False
)
#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(5, input_shape = (2, ), activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))
# model.summary()
#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

#4. 평가 예측
mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_test)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

### 4) 1:다
#1. 데이터
import numpy as np
x = np.array([range(100)])
y = np.array([range(201,301), range(301,401)])
x = np.transpose(x)
y = np.transpose(y)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, random_state=66, test_size=0.5, shuffle=False
)
print(x_test.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
# model.add(Dense(5, input_dim = 3, activation ='relu'))
model.add(Dense(5, input_shape = (1, ), activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x, y, epochs=100, batch_size=3)
model.fit(x_train, y_train, epochs=100, batch_size=1,
validation_data=(x_val, y_val))

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)



########### Chap5 앙상블 ###########
##### concatenate
#1. 데이터
import numpy as np
x1 = np.array([range(100), range(311,411), range(100)])
x2 = np.array([range(101,201), range(311,411), range(101,201)])
y = np.array([range(501,601)]) #, range(711,811), range(100)]

x1 = np.transpose(x1)
y = np.transpose(y)
x2 = np.transpose(x2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y_train, y_test = train_test_split(
x1, y, random_state=66, test_size=0.4, shuffle=False
)
x1_val, x1_test, y_val, y_test = train_test_split(
x1_test, y_test, random_state=66, test_size=0.5, shuffle=False
)
x2_train, x2_test = train_test_split(
x2, random_state=66, test_size=0.4, shuffle=False
)
x2_val, x2_test= train_test_split(
x2_test, random_state=66, test_size=0.5, shuffle = False
)

#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(100, activation='relu')(input1)
dense1_2 = Dense(30)(dense1)
dense1_3 = Dense(7)(dense1_2)

input2 = Input(shape=(3,))
dense2 = Dense(50, activation='relu')(input2)
dense2_2 = Dense(7)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_2])

from keras.layers.merge import Concatenate
merge1 = Concatenate()([dense1_3, dense2_2])

model1 = Dense(10)(merge1)
model2 = Dense(5)(model1)
output = Dense(1)(model2)

model = Model(inputs = [input1, input2], outputs = output)
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], y_train,
epochs=100, batch_size=1,
validation_data=([x1_val, x2_val] , y_val))

#4. 평가 예측
mse = model.evaluate([x1_test, x2_test],
y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict([x1_test, x2_test])
for i in range(len(y_predict)):
    print(y_test[i], y_predict[i])


##### Merge Layer
### 1) Add
import keras
input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# equivalent to added = keras.layers.add([x1, x2])
added = keras.layers.Add()([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)

### 2) Subtract
import keras
input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# Equivalent to subtracted = keras.layers.subtract([x1, x2])
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)

### 3) Multiply
# keras.layers.Multiply()

### 4) Average
# keras.layers.Average()

### 5) Maximum
# keras.layers.Maximum()

### 6) Minimum
# keras.layers.Minimum()

### 7) Concatenate
# keras.layers.Concatenate(axis=-1)

### 8) Dot
# keras.layers.Dot(axex, normalize=False)

### 9) add
# keras.layers.Add(inputs)
import keras
input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)

### 10) subtract
# keras.layers.subtract(inputs)
import keras
input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
subtracted = keras.layers.subtract([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)

### 11) multiply
# keras.layers.multiply(inputs)

### 12) average
# keras.layers.average(inputs)

### 13) maximum
# keras.layers.maximum(inputs)

### 14) minimum
# keras.layers.minimum(inputs)

### 15) concatenate
# keras.layers.concatenate(inputs, axis=-1)

### 16) dot
# keras.layers.dot(inputs, axes, normalize=False)



########### Chap6 회귀 모델 총정리 ###########
##### Sequential 모델
### 1) Sequential 1:1모델
#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,])
y_train = np.array([1,2,3,4,5,6,7,])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])
x_predict = np.array([11,12,13])

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(100, input_dim = 1, activation ='relu'))
model.add(Dense(30))
model.add(Dense(5))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)

### 2) Sequential 다:다 모델
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5,6,7,], [11,12,13,14,15,16,17]])
y_train = np.array([[1,2,3,4,5,6,7,], [11,12,13,14,15,16,17]])
x_test = np.array([[8,9,10], [18,19,20]])
y_test = np.array([[8,9,10], [18,19,20]])
x_predict = np.array([[21,22,23], [31,32,33]])

print(x_train.shape)
print(x_test.shape)
print(x_predict.shape)

x_train = np.transpose(x_train)
y_train = np.transpose(y_train)
x_test = np.transpose(x_test)
y_test = np.transpose(y_test)
x_predict = np.transpose(x_predict)
print(x_train.shape)
print(x_test.shape)
print(x_predict.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(100, input_dim = 2, activation ='relu'))
model.add(Dense(30))
model.add(Dense(5))
model.add(Dense(2))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)
#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)

### Sequential 다:1 모델
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5,6,7,], [11,12,13,14,15,16,17]])
y_train = np.array([1,2,3,4,5,6,7,])
x_test = np.array([[8,9,10], [18,19,20]])
y_test = np.array([8,9,10])
x_predict = np.array([[21,22,23], [31,32,33]])

print('x_train.shape : ', x_train.shape)
print('y_train.shape : ', y_train.shape)
print('x_test.shape : ' , x_test.shape)
print('y_test.shape : ' , y_test.shape)
print('x_predict.shape : ', x_predict.shape )

x_train = np.transpose(x_train)
x_test = np.transpose(x_test)
x_predict = np.transpose(x_predict)
print('x_train.shape : ', x_train.shape)
print('y_train.shape : ', y_train.shape)
print('x_test.shape : ' , x_test.shape)
print('y_test.shape : ' , y_test.shape)
print('x_predict.shape : ', x_predict.shape )

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(100, input_dim = 2, activation ='relu'))
model.add(Dense(30))
model.add(Dense(5))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)
#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)

### 4) Sequential 1:다 모델
#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,])
y_train = np.array([[1,2,3,4,5,6,7,], [11,12,13,14,15,16,17]])
x_test = np.array([8,9,10])
y_test = np.array([[8,9,10], [18,19,20]])
x_predict = np.array([11,12,13])
print('x_train.shape : ', x_train.shape)
print('y_train.shape : ', y_train.shape)
print('x_test.shape : ' , x_test.shape)
print('y_test.shape : ' , y_test.shape)
print('x_predict.shape : ', x_predict.shape )

y_train = np.transpose(y_train)
y_test = np.transpose(y_test)
print('x_train.shape : ', x_train.shape)
print('y_train.shape : ', y_train.shape)
print('x_test.shape : ' , x_test.shape)
print('y_test.shape : ' , y_test.shape)
print('x_predict.shape : ', x_predict.shape )

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(100, input_dim = 1, activation ='relu'))
model.add(Dense(30))
model.add(Dense(5))
model.add(Dense(2))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)
#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)


##### 함수형 모델
### 1) 함수형 1:1모델
#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,])
y_train = np.array([1,2,3,4,5,6,7,])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])
x_predict = np.array([11,12,13])

#2. 모델 구성
from keras.models import Model
from keras.layers import Dense, Input
input1 = Input(shape=(1,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(5)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs= output1)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)

### 2) 함수형 다:다 모델
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5,6,7,], [11,12,13,14,15,16,17]])
y_train = np.array([[1,2,3,4,5,6,7,], [11,12,13,14,15,16,17]])
x_test = np.array([[8,9,10], [18,19,20]])
y_test = np.array([[8,9,10], [18,19,20]])
x_predict = np.array([[21,22,23], [31,32,33]])
print(x_train.shape)
print(x_test.shape)
print(x_predict.shape)

x_train = np.transpose(x_train)
y_train = np.transpose(y_train)
x_test = np.transpose(x_test)
y_test = np.transpose(y_test)
x_predict = np.transpose(x_predict)
print(x_train.shape)
print(x_test.shape)
print(x_predict.shape)

#2. 모델 구성
from keras.models import Model
from keras.layers import Dense, Input
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(5)(dense2)
output1 = Dense(2)(dense3)
model = Model(inputs = input1, outputs= output1)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)

### 3)함수형 다:1 모델
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5,6,7,], [11,12,13,14,15,16,17]])
y_train = np.array([1,2,3,4,5,6,7,])
x_test = np.array([[8,9,10], [18,19,20]])

y_test = np.array([8,9,10])
x_predict = np.array([[21,22,23], [31,32,33]])

print('x_train.shape : ', x_train.shape)
print('y_train.shape : ', y_train.shape)
print('x_test.shape : ' , x_test.shape)
print('y_test.shape : ' , y_test.shape)
print('x_predict.shape : ', x_predict.shape )

x_train = np.transpose(x_train)
x_test = np.transpose(x_test)
x_predict = np.transpose(x_predict)
print('x_train.shape : ', x_train.shape)
print('y_train.shape : ', y_train.shape)

print('x_test.shape : ' , x_test.shape)
print('y_test.shape : ' , y_test.shape)
print('x_predict.shape : ', x_predict.shape )

#2. 모델 구성
from keras.models import Model
from keras.layers import Dense, Input
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(5)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs= output1)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)

### 4) 함수형 1:다 모델
#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,])
y_train = np.array([[1,2,3,4,5,6,7,], [11,12,13,14,15,16,17]])
x_test = np.array([8,9,10])
y_test = np.array([[8,9,10], [18,19,20]])
x_predict = np.array([11,12,13])
print('x_train.shape : ', x_train.shape)
print('y_train.shape : ', y_train.shape)
print('x_test.shape : ' , x_test.shape)
print('y_test.shape : ' , y_test.shape)
print('x_predict.shape : ', x_predict.shape )

y_train = np.transpose(y_train)
y_test = np.transpose(y_test)
print('x_train.shape : ', x_train.shape)

print('y_train.shape : ', y_train.shape)
print('x_test.shape : ' , x_test.shape)
print('y_test.shape : ' , y_test.shape)
print('x_predict.shape : ', x_predict.shape )

#2. 모델 구성
from keras.models import Model
from keras.layers import Dense, Input
input1 = Input(shape=(1,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(5)(dense2)
output1 = Dense(2)(dense3)
model = Model(inputs = input1, outputs= output1)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)


##### 앙상블 및 기타 모델
### 1) 앙상블 다:다 모델
#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101, 201)])
y1 = np.array([range(1, 101), range(101, 201)])
x2 = np.array([range(501, 601), range(601, 701)])
y2 = np.array([range(501, 601), range(601, 701)])
print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
x1, y1, random_state=66, test_size=0.2, shuffle = False
)
x1_val, x1_test, y1_val, y1_test = train_test_split(
x1_test, y1_test, random_state=66, test_size=0.5, shuffle = False
)
x2_train, x2_test, y2_train, y2_test = train_test_split(
x2, y2, random_state=66, test_size=0.2, shuffle = False
)
x2_val, x2_test, y2_val, y2_test = train_test_split(
x2_test, y2_test, random_state=66, test_size=0.5, shuffle = False
)

print('x2_train.shape : ', x2_train.shape)
print('x2_val.shape : ', x2_val.shape)
print('x2_test.shape : ', x2_test.shape)

#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu')(input1)
dense1 = Dense(30)(dense1)
dense1 = Dense(7)(dense1)

input2 = Input(shape=(2,))
dense2 = Dense(50, activation='relu')(input2)
dense2 = Dense(30)(dense2)
dense2 = Dense(7)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(30)(middle2)

output1 = Dense(30)(middle3)
output1 = Dense(7)(output1)
output1 = Dense(2)(output1)

output2 = Dense(20)(middle3)
output2 = Dense(70)(output2)
output2 = Dense(2)(output2)

model = Model(inputs = [input1, input2],
outputs = [output1, output2]
)
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train],
epochs=50, batch_size=1,
validation_data=([x1_val, x2_val] , [y1_val, y2_val]))

#4. 평가 예측
mse = model.evaluate([x1_test, x2_test],
                     [y1_test, y2_test], batch_size=1)
print("mse : ", mse)
y1_predict, y2_predict = model.predict([x1_test, x2_test])
print("y1 예측값 : \n", y1_predict, "\n y2 예측값 : \n", y2_predict)

### 2) 앙상블 다:다 모델2
#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101, 201)])
y1 = np.array([range(1, 101), range(101, 201)])
x2 = np.array([range(501, 601), range(601, 701)])
y2 = np.array([range(501, 601), range(601, 701)])
y3 = np.array([range(701, 801), range(801, 901)])

print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)
print(y3.shape)

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)
y3 = np.transpose(y3)
print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)
print(y3.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
x1, y1, random_state=66, test_size=0.2, shuffle = False
)
x1_val, x1_test, y1_val, y1_test = train_test_split(
x1_test, y1_test, random_state=66, test_size=0.5, shuffle = False
)
x2_train, x2_test, y2_train, y2_test = train_test_split(
x2, y2, random_state=66, test_size=0.2, shuffle = False
)
x2_val, x2_test, y2_val, y2_test = train_test_split(
x2_test, y2_test, random_state=66, test_size=0.5, shuffle = False
)
# y3 데이터의 분리
y3_train, y3_test = train_test_split(
y3 , random_state=66, test_size=0.2, shuffle=False
)
y3_val, y3_test = train_test_split(
y3_test , random_state=66, test_size=0.5, shuffle=False
)

y3_train.shape : (80, 2)
y3_val.shape : (10, 2)
y3_test.shape : (10, 2)

#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu')(input1)
dense1 = Dense(30)(dense1)
dense1 = Dense(7)(dense1)

input2 = Input(shape=(2,))
dense2 = Dense(50, activation='relu')(input2)
dense2 = Dense(30)(dense2)
dense2 = Dense(7)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(30)(middle2)

output1 = Dense(30)(middle3)
output1 = Dense(7)(output1)
output1 = Dense(2)(output1)

output2 = Dense(20)(middle3)
output2 = Dense(70)(output2)
output2 = Dense(2)(output2)

output3 = Dense(25)(middle3)
output3 = Dense(5)(output3)
output3 = Dense(2)(output3)

model = Model(inputs = [input1, input2],
outputs = [output1, output2, output3]
)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train],
          epochs=50, batch_size=1,
          validation_data=([x1_val, x2_val], [y1_val, y2_val, y3_val]))

#4. 평가 예측
mse = model.evaluate([x1_test, x2_test],
[y1_test, y2_test, y3_test], batch_size=1)

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])
print("y1 예측값 : \n", y1_predict,
      "\n y2 예측값 : \n", y2_predict, "\n y3 예측값 : \n", y3_predict)

### 3) 앙상블 다:1 모델
#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101, 201)])
x2 = np.array([range(501, 601), range(601, 701)])
y = np.array([range(1, 101), range(101, 201)])
print(x1.shape)
print(x2.shape)
print(y.shape)

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.transpose(y)
print(x1.shape)
print(x2.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y_train, y_test = train_test_split(
x1, y, random_state=66, test_size=0.2, shuffle = False
)
x1_val, x1_test, y_val, y_test = train_test_split(
x1_test, y_test, random_state=66, test_size=0.5, shuffle = False
)
x2_train, x2_test = train_test_split(
x2, random_state=66, test_size=0.2, shuffle = False
)
x2_val, x2_test = train_test_split(
x2_test, random_state=66, test_size=0.5, shuffle = False
)

print('x2_train.shape : ', x2_train.shape)
print('x2_val.shape : ', x2_val.shape)
print('x2_test.shape : ', x2_test.shape)

#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu')(input1)
dense1 = Dense(30)(dense1)
dense1 = Dense(7)(dense1)

input2 = Input(shape=(2,))
dense2 = Dense(50, activation='relu')(input2)
dense2 = Dense(30)(dense2)
dense2 = Dense(7)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1, dense2])

output1 = Dense(30)(merge1)
output1 = Dense(7)(output1)
output1 = Dense(2)(output1)

model = Model(inputs = [input1, input2],
outputs = output1
)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], y_train,
epochs=50, batch_size=1,
validation_data=([x1_val, x2_val], y_val))

#4. 평가 예측
mse = model.evaluate([x1_test, x2_test], y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict([x1_test, x2_test])
print("y1 예측값 : \n", y_predict)

### 4) 1:다 모델
#1. 데이터
import numpy as np
x = np.array([range(1, 101), range(101, 201)])
y1 = np.array([range(501, 601), range(601, 701)])
y2 = np.array([range(1, 101), range(101, 201)])
print(x.shape)
print(y1.shape)
print(y2.shape)
x = np.transpose(x)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
print(x.shape)
print(y1.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test = train_test_split(
x, y1, random_state=66, test_size=0.2, shuffle = False
)
x_val, x_test, y1_val, y1_test = train_test_split(
x_test, y1_test, random_state=66, test_size=0.5, shuffle = False
)
y2_train, y2_test = train_test_split(
y2, random_state=66, test_size=0.2, shuffle = False
)
y2_val, y2_test = train_test_split(
y2_test, random_state=66, test_size=0.5, shuffle = False
)
print('y2_train.shape : ', y2_train.shape)
print('y2_val.shape : ', y2_val.shape)
print('y2_test.shape : ', y2_test.shape)

#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu')(input1)
dense1 = Dense(30)(dense1)
dense1 = Dense(7)(dense1)

output1 = Dense(30)(dense1)
output1 = Dense(7)(output1)
output1 = Dense(2)(output1)

output2 = Dense(30)(dense1)
output2 = Dense(7)(output1)
output2 = Dense(2)(output1)

model = Model(inputs = input1,
outputs = [output1, output2]
)
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, [y1_train, y2_train],
epochs=50, batch_size=1,
validation_data=(x_val, [y1_val, y2_val]))
#4. 평가 예측
mse = model.evaluate(x_test, [y1_test, y2_test], batch_size=1)
print("mse : ", mse)
y1_predict, y2_predict = model.predict(x_test)
print("y1 예측값 : \n", y1_predict, "\n y2 예측값 : \n", y2_predict)



########### Chap7 RNN:시계열 ###########
##### SimpleRNN
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
y_train = np.array([6,7,8])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print("x_train.shape : ", x_train.shape) #(3, 5, 1)
print("y_train.shape : ", y_train.shape) #(3, )

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
model = Sequential()
model.add(SimpleRNN(7, input_shape = (5, 1), activation ='relu')) # input_shape = (None, 5, 1) 즉, 5열 1피처
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 예측
x_predict = np.array([[4,5,6,7,8]])
print(x_predict.shape) #(1, 5)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print("x_predict.shape : ", x_predict.shape) # (1, 5, 1)

y_predict = model.predict(x_predict)
print("예측값 : ", y_predict)


##### LSTM
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
y_train = np.array([6,7,8])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print("x_train.shape : ", x_train.shape) #(3, 5, 1)
print("y_train.shape : ", y_train.shape) #(3, )

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(7, input_shape = (5, 1), activation ='relu'))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
x_predict = np.array([[4,5,6,7,8]])
print(x_predict.shape) #(1, 5)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print("x_predict.shape : ", x_predict.shape) # (1, 5, 1)

y_predict = model.predict(x_predict)
print("예측값 : ", y_predict)


##### GRU
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
y_train = np.array([6,7,8])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print("x_train.shape : ", x_train.shape) #(3, 5, 1)
print("y_train.shape : ", y_train.shape) #(3, )

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, GRU
model = Sequential()
model.add(GRU(7, input_shape = (5, 1), activation ='relu'))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
x_predict = np.array([[4,5,6,7,8]])
print(x_predict.shape) #(1, 5)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print("x_predict.shape : ", x_predict.shape) # (1, 5, 1)

y_predict = model.predict(x_predict)
print("예측값 : ", y_predict)


##### Bidirectional
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
y_train = np.array([6,7,8])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print("x_train.shape : ", x_train.shape) #(3, 5, 1)
print("y_train.shape : ", y_train.shape) #(3, )

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
model = Sequential()
model.add(Bidirectional(LSTM(7, activation ='relu'), input_shape=(5, 1)))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
x_predict = np.array([[4,5,6,7,8]])
print(x_predict.shape) #(1, 5)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print("x_predict.shape : ", x_predict.shape) # (1, 5, 1)

y_predict = model.predict(x_predict)
print("예측값 : ", y_predict)


##### LSTM 레이어 연결
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
y_train = np.array([6,7,8])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print("x_train.shape : ", x_train.shape) #(3, 5, 1)
print("y_train.shape : ", y_train.shape) #(3, )

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(7, input_shape = (5, 1), activation ='relu',
          return_sequences=True )) # return_sequences=True 이전 차원을 유지
model.add(LSTM(8)) # 추가
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
x_predict = np.array([[4,5,6,7,8]])
print(x_predict.shape) #(1, 5)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print("x_predict.shape : ", x_predict.shape) # (1, 5, 1)
y_predict = model.predict(x_predict)
print("예측값 : ", y_predict)



########### Chap8 케라스 모델의 파라미터들과 기타 기법들 ###########
##### verbose
'''
verbose=0 : 훈련되는 모습 표시x
verbose=1 기본값 : 진행상황, 훈련되고 있는 두가지에 대해 출력값을 보여줌
verbose=2 : Epoch와 loss, mse만 화면에 간단히 출력
'''

##### EarlyStopping
'''
monitor : loss값을 사용
patience : 성능이 증가하지 않는 epoch가 10번 이상 반복되면 중지
mode : monitor의 loss값이 최솟값일때 적용
'''
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
y_train = np.array([6,7,8])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print("x_train.shape : ", x_train.shape) #(3, 5, 1)
print("y_train.shape : ", y_train.shape) #(3, )

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(7, input_shape = (5, 1), activation ='relu'))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=2, callbacks=[early_stopping])

#4. 평가, 예측
x_predict = np.array([[4,5,6,7,8]])
print(x_predict.shape) #(1, 5)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print("x_predict.shape : ", x_predict.shape) # (1, 5, 1)

y_predict = model.predict(x_predict)
print("예측값 : ", y_predict)

##### TensorBoard
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
y_train = np.array([6,7,8])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print("x_train.shape : ", x_train.shape) #(3, 5, 1)
print("y_train.shape : ", y_train.shape) #(3, )

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(7, input_shape = (5, 1), activation ='relu'))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(
log_dir='./graph', histogram_freq=0,
write_graph=True, write_images=True)

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
verbose=2, callbacks=[early_stopping, tb_hist])

#4. 평가, 예측
x_predict = np.array([[4,5,6,7,8]])
print(x_predict.shape) #(1, 5)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print("x_predict.shape : ", x_predict.shape) # (1, 5, 1)

y_predict = model.predict(x_predict)
print("예측값 : ", y_predict)

##### 모델의 Save
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(7, input_shape = (5, 1), activation ='relu'))
model.add(Dense(4))
model.add(Dense(1))

model.save('savetest01.h5')
print("저장 잘 됐음")

##### 모델의 Load
from keras.models import load_model
model = load_model("savetest01.h5")

model.summary()

### 불러오기 후 레이어 추가
from keras.models import load_model
model = load_model("savetest01.h5")

from keras.layers import Dense # 추가
model.add(Dense(1)) # 추가

model.summary()

'''
에러 발생.
'''

from keras.models import load_model
model = load_model("savetest01.h5")

from keras.layers import Dense
model.add(Dense(1, name='dense_x')) # 수정

model.summary()

### 훈련과 평과, 예측을 붙여 실행
#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
y_train = np.array([6,7,8])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print("x_train.shape : ", x_train.shape) #(3, 5, 1)
print("y_train.shape : ", y_train.shape) #(3, )

#2. 모델 구성
from keras.models import load_model
model = load_model("savetest01.h5")

from keras.layers import Dense
model.add(Dense(1, name='dense_x'))
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(
log_dir='./graph', histogram_freq=0,
write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')

model.fit(x_train, y_train, epochs=1000, batch_size=1,
verbose=2, callbacks=[early_stopping, tb_hist])

#4. 평가, 예측
x_predict = np.array([[4,5,6,7,8]])
print(x_predict.shape) #(1, 5)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print("x_predict.shape : ", x_predict.shape) # (1, 5, 1)

y_predict = model.predict(x_predict)
print("예측값 : ", y_predict)



########### Chap9 RNN용 데이터 자르기 ###########
##### split 함수 만들기(다:1)
import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy1(dataset, time_steps):
    x, y = list(), list()
    for i in range(len(dataset)):
       end_number = i + time_steps
       if end_number > len(dataset) -1:
          break
       tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
       x.append(tmp_x)
       y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy1(dataset, 4)
print(x, "\n", y)


##### split 함수 만들기2(다:다)
import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy2(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 추가
        # if end_number > len(dataset) -1:
        # break
        if y_end_number > len(dataset): # 수정
           break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number] # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

time_steps = 4
y_column = 2
x, y = split_xy2(dataset, time_steps, y_column)
print(x, "\n", y)
print("x.shape : ", x.shape)
print("y.shape : ", y.shape)


##### split 함수 만들기3(다입력, 다:1)
import numpy as np

dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                    [11,12,13,14,15,16,17,18,19,20],
                    [21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)
dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1 # 수정
        if y_end_number > len(dataset): # 수정
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1] # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy3(dataset, 3, 1)
print(x, "\n", y)
print(x.shape)
print(y.shape)


##### split 함수 만들기4(다입력, 다:다)
import numpy as np

dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                    [11,12,13,14,15,16,17,18,19,20],
                    [21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)
dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1 # 수정
        if y_end_number > len(dataset): # 수정
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1] # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy3(dataset, 3, 2) # 2를 1로 수정
print(x, "\n", y)
print(x.shape)
print(y.shape)


##### split 함수 만들기5(다입력, 다:다 두번째)
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                   [11,12,13,14,15,16,17,18,19,20],
                   [21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)

dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정

        if y_end_number > len(dataset): # 수정
            break
        tmp_x = dataset[i:x_end_number, :] # 수정
        tmp_y = dataset[x_end_number:y_end_number, :] # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(dataset, 3, 1)
print(x, "\n", y)
print(x.shape)
print(y.shape)


##### split 함수 만들기6(다입력, 다:다 세번째)
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                   [11,12,13,14,15,16,17,18,19,20],
                   [21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)

dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy6(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정

        if y_end_number > len(dataset): # 수정
            break
        tmp_x = dataset[i:x_end_number, :] # 수정
        tmp_y = dataset[x_end_number:y_end_number, :] # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy6(dataset, 3, 2)
print(x, "\n", y)
print(x.shape)
print(y.shape)



########### Chap10 RNN 모델 정리 ###########
##### MLP DNN 모델(다:1)
#1. 데이터
import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy1(dataset, time_steps):
    x, y = list(), list()
    for i in range(len(dataset)):
        end_number = i + time_steps
        if end_number > len(dataset) -1:
            break
        tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy1(dataset, 4)
print(x, "\n", y)
print(x.shape)
print(y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_shape=(4, )))
# model.add(Dense(64, input_dim=4))
model.add(Dense(1))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
mse = model.evaluate(x, y )
print("mse : ", mse)
x_pred = np.array([7, 8, 9, 10])
x_pred = x_pred.reshape(1, x_pred.shape[0])
print(x_pred.shape)

y_pred = model.predict(x_pred)
print(y_pred)


##### MLP RNN 모델(다:1)
#1. 데이터
import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])
def split_xy1(dataset, time_steps):
    x, y = list(), list()
    for i in range(len(dataset)):
        end_number = i + time_steps
        if end_number > len(dataset) -1:
           break
        tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy1(dataset, 4)
print(x, "\n", y)
print(x.shape)
print(y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM
#2. 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(4, 1)))
model.add(Dense(1))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
mse = model.evaluate(x, y )
print("mse : ", mse)
x_pred = np.array([7, 8, 9, 10])
# x_pred = x_pred.reshape(1, x_pred.shape[0])
x_pred = x_pred.reshape(1, x_pred.shape[0], 1)

print(x_pred.shape)

y_pred = model.predict(x_pred)
print(y_pred)


##### MLP RNN 모델(다:다)
#1. 데이터
import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy2(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy2(dataset, 4, 2)
print(x, "\n", y)
print(x.shape)
print(y.shape)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM
#2. 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(4, 1)))
model.add(Dense(2))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
mse = model.evaluate(x, y )
print("mse : ", mse)
x_pred = np.array([6, 7, 8, 9])
x_pred = x_pred.reshape(1, x_pred.shape[0], 1)
# print(x_pred.shape)

y_pred = model.predict(x_pred)
print(y_pred)


##### MLP RNN 모델(다입력 다:1)
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                    [11,12,13,14,15,16,17,18,19,20],
                    [21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)

dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy3(dataset, 3, 1)
print(x, "\n", y)
print(x.shape)
print(y.shape)

y = y.reshape(y.shape[0])
print(y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(3, 2)))
model.add(Dense(1))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, batch_size=1)
#4. 평가, 예측
mse = model.evaluate(x, y, batch_size=1 )
print("mse : ", mse)

x_pred = np.array([[9, 10, 11], [19, 20, 21]])
x_pred = np.transpose(x_pred)
x_pred = x_pred.reshape(1, x_pred.shape[0], x_pred.shape[1])
print(x_pred.shape)

y_pred = model.predict(x_pred, batch_size=1)
print(y_pred)


##### MLP DNN 모델(다입력 다:1)
#1. 데이터
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
[11,12,13,14,15,16,17,18,19,20],
[21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)
dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1 # 수정
        if y_end_number > len(dataset): # 수정
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1] # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy3(dataset, 3, 1)
print(x, "\n", y)
print(x.shape)
print(y.shape)
y = y.reshape(y.shape[0])
print(y.shape)

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
print(x.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
# model.add(LSTM(64, input_shape=(3, 2)))
model.add(Dense(64, input_shape=(6, )))
model.add(Dense(1))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, batch_size=1)

#4. 평가, 예측
mse = model.evaluate(x, y, batch_size=1 )
print("mse : ", mse)

x_pred = np.array([[9, 10, 11], [19, 20, 21]])
print(x_pred.shape)

x_pred = x_pred.reshape(1, x_pred.shape[0] * x_pred.shape[1])
print(x_pred.shape)

y_pred = model.predict(x_pred, batch_size=1)
print(y_pred)


##### MLP RNN 모델(다입력 다:다)
#1. 데이터
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
[11,12,13,14,15,16,17,18,19,20],
[21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)

dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy3(dataset, 3, 2)
# print(x, "\n", y)
print(x.shape)
print(y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(3, 2)))
model.add(Dense(2))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, batch_size=1)

#4. 평가, 예측
mse = model.evaluate(x, y, batch_size=1 )
print("mse : ", mse)
x_pred = np.array([[9, 10, 11], [19, 20, 21]])
x_pred = np.transpose(x_pred)
x_pred = x_pred.reshape(1, x_pred.shape[0], x_pred.shape[1])
# print(x_pred.shape)

y_pred = model.predict(x_pred, batch_size=1)
print(y_pred)


##### MLP DNN 모델(다입력 다:다)
#1. 데이터
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                   [11,12,13,14,15,16,17,18,19,20],
                   [21,22,23,24,25,26,27,28,29,30]])

print("dataset.shape : ", dataset.shape)
dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy3(dataset, 3, 2)
print(x.shape)
print(y.shape)

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
print(x.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
# model.add(LSTM(64, input_shape=(3, 2)))
model.add(Dense(64, input_shape=(6, )))
model.add(Dense(2))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, batch_size=1)

#4. 평가, 예측
mse = model.evaluate(x, y, batch_size=1 )
print("mse : ", mse)
x_pred = np.array([[9, 10, 11], [19, 20, 21]])
x_pred = np.transpose(x_pred)
x_pred = x_pred.reshape(1, x_pred.shape[0] * x_pred.shape[1])
print(x_pred.shape)

y_pred = model.predict(x_pred, batch_size=1)
print(y_pred)


##### RNN 모델(다입력 다:다 두번째)
#1. 데이터
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                    [11,12,13,14,15,16,17,18,19,20],    
                    [21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)
dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(dataset, 3, 1)
print(x, "\n", y)
print(x.shape)
print(y.shape)
y = y.reshape(y.shape[0], y.shape[2])
print(y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(3, 3)))
model.add(Dense(3))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, batch_size=1)

#4. 평가, 예측
mse = model.evaluate(x, y, batch_size=1 )
print("mse : ", mse)
x_pred = np.array([[8, 9, 10],[18, 19, 20], [28, 29, 30]])
x_pred = np.transpose(x_pred)
x_pred = x_pred.reshape(1, x_pred.shape[0], x_pred.shape[1])
print(x_pred)
print(x_pred.shape)
y_pred = model.predict(x_pred, batch_size=1)
print(y_pred)


##### DNN 모델(다입력 다:다 두번째)
#1. 데이터
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                    [11,12,13,14,15,16,17,18,19,20],    
                    [21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)
dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(dataset, 3, 1)
print(x, "\n", y)
print(x.shape)
print(y.shape)
y = y.reshape(y.shape[0], y.shape[2])
print(y.shape)

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
print(x.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
# model.add(LSTM(64, input_shape=(3, 3)))
model.add(Dense(64, input_shape=(9,)))
model.add(Dense(3))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, batch_size=1)

#4. 평가, 예측
mse = model.evaluate(x, y, batch_size=1 )
print("mse : ", mse)
x_pred = np.array([[8, 9, 10],[18, 19, 20], [28, 29, 30]])
x_pred = np.transpose(x_pred)
# x_pred = x_pred.reshape(1, x_pred.shape[0], x_pred.shape[1])
x_pred = x_pred.reshape(1, x_pred.shape[0] * x_pred.shape[1]) # 수정
print(x_pred)
print(x_pred.shape)
y_pred = model.predict(x_pred, batch_size=1)
print(y_pred)


##### RNN 모델(다입력 다:다 세번째)
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                    [11,12,13,14,15,16,17,18,19,20],
                    [21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)
dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :] # 수정
        tmp_y = dataset[x_end_number:y_end_number, :] # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(dataset, 3, 2)
print(x, "\n", y)
print(x.shape)
print(y.shape)

y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])
print(y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(3, 3)))
model.add(Dense(6))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, batch_size=1)

#4. 평가, 예측
mse = model.evaluate(x, y, batch_size=1 )
print("mse : ", mse)

x_pred = np.array([[8, 9, 10],[18, 19, 20], [28, 29, 30]])
x_pred = np.transpose(x_pred)
x_pred = x_pred.reshape(1, x_pred.shape[0], x_pred.shape[1])
# print(x_pred)
# print(x_pred.shape)
y_pred = model.predict(x_pred, batch_size=1)
print(y_pred)


##### DNN 모델(다입력 다:다 세번째)
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                   [11,12,13,14,15,16,17,18,19,20],
                   [21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)
dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :] # 수정
        tmp_y = dataset[x_end_number:y_end_number, :] # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(dataset, 3, 2)

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
print(x.shape)
y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])
print(y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
# model.add(LSTM(64, input_shape=(3, 3)))
model.add(Dense(64, input_shape=(9, )))
model.add(Dense(6))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, batch_size=1)

#4. 평가, 예측
mse = model.evaluate(x, y, batch_size=1 )
print("mse : ", mse)

x_pred = np.array([[8, 9, 10],[18, 19, 20], [28, 29, 30]])
x_pred = np.transpose(x_pred)
x_pred = x_pred.reshape(1, x_pred.shape[0] * x_pred.shape[1])
print(x_pred)
print(x_pred.shape)
y_pred = model.predict(x_pred, batch_size=1)
print(y_pred)



########### Chap11 KOSPI 200 데이터를 이용한 삼성전자 주가 예측 ###########
##### 데이터 저장
import numpy as np
import pandas as pd

df1 = pd.read_csv("kospi200.csv", index_col=0, 
                  header=0, encoding='cp949', sep=',')
                
print(df1)
print(df1.shape)

df2 = pd.read_csv("samsung.csv", index_col=0, 
                  header=0, encoding='cp949', sep=',')   
print(df2)
print(df2.shape)

# kospi200의 거래량
for i in range(len(df1.index)):     # 거래량 str -> int 변경
        df1.iloc[i,4] = int(df1.iloc[i,4].replace(',', ''))  
# 삼성전자의 모든 데이터 
for i in range(len(df2.index)):     # 모든 str -> int 변경
        for j in range(len(df2.iloc[i])):
                df2.iloc[i,j] = int(df2.iloc[i,j].replace(',', ''))  

df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])
print(df1)
print(df2)


##### pandas를 numpy로 변경 후 저장
df1 = df1.values
df2 = df2.values
print(type(df1), type(df2))
print(df1.shape, df2.shape)

np.save('kospi200.npy', arr=df1)
np.save('samsung.npy', arr=df2)


##### numpy 데이터 불러오기
import numpy as np
import pandas as pd

kospi200 = np.load('kospi200.npy', allow_pickle=True)
samsung = np.load('samsung.npy', allow_pickle=True)

print(kospi200)
print(samsung)
print(kospi200.shape)
print(samsung.shape)


##### DNN 구성하기
def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정

        if y_end_number > len(dataset):  # 수정
            break
        tmp_x = dataset[i:x_end_number, :]  # 수정
        tmp_y = dataset[x_end_number:y_end_number, 3]    # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy5(samsung, 5, 1) 
print(x[0,:], "\n", y[0])
print(x.shape)
print(y.shape)

### 1) 데이터 전처리
# 데이터 셋 나누기
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size = 0.3)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = np.reshape(x_train,
    (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test,
    (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
print(x_train.shape)
print(x_test.shape)

# 데이터 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0, :])

from keras.models import Sequential
from keras.layers import Dense

# 모델구성
model = Sequential()
model.add(Dense(64, input_shape=(25, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

### 2) 컴파일 및 훈련, 완성
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit(x_train_scaled, y_train, validation_split=0.2, verbose=1, # validation_split : train셋의 20%를 validation셋에 할당
          batch_size=1, epochs=100, callbacks=[early_stopping])

loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

y_pred = model.predict(x_test_scaled)

for i in range(5):
    print('종가 : ', y_test[i], '/ 예측가 : ', y_pred[i])


##### LSTM 구성하기
import numpy as np
import pandas as pd

kospi200 = np.load('kospi200.npy', allow_pickle=True)
samsung = np.load('samsung.npy', allow_pickle=True)

print(kospi200)
print(samsung)
print(kospi200.shape)
print(samsung.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정

        if y_end_number > len(dataset):  # 수정
            break
        tmp_x = dataset[i:x_end_number, :]  # 수정
        tmp_y = dataset[x_end_number:y_end_number, 3]    # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy5(samsung, 5, 1) 
print(x[0,:], "\n", y[0])
print(x.shape)
print(y.shape)

# 데이터 셋 나누기
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size = 0.3)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = np.reshape(x_train,
    (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test,
    (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
print(x_train.shape)
print(x_test.shape)

# 데이터 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0, :])

x_train_scaled = np.reshape(x_train_scaled,
    (x_train_scaled.shape[0], 5, 5))
x_test_scaled = np.reshape(x_test_scaled,
    (x_test_scaled.shape[0], 5, 5))
print(x_train_scaled.shape)
print(x_test_scaled.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM

# 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(5, 5)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit(x_train_scaled, y_train, validation_split=0.2, verbose=1,
          batch_size=1, epochs=100, callbacks=[early_stopping])

loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

y_pred = model.predict(x_test_scaled)

for i in range(5):
    print('종가 : ', y_test[i], '/ 예측가 : ', y_pred[i])


##### DNN 앙상블 구현하기
import numpy as np
import pandas as pd

kospi200 = np.load('kospi200.npy', allow_pickle=True)
samsung = np.load('samsung.npy', allow_pickle=True)

print(kospi200)
print(samsung)
print(kospi200.shape)
print(samsung.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정

        if y_end_number > len(dataset):  # 수정
            break
        tmp_x = dataset[i:x_end_number, :]  # 수정
        tmp_y = dataset[x_end_number:y_end_number, 3]    # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(samsung, 5, 1) 
x2, y2 = split_xy5(kospi200, 5, 1) 
print(x2[0,:], "\n", y2[0])
print(x2.shape)
print(y2.shape)


# 데이터 셋 나누기
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=1, test_size = 0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=2, test_size = 0.3)

print(x2_train.shape)
print(x2_test.shape)
print(y2_train.shape)
print(y2_test.shape)

x1_train = np.reshape(x1_train,
    (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test,
    (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train,
    (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test,
    (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))
print(x2_train.shape)
print(x2_test.shape)


# 데이터 전처리
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x1_train)
x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)
scaler2 = StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)
print(x2_train_scaled[0, :])


from keras.models import Model
from keras.layers import Dense, Input

# 모델구성
input1 = Input(shape=(25, ))
dense1 = Dense(64)(input1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
output1 = Dense(32)(dense1)

input2 = Input(shape=(25, ))
dense2 = Dense(64)(input2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
output2 = Dense(32)(dense2)

from keras.layers.merge import concatenate
merge = concatenate([output1, output2])
output3 = Dense(1)(merge)

model = Model(inputs=[input1, input2],
              outputs = output3 )


model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2, 
          verbose=1, batch_size=1, epochs=100, 
          callbacks=[early_stopping])

loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

y1_pred = model.predict([x1_test_scaled, x2_test_scaled])

for i in range(5):
    print('종가 : ', y1_test[i], '/ 예측가 : ', y1_pred[i])


##### LSTM 앙상블 구현하기
import numpy as np
import pandas as pd

kospi200 = np.load('kospi200.npy', allow_pickle=True)
samsung = np.load('samsung.npy', allow_pickle=True)

print(kospi200)
print(samsung)
print(kospi200.shape)
print(samsung.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정

        if y_end_number > len(dataset):  # 수정
            break
        tmp_x = dataset[i:x_end_number, :]  # 수정
        tmp_y = dataset[x_end_number:y_end_number, 3]    # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(samsung, 5, 1) 
x2, y2 = split_xy5(kospi200, 5, 1) 
print(x2[0,:], "\n", y2[0])
print(x2.shape)
print(y2.shape)


# 데이터 셋 나누기
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=1, test_size = 0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=2, test_size = 0.3)

print(x2_train.shape)
print(x2_test.shape)
print(y2_train.shape)
print(y2_test.shape)

x1_train = np.reshape(x1_train,
    (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test,
    (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train,
    (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test,
    (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))
print(x2_train.shape)
print(x2_test.shape)


# 데이터 전처리
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x1_train)
x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)
scaler2 = StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)
print(x2_train_scaled[0, :])

x1_train_scaled = np.reshape(x1_train_scaled,
    (x1_train_scaled.shape[0], 5, 5))
x1_test_scaled = np.reshape(x1_test_scaled,
    (x1_test_scaled.shape[0], 5, 5))
x2_train_scaled = np.reshape(x2_train_scaled,
    (x2_train_scaled.shape[0], 5, 5))
x2_test_scaled = np.reshape(x2_test_scaled,
    (x2_test_scaled.shape[0], 5, 5))
print(x2_train_scaled.shape)
print(x2_test_scaled.shape)


from keras.models import Model
from keras.layers import Dense, Input, LSTM

# 모델구성
input1 = Input(shape=(5, 5))
dense1 = LSTM(64)(input1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
output1 = Dense(32)(dense1)

input2 = Input(shape=(5, 5))
dense2 = LSTM(64)(input2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
output2 = Dense(32)(dense2)

from keras.layers.merge import concatenate
merge = concatenate([output1, output2])
output3 = Dense(1)(merge)

model = Model(inputs=[input1, input2],
              outputs = output3 )


model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2, 
          verbose=1, batch_size=1, epochs=100, 
          callbacks=[early_stopping])

loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

y1_pred = model.predict([x1_test_scaled, x2_test_scaled])

for i in range(5):
    print('종가 : ', y1_test[i], '/ 예측가 : ', y1_pred[i])

