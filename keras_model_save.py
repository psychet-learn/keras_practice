# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax

# 1. 데이터셋 생성하기
# 훈련셋과 시험셋 불러오기
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 데이터셋 전처리
X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
# 원핫인코딩 (one-hot encoding) 처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# 검증셋
X_val = X_train[42000:]
y_val = y_train[42000:]
# 훈련셋
X_train = X_train[:42000]
y_train = y_train[:42000]

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# 5. 모델 평가하기
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
print('\nloss_andmetrics: ' + str(loss_and_metrics))


# 6. 모델 저장하기
model.save('mnist_mlp_model.h5')

