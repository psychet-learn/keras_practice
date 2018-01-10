# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax

# 1. 실 데이터 준비하기
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
y_test = np_utils.to_categorical(y_test)
Xhat_idx = np.random.choice(X_test.shape[0], 5)
Xhat = X_test[Xhat_idx]

# 2. 모델 불러오기
from keras.models import load_model
model = load_model('mnist_mlp_model.h5')

# 3. 모델 사용하기
yhat = model.predict_classes(Xhat)

for i in range(5):
    print('True: ' + str(argmax(y_test[Xhat_idx[i]])) + ', Predict: ' + str(yhat[i]))

