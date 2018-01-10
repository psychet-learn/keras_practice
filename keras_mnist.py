# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# 1. 데이터 준비하기
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. 데이터셋 생성하기
X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 3. 모델 구성하기
model = Sequential()
model.add(Dense(64, input_dim=28*28, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 5. 모델 학습시키기
fitted = model.fit(X_train, y_train, epochs=5, batch_size=32)

# 6. 학습과정 살펴보기
print('\n## training loss and accuracy ##')
print(fitted.history['loss'])
print(fitted.history['acc'])

# 7. 모델 평가하기
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
print("\n## evaluation loss and metrics ##")
print(loss_and_metrics)

# 8. ahepf tkdydgkrl
Xhat = X_test[0:1]
yhat = model.predict(Xhat)
print('## yhat ##')
print(yhat)
