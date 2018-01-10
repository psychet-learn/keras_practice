# Iris 데이터 가져오기
# 기본적인 데이터 나누기, 정규화
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, 0:4]
y = iris.target

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_Y = np_utils.to_categorical(encoded_Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print(X_train_std)
print(Y_train)

from keras.models import Sequential

# Keras의 레이어들을 구성하는 방법으로 사용되는 핵심적인 자료구조형은 model임.
# 그 중 가장 간단한 형태의 모델은 레이어를 선형으로 쌓는 Sequential model.
# 더 복잡한 구조를 원한다면 Keras의 레이어의 임의의 그래프를 작성할 수 있는 functional API를 사용할 수 있음.
model = Sequential()

from keras.layers import Dense

# 레이어는 다음과 같이 .add()를 통해 추가할 수 있음
# units: 출력 데이터의 차원 개수
# activation: 활성 함수
# input_dim: 입력 데이터의 차원 개
model.add(Dense(units=16, input_dim=4, activation='sigmoid'))
model.add(Dense(units=3, activation='softmax'))

# 모델이 어느 정도 모양을 갖추면, .compile()을 통해 학습 방법을 설정할 수 있음.
# loss: 목적함수(비용함수, 손실함수)
# optimizer: 최적화기
# metrics: 평가 기준
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())

from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
# 프로그래머가 원한다면 더 복잡하게 최적화(optimizer parameter) 설정 등을 할 수 있음
# Keras의 궁극적인 원칙 상황을 간편하게 하고 사용자에게 필요한 모든 제어권을 주는 것임
# 사용자가 필요한 상황에서 모든 제어권을 가지고 있다는 말은 코드의 확장성이 좋다는 말과 같음
model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

# 모델 컴파일까지 완료되면 이제 학습시킬 수 있음
print("\n\n Model Fit\n")
# epochs: 학습을 반복하는 회수
# batch_size: 데이터를 일정한 크기(행)를 학습하고 싶을 때
model.fit(X_train_std, Y_train, epochs=10, batch_size=105)

# 또는 배치에 수동으로 모델에 피드를 줄 수 있음
X_batch_std = X_train_std[0:15, :]
y_batch = Y_train[0:15, :]
model.train_on_batch(X_batch_std, y_batch)

# 한 줄로 모델 평가 가능
print("\n\n Model Evaluate\n")
loss_and_metrics = model.evaluate(X_test_std, Y_test, batch_size=128)
print(loss_and_metrics)

#print(loss_and_metrics)

# 예측도 가능
classes = model.predict(X_test_std, batch_size=105)
