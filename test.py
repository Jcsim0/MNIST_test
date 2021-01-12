# -*- coding: utf-8 -*-
"""
Created with PyCharm
@Auth Jcsim
@Date 2021-1-12 16:35
@File test.py
@Description 
"""
from keras.datasets import mnist
# import CNN_MNIST
from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# import input_data
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation

# 装载数据
nb_classes = 10
(train_x, train_y), (test_x, test_y) = mnist.load_data()

for i in range(20):
    #  plt.subplot(行，列，索引)
    plt.subplot(4, 5, i + 1)
    plt.imshow(train_x[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(train_y[i]))
plt.show()

#  格式化数据
# 对于每一个训练样本我们的神经网络得到单个的数组，所以我们需要将28x28的图片变形成784的向量，我们还将输入从[0,255]缩到[0,1].
train_x = train_x.reshape(60000, 784)
test_x = test_x.reshape(10000, 784)
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x /= 255
test_x /= 255
print("Training matrix shape", train_x.shape)
print("Testing matrix shape", test_x.shape)
# 将目标矩阵变成one-hot格式
train_y = np_utils.to_categorical(train_y, nb_classes)
test_y = np_utils.to_categorical(test_y, nb_classes)

# 创建模型
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))  # An "activation" is just a non-linear function applied to the output
# of the layer above. Here, with a "rectified linear unit",
# we clamp all values below 0 to 0.

model.add(Dropout(0.2))  # Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))  # This special "softmax" activation among other things,
# ensures the output is a valid probaility distribution, that is
# that its values are all non-negative and sum to 1.

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam')

# 训练模型
model.fit(train_x, train_y,
          batch_size=128, epochs=20,
          verbose=1,
          validation_data=(test_x, test_y))

# 评估模型
score = model.evaluate(test_x, test_y, verbose=0)
print('Test score:', score)

predicted_classes = model.predict_classes(test_x)
# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == test_y)[0]
incorrect_indices = np.nonzero(predicted_classes != test_y)[0]
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_x[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_y[correct]))
plt.show()
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_x[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_y[incorrect]))
plt.show()