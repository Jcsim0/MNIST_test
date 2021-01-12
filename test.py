# -*- coding: utf-8 -*-
"""
Created with PyCharm
@Auth Jcsim
@Date 2021-1-12 16:35
@File test.py
@Description 
"""
from keras.datasets import mnist
import matplotlib.pyplot as plt
import CNN_MNIST

# train_x, train_y, valid_x, valid_y, test_x, test_y = CNN_MNIST.read_data('MNIST_data')
(X_train, y_train), (X_test, y_test) = mnist.load_data()
for i in range(20):
    #  plt.subplot(行，列，索引)
    plt.subplot(4, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))
plt.show()


plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))