import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import input_data
import numpy as np
import matplotlib.pyplot as plt




def read_data(path):
    # 载入数据
    mnist = input_data.read_data_sets(path, one_hot=True)
    train_x = mnist.train.images.reshape(-1, 28, 28, 1)
    train_y = mnist.train.labels
    valid_x = mnist.validation.images.reshape(-1, 28, 28, 1)
    valid_y = mnist.validation.labels
    test_x = mnist.test.images.reshape(-1, 28, 28, 1)
    test_y = mnist.test.labels
    return train_x, train_y, valid_x, valid_y, test_x, test_y


# 序列模型
def CNN_2D(train_x, train_y, valid_x, valid_y, test_x, test_y):
    # 创建模型
    model = Sequential()
    model.add(Conv2D(input_shape=(28, 28, 1), filters=16, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # 最大池化
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # 最大池化
    model.add(Flatten())  # 扁平化
    model.add(Dense(10, activation='softmax'))
    # 查看网络结构
    model.summary()
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 训练模型
    model.fit(train_x, train_y, batch_size=500, nb_epoch=20, verbose=2, validation_data=(valid_x, valid_y))
    # 评估模型
    pre = model.evaluate(test_x, test_y, batch_size=500, verbose=2)  # 评估模型
    print('test_loss:', pre[0], '- test_acc:', pre[1])

    # predicted_classes = model.predict_classes(test_x)
    # # Check which items we got right / wrong
    # correct_indices = np.nonzero(predicted_classes == test_y)[0]
    # incorrect_indices = np.nonzero(predicted_classes != test_y)[0]
    # plt.figure()
    # for i, correct in enumerate(correct_indices[:9]):
    #     plt.subplot(3, 3, i + 1)
    #     plt.imshow(test_x[correct].reshape(28, 28), cmap='gray', interpolation='none')
    #     plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_y[correct]))
    # plt.show()
    # plt.figure()
    # for i, incorrect in enumerate(incorrect_indices[:9]):
    #     plt.subplot(3, 3, i + 1)
    #     plt.imshow(test_x[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    #     plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_y[incorrect]))
    # plt.show()


if __name__ == '__main__':
    train_x, train_y, valid_x, valid_y, test_x, test_y = read_data('MNIST_data')
    CNN_2D(train_x, train_y, valid_x, valid_y, test_x, test_y)