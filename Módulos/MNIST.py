import tensorflow as tf
import numpy as np
(x_train_aux, y_train), (x_test_aux, y_test) = tf.keras.datasets.mnist.load_data()
x_train_aux = x_train_aux.tolist()
x_test_aux = x_test_aux.tolist()
x_train = list()
for i in range (len(x_train_aux)):
    imagen = list()
    for j in range (28):
        imagen += x_train_aux[i][j]
    x_train.append(imagen)
x_test = list()
for i in range (len(x_test_aux)):
    imagen = list()
    for j in range (28):
        imagen += x_test_aux[i][j]
    x_test.append(imagen)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)