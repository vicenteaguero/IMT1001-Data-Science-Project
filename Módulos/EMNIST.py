import tensorflow_datasets as tdfs
import pandas as pd
data_train = tdfs.load("emnist", split="train")
data_test = tdfs.load("emnist", split="test")
data_train_numpy = tdfs.as_numpy(data_train)
data_test_numpy = tdfs.as_numpy(data_test)
dataframe_train = pd.DataFrame(data_train_numpy)
dataframe_test = pd.DataFrame(data_test_numpy)
x_train = dataframe_train["image"].to_numpy()
y_train = dataframe_train["label"].to_numpy()
x_test = dataframe_test["image"].to_numpy()
y_test = dataframe_test["label"].to_numpy()
for image in range (len(x_train)):
  x_train[image] = x_train[image].reshape(28, 28).transpose()
  x_train[image] = x_train[image].reshape(784)
for image in range (len(x_test)):
  x_test[image] = x_test[image].reshape(28, 28).transpose()
  x_test[image] = x_test[image].reshape(784)