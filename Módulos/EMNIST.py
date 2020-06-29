import numpy as np
import pandas as pd
df_train = pd.read_csv("EMNIST\emnist-byclass-train.csv", names = np.arange(-1,784))
df_train = df_train.rename({-1:"labels"}, axis="columns")
df_test = pd.read_csv("EMNIST\emnist-byclass-test.csv", names = np.arange(-1,784)) 
df_test = df_test.rename({-1:"labels"}, axis="columns")
y_train= np.array(df_train['labels'])
x_train= np.array(df_train.drop('labels', axis=1))
x_test= np.array(df_test.drop('labels', axis=1))
y_test= np.array(df_test['labels'])