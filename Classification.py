from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import numpy as np


def load_data():
    #fetch data set from open ml and it return not in dataframe
    data = fetch_openml("mnist_784", as_frame=False)
    return data

#split data into train and trust
def split_data(d):
    x = data.data
    y = data.target
    x_train = x[:60000]
    x_test = x[60000:]
    y_train = y[:60000]
    y_test = y[60000:]
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    data = load_data()
    x_train, x_test, y_train, y_test = split_data(data)

    #we will use stochastic gradient descent classifier with logistic regression
    #we will do binary classification initially which is whether it is digit 4 or not
    y_train_4 = (y_train == '4') #labelled all 4 digit to 1 and others to 0
    y_test_4 = (y_test == '4')

    #log_loss option leads to usage of logistic regression. By default what is used is SVM
    sgd_c = SGDClassifier(loss="log_loss", random_state=42)
    sgd_c.fit(x_train, y_train_4)

    #just pass the first sample and check the prediction
    res = sgd_c.predict(x_train[0])
    print(res)
    print(x_test[0][-1])





