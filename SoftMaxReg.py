

import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

def plot_soft_max_classification(X,y):
    from matplotlib.colors import ListedColormap

    custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

    #generate points for len and width and pass through to get regions
    x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                         np.linspace(0, 3.5, 200).reshape(-1, 1))
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_proba = softmax_reg.predict_proba(X_new)
    y_predict = softmax_reg.predict(X_new)

    zz1 = y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)

    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
    plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz1, cmap="hot")
    plt.clabel(contour, inline=1)
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.legend(loc="center left")
    plt.axis([0.5, 7, 0, 3.5])
    plt.grid()

    plt.show()


if __name__ == '__main__':
    iris = load_iris(as_frame=True)
    X = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y = iris["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    """
    Algorithm Family               Most Compute-Efficient Approach                    Why?
    Linear Models / 
    SGD (SGDClassifier, linear loss)      OvR or Softmax                         Training scales linearly 
                                                                                $\mathcal{O}(m)$; OvR only needs $K$ models, 
                                                                                avoiding the overhead of 45 models.
    Kernel / Quadratic Models 
    (SVC with RBF/poly kernel)         OvO (by a massive margin!)               SVM scales super-linearly 
                                                                              $\mathcal{O}(m^2)$ to $\mathcal{O}(m^3)$. 
                                                                              Training on small pairwise subsets is vastly faster.
   At Inference / Prediction Time         Softmax                             Only 1 matrix multiplication across 
                                                                              $K$ classes; OvO is slowest 
                                                                              ($45$ evaluations).
    """

    softmax_reg = LogisticRegression(C=30, random_state=42)
    softmax_reg.fit(X_train, y_train)

    res = softmax_reg.predict([[5, 2]])
    print(res)

    #probability for each class
    softmax_reg.predict_proba([[5, 2]]).round(2)
    print(res)

    plot_soft_max_classification(X,y)