import graphviz
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.datasets import make_moons

def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data[["petal width (cm)", "petal length (cm)"]].values
    y = iris.target
    return iris, X, y

def render_dec_tree(iris_df, X,y):
    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X, y)

    #render and export the dec tree graph
    plt.figure(figsize=(10, 10))
    plot_tree(
        tree_clf,
        feature_names=["petal width (cm)", "petal length (cm)"],
        class_names=iris_df.target_names,
        filled=True,
        rounded=True
    )
    plt.savefig("iris_dec_tree.png", dpi=300, bbox_inches="tight")
    plt.close()

    #perdict class level probablity
    res = tree_clf.predict_proba([[5, 1.5]]).round(3)
    print(res)

    #Do the class prediction
    res = tree_clf.predict([[5, 1.5]])
    print(res)

"""
    Decision Trees usually have high variance. We can use regularisation to reduce variance
    1. Min Leaf size
    2. Max depth
    3. Max no of nodes
    4. Grow the whole tree and prune it
    Check the difference in variance between with and without regularisation
"""
def compare_dec_tree_reg():
    #random poin generation
    X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

    tree_clf1 = DecisionTreeClassifier(random_state=42)
    #regularised using min samples in leaf
    tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
    tree_clf1.fit(X_moons, y_moons)
    tree_clf2.fit(X_moons, y_moons)
    #generate test data
    X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2,
                                            random_state=43)

    res = tree_clf1.score(X_moons_test, y_moons_test)
    print("scope without regularisation: ", res)

    res = tree_clf2.score(X_moons_test, y_moons_test)
    print("scope with regularisation: ", res)

if __name__ == '__main__':
    i_df,X,y = load_data()
    render_dec_tree(i_df,X,y)
    compare_dec_tree_reg()
    #Similarly we can use DecisionTreeRegressor for regression with similar regularisation