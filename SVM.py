import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv('./datasets/churn/Telco-Customer-Churn.csv')
    return df

def data_cleanup(df):
    #remove rows which have total charges as NaN
    df = df.dropna(subset=["TotalCharges"])
    #remove custId col
    df = df.drop(columns=["customerID"])

    #Create label vector and remove it from train set
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    #convert all class cols to numeric. If there are 3 classes for a feature then one column is added
    # for each class stating is true or not. One  col will have true out of the three for each sample
    # This is one hot encoding
    X = pd.get_dummies(X, drop_first=True, dtype=int)

    return X, y




def plot_hist_churners_based_on_features(df):
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    #plot churns against the above features
    for ax, col in zip(axes, num_cols):
        df[df["Churn"] == "No"][col].plot(kind="hist", bins=30, alpha=0.6, ax=ax, label="Stayed")
        df[df["Churn"] == "Yes"][col].plot(kind="hist", bins=30, alpha=0.6, ax=ax, label="Churned", color="tab:red")
        ax.set_title(col)
        ax.legend()
    plt.tight_layout()
    plt.show()

def plot_hist_churners_by_group(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
        #convert the churn col to bool value and then take mean
        churn_rate = (df["Churn"] == "Yes").groupby(df[col]).mean().sort_values()
        churn_rate.plot(kind="barh", ax=ax, color="tab:red")
        ax.set_title(f"Churn rate by {col}")
        ax.set_xlabel("Churn rate")
    plt.tight_layout()
    plt.show()

#evaluate precision,recall, f1 score and accuracy for train and test for given model
def evaluate_svm(name, model, X_train_svm, X_test_svm, y_train, y_test):
    #Get predictions
    yhat_tr, yhat_te = model.predict(X_train_svm), model.predict(X_test_svm)

    def block(yt, yp):
        return (accuracy_score(yt, yp), precision_score(yt, yp),
                recall_score(yt, yp), f1_score(yt, yp))

    tr, te = block(y_train, yhat_tr), block(y_test, yhat_te)
    print(f"  {name}")
    print(f"  {'metric':<12}{'train':>8}{'test':>8}")
    for lbl, i in [("Accuracy", 0), ("Precision", 1), ("Recall", 2), ("F1 (Churn)", 3)]:
        print(f"  {lbl:<12}{tr[i]:>8.3f}{te[i]:>8.3f}")
    print(f"  {'F1 gap':<12}{'':>8}{tr[3] - te[3]:>8.3f}")

#Linear SVC does not use kernel functions and higher degree features. It just uses the geometric margin compute
#for the given feature set to do the classification
def classify_using_linearSVC(X_train, y_train, X_test, y_test):
    """
        C (Regularisation or Margin Strictness)
        What it does: Controls the trade-off between maximizing the margin (street width) and
        minimizing margin violations (points falling inside the street or on the wrong side).
        •
        Why 100000: A very large C places an extremely high penalty on any classification errors.
        ◦
        Low C (e.g., 0.01, 1): Soft margin — allows wider margins and accepts violations for better generalization.
        ◦
        Very High C (e.g., 100,000): Hard margin approximation — forces the model to create a margin
        with almost zero violations, fitting the training points as strictly as possible.
        (Hence your variable name svm_hard_linear).

        Loss function90
        What it does: Specifies the loss function used during optimization.
        ◦
        Formula: L(y, f(x)) = max(0, 1 - y * f(x))
        •
        Default in LinearSVC: The default is 'squared_hinge'.
        •
        Why specify 'hinge': Standard SVM theory (such as in sklearn.svm.SVC(kernel='linear')) strictly uses
        standard hinge loss. When points are outside the margin and correctly classified, the loss is exactly 0.

        class_weight="balanced" (Handling Class Imbalance)
        •
        What it does: Automatically adjusts weights inversely proportional to class frequencies.
        ◦
        Weight formula: n_samples / (n_classes * n_samples_per_class)
        •
        Why it matters for Churn: In customer churn datasets, the classes are typically imbalanced
        (e.g., ~73% stayed, ~27% churned).
        ◦
        Without this, standard SVM treats all errors equally, which leads the model to favor the majority class
        ("Stayed").
        ◦
        With "balanced", misclassifying a churner ("Yes") is penalized roughly 3x more heavily
        than misclassifying a non-churner, boosting the Recall of churn detection.

        Why set max_iter to 200000: Because you set C=100000 and loss='hinge', finding a strict separating boundary on noisy,
        overlapping customer data is extremely difficult. The default 1,000 iterations would almost certainly trigger a
        ConvergenceWarning: Liblinear failed to converge. A high value gives the optimization solver enough cycles to
        converge (or stop when tolerance tol is met).

    """
    #hard_margin
    svm_hard_linear = LinearSVC(
        C=100000,
        loss='hinge',
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=200000
    )

    svm_hard_linear.fit(X_train, y_train)
    evaluate_svm("Hard SVM (Linear Kernel)", svm_hard_linear, X_train, X_test, y_train, y_test)

    #soft_margin
    svm_soft_linear = LinearSVC(
        C=1, #Value is low allowing some classification errors and reducing overfitting
        loss='hinge',
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2000 #iterations can be reduced
    )

    svm_soft_linear.fit(X_train_svm, y_train)
    evaluate_svm("Soft SVM (Linear Kernel)", svm_soft_linear, X_train, X_test, y_train, y_test)

#Gussian kernel which will use higher dimensions to infinite
def classify_using_rbf(X_train, y_train, X_test, y_test):

    #hard_margin
    svm_hard_rbf = SVC(
        kernel="rbf",
        C=1000,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    svm_hard_rbf.fit(X_train, y_train)
    evaluate_svm("Hard SVM (Gaussian Kernel)", svm_hard_rbf,  X_train, X_test, y_train, y_test)

    svm_soft_rbf = SVC(
        kernel="rbf",
        C=1,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    svm_soft_rbf.fit(X_train_svm, y_train)
    evaluate_svm("Soft SVM (Gaussian Kernel)", svm_soft_rbf,  X_train, X_test, y_train, y_test)

if __name__ == '__main__':

    df_data = load_data()
    #convert text to number
    # text that isn't a number becomes NaN
    df_data["TotalCharges"] = pd.to_numeric(df_data["TotalCharges"], errors="coerce")

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    plot_hist_churners_based_on_features(df_data)
    plot_hist_churners_by_group(df_data)

    X,y = data_cleanup(df_data)
    #stratified split of train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    #Standard scale of the training and test set before training and validation
    scaler = StandardScaler()

    # Scale ONLY the numeric columns (the one-hot columns are already 0/1).
    X_train_svm = X_train.copy()
    X_test_svm = X_test.copy()
    X_train_svm[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_svm[num_cols] = scaler.transform(X_test[num_cols])

    #linear SV Classifier
    classify_using_linearSVC(X_train_svm, y_train, X_test_svm, y_test)

    #Gaussian SV Classifier using radial basis function has kernel function
    classify_using_rbf(X_train_svm, y_train, X_test_svm, y_test)

