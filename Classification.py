from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


"""
A Precision-Recall (PR) Curve evaluates the trade-off between Precision and Recall 
across all possible decision thresholds of a classifier.

The Two Metrics Being Measured
Precision
TP / (TP + FP)
When the model predicts an image is a '4', how often is it actually right?
Recall (Sensitivity)
TP / (TP + FN)
Out of all the real '4' images in the dataset, how many did the model detect?
"""


def plot_prec_rec_curve(precs, recs, thres):
    plt.figure(figsize=(8, 4))  # extra code – it's not needed, just formatting
    plt.plot(thres, precs[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thres, recs[:-1], "g-", label="Recall", linewidth=2)
    idx = (thres >= thres).argmax()  # first index ≥ threshold
    plt.plot(thres[idx], precs[idx], "bo")
    plt.plot(thres[idx], recs[idx], "go")
    plt.axis([-50000, 50000, 0, 1])
    plt.grid()
    plt.xlabel("Threshold")
    plt.legend(loc="center right")

    plt.show()


"""
    The Two Quantities Being Plotted
        The ROC curve plots:
        •
        Y-Axis: True Positive Rate (TPR) — also known as Recall or Sensitivity.
        ◦
        Formula: TPR = TP / (TP + FN)
        ◦
        Plain English: Out of all the real '4's in the dataset, what fraction did the model successfully find?
        •
        X-Axis: False Positive Rate (FPR) — also called the False Alarm Rate or (1 - Specificity).
        ◦
        Formula: FPR = FP / (FP + TN)
        ◦
        Plain English: Out of all the non-'4' images, what fraction did the model mistakenly label as a '4'?

        The AUC (area under curve) equals the probability that the classifier will give a higher score to a randomly 
        chosen positive sample (a real '4') than to a randomly chosen negative sample (a non-'4').
"""


def plot_roc_curve(precs, recs, idx):
    plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

    plt.plot(recs, precs, linewidth=2, label="Precision/Recall curve")

    # extra code – just beautifies Figure 3–6
    plt.plot([recs[idx], recs[idx]], [0., precs[idx]], "k:")
    plt.plot([0.0, recs[idx]], [precs[idx], precs[idx]], "k:")
    plt.plot([recs[idx]], [precs[idx]], "ko",
             label="Point at threshold 3,000")
    """
    plt.gca().add_patch(patches.FancyArrowPatch(
        (0.79, 0.60), (0.61, 0.78),
        connectionstyle="arc3,rad=.2",
        arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
        color="#444444"))
    """
    plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.legend(loc="lower left")

    plt.show()

def get_conf_mat_multi_class(x_train,y_train):
    #std scaling of features using z-score
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train.astype("float64"))
    #This will take time because this will do OvR approach so that will lead to
    #10 classifiers (each will need many epochs for convergence this gradient descent)
    #These 10 classifiers are done 3 times because there are 3 folds.
    sgd_c = SGDClassifier(loss="log_loss",  max_iter=2000, random_state=42)
    y_train_p = cross_val_predict(sgd_c, x_train_s, y_train, cv=3)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_p) # Here we can see 8 has many
    #wrong classifications. Which means probably we may need more samples of 8 to train the model
    # better
    plt.show()



if __name__ == '__main__':
    data = load_data()
    x_train, x_test, y_train, y_test = split_data(data)

    #we will use stochastic gradient descent classifier with logistic regression
    #we will do binary classification initially which is whether it is digit 4 or not
    y_train_4 = (y_train == '4')  #labelled all 4 digit to 1 and others to 0
    y_test_4 = (y_test == '4')

    #log_loss option leads to usage of logistic regression. By default what is used is SVM
    sgd_c = SGDClassifier(loss="log_loss", random_state=42)
    sgd_c.fit(x_train, y_train_4)

    #just pass the first sample and check the prediction. Here we are passing the third image from training
    #set which is 4 and it classified properly
    """
        SGDClassifier(loss="log_loss") - Stochastic Gradient Descent (SGD): updates weights one sample (or batch) at a time.
        LogisticRegression - Batch Solvers (lbfgs, saga, liblinear): use entire dataset to compute gradients and 
        Hessian/curvature estimates.
        Both optimises the cross entropy function
    """
    res = sgd_c.predict([x_train[2]])
    print(res)

    #Let us check the cross validation to see the accuracy of the model, split into 3 folds
    #It is StratifiedKFold where the proportions are maintained in each fold
    res = cross_val_score(sgd_c, x_train, y_train_4, cv=3, scoring="accuracy")
    print(res)  #97% accuracy is great

    #Let us compare with dummy classifier. The default option will use strategy as "prior" which is always
    #return most frequent occurrence which is non_4
    dummy_c = DummyClassifier()
    dummy_c.fit(x_train, y_train_4)
    #predict will always gives False because most frequent is non 4

    #Let us check the cross validation score
    res = cross_val_score(dummy_c, x_train, y_train_4, cv=3, scoring="accuracy")
    print(res)  #This gives accuracy of 90%. which is good too. But that is because the data set is skewed towards
    # occurrence of non 4

    #So for classifiers accuracy alone is not a good measure. Confusion matrix is what has to be used
    #confusion matrix gives measure of false positives (type 1 error) and false negatives (type 2 error)
    #We can get predictions from k fold cross validation
    pred = cross_val_predict(sgd_c, x_train, y_train_4, cv=3)

    #Now we have predictions let us get the confusion matrix
    conf_mat = confusion_matrix(y_train_4, pred)
    print(conf_mat)

    disp = ConfusionMatrixDisplay(conf_mat, display_labels=['non 4', '4'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()  #53801 true negative, 357 false positive, 1070 false negative, 4772 true positive

    print("precision " + str(precision_score(y_train_4, pred)))
    print("recall " + str(recall_score(y_train_4, pred)))
    print("f1 score " + str(f1_score(y_train_4, pred)))  #f1 score is 0.86 which is not really high
    #when precision increases recall will reduce. So there is a trade off. Sometimes precision is more important
    #sometimes recall is more important. It depends on the classification task

    #check out effect of thresholds in predictions and in turn effect on the scores. So this another parameter
    #which could be tweaked. Get the decision scores and effect of threshold. By default threshold is 0. Based on
    #change in threshold classifications will change. When score is higher than threshold then it classifies as positive
    #and vice versa

    y_scores = cross_val_predict(sgd_c, x_train, y_train_4, cv=3, method="decision_function")

    #we can draw the precision, recall curves based on threshold. Get the precision and recall values
    #for different thresholds

    precs, recalls, thresh = precision_recall_curve(y_train_4, y_scores)
    #plot the prec recall curve
    plot_prec_rec_curve(precs, recalls, thresh)

    #If we want to get threshold where precision is atleast 90%
    i_90_prec = (precs >= 90).argmax()
    thres_90_prec = thresh[i_90_prec]
    print(thres_90_prec)

    #Other option which is available is to use receiver operatinf characterstic (ROC). This curve
    #shows how well classifier seperates positive from negative

    plot_roc_curve(precs, recalls, i_90_prec)

    #prefer precision recall curve when positive class is rarer or low in number where what matters to you
    #more about false positives than false negatives

    #multiclass classification
    """
    One Over Rest approach
    You train one binary classifier for each class (n classifiers in total).
    For each classifier, that specific class is treated as positive and all other classes combined 
    are treated as negative
    Making a prediction: When given an image, you run it through all N classifiers. 
    You pick the class whose classifier produces the highest decision score / probability
    
    One Over One Approach
    You train one binary classifier for every unique pair of classes.
    Making a prediction: You run the image through all pair-wise classifiers and hold a "tournament." 
    Each classifier casts a vote for its winner. The digit that receives the most votes wins.
    
    """
    #using Support Vector Classifier
    sv_clf = SVC(random_state=42)
    #train using 200 samples. SVM uses OvO approach and hence will run (N(N-1)/2) number of classifiers
    sv_clf.fit(x_train[:2000], y_train[:2000])
    res = sv_clf.predict([x_train[2]])
    print(res)
    res = sv_clf.predict([x_train[0]])
    print(res)

    #to use OvR approach for SVC then we can do that as follows
    ovr_clf = OneVsRestClassifier(SVC(random_state=42))
    ovr_clf.fit(x_train[:2000], y_train[:2000])
    res = ovr_clf.predict([x_train[2]])
    print(res)

    #Error Analysis using confusion matric for multi class classification
    get_conf_mat_multi_class(x_train, y_train)



