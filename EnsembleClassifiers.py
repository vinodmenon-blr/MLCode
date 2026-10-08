from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss, accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

def gent_data():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test

def compare_hard_vote_class_vs_individual(X_tr,X_te,y_tr,y_te):
    """
        Class with max votes across all classifiers is chosen. For getting better results choose diverse classifiers
    """

    print("Hard Voting")
    v_clf = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('svc', SVC(random_state=42))
        ]
    )
    v_clf.fit(X_tr, y_tr)

    #get individual estimators of each classifier and check their score
    #Check out score for each estimator for test set. Here score is just running accuracy_score
    for name, clf in v_clf.named_estimators_.items():
        print(name, "=", clf.score(X_te, y_te))

    #check the score for voter classifier
    print("For voter classifier", v_clf.score(X_te, y_te)) #prediction is better for voter classifier than each of the
                                                           #classifiers individually showing the effect of ensembling

    #compare predictions
    print("Voting Classifier prediction", v_clf.predict(X_te[:1]))

    print("Each classifier prediction", [clf.predict(X_te[:1]) for clf in v_clf.estimators_])

def check_soft_vote_classifier(X_tr, X_te, y_tr, y_te):
    """
        In soft voting if all classifiers give class probabilities then we find average of the class probabilities
        and use that for the classification. This will be more accurate than hard voting
    """

    print("Soft Voting")
    v_clf = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('svc', SVC(random_state=42))
        ]
    )
    v_clf.voting = "soft"
    v_clf.named_estimators["svc"].probability = True #Give class probablity for support vector classifier
    v_clf.fit(X_tr, y_tr)
    res = v_clf.score(X_te, y_te)
    print("Soft Voting score", res) #You can see the score has improved over hard voting

"""
    The standard diagnostic tool is the Calibration Curve (available in scikit-learn via CalibrationDisplay).
    How it works:
    1.
    The test instances are sorted and divided into bins based on their predicted probabilities 
    (e.g., 10 bins: 0.0–0.1, 0.1–0.2, ..., 0.9–1.0).
    2.
    For each bin, two values are calculated:
    ◦
    Mean predicted probability (X-axis)
    ◦
    Fraction of actual positives (true observed frequency) (Y-axis)
    3.
    The points are plotted against a 45-degree diagonal line (y = x) representing perfect calibration.
    How to read the plot:
    •
    Along the diagonal (y = x): Perfectly calibrated. (If the model predicts 0.70 confidence, exactly 
    70% of those samples are positive).
    •
    Below the diagonal line: The model is overconfident. Its predicted probability is higher than the 
    true positive rate (e.g., predicted 0.90, but only 60% are positive).
    •
    Above the diagonal line: The model is underconfident. Its predicted probability is lower than the true 
    positive rate (e.g., predicted 0.60, but 85% are positive).
    •
    S-shaped curve: Typical for models like RandomForestClassifier, where probabilities are compressed 
    toward the middle (rarely approaching 0.0 or 1.0).
    
    •
    If a classifier's curve hugs the dashed diagonal reference line and has a low Brier score, 
    it is well-calibrated and will work well in soft voting.
    •
    If its curve deviates significantly from the diagonal, wrapping it with CalibratedClassifierCV
    will align it back to the diagonal line.
"""

def check_model_calibration(X_tr, X_te, y_tr, y_te):
    fig, ax = plt.subplots(figsize=(8, 6))

    #We will do it just for logistic regression
    # 1. Get predicted probabilities for positive class (Class 1)
    model = LogisticRegression(random_state=42)
    model.fit(X_tr, y_tr)
    prob_pos = model.predict_proba(X_te)[:, 1]

    # 2. Compute Brier score (lower is better)
    brier = brier_score_loss(y_te, prob_pos)
    print(f"Logistic Brier Score: {brier:.4f}")

    # 3. Plot calibration curve
    CalibrationDisplay.from_predictions(
        y_te,
        prob_pos,
        n_bins=10,
        name=f"Logistic (Brier: {brier:.3f})",
        ax=ax
    )
    ax.set_title("Calibration Curve (Reliability Diagram)")
    #plt.show()

def using_bagging_classifiers(X_tr,X_te,y_tr,y_te):

    #n_jobs=-1 means using multi threads for all CPU cores
    bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                                max_samples=100, n_jobs=-1, random_state=42)
    bag_clf.fit(X_tr, y_tr)


    bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                                max_samples=100, oob_score=True, n_jobs=-1, random_state=42)
    bag_clf.fit(X_tr, y_tr)
    print("Score from classification using OOB", bag_clf.oob_score_)
    print("Score from Bagging Classifier", bag_clf.score(X_te, y_te))

def using_random_forests(X_tr,X_te,y_tr,y_te):
    """
        If there are n features then chose root n features randomly for each partitioning
    """
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                     n_jobs=-1, random_state=42)
    rnd_clf.fit(X_tr, y_tr)
    y_pred_rf = rnd_clf.predict(X_te)
    print("Score for Random Forest Classifier", rnd_clf.score(X_te, y_te))

def check_feature_importance(X_tr, X_te, y_tr, y_te):
    ir_df = load_iris(as_frame=True)
    r_clf = RandomForestClassifier(n_estimators=500, random_state=42)
    r_clf.fit(ir_df.data, ir_df.target)
    """
        Based in which features has been more effective in reducing Gini impurity while
        being used in partitioning is measured across the tree. feature importance is giving
        normalised value for these comparisons
    """
    for score, name in zip(r_clf.feature_importances_, ir_df.data.columns):
        print(round(score, 2), name)

def using_adaptive_boost(X_tr, X_te, y_tr, y_te):

    #learning rate is used in computing predictor weight for samples
    #SAMME is be used since it can be used for multi class classification
    #Base Estimator is Decition Tree
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3), n_estimators=30,
        learning_rate=0.5, random_state=42)
    ada_clf.fit(X_tr, y_tr)
    print("Score for AdaBoost Classifier", ada_clf.score(X_te, y_te))

def gen_quad_dat_with_noise():
    m = 100
    rng = np.random.default_rng(seed=42)
    X = rng.random((m, 1)) - 0.5
    noise = 0.05 * rng.standard_normal(m)
    y = 3 * X[:, 0] ** 2 + noise  # y = 3x² + Gaussian noise
    return X, y

def train_using_boosting(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # First regressor
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(X_train, y_train)

    # Compute residual error
    y2 = y_train - tree_reg1.predict(X_train)
    tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
    tree_reg2.fit(X_train, y2)

    y3 = y2 - tree_reg2.predict(X_train)
    tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
    tree_reg3.fit(X_train, y3)

    # Find prediction after boost (sum of residual predictions)
    y_pred_boost = sum(tree.predict(X_test) for tree in (tree_reg1, tree_reg2, tree_reg3))

    # Find prediction without boost (first tree alone)
    y_pred = tree_reg1.predict(X_test)

    # For regression, evaluate with MSE (or RMSE) and R2 score instead of accuracy_score
    print("MSE using boost:", mean_squared_error(y_test, y_pred_boost))
    print("MSE without boost:", mean_squared_error(y_test, y_pred)) # much higher
    print("R2 score using boost:", r2_score(y_test, y_pred_boost)) #higher r2 score showing better prediction
    print("R2 score without boost:", r2_score(y_test, y_pred))

def train_using_Gradient_Boost(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,
                                     learning_rate=1.0, random_state=42)
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_test)
    print("MSE using Gradient Boost with small number of estimators:", mean_squared_error(y_test, y_pred))

    #With more number of estimators. Algo checks whether last 10 trees helped at all. If not it stops
    #here we have more number of estimators. In that case the learning rate has to be reduced
    gbrt_best = GradientBoostingRegressor(
        max_depth=2, learning_rate=0.05, n_estimators=500,
        n_iter_no_change=10, random_state=42)
    gbrt_best.fit(X, y)
    y_pred_best = gbrt_best.predict(X_test)
    print("MSE using Gradient Boost with large number of estimators:", mean_squared_error(y_test, y_pred_best))
    print("Num of estimators used:", gbrt_best.n_estimators_)
    print("R2 score using less estimators:", r2_score(y_test, y_pred)) #higher r2 score showing better prediction
    print("R2 score without more estimators:", r2_score(y_test, y_pred_best))

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = gent_data()
    compare_hard_vote_class_vs_individual(X_train, X_test, y_train, y_test)
    check_soft_vote_classifier(X_train, X_test, y_train, y_test)
    """
        If classifiers are not well calibrated then we will have to calibrate the results
        ----------------------------------------------------------------------------------
        Accuracy vs. Calibration
        A model can have low bias, low variance, and great classification accuracy, and still be completely miscalibrated.
        •
        Classification accuracy only depends on whether the decision threshold is right (e.g., is the score 
        for Class 1 greater than Class 0?).
        •
        Calibration depends on whether the exact numerical score represents the real-world probability.
        Many algorithms are simply not designed to produce probabilities:
        •
        SVC optimizes a geometric margin (distance to a boundary), not a probability.
        •
        RandomForestClassifier averages predictions across many decision trees. This averaging 
        pulls extreme probabilities inward toward 0.5, rarely outputting confident 0.0 or 1.0 values.
        
        How CalibratedClassifierCV Solves This
        CalibratedClassifierCV fixes this by acting as a post-processor. It does not change how the underlying trees 
        or SVM boundaries are built. Instead, it learns a 1-dimensional correction function that maps the model's raw 
        output score into a reliable probability.
        Step 1: Why the "CV" (Cross-Validation) Matters
        If you calibrated the model using the same training set it was trained on, the calibrator would inherit 
        the model's overfitting (high variance).
        To prevent this:
        1.
        It splits the training data into K folds (for example, 5 folds).
        2.
        It trains the classifier on 4 folds and predicts scores on the remaining 5th fold (unseen data).
        3.
        It repeats this so that every data point gets an out-of-fold score.
        4.
        By using predictions on unseen data, the calibrator sees how the model actually behaves in practice, 
        stripping away the artificial confidence caused by overfitting.
        Step 2: Learning the Mapping Function
        Using those out-of-fold scores and the real labels (0 or 1), it fits a correction curve using one of two methods:
        •
        Sigmoid (Platt Scaling) - method='sigmoid': It fits a logistic regression curve on the raw scores.
        In plain equation form:
        P(y = 1 | score) = 1 / (1 + exp(A * score + B))
        Here, A and B are parameters learned during calibration. If the model was overconfident 
        (saying 0.95 when the true success rate was only 0.70), the curve scales that 0.95 down to 0.70.
        •
        Isotonic Regression - method='isotonic': It fits a non-parametric, piece-wise increasing step function. 
        It makes no mathematical assumptions about the curve shape and can correct arbitrary distortions 
        (useful for random forests), though it requires more data.
        Step 3: Preserving Rank and Decision Power
        Because the mapping functions are monotonic (order-preserving):
        •
        If sample A had a higher raw score than sample B before calibration, sample A will still have a higher 
        calibrated probability after calibration.
        •
        Your underlying classifier's ranking and discriminative power remain intact.
        •
        Only the numbers are recalibrated to match real observed frequencies.
        3. Application to EnsembleClassifiers.py
        In your VotingClassifier, when you switch to voting='soft':
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # Wrap SVC so its raw distance margins are turned into calibrated probabilities
        calibrated_svc = CalibratedClassifierCV(
            estimator=SVC(random_state=42), 
            method='sigmoid', 
            cv=5
        )
        
        v_clf = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(random_state=42)),
                ('rf', RandomForestClassifier(random_state=42)),
                ('svc', calibrated_svc)
            ],
            voting='soft'
        )
        With calibration in place, no single model can dominate or skew the soft vote purely through artificial 
        overconfidence.
    """
    #check the calibration curve to see the classifier does good calibration or not
    check_model_calibration(X_train,  X_test, y_train, y_test)

    using_bagging_classifiers(X_train, X_test, y_train, y_test)

    using_random_forests(X_train, X_test, y_train, y_test) #score has improved from bagging classifier because
                                                            #reduction in variance because random choice of features
    check_feature_importance(X_train, X_test, y_train, y_test)

    using_adaptive_boost(X_train, X_test, y_train, y_test)

    X,y = gen_quad_dat_with_noise()
    print("shape of X", X.shape)
    print("shape of y", y.shape)

    train_using_boosting(X, y)

    train_using_Gradient_Boost(X,y)
