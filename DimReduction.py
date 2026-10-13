from pandas.core.common import random_state
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

def load_data():
    m_n = fetch_openml('mnist_784', as_frame=False)
    X_tr, y_tr = m_n.data[:60000], m_n.target[:60000]
    X_te, y_te = m_n.data[60000:], m_n.target[60000:]
    return X_tr, y_tr, X_te, y_te

def get_data_pca_reduced(X_tr):
    #First get the transform and see what are the variances to choose the right number of dimensions
    pca = PCA()
    pca.fit(X_tr)
    #get the cumulative variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    #Get the number of dimensions which captures cumulative variance of 0.95
    d = np.argmax(cumvar >= 0.95) + 1
    print(f"Dimensions reduced from: {X_tr.shape[1]} to {d}")
    X_red = pca.transform(X_tr)
    return X_red, d

def train_with_hyper_param_tune(X_tr, y_tr):
    clf = make_pipeline(PCA(random_state=42),
                        RandomForestClassifier(random_state=42))
    param_distrib = {
        "pca__n_components": np.arange(10, 80),
        "randomforestclassifier__n_estimators": np.arange(50, 500)
    }

    """
    [31,500 Parameter Combinations Available]
                    │
                    ▼
       Sample 10 random combinations (n_iter=10)
                    │
   ┌────────────────┴────────────────┐
   │ For each chosen combination:    │
   │  ├─ Fit on Folds 1 & 2 -> Test on Fold 3
   │  ├─ Fit on Folds 1 & 3 -> Test on Fold 2
   │  └─ Fit on Folds 2 & 3 -> Test on Fold 1
   │  Calculate Mean Validation Score│
   └────────────────┬────────────────┘
                    ▼
     Pick combination with best score
                    │
                    ▼
  Refit best model on all training data (refit=True)
  
  Once RandomizedSearchCV identifies the winning hyperparameter combination (e.g., pca__n_components=45,
   n_estimators=200), you want your final deployed model to learn from all 1,000 samples, not just a 667-sample slice.
    Because the parameter refit=True is enabled by default in RandomizedSearchCV:
    1.
    Retrains on 100% of the data: It automatically builds a brand new model using the best hyperparameters 
    found and trains it on the entire X_tr[:1000], y_tr[:1000].
    2.
    Stores the final model: It saves this trained model in the attribute rnd_search.best_estimator_.
    """
    rnd_search = RandomizedSearchCV(clf, param_distrib, n_iter=10, cv=3,
                                    random_state=42)
    rnd_search.fit(X_tr[:1000], y_tr[:1000])
    print("Best Parameters: Random Search: Random Forest ", rnd_search.best_params_)

    #You can use PCA(svd_sover="randomized", randon_state=42) to reduce compute time
    clf = make_pipeline(PCA(random_state=42), SGDClassifier())
    param_grid = {
        "pca__n_components": [20, 40, 60],
        "sgdclassifier__loss": ["hinge", "log_loss"],
        "sgdclassifier__alpha": [1e-4, 1e-3, 1e-2]
    }

    """
      When you pass a dictionary with multiple keys to GridSearchCV, it takes the full Cartesian product of all lists.
    What the combinations look like:
    Every single combination is tested:
    1.
    n_components=20, loss='hinge', alpha=0.0001
    2.
    n_components=20, loss='hinge', alpha=0.001
    3.
    n_components=20, loss='hinge', alpha=0.01
    4.
    n_components=20, loss='log_loss', alpha=0.0001
    5.
    n_components=20, loss='log_loss', alpha=0.001
    6.
    n_components=20, loss='log_loss', alpha=0.01
    7.
    n_components=40, loss='hinge', alpha=0.0001 ... all the way to:
    8.
    n_components=60, loss='log_loss', alpha=0.01
    """
    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X_tr[:1000], y_tr[:1000])
    print("Best Parameters: Grid Search: SGD ", grid_search.best_params_)

#Incremental PCA can be used if the training set is huge, and we cannot have that read completely
#into memory. This method shows how we can use PCA in mini batches.
def inc_pca_for_large_train_set(X_tr, y_tr):
    #using mini batches
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154)
    #Fit using mini batches
    for X_batch in np.array_split(X_tr, n_batches):
        inc_pca.partial_fit(X_batch)

    X_tr_red = inc_pca.transform(X_tr)

    #other approach is using a memory mapped file where only a chunk in
    #the file is loaded at a time into memory

    #create them memmap file with chunks
    filename = "test_mnist.mmap"
    X_mmap_wr = np.memmap(filename, dtype='float32', mode='write', shape=X_tr.shape)
    X_mmap_wr[:] = X_tr
    print("mmap shape", X_mmap_wr.shape)
    X_mmap_wr.flush()


    X_mmap_re = np.memmap(filename, dtype="float32", mode="readonly").reshape(-1, 784)
    batch_size = X_mmap_re.shape[0] // n_batches
    print("batch_size", batch_size)
    inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
    inc_pca.fit(X_mmap_re)

    #compare the accuracy between trained using complete training set vs reduced training set
    clf = SGDClassifier()
    #train with full training set
    clf.fit(X_tr, y_tr)
    accuracy = clf.score(X_te, y_te)
    print("Accuracy with full training set : ", accuracy)

    clf.fit(X_tr_red, y_tr)
    X_te_red = inc_pca.transform(X_te)
    accuracy = clf.score(X_te_red, y_te)
    print("Accuracy with reduced training set : ", accuracy) #accuracy infact has gone up with reduced dimension

"""
When to use PCA:
1.
Moderate number of features: PCA works best when your feature count is in the hundreds or a few thousands, 
such as image pixels (like the 784 features in your DimReduction.py MNIST script), sensor readings, or tabular data.
2.
Maximum information preservation: PCA is mathematically optimal for finding the directions of highest variance. 
If you want the smallest possible number of dimensions that still captures 95 percent of the variance, PCA will always 
beat Random Projection.
3.
Interpretability and noise reduction: PCA gives you explained variance ratios for each component, 
letting you inspect how much information is retained. Because it discards components with tiny variances, PCA often 
filters out random noise from the data.

When to use Random Projection:
1.
Very large number of features: Use Random Projection when you have tens of thousands, hundreds of thousands, 
or millions of features. Examples include:
•
Natural Language Processing: large vocabularies, TF-IDF matrices, or word n-grams with 50,000 to 100,000 terms.
•
Bioinformatics: gene expression data, DNA microarrays, and genome sequences where features vastly outnumber samples.
•
Recommendation systems and high-cardinality sparse tables.
2.
When PCA is too slow or runs out of memory: PCA requires computing covariance matrices or singular value 
decomposition, which takes cubic time relative to the number of features. When the feature count reaches 50,000, 
PCA will either run out of RAM or take hours to compute. Random Projection does not fit or learn from the data; 
it simply generates a random matrix and multiplies it, taking only seconds even on massive matrices.
3.
Distance-based algorithms and nearest neighbors: Because of the Johnson-Lindenstrauss lemma, Random Projection is 
guaranteed to preserve the relative Euclidean distances between data points. This makes it especially useful as a 
fast pre-processing step for K-Nearest Neighbors, clustering, or vector similarity search.
"""

def dim_red_using_random_proj():
    n = 20000
    m, ε = 5_000, 0.1
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((m, n))  #so 5000 samples with 20000 features. Just a random training set for demonstration
    #Using the jonhson lindenstrauss lemma to find the number of dimensions. Choses the feature projection
    #matrix considering the number of features using a gaussian distribution
    gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
    X_reduced = gaussian_rnd_proj.fit_transform(X)

    #creates a sparse projection matrix with far lower memory requrement and project compute will be much faster
    sparse_rnd_proj = SparseRandomProjection(eps=ε, random_state=42)
    X_reduced = sparse_rnd_proj.fit_transform(X)



if __name__ == '__main__':
    X_tr, y_tr, X_te, y_te = load_data()
    X_tr_red, d = get_data_pca_reduced(X_tr)
    train_with_hyper_param_tune(X_tr,y_tr)
    inc_pca_for_large_train_set(X_tr, y_tr)
    dim_red_using_random_proj()


