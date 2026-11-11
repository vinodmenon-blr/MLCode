import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from pathlib import Path
import PIL
from PIL import Image

def gen_data_with_clusters():
    blob_centers = np.array([[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8],
                             [-2.8, 2.8], [-2.8, 1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std,
                      random_state=7)

    k = 5  # There will be 5 clusters found
    kmeans = KMeans(n_clusters=k, random_state=43)
    y_pred = kmeans.fit_predict(X)

    # check whether clusters found are same as what was generated
    print("Are the clusters found same as what was generated ", y_pred is kmeans.labels_)

    #predict clusters based on clustering done
    X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
    res = kmeans.predict(X_new)
    print("predicted clusters are ", res)

    return X, y

"""
KMeans Clustering logic (lloyds Alog)
### 1. Distance Calculation (for each data instance to each centroid)

* **Squared Euclidean Distance:**
 
  distance(x_i, mu_j) = sum_over_features( (x_i[f] - mu_j[f])^2 )
  
  *(where `x_i` is the data instance, `mu_j` is centroid `j`, and `f` indexes through all feature dimensions)*

### 2. Cluster Assignment (Labeling each data instance)

* **Nearest Centroid Assignment:**
 
  cluster_label(x_i) = argmin_j( distance(x_i, mu_j) )   for j in {1, 2, ..., k}
 
  *(finds which centroid `mu_j` has the lowest distance to `x_i` and assigns that cluster index to `x_i`)*

---

### 3. Centroid Update (Mean of assigned instances)

* **New Centroid Calculation:**
  
  new_mu_j = (1 / N_j) * sum_over_assigned_points( x_i )
  
  *(where `N_j` is the count of data points assigned to cluster `j`, and the sum adds up all vectors `x_i` assigned to cluster `j`)*
---

### 4. Overall Objective (Inertia / Within-Cluster Sum of Squares)

* **Objective Function minimized across all iterations:**
  
  Intend is to minimise Inertia.
  Inertia = sum_over_all_clusters_j( sum_over_points_in_cluster_j( distance(x_i, mu_j) ) )
  
  Score is nothing but negative of inertia
 
---
"""
def kmeans_cluster_init_dependency(X):
    #Let us do random initialisation and see the effect on inertia

    kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=1,
                          random_state=18)
    kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=2,
                          random_state=18)
    kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=3,
                          random_state=18)

    kmeans_iter1.fit(X)
    kmeans_iter2.fit(X)
    kmeans_iter3.fit(X)

    #greater the inertia score the better. You can see with different inits chosen the end result is different
    # and inertias are different

    print("Iter 1 score", kmeans_iter1.score(X))
    print("Iter 2 score", kmeans_iter2.score(X))
    print("Iter 3 score", kmeans_iter3.score(X))

"""
Step 1: Choose the First Centroid
•
Select the first centroid c_1 uniformly at random from the dataset instances X.

Step 2: Compute Distance to Nearest Existing Centroid
For each data instance x_i in the dataset, compute D(x_i): the distance between x_i and the closest centroid 
already selected:
D(x_i) = min( distance(x_i, c_1), distance(x_i, c_2), ..., distance(x_i, c_m) )
(where c_1, ..., c_m are all the centroids chosen so far, and distance is Euclidean distance)

Step 3: Compute Selection Probability for Each Point
Assign each data point x_i a probability of being selected as the next centroid. 
The probability is proportional to the square of its distance, D(x_i)^2:
P(x_i) = ( D(x_i)^2 ) / ( sum_over_all_points_j( D(x_j)^2 ) )
•
What this means in practice:
◦
Points far from any existing centroid have a high D(x_i)^2, giving them a very high chance of being selected.
◦
Points close to an already chosen centroid have D(x_i) ~ 0, meaning their chance of being chosen is almost zero.

Step 4: Pick the Next Centroid
•
Sample the next centroid c_(m+1) from the dataset using the weighted probability distribution P(x_i).
Step 5: Repeat Until All k Centroids Are Selected
•
Repeat Steps 2 through 4 until all k centroids have been chosen.
•
Once all k starting points are chosen, proceed with regular K-Means iterations (distance calculation, 
point assignment, and centroid mean updating).

Here is kmeans++ is greedy kmeans++ in every iteration it does to sample just one centroid from prob distribution
it samples multiple and then chose the best

"""

def compare_kmeans_kmeans_plus(X):
    # n_init stands for "Number of Initializations" if it set as 1 then there is only one start with randon
    # init which is tried. If set to auto it will try 100 different starts and for each start it will try multiple
    # iterations
    kmeans = KMeans(n_clusters=5, init="random", n_init=1, max_iter=3,
                      random_state=18)

    kmeans_plus = KMeans(n_clusters=5, init="k-means++", n_init=1, max_iter=3, random_state=18)

    kmeans.fit(X)
    kmeans_plus.fit(X)

    #check the score between the random init vs greedy kmean++ approach and you can observe the scores when random is just one iteration!!!
    print("score for kmeans with random init ", kmeans.score(X))
    print("score for kmeans with k-means++", kmeans_plus.score(X))

def silhouette_score_compare(X):

    #compute silhouette scores for multiple number of clusters and check
    # what is the optimal value
    # Do clustering for different values of k
    kmeans_per_k = [KMeans(n_clusters=k, random_state=43).fit(X)
                    for k in range(1, 10)]

    silhouette_scores = [silhouette_score(X, model.labels_)
                         for model in kmeans_per_k[1:]]

    #check out the scores in this plot. You can see k=4 and k=5 looks good
    plt.figure(figsize=(8, 3))
    plt.plot(range(2, 10), silhouette_scores, "bo-")
    plt.xlabel("$k$")
    plt.ylabel("Silhouette score")
    plt.axis([1.8, 8.5, 0.55, 0.7])
    plt.grid()
    plt.show()

def img_seg_using_clusters():
    filename = "ladybug.png"
    filepath = Path(f"./datasets/{filename}")
    image = np.asarray(Image.open(filepath))
    #reshape the input where H,W are multiplied into one number
    #uniquely identifying each pixel and then RGB for that pixel
    X = image.reshape(-1, 3)

    #Do image segmentation considering 10,8,6,4,2 colours
    #Here we are taking cluster centroids and considering distances
    #RGB values of pixels
    segmented_imgs = []
    n_colors = (10, 8, 6, 4, 2)
    for n_clusters in n_colors:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        segmented_imgs.append(segmented_img.reshape(image.shape))

    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original image")
    plt.axis('off')
    #plotting the image with each of the cluster values
    for idx, n_clusters in enumerate(n_colors):
        plt.subplot(2, 3, 2 + idx)
        plt.imshow(segmented_imgs[idx] / 255)
        plt.title(f"{n_clusters} colors")
        plt.axis('off')

    plt.show()

"""
KMeans Clustering - Elkans ALgo
Imagine:
•
x is a data instance currently assigned to centroid c_1.

•
c_2 is another centroid.
By the triangle inequality:
distance(x, c_2) >= distance(c_1, c_2) - distance(x, c_1)
Now, suppose the distance between the two centroids distance(c_1, c_2) is large:
If:   distance(c_1, c_2) >= 2 * distance(x, c_1)

Then: distance(x, c_2) >= 2 * distance(x, c_1) - distance(x, c_1)
      distance(x, c_2) >= distance(x, c_1)
      
Using triangular inequality it can skip computing distances for many points in the space leading accelerating 
Kmeans algo
"""


if __name__ == '__main__':
   X,y = gen_data_with_clusters()
   kmeans_cluster_init_dependency(X)
   compare_kmeans_kmeans_plus(X)
   silhouette_score_compare(X)
   img_seg_using_clusters()
