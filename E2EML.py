# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys

from pathlib import Path

import numpy as np
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Check out https://www.kaggle.com/datasets/camnugent/california-housing-prices/data for housing prises data
# Since I have a tarball with this csv file I will be using that from local folder
# Load data into DataFrame
def load_h_data():
    tar_path = Path("datasets/housing.tgz")
    #Extract csv file
    with tarfile.open(tar_path, "r:gz") as h_tgz:
        h_tgz.extractall(Path("datasets"), filter="data")

    return pd.read_csv(Path("datasets/housing/housing.csv"))

# Plot each feature to check the distribution
def plot_features(df):
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    #Binned Histograms of each feature
    df.hist(bins=50, figsize=(15,10))
    plt.show()

#Plot a bar chart with income category
def plot_inc_cats(df):
    #Binning of mediam income based on the bins defined
    df["income_cat"] = pd.cut(df["median_income"],
                              bins=[0,1.5,3.0,4.5,6,np.inf],
                              labels=[1,2,3,4,5])
    cat_cnts = df["income_cat"].value_counts().sort_index()
    cat_cnts.plot.bar(rot=0, grid=True)
    plt.xlabel("Cat")
    plt.ylabel("Num Houses")
    plt.show()

#Some attributes combinations will be better than attributes themselves
def add_attr_combs(df):
    df["rooms_per_home"] = df["total_rooms"]/df["households"]
    df["bed_to_rooms"] = df["total_bedrooms"]/df["total_rooms"]
    df["people_per_home"] = df["population"]/df["households"]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    housing_df = load_h_data()
    print(housing_df.describe())
    plot_features(housing_df)
    plot_inc_cats(housing_df)
    #Stratified sampling, right number of samples is taken from each stratum. Here it is based on the income
    #category as the stratum. Proportions of samples in train and test based on income category, will be similar to
    #proportions in the complete data set
    s_train_set, s_test_set = train_test_split(housing_df, test_size=0.2,
                                               stratify=housing_df["income_cat"], random_state=42)

    #inspect the training data to understand it better. We can get correlation matrix to check on
    #correlations between features
    corr_matrix = housing_df.corr(numeric_only=True)
    print(corr_matrix["median_income"])

    #For median income there is positive correlation with median_house_value. postive correlation direct relation
    #negative correlation is inverse relation. neary zero means there is no relation. This all is indicative of linear
    #relation and is not indicative any non-linear relation
    #to show that close relation between median_income and median_house_value let us do a scatter plot
    housing_df.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1,grid=True)
    plt.show()

    add_attr_combs(housing_df)

    #Check the correlation of added attribute combinations. Check the correlation of median_income with
    # rooms per home Vs total_rooms, total_bedrooms Vs bed_to_rooms. Correlations have improved.
    corr_matrix = housing_df.corr(numeric_only=True)
    print(corr_matrix["median_income"])

    #drop unnecessary colums
    housing_df = s_train_set.drop("median_house_value", axis=1)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
