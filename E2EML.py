

from pathlib import Path

import numpy as np
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
from fontTools.otlLib.optimize.gpos import cluster_pairs_by_class2_coverage_custom_cost
from sklearn import linear_model
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


# Check out https://www.kaggle.com/datasets/camnugent/california-housing-prices/data for housing prises data
# Since I have a tarball with this csv file I will be using that from local folder
# Load data into DataFrame
def load_h_data():
   # tar_path = Path("datasets/housing.tgz")
    #Extract csv file
  #  with tarfile.open(tar_path, "r:gz") as h_tgz:
   #     h_tgz.extractall(Path("datasets"), filter="data")

    return pd.read_csv(Path("./datasets/housing/housing.csv"))

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

#data cleaning where missing values are imputed with measures of central tendency
def data_clean_up(df):
    #drop rows with null total_bedrooms in place
    df.dropna(subset="total_bedrooms", inplace=True)
    #compute median to impute
    median = df["total_bedrooms"].median()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(median)

    #imputer can be used to impute null values of features using a strategy.
    #impute missing values with median of the specific feature
    imputer = SimpleImputer(strategy="median")
    #get a dataframe with only numerical columns
    df_num = df.select_dtypes(include=[np.number])
    imputer.fit(df_num)
    #transform returns ndarray
    imput = imputer.transform(df_num)
    #ndarray converted to df
    df_clean = pd.DataFrame(imput, columns=df_num.columns, index=df_num.index)
    return df_clean

def clean_outliers(df,df_label):
    #IsolationForest algorithm could be used to detect outliers or anomalies in the data
    #check out https://medium.com/@corymaklin/isolation-forest-799fceacdda4
    iso_forest = IsolationForest(random_state=42)
    #returns a sparse matrix with labels based on prediction
    outl_pred = iso_forest.fit_predict(df)
    print(outl_pred)
    df = df.iloc[outl_pred==1]
    df_label = df_label.iloc[outl_pred==1]
    return df,df_label

def column_ratio(X):
    return X[:,[0]]/X[:,[1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )


def pre_processing_pipeline_creation():
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    preprocess = ColumnTransformer(
        [
            #use ratio of features
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            #for features with long tail use log of feature
            ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                                   "households", "median_income"]),
            #process the category column with onehotencoder
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
    remainder=default_num_pipeline
    )
    return preprocess


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

    #add some combination features which may have better correlation
    add_attr_combs(housing_df)

    #Check the correlation of added attribute combinations. Check the correlation of median_income with
    # rooms per home Vs total_rooms, total_bedrooms Vs bed_to_rooms. Correlations have improved.
    corr_matrix = housing_df.corr(numeric_only=True)
    print(corr_matrix["median_income"])

    #drop unnecessary columns and labels
    housing_train = s_train_set.drop("median_house_value", axis=1)
    housing_train = housing_train.drop("income_cat", axis=1)
    housing_labels = s_train_set["median_house_value"].copy()
    print(housing_train.shape)
    print(housing_labels.shape)


    #create preprocessing pipeline
    pre_process_columns = pre_processing_pipeline_creation()

    #create pipeline with pre processing of columns followed by linear regression
    linear_regress = make_pipeline(pre_process_columns, LinearRegression())
    linear_regress.fit(housing_train, housing_labels)
    housing_predict = linear_regress.predict(housing_train)

    #compute rootmean square error between labeled values and prediction
    linear_rmse = root_mean_squared_error(housing_predict, housing_labels)
    print(linear_rmse) #the rmse value is pretty high which clearly states that model is undefitting and
    #is not giving good prediction

    #since the linear regression is underfitting we can check out using a more complex model
    #let us try using decision tree regrerssor instead of linear regression
    t_reg = make_pipeline(pre_process_columns, DecisionTreeRegressor(random_state=42))
    t_reg.fit(housing_train, housing_labels)
    #let us checkout rsme for this model
    t_predict = t_reg.predict(housing_train)
    t_rmse = root_mean_squared_error(t_predict, housing_labels)
    print(t_rmse) #This is having 0 as rmse which means model is fitting well. But question is whether it overfitting

    #cross validation can help to understand overfitting. We can use the k_fold cross validation. k here is 10 ie
    #training set is split into k portions and cross validation is done.
    t_rsmes = -cross_val_score(t_reg, housing_train, housing_labels,
                               scoring="neg_root_mean_squared_error", cv=10)
    #checkout the stats of rsme after cross validation with all iterations
    print(pd.Series(t_rsmes).describe()) #cross validation shows high value for rmse and hence model has over fitted
    #validation is failing pretty badly!!!!

    print("Completed end to end training")

