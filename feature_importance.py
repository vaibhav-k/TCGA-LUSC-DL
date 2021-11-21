from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, LeakyReLU
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical


def read_prepare_data(feature_vector_length):
    # read the datasets
    X_train = pd.read_csv("./X_train_clean.csv")
    X_test = pd.read_csv("./X_test_clean.csv")
    y_train = pd.read_csv("./y_train.csv")
    y_test = pd.read_csv("./y_test.csv")

    # reshape the datasets for CNN
    X_train.drop("case_id", inplace=True, axis=1)
    tmp = np.array(X_train)
    tmp = tmp.reshape(-1, 2, 4)
    X_train = pd.DataFrame(sum(map(list, tmp), []))
    tmp = []
    for i, g in X_train.groupby(np.arange(len(X_train)) // 2):
        tmp.append(g)
    tmp = np.array([i.to_numpy() for i in tmp])
    # the number of images, shape of the image, and the number of channels
    X_train = tmp.reshape(402, 2, 4, 1)
    X_test.drop("case_id", inplace=True, axis=1)
    tmp = np.array(X_test)
    tmp = tmp.reshape(-1, 2, 4)
    X_test = pd.DataFrame(sum(map(list, tmp), []))
    tmp = []
    for i, g in X_test.groupby(np.arange(len(X_test)) // 2):
        tmp.append(g)
    tmp = np.array([i.to_numpy() for i in tmp])
    # the number of images, shape of the image, and the number of channels
    X_test = tmp.reshape(101, 2, 4, 1)

    # normalize the test and train data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
        X_train.shape
    )
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
        X_test.shape
    )

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train.vital_status)
    y_train_encoded = encoder.transform(y_train.vital_status)
    encoder = LabelEncoder()
    encoder.fit(y_test.vital_status)
    y_test_encoded = encoder.transform(y_test.vital_status)

    # reshape the data - MLPs do not understand such things as '2D'.
    # reshape to 2 x 4 pixels = 8 features
    X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
    X_test = X_test.reshape(X_test.shape[0], feature_vector_length)

    return X_train, X_test, y_train_encoded, y_test_encoded


# configuration options
feature_vector_length = 8

# read the datasets
X_train, X_test, y_train, y_test = read_prepare_data(feature_vector_length)

# test the feature importance using Boruta's algorithm
# define random forest classifier, with utilising all cores and sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators="auto", verbose=2, random_state=42)

# find all relevant features - 5 features should be selected
feat_selector.fit(X_train, y_train)

# check selected features - first 5 features are selected
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X_train)

# review the features
_, num_cols = X_train.shape

# zip the column indices, ranks, and decisions in a single iterable
feature_ranks = list(
    zip(range(num_cols), feat_selector.ranking_, feat_selector.support_)
)

# iterate through and print out the results
for feat in feature_ranks:
    print("Feature: {:<25} Rank: {},  Keep: {}".format(
        feat[0], feat[1], feat[2]))
