import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical


def read_prepare_data():
    # read the datasets
    X_train = pd.read_csv("./X_train_clean.csv")
    X_test = pd.read_csv("./X_test_clean.csv")
    y_train = pd.read_csv("./y_train.csv")
    y_test = pd.read_csv("./y_test.csv")

    # reshape the datasets for CNN
    X_train.drop("case_id", inplace=True, axis=1)
    X_test.drop("case_id", inplace=True, axis=1)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train.vital_status)
    y_train_encoded = encoder.transform(y_train.vital_status)
    encoder = LabelEncoder()
    encoder.fit(y_test.vital_status)
    y_test_encoded = encoder.transform(y_test.vital_status)

    return X_train, X_test, y_train_encoded, y_test_encoded


def train_kfold_model(X_train, X_test, y_train, y_test):
    # define per-fold score containers
    acc_per_fold = []

    # create the datasets and initialize the K-fold split
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train_index, test_index in kf.split(X):
        # create the model
        logisticRegr = LogisticRegression(solver="lbfgs", verbose=1)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # start the model training
        logisticRegr.fit(X_train, y_train)

        # test the model after training
        predictions = logisticRegr.predict(X_test)

        # generate a print
        print(
            "------------------------------------------------------------------------"
        )
        print(f"Training for fold {fold_no} ...")

        # generate generalization metrics
        score = logisticRegr.score(X_test, y_test)
        print(f"Score for fold {fold_no}: accuracy of {score * 100}%")
        acc_per_fold.append(score * 100)

        # increase fold number
        fold_no += 1

    # provide the average scores
    print("\n------------------------------------------------------------------------")
    print("Score per fold")
    for i in range(0, len(acc_per_fold)):
        print(
            "------------------------------------------------------------------------"
        )
        print(f"> Fold {i+1} - Accuracy: {acc_per_fold[i]}%")
    print("------------------------------------------------------------------------")
    print("Average scores for all folds:")
    print(f"> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})")
    print("------------------------------------------------------------------------")


def train_evaluate_model(X_train, X_test, y_train, y_test):
    # create the model
    # all parameters not specified are set to their defaults
    # default solver is incredibly slow thats why we change it
    logisticRegr = LogisticRegression(solver="lbfgs")

    # # configure the model and start training
    logisticRegr.fit(X_train, y_train)

    # # test the model after training
    predictions = logisticRegr.predict(X_test)
    score = logisticRegr.score(X_test, y_test)
    print(score * 100)


# read the datasets
X_train, X_test, y_train, y_test = read_prepare_data()

# # test the model after training
# # train_evaluate_model(X_train, X_test, y_train, y_test)
train_kfold_model(X_train, X_test, y_train, y_test)
