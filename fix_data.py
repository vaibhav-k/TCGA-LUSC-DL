import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OrdinalEncoder

# read the datasets
X_train = pd.read_csv("./X_train.csv")
X_test = pd.read_csv("./X_test.csv")
y_train = pd.read_csv("./y_train.csv")
y_test = pd.read_csv("./y_test.csv")

# convert the categorical variables to numeric
X_train["ajcc_pathologic_stage"] = (
    pd.factorize(X_train["ajcc_pathologic_stage"], sort=True)[0] + 1
)
X_train["tissue_or_organ_of_origin"] = (
    pd.factorize(X_train["tissue_or_organ_of_origin"], sort=True)[0] + 1
)
X_train["primary_diagnosis"] = (
    pd.factorize(X_train["primary_diagnosis"], sort=True)[0] + 1
)
X_train["prior_malignancy"] = (
    pd.factorize(X_train["prior_malignancy"], sort=True)[0] + 1
)
X_train["prior_treatment"] = pd.factorize(X_train["prior_treatment"], sort=True)[0] + 1
X_train["treatment_or_therapy"] = (
    pd.factorize(X_train["treatment_or_therapy"], sort=True)[0] + 1
)
X_train["synchronous_malignancy"] = (
    pd.factorize(X_train["synchronous_malignancy"], sort=True)[0] + 1
)
X_test["ajcc_pathologic_stage"] = (
    pd.factorize(X_test["ajcc_pathologic_stage"], sort=True)[0] + 1
)
X_test["tissue_or_organ_of_origin"] = (
    pd.factorize(X_test["tissue_or_organ_of_origin"], sort=True)[0] + 1
)
X_test["primary_diagnosis"] = (
    pd.factorize(X_test["primary_diagnosis"], sort=True)[0] + 1
)
X_test["prior_malignancy"] = pd.factorize(X_test["prior_malignancy"], sort=True)[0] + 1
X_test["prior_treatment"] = pd.factorize(X_test["prior_treatment"], sort=True)[0] + 1
X_test["treatment_or_therapy"] = (
    pd.factorize(X_test["treatment_or_therapy"], sort=True)[0] + 1
)
X_test["synchronous_malignancy"] = (
    pd.factorize(X_test["synchronous_malignancy"], sort=True)[0] + 1
)

# ordinal encoding
tmp = y_train.vital_status.to_numpy().reshape(-1, 1)
encoder = OrdinalEncoder()
y_train_ordinal = encoder.fit_transform(tmp)
y_train_ordinal = pd.DataFrame(y_train_ordinal)
tmp = y_test.vital_status.to_numpy().reshape(-1, 1)
encoder = OrdinalEncoder()
y_test_ordinal = encoder.fit_transform(tmp)
y_test_ordinal = pd.DataFrame(y_test_ordinal)
y_train_ordinal = [
    1 if i == 0.0 else 0 for i in sum(y_train_ordinal.values.tolist(), [])
]
y_test_ordinal = [1 if i == 0.0 else 0 for i in sum(y_test_ordinal.values.tolist(), [])]
y_train_ordinal = pd.DataFrame(y_train_ordinal)
y_test_ordinal = pd.DataFrame(y_test_ordinal)
y_train_ordinal.to_csv("y_train_ordinal.csv", index=False)
y_test_ordinal.to_csv("y_test_ordinal.csv", index=False)

# one-hot encoding
y_train = [1 if i == "Alive" else 0 for i in y_train.vital_status]
y_test = [1 if i == "Alive" else 0 for i in y_test.vital_status]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

# save the cleaned DataFrames as CSV
X_train.to_csv("X_train_clean.csv", index=False)
X_test.to_csv("X_test_clean.csv", index=False)
y_train.to_csv("y_train_clean.csv", index=False)
y_test.to_csv("y_test_clean.csv", index=False)

# getting the str labels back
# y_train.columns = ["Dead", "Alive"]
# y_train = y_train.idxmax(axis=1)
