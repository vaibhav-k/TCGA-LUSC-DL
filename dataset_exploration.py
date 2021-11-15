import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
from collections import Counter


# read the datasets
X_train = pd.read_csv("./X_train_clean.csv")
X_test = pd.read_csv("./X_test_clean.csv")
y_train = pd.read_csv("./y_train_clean.csv")
y_test = pd.read_csv("./y_test_clean.csv")

print(X_train.columns, "\n")

# check the percentiles of the age at diagnosis
tmp = X_train.age_at_diagnosis.to_numpy()
print(
    np.percentile(tmp, 5) / 365,
    np.percentile(tmp, 25) / 365,
    np.percentile(tmp, 50) / 365,
    np.percentile(tmp, 75) / 365,
    np.percentile(tmp, 95) / 365,
    "\n",
)
# visualize the age at diagnosis
# plt.scatter(X_train.case_id, X_train.age_at_diagnosis / 365)
# plt.title("age_at_diagnosis")
# plt.xlabel("case_id")
# plt.ylabel("age_at_diagnosis")
# plt.show()

# find the number of cases for each pathologic stage
tmp = []
for i in X_train.ajcc_pathologic_stage:
    if i == 2:
        tmp.append("Stage I")
    if i == 3:
        tmp.append("Stage IA")
    if i == 4:
        tmp.append("Stage IB")
    if i == 5:
        tmp.append("Stage II")
    if i == 6:
        tmp.append("Stage IIA")
    if i == 7:
        tmp.append("Stage IIB")
    if i == 8:
        tmp.append("Stage III")
    if i == 9:
        tmp.append("Stage IIIA")
    if i == 10:
        tmp.append("Stage IIIB")
print(dict(Counter(tmp)), "\n")

# find the number of cases for each primary diagnosis
tmp = []
for i in X_train.primary_diagnosis:
    if i == 1:
        tmp.append("Basaloid squamous cell carcinoma")
    if i == 2:
        tmp.append("Papillary squamous cell carcinoma")
    if i == 3:
        tmp.append("Squamous cell carcinoma, NOS")
    if i == 4:
        tmp.append("Squamous cell carcinoma, keratinizing, NOS")
print(dict(Counter(tmp)), "\n")

# find the number of cases for each prior malignancy
tmp = []
for i in X_train.prior_malignancy:
    if i == 1:
        tmp.append("No")
    if i == 2:
        tmp.append("Yes")
print(dict(Counter(tmp)), "\n")

# find the number of cases for each prior treatment
tmp = []
for i in X_train.prior_treatment:
    if i == 1:
        tmp.append("No")
    if i == 2:
        tmp.append("Yes")
print(dict(Counter(tmp)), "\n")

# find the number of cases for each treatment or therapy
tmp = []
for i in X_train.treatment_or_therapy:
    if i == 1:
        tmp.append("No")
    if i == 2:
        tmp.append("Yes")
print(dict(Counter(tmp)), "\n")

# find the number of cases for each tissue or organ of origin
tmp = []
for i in X_train.tissue_or_organ_of_origin:
    if i == 1:
        tmp.append("Lower lobe, lung")
    if i == 2:
        tmp.append("Lung, NOS")
    if i == 3:
        tmp.append("Main bronchus")
    if i == 4:
        tmp.append("Middle lobe, lung")
    if i == 3:
        tmp.append("Overlapping lesion of lung")
    if i == 4:
        tmp.append("Upper lobe, lung")
print(dict(Counter(tmp)), "\n")

# find the number of cases for each synchronous malignancy
tmp = []
for i in X_train.synchronous_malignancy:
    if i == 1:
        tmp.append("No")
    if i == 2:
        tmp.append("Yes")
print(dict(Counter(tmp)), "\n")
