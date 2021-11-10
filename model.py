import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")


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
# X_train = X_train.reshape(len(X_train), 2, 4)
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
# X_test = X_test.reshape(len(X_test), 2, 4)

# take only the relevant column from the target variabel DataFrame
# convert it to a numerical value
y_train = [1 if i == "Alive" else 0 for i in y_train.vital_status]
y_test = [1 if i == "Alive" else 0 for i in y_test.vital_status]

# one-hot encode the target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# sanity check printing the data as an image
# plt.imshow(X_test[0])
# plt.show()

# create model
model = Sequential()

# add model layers
model.add(Conv2D(64, kernel_size=2, activation="relu", input_shape=(2, 4, 1)))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))
print(model.summary())

# compile model using accuracy to measure model performance
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

# make the predictions
y_pred = model.predict(X_test)
y_pred = np.round_(y_pred)

# get the metrics
target_names = ["Dead", "Alive"]
print(classification_report(y_test, y_pred, target_names=target_names))

# the metrics measured by the model's training
print(history.history.keys())

# plot some of the above-mentioned metrics
# accuracy
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model's accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")
plt.show()
# loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model's loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")
plt.show()

# predict first 4 patients in the test set
print("Predicted the first 4 patients in the test set:\n", y_pred[:4])

# actual results for first 4 patients in test set
print("The actual results for the first 4 patients in test set:\n", y_test[:4])

# plot the confusion matrix
conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("The confusion matrix for this model is:\n", conf_mat)
