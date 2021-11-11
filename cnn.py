import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

warnings.filterwarnings("ignore")
np.random.seed(seed=42)


# read the datasets
X_train = pd.read_csv("./X_train_clean.csv")
X_test = pd.read_csv("./X_test_clean.csv")
y_train = pd.read_csv("./y_train_clean.csv")
y_test = pd.read_csv("./y_test_clean.csv")

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
history = model.fit(
    X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test)
)

# make the predictions
y_pred = model.predict(X_test)
y_pred = np.round_(y_pred)

# get the metrics
target_names = ["Dead", "Alive"]
print(classification_report(y_test, y_pred, target_names=target_names))

# plot the confusion matrix
conf_mat = confusion_matrix(y_test.to_numpy().argmax(axis=1), y_pred.argmax(axis=1))
print("The confusion matrix for this model is:\n", conf_mat)

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
