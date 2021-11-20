import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical


def read_prepare_data():
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
    return X_train, X_test, y_train_encoded, y_test_encoded


# configuration options
feature_vector_length = 8
num_classes = 2

# read the datasets
X_train, X_test, y_train_encoded, y_test_encoded = read_prepare_data()

# reshape the data - MLPs do not understand such things as '2D'.
# reshape to 2 x 4 pixels = 8 features
X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
X_test = X_test.reshape(X_test.shape[0], feature_vector_length)
print(X_train[0])

# Convert target classes to categorical ones
y_train = to_categorical(y_train_encoded, num_classes)
y_test = to_categorical(y_test_encoded, num_classes)
print(y_train[0])

# Set the input shape
input_shape = (feature_vector_length,)
print(f"Feature shape: {input_shape}")

# Create the model
model = Sequential()
model.add(Dense(10, input_shape=input_shape, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

# Configure the model and start training
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
history = model.fit(
    X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2
)

# Test the model after training
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f"Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%")
