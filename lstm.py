import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    LeakyReLU,
    LSTM,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


def read_prepare_data():
    # read the datasets
    X_train = pd.read_csv("./X_train_clean.csv")
    X_test = pd.read_csv("./X_test_clean.csv")
    y_train = pd.read_csv("./y_train.csv")
    y_test = pd.read_csv("./y_test.csv")

    # reshape the datasets for CNN
    X_train.drop("case_id", inplace=True, axis=1)
    X_train = X_train.to_numpy()
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
    #     X_train.shape
    # )
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] // 2, 2)
    X_test.drop("case_id", inplace=True, axis=1)
    X_test = X_test.to_numpy()
    # X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
    #     X_test.shape
    # )
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] // 2, 2)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train.vital_status)
    y_train_encoded = encoder.transform(y_train.vital_status)
    encoder = LabelEncoder()
    encoder.fit(y_test.vital_status)
    y_test_encoded = encoder.transform(y_test.vital_status)
    return X_train, X_test, y_train_encoded, y_test_encoded


def train_model(X, y):
    # define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(X, y):
        # define the model architecture
        model = Sequential()
        model.add(
            LSTM(1, input_shape=(X.shape[1], 2),
                 dropout=0.4, return_sequences=True)
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(LSTM(1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))

        # compile the model
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        # define Tensorboard as a Keras callback
        tensorboard = TensorBoard(
            log_dir="logs/fit/lstm/", histogram_freq=1, write_images=True
        )
        keras_callbacks = [tensorboard]

        # generate a print
        print(
            "------------------------------------------------------------------------"
        )
        print(f"Training for fold {fold_no} ...")

        # Fit data to model
        history = model.fit(
            X[train],
            y[train],
            batch_size=32,
            epochs=10,
            verbose=1,
            callbacks=keras_callbacks,
        )

        # get the incorrect predictions made by the model
        predictions = model.predict(X[test])
        indices = [i for i, v in enumerate(
            predictions) if predictions[i] != y[test][i]]
        subset_of_wrongly_predicted = [X_test[i] for i in indices]
        print(subset_of_wrongly_predicted)

        # generate generalization metrics
        try:
            scores = model.evaluate(X[test], y[test], verbose=0)
            print(
                f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%"
            )
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
        except IndexError:
            pass

        # increase fold number
        fold_no += 1

    # provide the average scores
    print("\n------------------------------------------------------------------------")
    print("Score per fold")
    for i in range(0, len(acc_per_fold)):
        print(
            "------------------------------------------------------------------------"
        )
        print(
            f"> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%")
    print("------------------------------------------------------------------------")
    print("Average scores for all folds:")
    print(f"> Accuracy: {np.mean(acc_per_fold)}% (+- {np.std(acc_per_fold)}%)")
    print(f"> Loss: {np.mean(loss_per_fold)}")
    print("------------------------------------------------------------------------")

    plot_model(model, to_file="lstm-kfold.png", show_shapes=True)


def train_time_distributed(X, y):
    X = np.array([i.flatten().reshape(
        1, X[0].shape[0] * X[0].shape[1], 1) for i in X])
    X = X.reshape(1, len(X) * X.shape[2], 1)
    y = y.reshape(1, y.shape[0], 1)
    y = np.resize(y, (1, X.shape[1], 1))

    # define the model architecture
    model = Sequential()
    model.add(LSTM(8, input_shape=(
        X.shape[1], 1), dropout=0.4, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation="softmax")))

    # compile the model
    model.compile(loss="binary_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    # define Tensorboard as a Keras callback
    tensorboard = TensorBoard(
        log_dir="logs/fit/lstm-time-distributed/", histogram_freq=1, write_images=True
    )
    keras_callbacks = [tensorboard]

    # fit data to model
    history = model.fit(
        X, y, batch_size=32, epochs=10, verbose=1, callbacks=keras_callbacks,
    )

    # generate generalization metrics
    scores = model.evaluate(X, y, verbose=0)
    print(
        f"The {model.metrics_names[0]} is {scores[0]}.\nThe {model.metrics_names[1]} is {scores[1] * 100}%"
    )


# read the datasets
X_train, X_test, y_train_encoded, y_test_encoded = read_prepare_data()

# join the training and testing data for cross validation
X = np.vstack((X_train, X_test))
y = np.append(y_train_encoded, y_test_encoded)

# evaluate the model with standardized dataset
train_model(X, y)
# train_time_distributed(X, y)
