import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization, Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical


def read_prepare_data(feature_vector_length):
    # read the datasets
    X_train = pd.read_csv("./X_train_clean.csv")
    X_test = pd.read_csv("./X_test_clean.csv")
    y_train = pd.read_csv("./y_train.csv")
    y_test = pd.read_csv("./y_test.csv")
    print(X_train.columns)

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

    # # normalize the test and train data
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
    #     X_train.shape
    # )
    # X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
    #     X_test.shape
    # )

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

    # convert target classes to categorical ones
    y_train = to_categorical(y_train_encoded, num_classes)
    y_test = to_categorical(y_test_encoded, num_classes)

    return X_train, X_test, y_train_encoded, y_test_encoded


def train_kfold_model(X_train, X_test, y_train, y_test, input_shape):
    # define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # create the datasets and initialize the K-fold split
    X = np.vstack((X_train, X_test))
    y = np.append(y_train, y_test)
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = Sequential()
        model.add(Dense(10, input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.05))
        model.add(BatchNormalization())
        model.add(Dense(5))
        model.add(LeakyReLU(alpha=0.05))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))

        # compile the model
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        # Define Tensorboard as a Keras callback
        tensorboard = TensorBoard(
            log_dir="logs/fit/mlp/", histogram_freq=1, write_images=True
        )
        keras_callbacks = [tensorboard]

        # generate a print
        print(
            "------------------------------------------------------------------------"
        )
        print(f"Training for fold {fold_no} ...")

        # fit the data to model
        history = model.fit(
            X_train,
            y_train,
            batch_size=32,
            epochs=30,
            verbose=1,
            callbacks=keras_callbacks,
        )

        # get the incorrect predictions made by the model
        predictions = model.predict(X_test)
        indices = [i for i, v in enumerate(
            predictions) if predictions[i] != y_test[i]]
        subset_of_wrongly_predicted = [X_test[i] for i in indices]
        print(subset_of_wrongly_predicted)

        # generate generalization metrics
        scores = model.evaluate(X_test, y_test, verbose=0)
        print(
            f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%"
        )
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

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
    print(f"> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})")
    print(f"> Loss: {np.mean(loss_per_fold)}")
    print("------------------------------------------------------------------------")

    plot_model(model, to_file="mlp-kfold.png", show_shapes=True)


# configuration options
feature_vector_length = 8
num_classes = 2

# read the datasets
X_train, X_test, y_train, y_test = read_prepare_data(feature_vector_length)

# set the input shape
input_shape = (feature_vector_length,)

# test the model after training
train_kfold_model(X_train, X_test, y_train, y_test, input_shape)
