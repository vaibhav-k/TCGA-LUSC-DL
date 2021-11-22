import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Conv2D,
    Dropout,
    Flatten,
    LeakyReLU,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications.vgg16 import VGG16


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


def grayscale_color(X_train, X_test):
    X_train_rgb = []
    for i in X_train:
        X_train_rgb.append(np.stack((i, i, i), axis=2))
    X_test_rgb = []
    for i in X_test:
        X_test_rgb.append(np.stack((i, i, i), axis=2))
    return np.array(X_train_rgb), np.array(X_test_rgb)


def save_data_images(X_train_rgb, X_test_rgb):
    for ind, i in enumerate(X_train_rgb):
        plt.imsave("Train and test data/X_train_rgb_" + str(ind) + ".jpeg", i)
    for ind, i in enumerate(X_test_rgb):
        plt.imsave("Train and test data/X_test_rgb_" + str(ind) + ".jpeg", i)


def resize_images():
    imgs = glob.glob("**/*.*")
    for i in imgs:
        baseheight = 32
        img = Image.open(i)
        hpercent = baseheight / float(img.size[1])
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, baseheight), Image.ANTIALIAS)
        img.save("Train and test data resized/" + i[20:])


def read_resized_images():
    X_train_rgb_resized = []
    X_test_rgb_resized = []
    for f in glob.glob("Train and test data resized/*.jpeg"):
        if "train" in f:
            X_train_rgb_resized.append(imread(f))
        elif "test" in f:
            X_test_rgb_resized.append(imread(f))
    return np.array(X_train_rgb_resized), np.array(X_test_rgb_resized)


def train_vanilla_model(X, y):
    # define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(X, y):
        model = Sequential()
        model.add(Conv2D(1, kernel_size=2,
                  activation="relu", input_shape=(2, 4, 1)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(1, input_shape=(2, 4, 1)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(BatchNormalization())

        # compile the model
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        # Define Tensorboard as a Keras callback
        tensorboard = TensorBoard(
            log_dir="logs/fit/cnn-vanilla/", histogram_freq=1, write_images=True
        )
        keras_callbacks = [tensorboard]

        # generate a print
        print(
            "------------------------------------------------------------------------"
        )
        print(f"Training for fold {fold_no} ...")

        # fit data to model
        history = model.fit(
            X[train],
            y[train],
            batch_size=32,
            epochs=20,
            verbose=1,
            callbacks=keras_callbacks,
        )

        # generate generalization metrics
        scores = model.evaluate(X[test], y[test], verbose=0)
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

    plot_model(model, to_file="cnn-vanilla-kfold.png", show_shapes=True)


def train_transfer_model(X, y):
    # define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(X, y):
        # loading the VGG16 model
        base_model = VGG16(
            weights="imagenet", include_top=False, input_shape=(32, 64, 3)
        )
        base_model.trainable = False  # not trainable weights
        # defining the model architecture
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.add(LeakyReLU(alpha=0.05))

        # compile the model
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        # Define Tensorboard as a Keras callback
        tensorboard = TensorBoard(
            log_dir="logs/fit/cnn-transfer/", histogram_freq=1, write_images=True
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
            epochs=20,
            verbose=1,
            callbacks=keras_callbacks,
        )

        # Generate generalization metrics
        scores = model.evaluate(X[test], y[test], verbose=0)
        print(
            f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%"
        )
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
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

    plot_model(model, to_file="cnn-transfer-kfold.png", show_shapes=True)


# read the datasets
X_train, X_test, y_train_encoded, y_test_encoded = read_prepare_data()
X_train_rgb_resized, X_test_rgb_resized = read_resized_images()

# join the training and testing data for cross validation
X =  np.vstack((X_train, X_test))
X_rgb_resized = np.vstack((X_train_rgb_resized, X_test_rgb_resized))
y = np.append(y_train_encoded, y_test_encoded)

# convert the grayscale images to RGB for transfer learning
# X_train_rgb, X_test_rgb = grayscale_color(X_train, X_test)

# resize and save the images
# save_data_images(X_train_rgb, X_test_rgb)
# resize_images()

# evaluate the model using the vanilla CNN model
# train_vanilla_model(X, y)

# evaluate the model using transfer learning
train_transfer_model(X_rgb_resized, y)
