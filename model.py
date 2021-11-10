# Importing core libraries
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

# suppressing warnings because of skopt verbosity
import warnings

warnings.filterwarnings("ignore")

# model selection
from sklearn.model_selection import StratifiedKFold

# metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import make_scorer

# deep learning network
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Reshape,
    GlobalAveragePooling1D,
    Flatten,
    concatenate,
    Concatenate,
    Lambda,
    Dropout,
    SpatialDropout1D,
    Reshape,
    MaxPooling1D,
    BatchNormalization,
    AveragePooling1D,
    Conv1D,
    Activation,
    LeakyReLU,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.metrics import binary_crossentropy
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


# deep learning data manipulation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
    LabelEncoder,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.compose import ColumnTransformer

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
X_train = tmp.reshape(402, 2, 4, 1)
X_test.drop("case_id", inplace=True, axis=1)
tmp = np.array(X_test)
tmp = tmp.reshape(-1, 2, 4)
X_test = pd.DataFrame(sum(map(list, tmp), []))
tmp = []
for i, g in X_test.groupby(np.arange(len(X_test)) // 2):
    tmp.append(g)
tmp = np.array([i.to_numpy() for i in tmp])
X_test = tmp.reshape(101, 2, 4, 1)

# sanity check printing as an image
# plt.imshow(X_test[0])
# plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# create model
model = Sequential()
# add model layers
model.add(Conv2D(64, kernel_size=2, activation="relu", input_shape=(2, 4, 1)))
print(model.summary())
# model.add(Flatten())
# model.add(Dense(2, activation="softmax"))

# # compile model using accuracy to measure model performance
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# # predict first 4 images in the test set
# print(model.predict(X_test[:4]))

# # actual results for first 4 images in test set
# print(y_test[:4])

# for ind, i in enumerate(tmp):
#     imageio.imwrite(str(ind) + ".jpeg", i)

# df = pd.DataFrame([list(l) for l in tmp]).stack().apply(pd.Series).reset_index(1, drop=True)
# print(df.iloc[0])

# # print(y_train.set_index([pd.Index([i for i in range(len(y_train))])]))

# # counting unique values of categorical variables
# categorical_vars = [
#     "ajcc_pathologic_stage",
#     "primary_diagnosis",
#     "prior_malignancy",
#     "prior_treatment",
#     "synchronous_malignancy",
#     "tissue_or_organ_of_origin",
#     "treatment_or_therapy",
# ]
# print(
#     "Count of the unique values in the categorical columns",
#     X_train[categorical_vars].nunique(),
# )

# # describing numeric variables
# numerical_vars = ["age_at_diagnosis"]
# print(X_train[numerical_vars].describe())

# # parametric architecture
# def tabular_dnn(
#     numeric_variables,
#     categorical_variables,
#     categorical_counts,
#     feature_selection_dropout=0.2,
#     categorical_dropout=0.1,
#     first_dense=256,
#     second_dense=256,
#     dense_dropout=0.2,
# ):

#     numerical_inputs = Input(shape=(len(numeric_variables),))
#     numerical_normalization = BatchNormalization()(numerical_inputs)
#     numerical_feature_selection = Dropout(feature_selection_dropout)(
#         numerical_normalization
#     )

#     categorical_inputs = []
#     categorical_embeddings = []
#     for category in categorical_variables:
#         categorical_inputs.append(Input(shape=[1], name=category))
#         category_counts = categorical_counts[category]
#         categorical_embeddings.append(
#             Embedding(
#                 category_counts + 1,
#                 int(np.log1p(category_counts) + 1),
#                 name=category + "_embed",
#             )(categorical_inputs[-1])
#         )

#     categorical_logits = Concatenate(name="categorical_conc")(
#         [
#             Flatten()(SpatialDropout1D(categorical_dropout)(cat_emb))
#             for cat_emb in categorical_embeddings
#         ]
#     )

#     x = concatenate([numerical_feature_selection, categorical_logits])
#     x = Dense(first_dense, activation="relu")(x)
#     x = Dropout(dense_dropout)(x)
#     x = Dense(second_dense, activation="relu")(x)
#     x = Dropout(dense_dropout)(x)
#     output = Dense(1, activation="sigmoid")(x)
#     model = Model([numerical_inputs] + categorical_inputs, output)

#     return model


# # defining some useful functions

# from tensorflow.keras.metrics import AUC


# def mAP(y_true, y_pred):
#     return tf.py_function(average_precision_score, (y_true, y_pred), tf.double)


# def compile_model(model, loss, metrics, optimizer):
#     model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
#     return model


# def plot_keras_history(history, measures):
#     rows = len(measures) // 2 + len(measures) % 2
#     fig, panels = plt.subplots(rows, 2, figsize=(15, 5))
#     plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0.4, wspace=0.2)
#     try:
#         panels = [item for sublist in panels for item in sublist]
#     except:
#         pass
#     for k, measure in enumerate(measures):
#         panel = panels[k]
#         panel.set_title(measure + " history")
#         panel.plot(history.epoch, history.history[measure], label="Train " + measure)
#         panel.plot(
#             history.epoch,
#             history.history["val_" + measure],
#             label="Validation " + measure,
#         )
#         panel.set(xlabel="epochs", ylabel=measure)
#         panel.legend()

#     plt.show(fig)


# # global training settings
# SEED = 42
# FOLDS = 5
# BATCH_SIZE = 32

# # defining callbacks
# measure_to_monitor = "val_auc"
# modality = "max"

# early_stopping = EarlyStopping(
#     monitor=measure_to_monitor, mode=modality, patience=5, verbose=0
# )

# model_checkpoint = ModelCheckpoint(
#     "best.model",
#     monitor=measure_to_monitor,
#     mode=modality,
#     save_best_only=True,
#     verbose=0,
# )

# # setting the CV strategy
# skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# # CV Iteration
# roc_auc = list()
# average_precision = list()
# oof = np.zeros(len(X_train))
# best_iteration = list()

# y_train.vital_status = pd.Categorical(pd.factorize(y_train.vital_status)[0])
# num_classes = len(np.unique(y_train["vital_status"]))
# y_train_categorical = tf.keras.utils.to_categorical(
#     y_train["vital_status"], num_classes
# )


# class DataGenerator(tf.keras.utils.Sequence):
#     """
#     Generates data for Keras
#     X: a pandas DataFrame
#     y: a pandas Series, a NumPy array or a List
#     """

#     def __init__(
#         self,
#         X,
#         y,
#         tabular_transformer=None,
#         batch_size=32,
#         shuffle=False,
#         dict_output=False,
#     ):

#         "Initialization"
#         self.X = X

#         try:
#             # If a pandas Series, converting to a NumPy array
#             self.y = y.values
#         except:
#             self.y = np.array(y)
#         self.tbt = tabular_transformer
#         self.tabular_transformer = tabular_transformer
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.dict_output = dict_output
#         self.indexes = self._build_index()
#         self.on_epoch_end()
#         self.item = 0

#     def _build_index(self):
#         """
#         Builds an index from data
#         """
#         return np.arange(len(self.y))

#     def on_epoch_end(self):
#         """
#         At the end of every epoch, shuffle if required
#         """
#         if self.shuffle:
#             np.random.shuffle(self.indexes)

#     def __len__(self):
#         """
#         Returns the number of batches per epoch
#         """
#         return int(len(self.indexes) / self.batch_size) + 1

#     def __iter__(self):
#         """
#         returns an iterable
#         """
#         for i in range(self.__len__()):
#             self.item = i
#             yield self.__getitem__(index=i)

#         self.item = 0

#     def __next__(self):
#         return self.__getitem__(index=self.item)

#     def __call__(self):
#         return self.__iter__()

#     def __data_generation(self, selection):
#         if self.tbt is not None:
#             if self.dict_output:
#                 dct = {
#                     "input_" + str(j): arr
#                     for j, arr in enumerate(
#                         self.tbt.transform(self.X.iloc[selection, :])
#                     )
#                 }
#                 return dct, self.y[selection]
#             else:
#                 return self.tbt.transform(self.X.iloc[selection, :]), self.y[selection]
#         else:
#             return self.X.iloc[selection, :], self.y[selection]

#     def __getitem__(self, index):
#         indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
#         samples, labels = self.__data_generation(indexes)
#         return samples, labels


# class TabularTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, numeric=list(), ordinal=list(), lowcat=list(), highcat=list()):

#         self.numeric = numeric
#         self.ordinal = ordinal
#         self.lowcat = lowcat
#         self.highcat = highcat

#         self.mvi = multivariate_imputer = IterativeImputer(
#             estimator=ExtraTreesRegressor(n_estimators=300, n_jobs=-2),
#             initial_strategy="median",
#         )

#         self.uni = univariate_imputer = SimpleImputer(
#             strategy="median", add_indicator=True
#         )

#         self.nmt = numeric_transformer = Pipeline(
#             steps=[
#                 (
#                     "normalizer",
#                     QuantileTransformer(
#                         n_quantiles=600, output_distribution="normal", random_state=42
#                     ),
#                 ),
#                 ("imputer", univariate_imputer),
#                 ("scaler", StandardScaler()),
#             ]
#         )

#         self.ohe = generic_categorical_transformer = Pipeline(
#             steps=[
#                 ("string_converter", ToString()),
#                 ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
#                 ("onehot", OneHotEncoder(handle_unknown="ignore")),
#             ]
#         )

#         self.lle = label_enc_transformer = Pipeline(
#             steps=[("string_converter", ToString()), ("label_encoder", LEncoder())]
#         )

#         self.ppl = ColumnTransformer(
#             transformers=[
#                 ("num", numeric_transformer, self.numeric + self.ordinal),
#                 ("ohe", generic_categorical_transformer, self.lowcat + self.ordinal),
#             ],
#             remainder="drop",
#         )

#     def fit(self, X, y=None, **fit_params):
#         _ = self.ppl.fit(X)
#         if len(self.highcat) > 0:
#             _ = self.lle.fit(X[self.highcat])
#         return self

#     def shape(self, X, y=None, **fit_params):
#         numeric_shape = self.ppl.transform(X.iloc[[0], :]).shape[1]
#         categorical_size = self.lle.named_steps["label_encoder"].dictionary_size
#         return [numeric_shape] + categorical_size

#     def transform(self, X, y=None, **fit_params):
#         Xn = self.ppl.transform(X)
#         if len(self.highcat) > 0:
#             return [Xn] + self.lle.transform(X[self.highcat])
#         else:
#             return Xn

#     def fit_transform(self, X, y=None, **fit_params):
#         self.fit(X, y, **fit_params)
#         return self.transform(X)


# class LEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.encoders = dict()
#         self.dictionary_size = list()
#         self.unk = -1

#     def fit(self, X, y=None, **fit_params):
#         for col in range(X.shape[1]):
#             le = LabelEncoder()
#             le.fit(X.iloc[:, col].fillna("_nan"))
#             le_dict = dict(zip(le.classes_, le.transform(le.classes_)))

#             if "_nan" not in le_dict:
#                 max_value = max(le_dict.values())
#                 le_dict["_nan"] = max_value

#             max_value = max(le_dict.values())
#             le_dict["_unk"] = max_value

#             self.unk = max_value
#             self.dictionary_size.append(len(le_dict))
#             col_name = X.columns[col]
#             self.encoders[col_name] = le_dict

#         return self

#     def transform(self, X, y=None, **fit_params):
#         output = list()
#         for col in range(X.shape[1]):
#             col_name = X.columns[col]
#             le_dict = self.encoders[col_name]
#             emb = (
#                 X.iloc[:, col]
#                 .fillna("_nan")
#                 .apply(lambda x: le_dict.get(x, le_dict["_unk"]))
#                 .values
#             )
#             output.append(emb)
#         return output

#     def fit_transform(self, X, y=None, **fit_params):
#         self.fit(X, y, **fit_params)
#         return self.transform(X)


# class ToString(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None, **fit_params):
#         return self

#     def transform(self, X, y=None, **fit_params):
#         return X.astype(str)

#     def fit_transform(self, X, y=None, **fit_params):
#         self.fit(X, y, **fit_params)
#         return self.transform(X)


# for fold, (train_idx, test_idx) in enumerate(
#     skf.split(X_train, y_train_categorical.argmax(1))
# ):
#     tb = TabularTransformer(
#         numeric=numerical_vars, ordinal=[], lowcat=[], highcat=categorical_vars,
#     )

#     tb.fit(X_train.iloc[train_idx])
#     sizes = tb.shape(X_train.iloc[train_idx])
#     categorical_levels = dict(zip(categorical_vars, sizes[1:]))
#     print(f"Input array sizes: {sizes}")
#     print(f"Categorical levels: {categorical_levels}\n")

#     model = tabular_dnn(
#         numerical_vars,
#         categorical_vars,
#         categorical_levels,
#         feature_selection_dropout=0.1,
#         categorical_dropout=0.1,
#         first_dense=256,
#         second_dense=256,
#         dense_dropout=0.1,
#     )

#     model = compile_model(
#         model, binary_crossentropy, [AUC(name="auc"), mAP], Adam(learning_rate=0.0001),
#     )

#     # train_batch = DataGenerator(
#     #     X_train.iloc[train_idx],
#     #     y_train[train_idx],
#     #     tabular_transformer=tb,
#     #     batch_size=BATCH_SIZE,
#     #     shuffle=True,
#     # )

#     history = model.fit(
#         X_train,
#         y_train,
#         # validation_data=(tb.transform(X_train.iloc[test_idx]), y_train[test_idx]),
#         epochs=30,
#         batch_size=64,
#         callbacks=[model_checkpoint, early_stopping],
#         # class_weight={0: 1.0, 1: (np.sum(y == 0) / np.sum(y == 1))},
#         verbose=1,
#     )

#     # print("\nFOLD %i" % fold)
#     # plot_keras_history(history, measures=["auc", "loss"])

#     # best_iteration.append(np.argmax(history.history["val_auc"]) + 1)
#     # preds = model.predict(
#     #     tb.transform(X_train.iloc[test_idx]), verbose=1, batch_size=1024
#     # ).flatten()

#     # oof[test_idx] = preds

#     # roc_auc.append(roc_auc_score(y_true=y_train[test_idx], y_score=preds))
#     # average_precision.append(
#     #     average_precision_score(y_true=y_train[test_idx], y_score=preds)
#     # )
