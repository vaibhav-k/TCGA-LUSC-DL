import csv
import pandas as pd
from sklearn.model_selection import train_test_split


# function to convert age in days to age in years
def read_tsv(file_read):
    # read the file
    clinical = []
    with open(file_read) as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for line in tsv_file:
            clinical.append(line)
    # convert the list to a DataFrame
    clinical_df = pd.DataFrame.from_records(clinical)
    return clinical_df


# function to clean-up the DataFrame read
def clean_df(df):
    # set the column headers
    df.columns = df.iloc[0]
    df = df.drop(0)
    # remove the columns with only one value repeated
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)
    # extract the columns with some biological value
    df = df[
        df.columns
        & [
            "case_id",
            "vital_status",
            "age_at_diagnosis",
            "ajcc_pathologic_stage",
            "primary_diagnosis",
            "prior_malignancy",
            "prior_treatment",
            "synchronous_malignancy",
            "tissue_or_organ_of_origin",
            "treatment_or_therapy",
        ]
    ]
    # remove the duplicate rows from the DataFrame
    df = df.drop_duplicates()
    df = df.set_index("case_id")
    # convert the age from days to years
    age_at_diagnosis_yrs = []
    for i in df["age_at_diagnosis"]:
        age_at_diagnosis_yrs.append(days_to_years(i))
    df["age_at_diagnosis"] = age_at_diagnosis_yrs
    # filter the DataFrame
    df = df.drop(df[df.prior_malignancy == "not reported"].index)
    df = df.drop(df[df.synchronous_malignancy == "Not Reported"].index)
    df = df.drop(df[df.treatment_or_therapy == "not reported"].index)
    df = df[df["age_at_diagnosis"].notna()]
    return df


# function to convert age in days to age in years
def days_to_years(num_days):
    if num_days != "'--":
        num_days = int(num_days)
        yrs = num_days // 365
        mnths = (num_days - yrs * 365) // 30
        days = num_days - yrs * 365 - mnths * 30
        return yrs, mnths, days


# function to save the formatted DataFrame in a CSV file
def save_df_csv(df, file_save):
    df.to_csv(file_save)


def read_saved_csv(file):
    # read the CSV files
    df = pd.read_csv(file)
    df = df.set_index("case_id")
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv")
    y_test = pd.read_csv("y_test.csv")
    # set the indices
    X_train = X_train.set_index("case_id")
    X_test = X_test.set_index("case_id")
    y_train.index = X_train.index
    y_train = y_train.drop("case_id", axis=1)
    y_test.index = X_test.index
    y_test = y_test.drop("case_id", axis=1)
    return df, X_train, X_test, y_train, y_test


def train_test_split_df(df):
    # shuffle the df
    df = df.sample(frac=1)
    # split and return the datasets
    X = df[
        [
            "age_at_diagnosis",
            "ajcc_pathologic_stage",
            "primary_diagnosis",
            "prior_malignancy",
            "prior_treatment",
            "synchronous_malignancy",
            "tissue_or_organ_of_origin",
            "treatment_or_therapy",
        ]
    ]
    y = df["vital_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# function to save the datasets for later use
def save_datasets(X_train, X_test, y_train, y_test):
    X_train.to_csv("X_train.csv")
    X_test.to_csv("X_test.csv")
    y_train.to_csv("y_train.csv")
    y_test.to_csv("y_test.csv")


# function to measure the baseline
def baseline(y_test):
    y_pred = ["Alive"] * len(y_test)
    print(
        len(set(y_test["vital_status"]) & set(y_pred))
        / float(len(set(y_test["vital_status"]) | set(y_pred)))
        * 100
    )


def filter_age(age):
    age = [x for x in age if x != "("]
    age = [x for x in age if x != ")"]
    age = [x for x in age if x != " "]
    q = [365, 12]
    age_days = 0
    tmp = ""
    for i in age:
        if i.isdigit():
            tmp += i
        else:
            age_days += int(tmp) * q.pop(0)
            tmp = ""
    age_days += int(tmp)
    return age_days


def main():
    # df = read_tsv("clinical.tsv")
    # df = clean_df(df)
    # save_df_csv(df, "clinical_df.csv")
    # X_train, X_test, y_train, y_test = train_test_split_df(df)
    # save_datasets(X_train, X_test, y_train, y_test)
    df, X_train, X_test, y_train, y_test = read_saved_csv("clinical_df.csv")
    # baseline(y_test)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(X_train.columns)


if __name__ == "__main__":
    main()
