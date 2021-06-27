import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ******************************************* define some functions ************************************************** #


# returns the indexes of samples in 'df', if his 'feature' is lower than 's'
def lower_than(df, features, s):
    indices = []
    # iterate over features(columns)
    for col in features:
        # Determine a list of indices
        list_col = df[(df[col] < float(s))].index
        # append the found outlier indices for col to the list of outlier indices
        indices.extend(list_col)
    return indices


# ******************************************* load / prepare the data ************************************************ #

path = os.path.abspath('.')

# get the tables of each sheets as DataFrames
data_v1 = pd.read_excel(path+"\\data.xlsx", sheet_name="data_v1")
data_v2 = pd.read_excel(path+"\\data.xlsx", sheet_name="data_v2")
data_v3 = pd.read_excel(path+"\\data.xlsx", sheet_name="data_v3")
data_vd = pd.read_excel(path+"\\data.xlsx", sheet_name="data_vd")

# add a column to mark the test series of the datasets
data_v1["Versuch"] = 1
data_v2["Versuch"] = 2
data_v3["Versuch"] = 3
data_vd["Versuch"] = 4

# combine the sheets to one full dataset
data_full = pd.concat([data_v1, data_v2, data_v3, data_vd], ignore_index=True)

# split the values of differences, just for the calculate
differences = pd.DataFrame(
    data_full,
    columns=[
        "1_Differenz",
        "2_Differenz",
        "3_Differenz",
        "4_Differenz",
        "5_Differenz",
        "6_Differenz",
        "7_Differenz",
        "8_Differenz",
        "9_Differenz",
        "10_Differenz",
    ],
)
# add a column of average/maximal value of all "differences" in each row
# data_full['Abtrag'] = differences.mean(axis=1)    # axis = 0 by default, for rows
data_full["Abtrag"] = differences.max(axis=1)
# add a column of standard deviation of all "differences"
data_full["StdAbw"] = differences.std(axis=1)
# add columns of measurement quality
# data_full['P'] = data_full.apply(lambda x: 1 if x.Abtrag >= 0 else 0, axis=1)
data_full["Qualität"] = np.where(data_full["Abtrag"] >= 0.01, 1, 0)
# delete the original "ID"-column
data_full = data_full.drop(["ID"], axis=1)

# the original data set has 272 rows, 4 of them have missing values, 3 of them have exceptional high "Abtrag"

# drop the outliers
data_full = data_full.drop(labels=[80, 246, 257], axis=0)
# removes the rows with missing values
data_full = data_full.dropna()
# build a new DataFrame, keep the needed columns only and reset the index
data = data_full[
    ["Drehzahl", "Vorschub", "Kraft", "Winkel", "Abtrag", "Qualität"]
].reset_index(drop=True)

# choose the data with effective "Abtrag"

# detect indexes of samples lower than 0.01
ind_lower = lower_than(data, ["Abtrag"], 0.000)  # 0.005!!!!
# drop the negatives, reset the index
data_sel = data.drop(ind_lower, axis=0).reset_index(drop=True)


# *************************************** initialize the input / output ********************************************** #

# split the feature-/label
feature_df = pd.DataFrame(data_sel, columns=["Drehzahl", "Vorschub", "Kraft", "Winkel"])
label_df = pd.DataFrame(data_sel, columns=["Abtrag"])

# normalize the parameters
feature_df["Drehzahl"] = feature_df["Drehzahl"].apply(lambda x: x / 10000)
feature_df["Vorschub"] = feature_df["Vorschub"].apply(lambda x: x / 500)
feature_df["Kraft"] = feature_df["Kraft"].apply(lambda x: x / 50)
feature_df["Winkel"] = feature_df["Winkel"].apply(lambda x: x / 10)

# transform the DataFrame to np.array
Feature = np.array(feature_df)
# Label = np.array(label_df)  # unused
# numpy.ravel(a)returns a contiguous flattened array
Labels = np.ravel(label_df)

# split the train/test data set
X_train, X_test, y_train, y_test = train_test_split(
    Feature, Labels, test_size=None, random_state=0
)  # numpy.ndarray



if __name__ == "__main__":

    print(100 * "=")
    print("Path:", path, "\n")
    print(
        "Der verwendete Datensatz besitzt insgesamt {} Datenpunkte.".format(
            Feature.shape[0]
        ), "\n"
    )
    print("Nach der Aufteilung von Train-/Test-Daten: ")
    print("Abmessung von X_train: {}".format(X_train.shape))
    print("Abmessung von y_train: {}".format(y_train.shape))
    print("Abmessung von X_test: {}".format(X_test.shape))
    print("Abmessung von y_test: {}".format(y_test.shape), "\n")
    print("Beispiele der Features: ")
    print(feature_df.sample(4))
    print("Beispiele der Labels: ")
    print(label_df.sample(4))
    print(100 * "=")
