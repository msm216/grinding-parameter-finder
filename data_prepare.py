import os
import itertools

import pandas as pd
import numpy as np
import pymysql.cursors
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ******************************************* define some functions ************************************************** #


local_file_name = "data.xls"

# MySQL to DataFrame
def read_from_mysql():
    # 连接数据库
    connection = pymysql.connect(
        host="localhost",  # host='127.0.0.1',
        port=3306,
        user="root",
        passwd="12345",
        db="grinding_parameter",
        charset="utf8",
    )
    try:
        # 获取会话指针
        with connection.cursor() as cursor:
            # 筛选语句
            # sql_select = """SELECT column_name,column_name FROM raw_data WHERE column_name IS NOT NULL"""
            # 全选语句
            sql_select = """SELECT * FROM data"""
            # 执行查询语句
            cursor.execute(sql_select)
            # 查询所有数据
            result = cursor.fetchall()  # tuple
            # 关闭指针
            cursor.close()
    finally:
        # 关闭数据库连接
        connection.close()
    # 转换为DataFrame
    df = pd.DataFrame(
        result,
        columns=[
            "ID",
            "Drehzahl",
            "Vorschub",
            "Kraft",
            "Winkel",
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
            "submission_date",
        ],
    )
    # date of last update
    date = df.loc[df.index[-1], "submission_date"]
    # delete the "date"-column
    df = df.drop(["submission_date"], axis=1)
    # delete the original "ID"-column
    df = df.drop(["ID"], axis=1)

    return df, date


# Excel to DataFrame
def read_local_excel(fn=local_file_name):
    # path
    file = os.getcwd() + "\\" + fn
    # 初始化空DataFrame
    df = pd.DataFrame()
    # 读取全表（直接dict中获取各个表？？？）
    reader = pd.read_excel(file, sheet_name=None)  # dict of DataFrame
    # 生成sheets表
    sheets = []
    for k in reader.keys():
        sheets.append(k)
    # 循环 sheets
    for sheet in sheets:
        # 提取 sheet 为 DataFrame
        df_sheet = pd.read_excel(file, sheet_name=sheet)
        # 删除含空值的行
        df_sheet.dropna(axis=0, how="any", inplace=True)
        # 删除原生索引列
        df_sheet = df_sheet.drop(["ID"], axis=1)
        # 添加到df
        df = pd.concat([df, df_sheet], ignore_index=True)

    return df


# choose data source
def choose_source(xls_name=local_file_name):
    my_input = input("Use data from MySQL? [y/n]: ")
    if len(my_input) == 0:
        df = read_local_excel()
        c = False
        print("Loading data from excel file: ", xls_name)
    elif "y" in my_input:
        df, date = read_from_mysql()
        c = True
        print("Loading data from MySQL...")
        print("Last update on:", date)
    else:
        df = read_local_excel()
        c = False
        print("Using data from excel file: ", xls_name)
    return df, c


# outlier detection using standard deviation
def detect_outliers_sd(df, features):
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # mean
        mean = df[col].mean()
        # standard deviation
        std = df[col].std()
        # the upper bound
        top = mean + std * 1.96
        #  the lower bound
        bot = mean - std * 1.96
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < bot) | (df[col] > top)].index
        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)
    return outlier_indices


# outlier detection using interquartile ranges
def detect_outliers_iqr(df, features):
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        iqr = q3 - q1
        # outlier step (extrem)
        outlier_step = 1.5 * iqr
        # the upper bound
        top = q3 + outlier_step
        #  the lower bound
        bot = q1 - outlier_step
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < bot) | (df[col] > top)].index
        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)
    return outlier_indices


# outlier detection by each feature combination
def outliers_by_feature(df, method="iqr"):
    features = ["Drehzahl", "Vorschub", "Kraft", "Winkel"]
    candi = []
    combi = []
    outlier_ind = []
    # list of candidates (list of 4 lists)
    for col in features:
        lis = list(set(df[col]))
        candi.append(lis)
    # list of all possible feature combinations (lists)
    for c in itertools.product(*candi):
        combi.append(c)
    # iterate over all possible feature combinations
    for ser in combi:
        ol_ind = []
        cutout = df.loc[
            (df["Drehzahl"] == ser[0])
            & (df["Vorschub"] == ser[1])
            & (df["Kraft"] == ser[2])
            & (df["Winkel"] == ser[3])
        ]
        # get index of outliers for this feature combination
        if len(cutout) != 0:
            if method == "iqr":
                ol_ind = detect_outliers_iqr(cutout, ["Abtrag"])
            if method == "sd":
                ol_ind = detect_outliers_sd(cutout, ["Abtrag"])
        # add result to list
        outlier_ind.extend(ol_ind)
    return outlier_ind


# *********************************************** load  the data **************************************************** #

# choose data source
print(80 * "=")
(
    data_load,
    sql_contr,
) = choose_source()  # <----------------------------------- controller
print("Length of loaded data: ", len(data_load))
print(data_load.sample(5))

# ********************************************* prepare the data ***************************************************** #

# split the values of differences, just for the calculate
differences = data_load[
    [
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
    ]
]
# add a column of average/maximal value of all "differences" in each row
data_load["Abtrag"] = differences.max(axis=1)
# data_load['Abtrag'] = differences.mean(axis=1)
# add a column of standard deviation
data_load["StdAbw"] = differences.std(axis=1)
# add columns of measurement quality
data_load["Qualitaet"] = np.where(data_load["Abtrag"] > 0, 1, 0)
# data_load['P'] = data_full.apply(lambda x: 1 if x.Abtrag >= 0 else 0, axis=1)

# ********************************************* clean the data ******************************************************* #

# the original dataset has 268 rows (4 with missing values dropped)
# keep the needed columns only, reset the index
data_sel = data_load[["Drehzahl", "Vorschub", "Kraft", "Winkel", "Abtrag"]]
print("Feature selected.")

# detect indexes of outliers overall [80, 246, 257]
outlier_ind_oa = detect_outliers_sd(data_sel, ["Abtrag"])
# detect indexes of outliers by features [54, 129, 212, 246, 248, 251, 257]
outlier_ind = outliers_by_feature(data_sel, "iqr")
# combine the index [54, 80, 129, 212, 246, 248, 251, 257]
outlier_ind.extend(outlier_ind_oa)
outlier_ind = list(set(outlier_ind))
outlier_ind.sort()
# drop the outliers
data_sel = data_sel.drop(outlier_ind, axis=0)
print("{} outliers dropped.".format(len(outlier_ind)))
# drop the negative data and reset index
data = data_sel.drop(
    data_sel[data_sel["Abtrag"] <= 0].index, axis=0
)  # .reset_index(drop=True)
print("{} negative data dropped.\n".format(len(data_sel)-len(data)))

print("Data set is ready\n", data.head(5))
print(80 * "=")

show = input("Show correlations? [y/n]: ")
if "y" in show:
    # Show the correlation between parameters and material removal
    colormap = plt.cm.RdBu
    plt.figure(figsize=(8, 8))
    plt.title("Correlations after Cleaning", y=1.05, size=15)
    sns.heatmap(
        data.astype(float).corr(),
        linewidths=1,
        vmax=1.0,
        vmin=-1.0,
        square=True,
        cmap=colormap,
        linecolor="white",
        annot=True,
    )
    plt.show()

    # linear regression
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.suptitle("Linear Regression across categorical Values")
    sns.regplot(x=data["Drehzahl"], y=data["Abtrag"], fit_reg=True, ax=ax[0, 0])
    # ax[0, 0].set_title("Abtrag bei Drehzahl")
    ax[0, 0].set_xlim([800, 5200])
    sns.regplot(x=data["Vorschub"], y=data["Abtrag"], fit_reg=True, ax=ax[0, 1])
    # ax[0, 1].set_title("Abtrag bei Vorschub")
    ax[0, 1].set_xlim([42, 208])
    sns.regplot(x=data["Kraft"], y=data["Abtrag"], fit_reg=True, ax=ax[1, 0])
    # ax[1, 0].set_title("Abtrag bei Kraft")
    ax[1, 0].set_xlim([8, 32])
    sns.regplot(x=data["Winkel"], y=data["Abtrag"], fit_reg=True, ax=ax[1, 1])
    # ax[1, 1].set_title("Abtrag bei Winkel")
    ax[1, 1].set_xlim([0.8, 5.2])
    plt.show()

else:
    print("Not showing correlations.")

# *************************************** initialize the input / output ********************************************** #

# split the feature-/label
feature_df = pd.DataFrame(data, columns=["Drehzahl", "Vorschub", "Kraft", "Winkel"])
label_df = pd.DataFrame(data, columns=["Abtrag"])

# normalize the parameters
# Transform features by scaling each feature to a given range between 0 and 1.
scaler = MinMaxScaler().fit(feature_df)
# scaler.data_max_
Feature = scaler.transform(feature_df)
print("Features normalized.")
# numpy.ravel(a)returns a contiguous flattened array
Label = np.ravel(label_df)

# 222
print("Loaded data after filtering has {} rows.\n".format(Feature.shape[0]))

# split the train/test data set
X_train, X_test, y_train, y_test = train_test_split(
    Feature, Label, test_size=None, random_state=0
)  # numpy.ndarray

print("Data has been in train- and testset split up.")
print("Size of X_train: {}".format(X_train.shape))
print("Size of y_train: {}".format(y_train.shape))
print("Size of X_test: {}".format(X_test.shape))
print("Size of y_test: {}".format(y_test.shape), "\n")
print(80 * "=")


if __name__ == "__main__":

    print("Current working path:", os.getcwd(), "\n")
    print("Samples of Features: ")
    print(feature_df.sample(4))
    print("Samples of Labels: ")
    print(label_df.sample(4))
    print(80 * "=")
