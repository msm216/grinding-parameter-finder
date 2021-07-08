import os
import pandas as pd
import numpy as np
import pymysql.cursors
import seaborn as sns
import matplotlib.pyplot as plt

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


# outlier detection Method 1 using Standard Deviation (wrong!)
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
        outlier_list_col = df[(df[col]<bot)|(df[col]>top)].index       
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
    # select observations containing more than n outliers
    #outlier_indices = Counter(outlier_indices)        
    #multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return outlier_indices


def read_from_mysql():
    # 连接数据库
    connection = pymysql.connect(
        host='localhost',  #host='127.0.0.1',
        port=3306,
        user='root',
        passwd='12345',
        db='grinding_parameter',
        charset='utf8'
    )
    try:    
        # 获取会话指针
        with connection.cursor() as cursor:        
            # 筛选语句
            '''
            sql_select = """SELECT column_name,column_name FROM data
                        WHERE column_name IS NOT NULL"""
            '''
            # 全选语句
            sql_select = """SELECT * FROM data"""        
            # 执行查询语句
            cursor.execute(sql_select)
            # 查询所有数据
            result = cursor.fetchall()    # tuple      
            # 关闭指针
            cursor.close()
    finally:    
        # 关闭数据库连接
        connection.close()
    
    df = pd.DataFrame(result, columns=['ID', 'Drehzahl', 'Vorschub', 'Kraft', 'Winkel', '1_Differenz',
            '2_Differenz', '3_Differenz', '4_Differenz', '5_Differenz',
            '6_Differenz', '7_Differenz', '8_Differenz', '9_Differenz',
            '10_Differenz', 'submission_date'])
    # get date of the last raw
    date = df.loc[df.index[-1], 'submission_date'] 
    # delete the "date"-column
    df = df.drop(['submission_date'], axis=1)
    # delete the original "ID"-column
    df = df.drop(['ID'], axis=1)
    
    return df, date


def read_local_excel():
    # path
    file= os.getcwd()+'\\data.xlsx'
    # 初始化空DataFrame
    df = pd.DataFrame()
    
    # 读取全表（直接dict中获取各个表???）
    reader = pd.read_excel(file, sheet_name=None)    # dict of DataFrame
    # 生成sheets表
    sheets = []
    for k in reader.keys():
        sheets.append(k)

    # 循环 sheets
    for sheet in sheets:
        # 提取 sheet 为 DataFrame
        df_sheet = pd.read_excel(file, sheet_name=sheet)    # DataFrame
        # 删除含空值的行
        df_sheet.dropna(axis=0, how='any', inplace=True)
        # 删除原生索引列
        df_sheet = df_sheet.drop(['ID'], axis=1)
        # 添加到df
        df = pd.concat([df, df_sheet], ignore_index=True)
    
    return df



# *********************************************** load  the data **************************************************** #


def choose_source():
    
    my_input = input("Use data from MySQL? [y/n]: ")
    
    if len(my_input) == 0:
        df = read_local_excel()
        c = False
        print("Using data from excel file: "+'data.xlsx')
    elif "y" in my_input:
        df, date = read_from_mysql()
        c = True
        print("Using data from MySQL...")
        print("Last update on:", date)
    else:
        df = read_local_excel()
        c = False
        print("Using data from excel file: "+'data.xlsx')
    return df, c


# choose data source
print(80 * "=")
data_full, controller = choose_source()    # <----------------------------------- controller
print('\nNumber of loaded data: ',len(data_full))
print(data_full.sample(5))


# ********************************************* prepare the data ***************************************************** #

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
#data_full['Abtrag'] = differences.mean(axis=1)    # axis = 0 by default, for rows
data_full['Abtrag'] = differences.max(axis=1)
# add a column of standard deviation of all "differences"
data_full['StdAbw'] = differences.std(axis=1)
# add columns of measurement quality
#data_full['P'] = data_full.apply(lambda x: 1 if x.Abtrag >= 0 else 0, axis=1)
data_full['Qualität'] = np.where(data_full['Abtrag'] >= 0.01, 1, 0)

# the original data set has 272 rows, 4 of them have missing values, 3 of them have exceptional high "Abtrag"

# detect indexes of outliers
ind_outliers = detect_outliers_sd(data_full, ["Abtrag"])    # <----------------------------------- 不严谨方法
# drop the outliers, build a new DataFrame
data_full = data_full.drop(ind_outliers, axis = 0)
# build a new DataFrame, keep the needed columns only and reset the index
data = data_full[
    ["Drehzahl", "Vorschub", "Kraft", "Winkel", "Abtrag", "Qualität"]
].reset_index(drop=True)


print("\nFeatures selected.")
#print(data.sample(5))


# choose the data with effective "Abtrag"

# detect indexes of samples lower than 0.01
ind_lower = lower_than(data, ["Abtrag"], 0)    # <----------------------------------- threshold
# show the bad data
print("Following data have been removed:\n", data.loc[ind_lower])
# drop the negatives, reset the index
data_sel = data.drop(ind_lower, axis=0).reset_index(drop=True)



# Show the correlation between parameters and material removal
data_heat = pd.DataFrame(
    data_sel, columns=["Drehzahl", "Vorschub", "Kraft", "Winkel", "Abtrag"]
)
colormap = plt.cm.RdBu
plt.figure(figsize=(8, 8))
plt.title("Korrelation der Parametern und Abtrag", y=1.05, size=15)
sns.heatmap(
    data_heat.astype(float).corr(),
    linewidths=1,
    vmax=1.0,
    vmin=-1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
)
plt.show()


# *************************************** initialize the input / output ********************************************** #

# split the feature-/label
feature_df = pd.DataFrame(data_sel, columns=["Drehzahl", "Vorschub", "Kraft", "Winkel"])
label_df = pd.DataFrame(data_sel, columns=["Abtrag"])

# normalize the parameters
feature_df["Drehzahl"] = feature_df["Drehzahl"].apply(lambda x: x / 10000)
feature_df["Vorschub"] = feature_df["Vorschub"].apply(lambda x: x / 500)
feature_df["Kraft"] = feature_df["Kraft"].apply(lambda x: x / 50)
feature_df["Winkel"] = feature_df["Winkel"].apply(lambda x: x / 10)


print("\nFeatures normalized.")


# transform the DataFrame to np.array
Feature = np.array(feature_df)
# Label = np.array(label_df)  # unused
# numpy.ravel(a)returns a contiguous flattened array
Labels = np.ravel(label_df)


print("Loaded data after filtering hast {} rows.\n".format(Feature.shape[0]))


# split the train/test data set
X_train, X_test, y_train, y_test = train_test_split(
    Feature, Labels, test_size=None, random_state=0
)  # numpy.ndarray


print("Data has been in train- and test set split up.")
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
