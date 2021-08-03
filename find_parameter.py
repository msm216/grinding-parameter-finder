import os
import pandas as pd
import numpy as np
import itertools
import csv
import random
import pymysql

from data_prepare import controller
from model_initialize import pred_method, method_name


# *************************************** define fake bench ********************************************** #


def fake_bench():
    csv_file = os.getcwd() + "\\p_recommend.csv"

    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        l_out = []
        l_para = []

        for r in reader:
            r.pop(0)
            # 确保list中的元素为int
            r = [int(x) for x in r]
            n = 0
            l_in = []

            while n < 10:
                n += 1
                # random bias [-0.005, 0.005]
                bias = random.uniform(0, 0.01) - 0.005
                removal = (
                    0.0456 * (r[0] / 10000)
                    - 0.1247 * (r[1] / 500)
                    + 0.0120 * (r[2] / 50)
                    - 0.0106 * (r[3] / 10)
                    + 0.0181
                    + bias
                )
                l_in.append(removal)

            l_para.append(r)
            l_out.append(l_in)

        # 存储为DataFrame
        df_para = pd.DataFrame(l_para, dtype="int64")
        df_diff = pd.DataFrame(l_out)
        # 横向合并parameter和differenz
        df = pd.concat([df_para, df_diff], axis=1)
        # 重写列
        df.columns = [
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
        ]

    return df


def df_to_sql(df=fake_bench()):
    # 连接数据库
    connection = pymysql.connect(
        host="localhost",  # host='127.0.0.1',
        port=3306,
        user="root",
        passwd="12345",
        db="grinding_parameter",
        charset="utf8",
    )
    # 获取会话指针（游标）
    cursor = connection.cursor()

    ## 准备写入 ##

    # 提取索引为list
    ind = df.index.values.tolist()
    # 获取 columns
    col = df.columns.values  # numpy.ndarray
    col = np.append(col, "submission_date")

    # 遍历整个 DataFrame 生成内容
    for i in ind:
        # 获取行数据
        row = df.iloc[i].tolist()  # seires to list
        row.append("NOW()")

        # 生成 INSERT 语句
        sql_insert = """INSERT INTO data ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}) 
                        VALUES ({:n}, {:n}, {:n}, {:n}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})""".format(
            *col, *row
        )
        # 添加一行数据
        cursor.execute(sql_insert)

    # 提交
    connection.commit()
    # 关闭指针
    cursor.close()
    # 关闭数据库连接
    connection.close()


# ***************************************** parameter pre-selection ************************************************** #


# parameter selector, find the fitting value from in lists, which is closest to the input value

# find the fitting value from the parameter lists, which is closest to the input value
def get_list(num, array):
    if len(num) == 0:
        # transform the array to list directly
        the_list = array.tolist()
    else:
        # returns the index of the minimal value in array
        idx = (np.abs(array - float(num))).argmin()
        # build a new list with one single value
        the_list = [array[idx]]
    return the_list


# pre-selection of parameter "Drehzahl"
array_d = np.arange(1000, 5001, 500)  # numpy.ndarray
num_d = input("Select a Werkzeugdrehzahl between 1000 and 5000: ")
Drehzahl = get_list(num_d, array_d)
print("The requested Werkzeugdrehzahl is: ", num_d, "[1/min]")
print("List for the parameter search:\n", Drehzahl)

print(50 * "=")

# pre-selection of parameter "Vorschub"
array_v = np.arange(50, 201, 10)  # numpy.ndarray
num_v = input("Select a Vorschubgeschwindigkeit between 50 and 200: ")
Vorschub = get_list(num_v, array_v)
print("The requested Vorschubgeschwindigkeit is: ", num_v, "[mm/s]")
print("List for the parameter search:\n", Vorschub)

print(50 * "=")

# pre-selection of parameter "Kraft"
array_k = np.arange(10, 31, 1)  # numpy.ndarray
num_k = input("Select a Anpresskraft between 10 and 30: ")
Kraft = get_list(num_k, array_k)
print("The requested Anpresskraft is: ", num_k, "[N]")
print("List for the parameter search:\n", Kraft)

print(50 * "=")

# pre-selection of parameter "Winkel"
array_w = np.arange(1, 6, 1)  # numpy.ndarray
num_w = input("Select a Anstellwinkel between 1 and 5: ")
Winkel = get_list(num_w, array_w)
print("The requested Anstellwinkel is: ", num_w, "[°]")
print("List for the parameter search:\n", Winkel)

print(80 * "=")


# ********************************************** parameter finder **************************************************** #

# original parameter lists
# Drehzahl = [1000, 3000, 5000]
# Vorschub = [50, 100, 200]
# Kraft = [10, 20, 30]
# Winkel = [1, 3, 5]

# iterates over all possible parameter options and finds the parameter constellation
# with the smallest deviation from the input
def find_parameter(estimator, expectation, *parameter_lists):
    dev_min = np.inf
    parameter = []
    for d, v, k, w in itertools.product(*parameter_lists):
        prediction = estimator(np.array([[d / 10000, v / 500, k / 50, w / 10]]))
        deviation = abs(prediction - expectation)
        if deviation < dev_min:
            parameter = [d, v, k, w]
            dev_min = deviation
    return parameter  # list


print("Please enter material removal in [mm] in sequence.")
print("Leave the input blank and press Enter to stop the input process.")

# run the parameter-finder for multiple times, put the results into a list
p_list = []
# list of all inputs
in_list = []
while True:
    the_target = input("Target material removal: ")
    if len(the_target) == 0:
        break
    else:
        res_list = find_parameter(
            pred_method, float(the_target), Drehzahl, Vorschub, Kraft, Winkel
        )
        p_list.append(res_list)
        Winkel = [res_list[3]]
        in_list.append(the_target)

# DataFrame with full information
df_01 = pd.DataFrame(p_list, columns=["Drehzahl", "Vorschub", "Kraft", "Winkel"])
df_01["Abtrag"] = df_01.apply(
    lambda x: pred_method(
        np.array(
            [
                [
                    x["Drehzahl"] / 10000,
                    x["Vorschub"] / 500,
                    x["Kraft"] / 50,
                    x["Winkel"] / 10,
                ]
            ]
        )
    )[0],
    axis=1,
)

df_01["Input"] = in_list
df_01["Methode"] = method_name
# DataFrame for the recommend
df_02 = df_01[["Drehzahl", "Vorschub", "Kraft", "Winkel"]]

print("Following parameter have been recommended:\n", df_01)
print(80 * "=")


# *********************************************** write into csv ***************************************************** #

# write the parameter into a csv-file
df_01.to_csv("run_history.csv", index=True, mode="a", header=False)
df_02.to_csv("p_recommend.csv", index=True, header=False)
print(
    "The recommended process parameter have been saved into csv-file under:\n"
    + os.getcwd()
)

csv_file = os.getcwd() + "\\p_recommend.csv"

if controller == True:
    df_to_sql()
    print("New data has been written into MySQL.")
else:
    print("Done, MySQL database not updated.")
