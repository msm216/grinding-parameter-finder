import os
import csv
import random
import pymysql
import pandas as pd
import numpy as np

# *************************************** define fake bench ********************************************** #

local_csv_file = os.getcwd() + "\\p_recommend.csv"

# read the recommended parameters on working bench, generate new results
def fake_bench(csv_file=local_csv_file):
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


# write the "measuring results" into mysql data base
def write_to_sql(df):
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
        # seires to list
        row = df.iloc[i].tolist()
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


print("Loading recommended parameters on working bench...")
new_measuring_results = fake_bench()

if __name__ == "__main__":

    print("Virtual measuring results:\n", new_measuring_results)
    print("MySQL database not updated.")
