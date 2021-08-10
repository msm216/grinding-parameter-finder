# -*- coding: utf-8 -*-
import pymysql
import pandas as pd
import numpy as np


### 全部写入测试 ###
### 共268行原始数据 ###




# 连接数据库
connection = pymysql.connect(
    host='localhost',  #host='127.0.0.1',
    port=3306,
    user='root',
    passwd='12345',
    db='grinding_parameter',
    charset='utf8'
)

# 生成 CREATE 语句
sql_create = """CREATE TABLE IF NOT EXISTS data (
    id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    Drehzahl SMALLINT UNSIGNED,
    Vorschub SMALLINT UNSIGNED,
    Kraft SMALLINT UNSIGNED,
    Winkel SMALLINT UNSIGNED,
    1_Differenz FLOAT, 
    2_Differenz FLOAT, 
    3_Differenz FLOAT, 
    4_Differenz FLOAT, 
    5_Differenz FLOAT, 
    6_Differenz FLOAT, 
    7_Differenz FLOAT, 
    8_Differenz FLOAT, 
    9_Differenz FLOAT, 
    10_Differenz FLOAT,
    submission_date DATE)
"""


## 准备写入 ##

# 文件名
file = 'data.xlsx'
# 获取 sheet names
reader = pd.read_excel(file, sheet_name=None)    # dict
# 选择遍历的 sheets (选择除data_vd外的sheet)
sheets = []
for k in reader.keys():
    sheets.append(k)
# 删去最后一个元素
#sheet_l.pop()

# 获取会话指针（游标）
with connection.cursor() as cursor:
        
    # 如果表存在则删除
    cursor.execute("DROP TABLE IF EXISTS data")
    # 创建数据表
    cursor.execute(sql_create)
    
    # 循环 sheets
    for sheet in sheets:
        
        # 提取 sheet 为 DataFrame
        df = pd.read_excel(file, sheet_name=sheet)    # DataFrame
        # 删除含空值的行
        df.dropna(axis=0, how='any', inplace=True)
        # 提取索引为list
        ind = df.index.values.tolist()
        
        # 获取 columns
        col = df.columns.values    # numpy.ndarray
        col = np.delete(col, 0)
        col = np.append(col, 'submission_date')
        
        # 循环 index
        for i in ind:
          
            # 获取单个 row
            row = df.iloc[i].tolist()    # seires to list
            row.remove(row[0])
            row.append('NOW()')
            
            # 生成 INSERT 语句
            sql_insert = """INSERT INTO data ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}) 
                             VALUES ({:n}, {:n}, {:n}, {:n}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})""".format(*col, *row)
                
            # 添加一行数据
            cursor.execute(sql_insert) 
        
    # 提交
    connection.commit()
    # 全选语句
    sql_select = """SELECT * FROM data"""        
    # 执行查询语句
    cursor.execute(sql_select)
    # 查询所有数据
    result = cursor.fetchall()    # tuple    
    # 关闭指针
    cursor.close()

# 关闭数据库连接
connection.close()

df = pd.DataFrame(result, columns=['ID', 'Drehzahl', 'Vorschub', 'Kraft', 'Winkel', '1_Differenz',
            '2_Differenz', '3_Differenz', '4_Differenz', '5_Differenz',
            '6_Differenz', '7_Differenz', '8_Differenz', '9_Differenz',
            '10_Differenz', 'submission_date'])
date = df.loc[df.index[-1], 'submission_date'] 
# delete the "date"-column
df = df.drop(['submission_date'], axis=1)
# delete the original "ID"-column
df = df.drop(['ID'], axis=1)

print(df)

