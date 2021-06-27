# -*- coding: utf-8 -*-
import pymysql
import pandas as pd



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
sql_create = """CREATE TABLE IF NOT EXISTS raw_data (
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
sheets = pd.read_excel(file, sheet_name=None)    # dict
# 选择遍历的 sheets (选择除data_vd外的sheet)
sheet_l = []
for k in sheets.keys():
    sheet_l.append(k)
# 删去最后一个元素
sheet_l.pop()

# 获取会话指针（游标）
with connection.cursor() as cursor:
        
    # 如果表存在则删除
    cursor.execute("DROP TABLE IF EXISTS raw_data")
    # 创建数据表
    cursor.execute(sql_create)
    
    # 循环 sheets
    for sheet in sheet_l:
        
        # 提取 sheet 为 DataFrame
        df = pd.read_excel(file, sheet_name=sheet)    # DataFrame
        # 提取索引为list
        ind = df.index.values.tolist()
        
        # 获取 columns
        col = df.columns.values    # numpy.ndarray
        col = np.delete(col, 0)
        col = np.append(col, 'submission_date')
        
        # 循环 ind
        for i in ind:
          
            # 获取单个 row
            row = df.iloc[i].tolist()    # seires to list
            row.remove(row[0])
            row.append('NOW()')
            
            # 生成 INSERT 语句
            sql_insert = """INSERT INTO raw_data ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}) 
                             VALUES ({:n}, {:n}, {:n}, {:n}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})""".format(*col, *row)
                
            # 添加一行数据
            cursor.execute(sql_insert)
        
    # 提交
    connection.commit()
    # 关闭指针
    cursor.close()

# 关闭数据库连接
connection.close()

# 建议添加 try...except...finally 表达

## 批量操作 https://blog.csdn.net/jy1690229913/article/details/79407224
#cursor.executemany(sql,values) 
# 添加的数据 values 的格式必须为list[tuple(),tuple(),tuple()]或者tuple(tuple(),tuple(),tuple())
# sql 语句中的占位符统一使用%s,且不能加上引号。例如:
# sql="insert into tablename (id,name) values (%s,%s)"
