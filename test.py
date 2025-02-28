import pandas as pd

import pyarrow.parquet as pq

# 读取 Parquet 文件
table = pq.read_table('/shared/nas2/ph16/open-r1/datasets/countdown/test.parquet')

# 将表格转换为 Python 字典，再获取第一条数据
data_dict = table.to_pydict()

# 打印第一条数据（注意：这里将各列的第一条数据组合成字典打印）
first_row = {key: value[0] for key, value in data_dict.items()}
print(first_row)
