import pandas as pd
import numpy as np
import torch
import time
from sklearn.preprocessing import StandardScaler
from functools import reduce
from tqdm import tqdm
start = time.time()
ori_data = pd.read_parquet('./train.parquet')
end = time.time()

print('cost time of reading: {}s'.format(end-start))

df = ori_data.groupby('time_id')
del ori_data
start = time.time()
data = []
# # count = df.agg('count')
# index = count.index
count = []
for key, value in tqdm(df):
    value.drop('row_id', axis=1, inplace=True)
    data.append(np.array(value, dtype='float32'))
    count.append(len(np.array(value)))
end = time.time()
print('cost time of rebuilding data: {}s'.format(end-start))
np.save('./count.npy',np.array(count))
del df
start = time.time()
temp_data = data[0]
for i in tqdm(range(1,len(data))):
    temp_data = np.concatenate((temp_data,data[i]))
end = time.time()
print('cost time of constructing final data: {}s'.format(end-start))

data = np.array(temp_data)
print(data.shape)
np.save('./train.npy', data)

# df = pd.DataFrame({'A': [1, 1, 2, 3, 1, 3, 1, 2],
#                   'B': [2, 8, 1, 4, 3, 2, 5, 9],
#                     'C': [102, 98, 107, 104, 115, 87, 92, 123]})
# df1 = df.groupby('A')
# # count = df1.agg('count')
# # data = np.empty([len(df1.index)], )
# data = []
# for key, value in df1:
#     data.append(np.array(value, dtype='float32'))
# temp_data = data[0]
# data = reduce(lambda x,y: np.concatenate((x,y)), data)
# data = np.array(data)
# print(data)





