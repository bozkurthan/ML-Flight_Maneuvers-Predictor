import pandas as pd
import numpy as np
import os

import glob

path = os.getcwd()+"\Datas\Flight1"
all_files = glob.glob(path + "/*.csv")

merged_dfs = pd.read_csv(all_files[0],index_col=None,header=0)
all_files = all_files[1:]
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df = df.set_index('timestamp').reindex(merged_dfs.set_index('timestamp').index, method='nearest').reset_index()
    merged_dfs = pd.merge(merged_dfs,df,on="timestamp")
print(merged_dfs)


print(merged_dfs.shape)
print(merged_dfs.head())
print("test")
