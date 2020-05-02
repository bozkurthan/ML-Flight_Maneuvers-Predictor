import pandas as pd
import numpy as np
import os
from sklearn.utils import resample

import glob

def resample_fixed(df, n_new):
    n_old, m = df.values.shape
    mat_old = df.values
    mat_new = np.zeros((n_new, m))
    x_old = np.linspace(df.index.min(), df.index.max(), n_old)
    x_new = np.linspace(df.index.min(), df.index.max(), n_new)

    for j in range(m):
        y_old = mat_old[:, j]
        y_new = np.interp(x_new, x_old, y_old)
        mat_new[:, j] = y_new

    return pd.DataFrame(mat_new, index=x_new, columns=df.columns)

def upsampleto5000():
    df = pd.read_excel(r"C:\Users\emir\Desktop\test.xlsx",index_col = None,header = 0)
    
    resampleddf = pd.DataFrame(resample(df,replace=True,n_samples=20)).T
    resampleddf.T.sort_values(by=['x1']).to_excel(r"C:\Users\emir\Desktop\outputtest.xlsx",sheet_name='Sheet_name_1')
    print(resampleddf.T.sort_values(by=['x1']))

def downsampleto5000(df):
    count = len(df.index)
    div = count / 5000

if __name__ == "__main__":
    upsampleto5000()
    path = os.getcwd() + "/not augmented data"
    all_files = glob.glob(path + "/*.xlsx")

    if not os.path.exists(os.getcwd() + "/augmented data"):
        os.makedirs(os.getcwd() + "/augmented data")

    for filename in all_files:
        df = pd.read_excel(filename, index_col=None, header=0)
        if len(df.index) > 5000:
            augmentedDf = downsampleto5000(df)
        else:
            augmentedDf = upsampleto5000()

# merged_dfs = pd.read_csv(all_files[0],index_col=None,header=0)
# all_files = all_files[1:]
# for filename in all_files:
#     df = pd.read_csv(filename, index_col=None, header=0)
#     df = df.set_index('timestamp').reindex(merged_dfs.set_index('timestamp').index, method='nearest').reset_index()
#     merged_dfs = pd.merge(merged_dfs,df,on="timestamp")
# print(merged_dfs)
#
#
# print(merged_dfs.shape)
# print(merged_dfs.head())
# print("test")
