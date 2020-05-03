import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
        if j == 0:
            rounded = [np.round(x) for x in y_new]
            mat_new[:,j] = rounded
        else:
            mat_new[:, j] = y_new

    return pd.DataFrame(mat_new, index=x_new, columns=df.columns)

def augmentTheData():
    path = os.getcwd() + "/Flight 10"
    all_files = glob.glob(path + "/*.csv")

    if not os.path.exists(os.getcwd() + "/Flight 10 Augmented"):
        os.makedirs(os.getcwd() + "/Flight 10 Augmented")

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        resampledDf = resample_fixed(df, 5000)
        baseName = os.path.basename(filename)
        resampledDf.to_excel(os.getcwd() + "/Flight 10 Augmented/" + baseName[:-5] + "augmented.xlsx", index=None)

def fixTimestampsOfAugmentedFiles():
    path = os.getcwd()+"/Flight 10 Augmented"
    all_files = glob.glob(path+"/*.xlsx")

    timestamps = pd.Series()

    for filename in all_files:
        df = pd.read_excel(filename,index_col=None,header=0)
        if timestamps.empty:
            timestamps = df.timestamp
        else:
            df.timestamp = timestamps
            baseName = os.path.basename(filename)
            df.to_excel(filename, index=None)


def makeFinalCSVFiles(i):
    path = os.getcwd()+"/Flight "+str(i)+" Augmented"
    all_files = glob.glob(path+"/*.xlsx")

    df = pd.read_excel(all_files[0],index_col=None,header=0)

    for i in range(1,len(all_files)):
        tempDf = pd.read_excel(all_files[i],index_col= None,header=0)
        tempDf = tempDf.drop(['timestamp'],axis=1)
        df = pd.concat([df, tempDf], axis=1)
    return df


if __name__ == "__main__":
    # filename = r"C:\Users\emir\PycharmProjects\ML-Flight_Maneuvers-Predictor\augmented data\actuator outputsaugmented.xlsx"
    # sns.set(rc={'figure.figsize': (20, 10)})
    # df = pd.read_excel(filename, index_col=None, header=0)
    # df.plot(x = "timestamp",y = ["output[2]","output[3]","output[4]","output[5]","output[6]","output[7]"])
    # plt.show()
    # augmentTheData()
    # print("augmentation finished")
    # fixTimestampsOfAugmentedFiles()
    if not os.path.exists(os.getcwd() + "/Flights Final"):
        os.makedirs(os.getcwd() + "/Flights Final")
    path = os.getcwd()+"/Flights Final"
    for i in range(1,11):
        df = makeFinalCSVFiles(i)
        # df.to_csv(path+"/flight_"+str(i)+"_augmented.csv",index=False)
        df.to_excel(path+"/flight_"+str(i)+"_augmented.xlsx",index=False)


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
