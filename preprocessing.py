import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

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

def augmentTheData(flightNum):
    path = os.getcwd() + "/Flight "+str(flightNum)
    all_files = glob.glob(path + "/*.csv")

    if not os.path.exists(os.getcwd() + "/Flight "+str(flightNum)+" Augmented"):
        os.makedirs(os.getcwd() + "/Flight "+str(flightNum)+" Augmented")

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        resampledDf = resample_fixed(df, 5000)
        baseName = os.path.basename(filename)
        resampledDf.to_excel(os.getcwd() + "/Flight "+str(flightNum)+" Augmented/" + baseName[:-5] + "augmented.xlsx", index=None)

def fixTimestampsOfAugmentedFiles(flightNum):
    path = os.getcwd()+"/Flight "+str(flightNum)+" Augmented"
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

def splitTaggedFiles():
    if not os.path.exists(os.getcwd() + "/Flights Takeoff"):
        os.makedirs(os.getcwd() + "/Flights Takeoff")
    if not os.path.exists(os.getcwd() + "/Flights Climbing"):
        os.makedirs(os.getcwd() + "/Flights Climbing")
    if not os.path.exists(os.getcwd() + "/Flights Descending"):
        os.makedirs(os.getcwd() + "/Flights Descending")
    if not os.path.exists(os.getcwd() + "/Flights Cruise"):
        os.makedirs(os.getcwd() + "/Flights Cruise")
    if not os.path.exists(os.getcwd() + "/Flights Landed"):
        os.makedirs(os.getcwd() + "/Flights Landed")
    path = os.getcwd()+"/Flights wTagged"
    all_files = glob.glob(path+"/*.csv")
    for i in range(1,len(all_files)+1):
        df = pd.read_csv(all_files[i-1],index_col=None,header=0)
        takeoff = df[df["isTakeoff"] == 1]
        climbing = df[df["isClimbing"] == 1]
        cruise = df[df["isCruise"] == 1]
        descending = df[df["isDescending"] == 1]
        landed = df[df["isLand"] == 1]
        takeoff.to_csv(os.getcwd()+"/Flights Takeoff/flight"+str(i)+"_takeoff.csv",index=None)
        climbing.to_csv(os.getcwd()+"/Flights Climbing/flight"+str(i)+"_climbing.csv",index=None)
        descending.to_csv(os.getcwd()+"/Flights Descending/flight"+str(i)+"_descending.csv",index=None)
        cruise.to_csv(os.getcwd()+"/Flights Cruise/flight"+str(i)+"_cruise.csv",index=None)
        landed.to_csv(os.getcwd()+"/Flights Landed/flight"+str(i)+"_landed.csv",index=None)

def interpolateTaggedData():
    base_dir = os.getcwd()
    folders = ('Flights Takeoff', 'Flights Landed', 'Flights Cruise', 'Flights Descending',"Flights Climbing")
    interpolated_dir = createFoldersForInterpolatedData()
    for folder in folders:
        path = base_dir+"/"+folder
        name = folder.split(" ")[1]
        all_files = glob.glob(path+"/*.csv")
        for i in range(1,len(all_files)+1):
            df= pd.read_csv(all_files[i-1],index_col=None,header=0)
            new_df = resample_fixed(df,500)
            new_df.to_csv(interpolated_dir+"/Flight "+str(i)+"/flight"+str(i)+"_"+name+"_interpolated.csv",index=None)

def createFoldersForInterpolatedData():
    base_dir = os.getcwd()
    interpolated_dir = base_dir+"/Interpolated Flights"
    if not os.path.exists(interpolated_dir):
        os.makedirs(interpolated_dir)
    for i in range(1,11):
        if not os.path.exists(interpolated_dir+"/Flight "+str(i)):
            os.makedirs(interpolated_dir+"/Flight "+str(i))
    return interpolated_dir


def scaleDataWithMinMaxScaler(base_dir):
    allFolders = os.listdir(base_dir)
    scaler = MinMaxScaler()
    for i in allFolders:
        all_files = glob.glob(base_dir+"/"+i+"/*.csv")
        for i in range(1,len(all_files)+1):
            df = pd.read_csv(all_files[i-1],index_col=None,header=0)
            arr = scaler.fit_transform(df.iloc[:,:-5])
            response_df = df.iloc[:,-5:]
            new_df = pd.DataFrame(data=arr,  # values
            index = None,
            columns = list(df.columns[:-5]))
            result = pd.concat([new_df, response_df], axis=1, join='inner')
            result.to_csv(all_files[i-1][:-4]+"_scaled.csv",index=None)
            print(new_df.head())

def normalizeDataWithPCA(path,components = 20):
    pca = PCA(n_components=components)
    df = pd.read_csv(path,index_col=None,header=0)
    # imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data =df.iloc[:,:-5]
    data = data.fillna(0)
    # imputer = imputer.fit(data)
    # data = imputer.transform(data)
    response = df.iloc[:,-5:]

    principalComponents = pca.fit_transform(data)
    cols = []
    for i in range(1,components+1):
        cols.append("col " + str(i))
    principalDf = pd.DataFrame(data = principalComponents
             , columns =cols)

    # corrMatrix = principalDf.corr()
    # sns.heatmap(corrMatrix, annot=True)
    # plt.show()

    finalDf = pd.concat([principalDf, response], axis=1)
    return finalDf

def vectoriseTheData(df):
    data =df.iloc[:,:-5:]
    response = df.iloc[:,-5:]

    onedimSamples = np.array(data).reshape(-1)
    onedimTargets = np.array(response.iloc[0]).reshape(-1)
    return onedimSamples,onedimTargets

def plotData():
    filename = r"C:\Users\emir\PycharmProjects\ML-Flight_Maneuvers-Predictor\augmented data\actuator outputsaugmented.xlsx"
    sns.set(rc={'figure.figsize': (20, 10)})
    df = pd.read_excel(filename, index_col=None, header=0)
    df.plot(x = "timestamp",y = ["output[2]","output[3]","output[4]","output[5]","output[6]","output[7]"])
    plt.show()

def handle_exception(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)
    return

def query():
    df = pd.read_csv(r"C:\Users\emir\PycharmProjects\ML-Flight_Maneuvers-Predictor\Interpolated Flights\Flight 1\flight1_Climbing_interpolated_scaled.csv",index_col=None, header=0)
    df2 = df["isClimbing"]
    print(df2)

if __name__ == "__main__":
    sys.excepthook = handle_exception
    fligthNum = 10
    components = 20

    train_targetMatris = []
    train_dataMatris = []
    test_targetMatris = []
    test_dataMatris = []

    interpolatedFlightDir = os.getcwd() + "/Interpolated Flights"
    scaledFlightDir = os.getcwd() + "/Scaled Flights"
    allFolders = os.listdir(interpolatedFlightDir)


    for j in allFolders:
        all_files = glob.glob(scaledFlightDir+"/"+j+"/*.csv")
        for i in range(1, len(all_files)+1):
            normalizedDfWithTarget = normalizeDataWithPCA(all_files[i-1],components)
            vectorisedSampleDf,vectorisedTargetDf = vectoriseTheData(normalizedDfWithTarget)
            if j.__contains__("Flight 10"):
                test_dataMatris.append(vectorisedSampleDf)
                test_targetMatris.append(vectorisedTargetDf)
            else:
                train_dataMatris.append(vectorisedSampleDf)
                train_targetMatris.append(vectorisedTargetDf)




    X_train = train_dataMatris
    y_train = train_targetMatris
    X_test = test_dataMatris
    y_test = test_targetMatris

    clf = svm.SVC(kernel='linear')  # Linear Kernel
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

    # if not os.path.exists(os.getcwd() + "/Flights Final"):
    #     os.makedirs(os.getcwd() + "/Flights Final")
    # finalPath = os.getcwd()+"/Flights Final"
    #
    # for i in range(1,fligthNum+1):
    #     augmentTheData(i)
    #     print("Augmentation of Flight "+str(i)+" finished.")
    #     fixTimestampsOfAugmentedFiles(i)
    #     print("Fixing timestamp of Flight "+str(i)+" finished")
    #     df = makeFinalCSVFiles(i)
    #     df.to_csv(finalPath+"/flight_"+str(i)+"_augmented.csv",index=False)
    #     df.to_excel(finalPath+"/flight_"+str(i)+"_augmented.xlsx",index=False)
    #     print("Making combined csv and excel files for Flight "+str(i)+"finished")
    # print("All flights combined and preprocessed.")
    # print("Do you want to split datas with tags. (Y/N) ?")
    # input1 = input()
    # if input1 == "Y" or input1 == "y":
    #     splitTaggedFiles()
    #     interpolateTaggedData()
    # else:
    #     pass







