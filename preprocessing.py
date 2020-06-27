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
import pickle
from sklearn.multiclass import OneVsRestClassifier
import re


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

def augmentTheData(baseDir,flightNum):
    path = baseDir+ "/Flight "+str(flightNum)
    all_files = glob.glob(path + "/*.csv")
    all_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    if not os.path.exists(baseDir + "/Flight "+str(flightNum)+" Augmented"):
        os.makedirs(baseDir + "/Flight "+str(flightNum)+" Augmented")

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        resampledDf = resample_fixed(df, 5000)
        baseName = os.path.basename(filename)
        resampledDf.to_excel(baseDir + "/Flight "+str(flightNum)+" Augmented/" + baseName[:-5] + "augmented.xlsx", index=None)

def fixTimestampsOfAugmentedFiles(flightNum):
    path =dataPath+"/Flight "+str(flightNum)+" Augmented"
    all_files = glob.glob(path+"/*.xlsx")

    timestamps = pd.Series()

    for filename in all_files:
        df = pd.read_excel(filename,index_col=None,header=0)
        if timestamps.empty:
            timestamps = df.timestamp
        else:
            df.timestamp = timestamps
            # baseName = os.path.basename(filename)
            df.to_excel(filename, index=None)

def makeFinalCSVFiles(i):
    path = dataPath+"/Flight "+str(i)+" Augmented"
    all_files = glob.glob(path+"/*.xlsx")

    df = pd.read_excel(all_files[0],index_col=None,header=0)

    for i in range(1,len(all_files)):
        tempDf = pd.read_excel(all_files[i],index_col= None,header=0)
        tempDf = tempDf.drop(['timestamp'],axis=1)
        df = pd.concat([df, tempDf], axis=1)
    return df

def splitTaggedFiles(dataPath):
    if not os.path.exists(dataPath + "/Flights Takeoff"):
        os.makedirs(dataPath + "/Flights Takeoff")
    if not os.path.exists(dataPath + "/Flights Climbing"):
        os.makedirs(dataPath+ "/Flights Climbing")
    if not os.path.exists(dataPath + "/Flights Descending"):
        os.makedirs(dataPath + "/Flights Descending")
    if not os.path.exists(dataPath + "/Flights Cruise"):
        os.makedirs(dataPath + "/Flights Cruise")
    if not os.path.exists(dataPath + "/Flights Landed"):
        os.makedirs(dataPath + "/Flights Landed")
    path = dataPath+"/Flights wTagged"
    all_files = glob.glob(path+"/*.csv")
    for i in range(1,len(all_files)+1):
        df = pd.read_csv(all_files[i-1],index_col=None,header=0)
        takeoff = df[df["isTakeoff"] == 1]
        climbing = df[df["isClimbing"] == 1]
        cruise = df[df["isCruise"] == 1]
        descending = df[df["isDescending"] == 1]
        landed = df[df["isLand"] == 1]
        takeoff.to_csv(dataPath+"/Flights Takeoff/flight"+str(i)+"_takeoff.csv",index=None)
        climbing.to_csv(dataPath+"/Flights Climbing/flight"+str(i)+"_climbing.csv",index=None)
        descending.to_csv(dataPath+"/Flights Descending/flight"+str(i)+"_descending.csv",index=None)
        cruise.to_csv(dataPath+"/Flights Cruise/flight"+str(i)+"_cruise.csv",index=None)
        landed.to_csv(dataPath+"/Flights Landed/flight"+str(i)+"_landed.csv",index=None)

def interpolateTaggedData(dataPath):
    base_dir = dataPath
    folders = ('Flights Takeoff', 'Flights Landed', 'Flights Cruise', 'Flights Descending',"Flights Climbing")
    interpolated_dir = createFoldersForInterpolatedData(dataPath)
    for folder in folders:
        path = base_dir+"/"+folder
        name = folder.split(" ")[1]
        all_files = glob.glob(path+"/*.csv")
        all_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        for i in range(1,len(all_files)+1):
            f = all_files[i-1]
            df= pd.read_csv(f,index_col=None,header=0)
            new_df = resample_fixed(df,500)
            new_df.to_csv(interpolated_dir+"/Flight "+str(i)+"/flight"+str(i)+"_"+name+"_interpolated.csv",index=None)

def createFoldersForInterpolatedData(dataPath):
    interpolated_dir = dataPath+"/Interpolated Flights"
    if not os.path.exists(interpolated_dir):
        os.makedirs(interpolated_dir)
    for i in range(1,13):
        if not os.path.exists(interpolated_dir+"/Flight "+str(i)):
            os.makedirs(interpolated_dir+"/Flight "+str(i))
    return interpolated_dir

def createFoldersForScaledData(dataPath):
    scaled_dir = dataPath+"/Scaled Flights"
    if not os.path.exists(scaled_dir):
        os.makedirs(scaled_dir)
    for i in range(1,13):
        if not os.path.exists(scaled_dir+"/Flight "+str(i)):
            os.makedirs(scaled_dir+"/Flight "+str(i))
    return scaled_dir

def scaleDataWithMinMaxScaler(base_dir,scaledDataDir):
    createFoldersForScaledData(dataPath)
    allFolders = os.listdir(base_dir)
    scaler = MinMaxScaler()
    for j in allFolders:
        all_files = glob.glob(base_dir+"/"+j+"/*.csv")
        for i in range(1,len(all_files)+1):
            df = pd.read_csv(all_files[i-1],index_col=None,header=0)
            arr = scaler.fit_transform(df.iloc[:,:-5])
            response_df = df.iloc[:,-5:]
            new_df = pd.DataFrame(data=arr,  # values
            index = None,
            columns = list(df.columns[:-5]))
            result = pd.concat([new_df, response_df], axis=1, join='inner')
            name = all_files[i-1].rsplit("\\",1)[1][:-4]
            # Need refactor
            saving = scaledDataDir+"/"+j+"/"+name+"_scaled.csv"
            result.to_csv(saving,index=None)
            print(new_df.head())

def normalizeDataWithPCA(path,components = 25):
    pca = PCA(n_components=components)
    df = pd.read_csv(path,index_col=None,header=0)
    data =df.iloc[:,:-5]
    data = data.fillna(0)
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
    onedimTargets = ["".join([str(int(x)) for x in onedimTargets])]
    return onedimSamples,onedimTargets

def plotData():
    filename = r"C:\Users\emir\PycharmProjects\ML-Flight_Maneuvers-Predictor\Datas\Scaled Flights\Flight 2\flight2_Climbing_interpolated_scaled.csv"
    sns.set(rc={'figure.figsize': (20, 10)})
    df = pd.read_csv(filename, index_col=None, header=0)
    df.plot(x = "timestamp",y = ["output[2]","output[3]","output[4]","output[5]","output[6]","output[7]"])
    plt.show()

def handle_exception(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)
    return

def query():
    df = pd.read_csv(r"C:\Users\emir\PycharmProjects\ML-Flight_Maneuvers-Predictor\Interpolated Flights\Flight 1\flight1_Climbing_interpolated_scaled.csv",index_col=None, header=0)
    df2 = df["isClimbing"]
    print(df2)

def train(scaledFlightDir,modelName):
    train_targetMatris = []
    train_dataMatris = []
    test_targetMatris = []
    test_dataMatris = []

    allFolders = os.listdir(scaledFlightDir)


    for j in allFolders:
        all_files = glob.glob(scaledFlightDir+"/"+j+"/*.csv")
        for i in range(1, len(all_files)+1):
            normalizedDfWithTarget = normalizeDataWithPCA(all_files[i-1],components)
            vectorisedSampleDf,vectorisedTargetDf = vectoriseTheData(normalizedDfWithTarget)
            if j.__contains__("Flight 5"):
                test_dataMatris.append(vectorisedSampleDf)
                test_targetMatris.append(vectorisedTargetDf)
            elif j.__contains__("Flight 7"):
                test_dataMatris.append(vectorisedSampleDf)
                test_targetMatris.append(vectorisedTargetDf)
            elif j.__contains__("Flight 11"):
                test_dataMatris.append(vectorisedSampleDf)
                test_targetMatris.append(vectorisedTargetDf)
            elif j.__contains__("Flight 9"):
                test_dataMatris.append(vectorisedSampleDf)
                test_targetMatris.append(vectorisedTargetDf)
            else:
                train_dataMatris.append(vectorisedSampleDf)
                train_targetMatris.append(vectorisedTargetDf)

    X_train = train_dataMatris
    y_train = train_targetMatris
    X_test = test_dataMatris
    y_test = test_targetMatris

    model =OneVsRestClassifier(svm.SVC())
    # svm.SVC(decision_function_shape='ovo')  # Linear Kernel
    model.fit(X_train, np.ravel(y_train))
    y_pred = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred,average="macro"))
    print("Recall:", metrics.recall_score(y_test, y_pred,average="macro"))
    pickle.dump(model,open(modelName,'wb'))

def test(scaledFlightDir,modelName):
    x_test = []
    y_test = []
    dir = scaledFlightDir + "\Flight 10"
    dir2 = scaledFlightDir + "\Flight 11"
    all_files = glob.glob(dir+"/*.csv")
    all_files.extend(glob.glob(dir2+"/*.csv"))


    for i in range(1, len(all_files) + 1):
        normalizedDfWithTarget = normalizeDataWithPCA(all_files[i - 1], components)
        vectorisedSampleDf, vectorisedTargetDf = vectoriseTheData(normalizedDfWithTarget)
        x_test.append(vectorisedSampleDf)
        y_test.append(vectorisedTargetDf)

    loaded_model = pickle.load(open(modelName, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)

if __name__ == "__main__":
    sys.excepthook = handle_exception
    dataPath = os.getcwd()+"/Datas"
    fligthNum = 12
    components = 20
    modelName = "finalModel.sav"
    interpolatedFlightDir = dataPath+"/Interpolated Flights"
    scaledFlightDir = dataPath + "/Scaled Flights"

    if not os.path.exists(dataPath + "/Flights Final"):
        os.makedirs(dataPath + "/Flights Final")
    finalPath = dataPath+"/Flights Final"
    for i in range(1,fligthNum+1):
        augmentTheData(dataPath,i)
        print("Augmentation of Flight "+str(i)+" finished.")
        fixTimestampsOfAugmentedFiles(i)
        print("Fixing timestamp of Flight "+str(i)+" finished")
        df = makeFinalCSVFiles(i)
        df.to_csv(finalPath+"/flight_"+str(i)+"_augmented.csv",index=False)
        df.to_excel(finalPath+"/flight_"+str(i)+"_augmented.xlsx",index=False)
        print("Making combined csv and excel files for Flight "+str(i)+"finished")
    print("All flights combined and preprocessed.")
    print("Do you want to split datas with tags. (Y/N) ?")
    input1 = input()
    if input1 == "Y" or input1 == "y":
        splitTaggedFiles(dataPath)
        interpolateTaggedData(dataPath)
        scaleDataWithMinMaxScaler(interpolatedFlightDir,scaledFlightDir)
        train(scaledFlightDir,modelName)
    else:
        pass







