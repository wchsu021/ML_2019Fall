import sys
import numpy as np
import pandas as pd
import csv
from sklearn.externals import joblib

def readFile(fileName):
    df = pd.read_csv(fileName)
    rawData = df.values
    rawData = rawData.astype(np.float)

    return rawData

def preProcess(data):
    data = data[:,:-42]
    data = np.delete(data, (1), axis=1)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    index = [0, 2, 3, 4]
    mean_vec = np.zeros(data.shape[1])
    std_vec = np.ones(data.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    for i in range(data.shape[1]):
        if std_vec[i] != 0:
            data[:,i] = (data[:,i] - mean_vec[i]) / std_vec[i]

    return data

def genFile(y_test,outFile):
    f = open(outFile,"w")
    w = csv.writer(f)
    title = ['id','label']
    w.writerow(title) 
    for i in range(y_test.shape[0]):
        content = [str(i+1), y_test[i][0]]
        w.writerow(content)
    f.close()

def main(testFile, modelFile, outFile):
    x_test = readFile(testFile)
    x_test = preProcess(x_test)
    model = joblib.load(modelFile)
    y_test = model.predict(x_test)
    y_test = y_test.reshape(-1,1)
    y_test[y_test >= 0.5] = 1
    y_test[y_test < 0.5] = 0
    y_test = y_test.astype(np.int)
    genFile(y_test,outFile)

if __name__ == '__main__':
    main(sys.argv[1],'best_param.pkl',sys.argv[2])