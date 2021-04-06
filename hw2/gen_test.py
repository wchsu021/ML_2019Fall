import numpy as np
import math
import pandas as pd
import csv
import sys

dim = 106

def load_data():
    x_train = pd.read_csv(sys.argv[1])
    x_test = pd.read_csv(sys.argv[3])

    x_train = x_train.values
    x_test = x_test.values

    y_train = pd.read_csv(sys.argv[2], header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)

    return x_train, y_train, x_test

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)
	
def train(x_train, y_train):
    cnt1 = 0
    cnt2 = 0
    
    mu1 = np.zeros((dim,))
    mu2 = np.zeros((dim,))
    
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            cnt1 += 1
            mu1 += x_train[i]
        else:
            cnt2 += 1
            mu2 += x_train[i]
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim,dim))
    sigma2 = np.zeros((dim,dim))
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2

    
    share_sigma = (cnt1 / x_train.shape[0]) * sigma1 + (cnt2 / x_train.shape[0]) * sigma2
    return mu1, mu2, share_sigma, cnt1, cnt2
	
def predict(x_test, mu1, mu2, share_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(share_sigma)

    w = np.dot( (mu1-mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse), mu2) + np.log(float(N1)/N2)

    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    return pred
	
if __name__ == '__main__':
    x_train,y_train,x_test = load_data()
    param = np.load('gen_param.npz')
    mu1 = param['mu1']
    mu2 = param['mu2']
    shared_sigma = param['shared_sigma']
    N1 = param['N1']
    N2 = param['N2']
    
    y_test = predict(x_test, mu1, mu2, shared_sigma, N1, N2)
    y_test = np.around(y_test)
    y_test = np.int_(y_test)
    #print(y_test)
    f = open(sys.argv[4],"w")
    w = csv.writer(f)
    title = ['id','label']
    w.writerow(title) 
    for i in range(y_test.shape[0]):
        content = [str(i+1),y_test[i]]
        w.writerow(content) 
    
    print("Finish!")