import numpy as np
import pandas as pd
import csv
import sys

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
	
def normalize(x_train, x_test):
    
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)

    index = [0, 1, 3, 4, 5]
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor
	
def train(x_train, y_train):
    b = 0.0
    w = np.zeros(x_train.shape[1])
    lr = 0.05
    epoch = 1000
    b_lr = 0
    w_lr = np.ones(x_train.shape[1])
    
    for e in range(epoch):
        z = np.dot(x_train, w) + b
        pred = sigmoid(z)
        loss = y_train - pred

        b_grad = -1*np.sum(loss)
        w_grad = -1*np.dot(loss, x_train)

        b_lr += b_grad**2
        w_lr += w_grad**2


        b = b-lr/np.sqrt(b_lr)*b_grad
        w = w-lr/np.sqrt(w_lr)*w_grad

        if(e+1)%500 == 0:
            loss = -1*np.mean(y_train*np.log(pred+1e-100) + (1-y_train)*np.log(1-pred+1e-100))
            print('epoch:{}\nloss:{}\n'.format(e+1,loss))
    return w, b
	
if __name__ == '__main__':
    x_train, y_train, x_test = load_data()
    x_train, x_test = normalize(x_train, x_test)
    param = np.load('log_param.npz')
    w = param['w']
    b = param['b']
    result = np.dot(w, x_test.T) + b
    y_test = sigmoid(result)
    y_test = np.around(y_test)
    y_test = np.int_(y_test)
    f = open(sys.argv[4],"w")
    w = csv.writer(f)
    title = ['id','label']
    w.writerow(title) 
    for i in range(y_test.shape[0]):
        
        content = [i+1,y_test[i]]
        w.writerow(content) 
        
    print("Finish!")