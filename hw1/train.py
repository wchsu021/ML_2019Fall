import math
import numpy as np
import pandas as pd
import csv
import sys

def readdata(data):
    
	# 把有些數字後面的奇怪符號刪除
	for col in list(data.columns[2:]):
		data[col] = data[col].astype(str).map(lambda x: x.rstrip('x*#A'))
	data = data.values
	
	# 刪除欄位名稱及日期
	data = np.delete(data, [0,1], 1)
	
	# 特殊值補0
	data[ data == 'NR'] = 0
	data[ data == ''] = 0
	data[ data == 'nan'] = 0
	data = data.astype(np.float)

	return data

def extract(data):
	N = data.shape[0] // 18

	temp = data[:18, :]
    
    # Shape 會變成 (x, 18) x = 取多少hours
	for i in range(1, N):
		temp = np.hstack((temp, data[i*18: i*18+18, :]))
	return temp

def valid(x, y):
	if y <= 1 or y > 80:
		return False
	for i in range(9):
		if x[9,i] <= 2 or x[9,i] > 100:
			return False
	return True

def parse2train(data):
	x = []
	y = []
	
	# 用前面9筆資料預測下一筆PM2.5 所以需要-9
	total_length = data.shape[1] - 9
	for i in range(total_length):
		x_tmp = data[:,i:i+9]
		y_tmp = data[9,i+9]
		if valid(x_tmp, y_tmp):
			x.append(x_tmp.reshape(-1,))
			y.append(y_tmp)
	# x 會是一個(n, 18, 9)的陣列， y 則是(n, 1) 
	x = np.array(x)
	y = np.array(y)
	return x,y
    
def minibatch(x, y):
    # 打亂data順序
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    
    # 訓練參數以及初始化
    batch_size = 128
    lr = 0.001
    lam = 0.001
    beta_1 = np.full(x[0].shape, 0.9).reshape(-1, 1)
    beta_2 = np.full(x[0].shape, 0.99).reshape(-1, 1)
    w = np.full(x[0].shape, 0.1).reshape(-1, 1)
    bias = 0.1
    m_t = np.full(x[0].shape, 0).reshape(-1, 1)
    v_t = np.full(x[0].shape, 0).reshape(-1, 1)
    m_t_b = 0.0
    v_t_b = 0.0
    t = 0
    epsilon = 1e-8
    
    
    for num in range(50):
        for b in range(int(x.shape[0]/batch_size)):
            t+=1
            x_batch = x[b*batch_size:(b+1)*batch_size]
            y_batch = y[b*batch_size:(b+1)*batch_size].reshape(-1,1)
            loss = y_batch - np.dot(x_batch,w) - bias
            
            # 計算gradient
            g_t = np.dot(x_batch.transpose(),loss) * (-2) +  2 * lam * np.sum(w)
            g_t_b = loss.sum(axis=0) * (2)
            m_t = beta_1*m_t + (1-beta_1)*g_t 
            v_t = beta_2*v_t + (1-beta_2)*np.multiply(g_t, g_t)
            m_cap = m_t/(1-(beta_1**t))
            v_cap = v_t/(1-(beta_2**t))
            m_t_b = 0.9*m_t_b + (1-0.9)*g_t_b
            v_t_b = 0.99*v_t_b + (1-0.99)*(g_t_b*g_t_b) 
            m_cap_b = m_t_b/(1-(0.9**t))
            v_cap_b = v_t_b/(1-(0.99**t))
            w_0 = np.copy(w)
            
            # 更新weight, bias
            w -= ((lr*m_cap)/(np.sqrt(v_cap)+epsilon)).reshape(-1, 1)
            bias -= (lr*m_cap_b)/(math.sqrt(v_cap_b)+epsilon)
            

    return w, bias
    
if __name__ == "__main__":
    
    #print(sys.argv[1])
    year1_pd = pd.read_csv(sys.argv[1])
    year1 = readdata(year1_pd)
    train_data1 = extract(year1)
    
    year2_pd = pd.read_csv(sys.argv[2])
    year2 = readdata(year2_pd)
    train_data2 = extract(year2)
    
    train_data = np.concatenate((train_data1, train_data2), axis = 1)
    train_x, train_y = parse2train(train_data)
    w, bias = minibatch(train_x, train_y)
    
    np.savez('param.npz',w = w, bias = bias)
    np.set_printoptions(suppress=True) 
    '''
    #print(w)
    #print(bias)
    for i in range(162):
        if w[i] < 0.001:
           w[i] = 0
    np.savez('param_g.npz',w = w, bias = bias)
    #print("------------------------")
    #print(w)
    '''
    print("Finish training!")