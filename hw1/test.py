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
    
if __name__ == "__main__":
    
    test_pd = pd.read_csv(sys.argv[1])
    test = readdata(test_pd)
    test_data = test.reshape(-1, 162)
    param = np.load('param.npz')
    #print(param['w'])
    #print(param['bias'])
    w = param['w']
    bias = param['bias']
    result = np.dot(test_data, w)+bias
   
    f = open(sys.argv[2],"w")
    w = csv.writer(f)
    title = ['id','value']
    w.writerow(title) 
    for i in range(500):
        content = ['id_'+str(i),result[i][0]]
        w.writerow(content) 
    
    print("Finish testing!")