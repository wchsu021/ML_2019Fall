import os
import random
import glob
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv
from process import hw3_dataset, Net

def load_data_test(img_path):
    test_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    test_label = [0 for i in range(len(test_image))]
    test_data = list(zip(test_image, test_label))
    return test_data
    

if __name__ == '__main__':    
    use_gpu = torch.cuda.is_available()
    result = []
    model = Net()
    #print("Loading model")
    model.load_state_dict(torch.load("model_21.pth"))
    model.eval()
    #print("Loading data")
    test_set = load_data_test(sys.argv[1])
    transform = transforms.Compose([
        #transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5,), std = (0.5,), inplace=False)
        ])
    test_dataset = hw3_dataset(test_set,transform)
    test_loader = DataLoader(test_dataset, shuffle=False)
    #test_loader = DataLoader(test_dataset, shuffle=False)
    #print("Start predict")
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            img, _ = data
            output = model(img)
            predict = torch.max(output, 1)[1].data.numpy()
            #print(predict)
            result.append(predict)
           
        #print(len(test_set))   
        #print(len(result))
        
    #print("Start output")
    f = open(sys.argv[2],"w")
    w = csv.writer(f)
    title = ['id','label']
    w.writerow(title) 

    for i in range(len(test_set)):
        content = [str(i),result[i][0]]
        w.writerow(content) 
    
    print("Finish!")


