import os
import random
import glob
import numpy as np
import pandas as pd
import time
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from process import hw3_dataset, Net

def load_data(img_path, label_path):
    train_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    train_label = pd.read_csv(label_path)
    train_label = train_label.iloc[:,1].values.tolist()
    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)
    
    train_set = train_data[4000:]
    valid_set = train_data[:4000]
    
    return train_set, valid_set
	

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()

    train_set, valid_set = load_data(sys.argv[1], sys.argv[2])
    
    #transform to tensor, data augmentation
    
    transform = transforms.Compose([
    #transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5,), std = (0.5,), inplace=False)
    ])
    
    train_dataset = hw3_dataset(train_set,transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    valid_dataset = hw3_dataset(valid_set,transform)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    model = Net()
    if use_gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    num_epoch = 30
    best_acc = 0.0
    for epoch in range(num_epoch):
        start =  time.time()
        model.train()
        train_loss = []
        train_acc = []
        for idx, (img, label) in enumerate(train_loader):
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}, time: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc), time.time()-start))

        start = time.time()
        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            for idx, (img, label) in enumerate(valid_loader):
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                output = model(img)
                loss = loss_fn(output, label)
                predict = torch.max(output, 1)[1]
                acc = np.mean((label == predict).cpu().numpy())
                valid_loss.append(loss.item())
                valid_acc.append(acc)
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}, time: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc), time.time()-start))
        
        if np.mean(train_acc) > 0.99:
            checkpoint_path = 'model_{}.pth'.format(epoch+1) 
            torch.save(model.state_dict(), checkpoint_path)
            print('model saved to %s' % checkpoint_path)
    

    #finish test code