import numpy as np 
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import manifold, cluster
from process import Autoencoder

if __name__ == '__main__':
 
    # detect is gpu available.
    use_gpu = torch.cuda.is_available()
 
    autoencoder = Autoencoder()
    
    # load data and normalize to [-1, 1]
    trainX = np.load('trainX.npy')
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
    trainX = torch.Tensor(trainX)
 
    # if use_gpu, send model / data to GPU.
    if use_gpu:
        autoencoder.cuda()
        trainX = trainX.cuda()
 
    # Dataloader: train shuffle = True
    train_dataloader = DataLoader(trainX, batch_size=128, shuffle=True)
 
    # We set criterion : L1 loss (or Mean Absolute Error, MAE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
 
    # Now, we train 20 epochs.
    best_loss = 100
    for epoch in range(30):
        start_time = time.time()
        cumulate_loss = 0
        for x in train_dataloader:
            
            latent, reconstruct = autoencoder(x)
            loss = criterion(reconstruct, x)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cumulate_loss = loss.item() * x.shape[0]
 
        print(f'Epoch { "%03d" % epoch }: Loss : { "%.5f" % (cumulate_loss / trainX.shape[0])}: Time :{ "%.5f" % (time.time()-start_time)}')
        if cumulate_loss < best_loss:
            best_loss = cumulate_loss
            checkpoint_path = 'model/model_{}.pth'.format(epoch) 
            torch.save(autoencoder.state_dict(), checkpoint_path)
            print('model saved to %s' % checkpoint_path)
            

    print("Finish training")
    