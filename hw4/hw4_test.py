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
import sys

if __name__ == '__main__':
 
    # detect is gpu available.
    use_gpu = torch.cuda.is_available()
    autoencoder = Autoencoder()
    
    # load data and normalize to [-1, 1]
    trainX = np.load(sys.argv[1])
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
    trainX = torch.Tensor(trainX)
 
    # if use_gpu, send model / data to GPU.
    if use_gpu:
        autoencoder.cuda()
        trainX = trainX.cuda()
 
    # Dataloader: train shuffle = True
    test_dataloader = DataLoader(trainX, batch_size=128, shuffle=False)
    
    autoencoder.load_state_dict(torch.load('model_22.pth'))
    autoencoder.eval()
    latents = []
    reconstructs = []
    for x in test_dataloader:
        latent, reconstruct = autoencoder(x)
        latents.append(latent.cpu().detach().numpy())
        reconstructs.append(reconstruct.cpu().detach().numpy())
 
    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)
 
    # Use PCA to lower dim of latents and use K-means to clustering.
    print("Start lower dim!")
    start_time = time.time()
    latents = PCA(n_components=128, whiten = True, svd_solver="full", random_state=0).fit_transform(latents)
    #latents = manifold.TSNE(n_components=2).fit_transform(latents)
    print(f'Time :{ "%.5f" % (time.time()-start_time)}')
    print("Start clustering!")
    start_time = time.time()
    #result = cluster.SpectralClustering(n_clusters = 2).fit(latents).labels_
    result = KMeans(n_clusters = 2).fit(latents).labels_
    print(f'Time :{ "%.5f" % (time.time()-start_time)}')
    
    # We know first 5 labels are zeros, it's a mechanism to check are your answers
    # need to be flipped or not.
    if np.sum(result[:5]) >= 3:
        result = 1 - result
        
    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(sys.argv[2],index=False)
    print("Finish testing")