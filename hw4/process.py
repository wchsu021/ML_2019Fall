import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # define: encoder
        self.encoder = nn.Sequential(
          nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(16),
          #nn.LeakyReLU(),
          nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(32),
          #nn.LeakyReLU(),
          nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(64),
          #nn.LeakyReLU(),
          nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(16),
          #nn.LeakyReLU(),
          
        )
        
        self.latent = nn.Sequential(
            nn.Linear(8*8*16, 2048),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048),
        )
        
        self.sample = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 8*8*16),
            nn.BatchNorm1d(32*32*16)
        )
 
        # define: decoder
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
          nn.BatchNorm2d(64),
          #nn.LeakyReLU(),
          nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(32),
          #nn.LeakyReLU(),
          nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
          nn.BatchNorm2d(16),
          #nn.LeakyReLU(),
          nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False),
          
        )
 
 
    def forward(self, x):
        x = self.encoder(x)
        #x = x.view(-1, 8*8*16)
        #x = x.view(-1, 8*8*16)
        #x = self.latent(x)
        #y = self.sample(x)
        #y = y.view(-1,16,32,32)
        #y = self.decoder(y)
        y = self.decoder(x)
        y = y / 2.0
        #y = y.view(-1, 3, 32, 32)
        return x, y