import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
from pathlib import Path
import sys
import os

import ffht_train
device = torch.device("cpu")
epochs  = 100000

L = 10
tStart = 0
tEnd = tStart+L
D = 0.5
nu = 1e-2
np.random.seed(1)
Data = np.load('txY.npz')
tx = Data['tx']
y = Data['y']
print('shape of Inner tx',tx.shape)
print('shape of Inner y',y.shape)
batchsize = 1001*51
Um = 1.0
g = 9.8
rho = 1
learning_rate = 1e-3
path = "Cases/"

ffht_train.ffht_train(device,tStart,tEnd,L,D,tx,y,Um,nu,rho,g,batchsize,learning_rate,epochs,path)





