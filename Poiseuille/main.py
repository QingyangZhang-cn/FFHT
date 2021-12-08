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
epochs  = 50000
L = 1
xStart = 0
xEnd = xStart+L
rInlet = 0.05
nu = 1e-3

np.random.seed(1)
Data = np.load('XYP.npz')
x = Data['x']
y = Data['y']
print('shape of Inner x',x.shape)
print('shape of Inner y',y.shape)

batchsize = 1001*41

dP = 0.1
g = 9.8
rho = 1
k = 1
Cp = 100
qs = 10
learning_rate = 1e-3


path = "Cases/"

ffht_train.ffht_train(device,xStart,xEnd,L,rInlet,x,y,dP,nu,rho,g,k,Cp,qs,batchsize,learning_rate,epochs,path)





