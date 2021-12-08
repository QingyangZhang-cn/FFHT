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

####################Setting############################
device = torch.device("cpu")
epochs  = 100000
batchsize = 1001*41
learning_rate = 1e-3
path = "Cases/"
####################Parameters#########################
L = 1
xStart = 0
xEnd = xStart+L
D = 0.5
nu = 1e-3
Um = 1.0
g = 9.8
rho = 1
####################Reading X Y########################
np.random.seed(1)
Data = np.load('XYC.npz')
x = Data['x']
y = Data['y']
print('shape of x',x.shape)
print('shape of y',y.shape)

####################Training###########################

ffht_train.ffht_train(device,xStart,xEnd,L,D,x,y,Um,nu,rho,g,batchsize,learning_rate,epochs,path)

print ("Training is over!\n")




