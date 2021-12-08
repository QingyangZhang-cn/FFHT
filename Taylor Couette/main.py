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
rStart = 17.8e-3
rEnd = 46.28e-3
L = rEnd - rStart
nu = 2e-4
rho = 1
wmax = 1

np.random.seed(1)

Data = np.load('ROC.npz')
r = Data['r']
O = Data['O']
print('shape of  r',r.shape)
print('shape of  O',O.shape)

batchsize = 1001*41

learning_rate = 1e-3


path = "Cases/"


ffht_train.ffht_train(device,rStart,rEnd,L,r,O,nu,rho,wmax,batchsize,learning_rate,epochs,path)







