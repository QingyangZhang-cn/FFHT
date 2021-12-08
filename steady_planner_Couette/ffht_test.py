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

def ffht_test(x,y,nu,Um,epochs,path1,device):
	h_nD = 30
	h_n = 20
	h_np = 20
	class Swish(nn.Module):
		def __init__(self, inplace=True):
			super(Swish, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			if self.inplace:
				x.mul_(torch.sigmoid(x))
				return x
			else:
				return x * torch.sigmoid(x)
	class Net1(nn.Module):
		def __init__(self):
			super(Net1, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(2,h_nD),
				nn.Tanh(),
				nn.Linear(h_nD,h_nD),
				nn.Tanh(),
				nn.Linear(h_nD,h_nD),
				nn.Tanh(),

				nn.Linear(h_nD,1),
			)
		def forward(self,x):
			output = self.main(x)
			return  output
	class Net2(nn.Module):
		def __init__(self):
			super(Net2, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(2,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),

				nn.Linear(h_n,1),
			)
		def forward(self,x):
			output = self.main(x)
			return  output

	class Net3(nn.Module):
		def __init__(self):
			super(Net3, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(2,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),

				nn.Linear(h_n,1),
			)
		def forward(self,x):
			output = self.main(x)
			return  output

	class Net4(nn.Module):
		def __init__(self):
			super(Net4, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(2,h_np),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_np,h_np),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_np,h_np),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_np,1),
			)
		def forward(self,x):
			output = self.main(x)
			return  output

	class Net5(nn.Module):
		def __init__(self):
			super(Net5, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(2,h_np),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_np,h_np),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_np,h_np),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_np,1),
			)
		def forward(self,x):
			output = self.main(x)
			return  output
	################################################################
	net1 = Net1()
	net2 = Net2()
	net3 = Net3()
	net4 = Net4()
	net5 = Net5()

	pre = ''
	net2.load_state_dict(torch.load(pre+path1+"PINN_FFHT_"+"hard_u.pt",map_location = 'cpu'))
	net3.load_state_dict(torch.load(pre+path1+"PINN_FFHT_"+"hard_v.pt",map_location = 'cpu'))
	net4.load_state_dict(torch.load(pre+path1+"PINN_FFHT_"+"hard_P.pt",map_location = 'cpu'))
	net5.load_state_dict(torch.load(pre+path1+"PINN_FFHT_"+"hard_T.pt",map_location = 'cpu'))

	net2.eval()
	net3.eval()
	net4.eval()
	net5.eval()

	D = 0.5
	########################################
	xt = torch.FloatTensor(x).to(device)
	yt = torch.FloatTensor(y).to(device)
	xt = xt.view(len(xt),-1)
	yt = yt.view(len(yt),-1)

	###################################
	xt.requires_grad = True
	yt.requires_grad = True

	net_in = torch.cat((xt,yt),1)

	u_t = net2(net_in)
	v_t = net3(net_in)
	P_t = net4(net_in)
	T_t = net5(net_in)

	u_hard = yt*u_t
	v_hard = yt*v_t
	T_hard = 300+T_t*yt
	P_hard = 0.0*P_t
	u_hard = u_hard.cpu().data.numpy()
	v_hard = v_hard.cpu().data.numpy()
	P_hard = P_hard.cpu().data.numpy()
	T_hard = T_hard.cpu().data.numpy()

	return u_hard,v_hard,P_hard,T_hard

