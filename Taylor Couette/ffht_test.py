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
from math import exp, sqrt,pi,log
import time
def ffht_test(r,O,nu,wmax,rStart,rEnd,L,epochs,path1,device):
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
	##
	net2.eval()
	net3.eval()
	net4.eval()
	net5.eval()

	D = 0.1
	########################################
	rt = torch.FloatTensor(r).to(device)
	Ot = torch.FloatTensor(O).to(device)
	rt = rt.view(len(rt),-1)
	Ot = Ot.view(len(Ot),-1)

	###################################
	rt.requires_grad = True
	Ot.requires_grad = True

	net_in = torch.cat((rt,Ot),1)

	u_t = net2(net_in)
	w_t = net3(net_in)
	P_t = net4(net_in)
	T_t = net5(net_in)

	u_hard = 0.0*u_t#(rt-rStart)/L*0 + (rEnd-rt)/L*0 + (rStart-rt)*(rEnd-rt)*u_t
	w_hard = (rEnd**2-rt**2)*w_t #/(rEnd**2-rStart**2)*w_t #+ (rStart**2-rt**2)*(rEnd**2-rt**2)*w_t #wmax*((rEnd**2-rt**2)/(rEnd**2-rStart**2))*(rStart**2/rt**2)+(rEnd-rt)*(rt-rStart)*w_t
	T_hard = 300.0 + T_t*(torch.log(rt/rStart)/log(rEnd/rStart))#(rt-rStart)/L*303 + (rEnd-rt)/L*353 + (rStart-rt)*(rEnd-rt)*T_t

	P_hard = 0.0*P_t
	u_hard = u_hard.cpu().data.numpy()
	w_hard = w_hard.cpu().data.numpy()
	T_hard = T_hard.cpu().data.numpy()

	return w_hard,T_hard

