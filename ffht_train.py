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
def ffht_train(device,xStart,xEnd,L,D,x,y,Um,nu,rho,g,batchsize,learning_rate,epochs,path):
	dataset = TensorDataset(torch.Tensor(x),torch.Tensor(y))
	dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=False,num_workers = 4,drop_last = True )
	h_nD = 30
	h_n = 20
	input_n = 2
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
				nn.Linear(input_n,h_n),
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
				nn.Linear(input_n,h_n),
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
				nn.Linear(input_n,h_n),
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
	################################################################
	#net for energy equation
	class Net5(nn.Module):
		def __init__(self):
			super(Net5, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
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
#############################################################################
	net1 = Net1().to(device)
	net2 = Net2().to(device)
	net3 = Net3().to(device)
	net4 = Net4().to(device)
	net5 = Net5().to(device)
 
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	# use the modules apply function to recursively apply the initialization
	net1.apply(init_normal)
	net2.apply(init_normal)
	net3.apply(init_normal)
	net4.apply(init_normal)
	net5.apply(init_normal)
	#############################Adam optimizer########################################

	optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer3	= optim.Adam(net3.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer4	= optim.Adam(net4.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer5	= optim.Adam(net5.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
 
	def criterion(x,y):

		x = torch.FloatTensor(x).to(device)
		y = torch.FloatTensor(y).to(device)

		x.requires_grad = True
		y.requires_grad = True
		
		net_in = torch.cat((x,y),1)
		u = net2(net_in)
		v = net3(net_in)
		P = net4(net_in)
		T = net5(net_in)
		u = u.view(len(u),-1)
		v = v.view(len(v),-1)
		P = P.view(len(P),-1)
		T = T.view(len(T),-1)

		u_hard = u*y
		v_hard = v*y
		T_hard = 300+T*y
		P_hard = 0.0*P


		
		u_x = torch.autograd.grad(u_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_y = torch.autograd.grad(u_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		P_x = torch.autograd.grad(P_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]

		

		v_x = torch.autograd.grad(v_hard,x,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_y = torch.autograd.grad(v_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		P_y = torch.autograd.grad(P_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		T_x = torch.autograd.grad(T_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		T_xx = torch.autograd.grad(T_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		T_y = torch.autograd.grad(T_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		T_yy = torch.autograd.grad(T_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

		loss_1 = (u_hard*u_x+v_hard*u_y-nu*(u_xx+u_yy)+1/rho*P_x)
		loss_2 = (u_hard*v_x+v_hard*v_y - nu*(v_xx+v_yy)+1/rho*P_y)
		loss_3 = (u_x + v_y)
		loss_4 = (u_hard*T_x+v_hard*T_y-(0.6/(4182*1))*(T_xx+T_yy))
		loss_BC = u_hard[1001:2002,:] - 1.0
		loss_BC1 = T_hard[1001:2002,:]- 350.0

	################# LOSS #################################
		loss_f = nn.MSELoss()

		loss = 10000*(loss_f(loss_1,torch.zeros_like(loss_1))+ loss_f(loss_2,torch.zeros_like(loss_2))+loss_f(loss_3,torch.zeros_like(loss_3))+loss_f(loss_4,torch.zeros_like(loss_4))) + loss_f(loss_BC,torch.zeros_like(loss_BC))+loss_f(loss_BC1,torch.zeros_like(loss_BC1))

		return loss

	###################################################################
	#training loop
	LOSS = []
	delta = 10.0

	for epoch in range(epochs):
		for batch_idx, (x_in,y_in) in enumerate(dataloader):
			net2.zero_grad()
			net3.zero_grad()
			net4.zero_grad()
			net5.zero_grad()
			loss = criterion(x_in,y_in)
			loss.backward()
			optimizer2.step() 
			optimizer3.step()
			optimizer4.step()
			optimizer5.step()
			if (batch_idx+1) % 1 == 0:
				print('Train Epoch: {} \tLoss: {:.10f}'.format(epoch, loss.item()))
				LOSS.append(loss.item())
			if (loss.item()<delta):
				delta = loss.item()
				torch.save(net2.state_dict(),path+"PINN_FFHT_"+"hard_u.pt")
				torch.save(net3.state_dict(),path+"PINN_FFHT_"+"hard_v.pt")
				torch.save(net4.state_dict(),path+"PINN_FFHT_"+"hard_P.pt")
				torch.save(net5.state_dict(),path+"PINN_FFHT_"+"hard_T.pt")
	############################################################
	LOSS = np.array(LOSS)
	np.savetxt('PINN_FFHT_LOSS.csv',LOSS)
	############################################################
	#save network
	torch.save(net2.state_dict(),path+"PINN_FFHT_"+str(epochs)+"hard_u.pt")
	torch.save(net3.state_dict(),path+"PINN_FFHT_"+str(epochs)+"hard_v.pt")
	torch.save(net4.state_dict(),path+"PINN_FFHT_"+str(epochs)+"hard_P.pt")
	torch.save(net5.state_dict(),path+"PINN_FFHT_"+str(epochs)+"hard_T.pt")
	#####################################################################
