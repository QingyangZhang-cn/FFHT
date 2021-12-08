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
				nn.Linear(4,h_nD),
				nn.Sigmoid(),
				nn.Linear(h_nD,h_nD),
				nn.Sigmoid(),
				nn.Linear(h_nD,4),
				nn.Sigmoid(),
			)
		def forward(self,x):
			output = 1.0+10*self.main(x)
			return output
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

	net1.apply(init_normal)
	net2.apply(init_normal)
	net3.apply(init_normal)
	net4.apply(init_normal)
	net5.apply(init_normal)
	############################################################################
	optimizer1 = optim.Adam(net1.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
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
		#v = net3(net_in)
		#P = net4(net_in)
		T = net5(net_in)
		u = u.view(len(u),-1)
		#v = v.view(len(v),-1)
		#P = P.view(len(P),-1)
		T = T.view(len(T),-1)
		u_hard = Um*y/D - u*(0.5-y)*y
		#v_hard = 0.0*v#(y-0)/D*0 + (D-y)/D*0 + (0-y)*(D-y)*v
		T_hard = 50*y/D - T*(0.5-y)*y
		
		u_t = torch.autograd.grad(u_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		#u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_y = torch.autograd.grad(u_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		#P_x = torch.autograd.grad(P_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		#P_xx = torch.autograd.grad(P_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]


		T_t = torch.autograd.grad(T_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		#T_xx = torch.autograd.grad(T_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		T_y = torch.autograd.grad(T_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		T_yy = torch.autograd.grad(T_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		loss_1 = u_t-nu*u_yy#(u_hard*u_x+v_hard*u_y-nu*(u_xx+u_yy)+1/rho*P_x)
		loss_4 = T_t - 0.01*T_yy#(u_hard*T_x+v_hard*T_y-(0.6/(4182*1))*(T_xx+T_yy))
		loss_BC1 = u_hard[1000:2000,:] - 1.0
		loss_BC2 = T_hard[1000:2000,:]- 50.0
		loss_BC3 = u_hard[0:1000,:]
		loss_BC4 = T_hard[0:1000,:]
		loss_IC1 = u_hard[2000:2051,:]
		loss_IC2 = T_hard[2000:2051,:]


		# MSE LOSS
		loss_f = nn.MSELoss()
		loss_eq = loss_f(loss_1,torch.zeros_like(loss_1))+loss_f(loss_4,torch.zeros_like(loss_4))
		loss_BCU = loss_f(loss_BC1,torch.zeros_like(loss_BC1))+loss_f(loss_BC3,torch.zeros_like(loss_BC3))
		loss_BCT = loss_f(loss_BC2,torch.zeros_like(loss_BC2))+loss_f(loss_BC4,torch.zeros_like(loss_BC4))
		loss_ICU = loss_f(loss_IC1,torch.zeros_like(loss_IC1))
		loss_ICT = loss_f(loss_IC2,torch.zeros_like(loss_IC2))

		loss_eq = loss_eq.view(1,-1)
		loss_BCU = loss_BCU.view(1,-1)
		loss_BCT = loss_BCT.view(1,-1)
		loss_ICU = loss_ICU.view(1,-1)
		loss_ICT = loss_ICT.view(1,-1)

		sigma_in = torch.cat((loss_BCU,loss_BCT,loss_ICU,loss_ICT),1)
		sigma_out = net1(sigma_in)
		sg1 = sigma_out[0,0]
		sg2 = sigma_out[0,1]
		sg3 = sigma_out[0,2]
		sg4 = sigma_out[0,3]

		loss = loss_eq + 100*((1/(sg1*sg1))*loss_BCU + (1/sg2*sg2)*loss_BCT + (1/sg3*sg3)*loss_ICU + (1/sg4*sg4)*loss_ICT + torch.log(sg1*sg2*sg3*sg4))

		return loss

	###################################################################

	# Main loop
	LOSS = []
	tic = time.time()
	delta = 100

	for epoch in range(epochs):
		for batch_idx, (x_in,y_in) in enumerate(dataloader):
			net1.zero_grad()
			net2.zero_grad()
			#net3.zero_grad()
			#net4.zero_grad()
			net5.zero_grad()
			loss = criterion(x_in,y_in)
			loss.backward()
			optimizer1.step() 
			optimizer2.step()
			#optimizer4.step()
			optimizer5.step()
			if (batch_idx+1) % 1 == 0:
				print('Train Epoch: {} \tLoss: {:.10f}'.format(epoch, loss.item()))
				LOSS.append(loss.item())
			if (loss.item()<delta):
				delta = loss.item()
				torch.save(net2.state_dict(),path+"PINN_FFHT_"+"hard_u.pt")
				torch.save(net5.state_dict(),path+"PINN_FFHT_"+"hard_T.pt")
	############################################################
	LOSS = np.array(LOSS)
	np.savetxt('PINN_FFHT_time_loss.csv',LOSS)
	############################################################
	#save network
	#torch.save(net1.state_dict(),"stenosis_para_axisy_sigma"+str(sigma)+"scale"+str(scale)+"_epoch"+str(epochs)+"boundary.pt")
	torch.save(net2.state_dict(),path+"geo_para_axisy_epoch"+str(epochs)+"hard_u.pt")
	#torch.save(net3.state_dict(),path+"geo_para_axisy_epoch"+str(epochs)+"hard_v.pt")
	#torch.save(net4.state_dict(),path+"geo_para_axisy_epoch"+str(epochs)+"hard_P.pt")
	torch.save(net5.state_dict(),path+"geo_para_axisy_epoch"+str(epochs)+"hard_T.pt")
	#####################################################################
