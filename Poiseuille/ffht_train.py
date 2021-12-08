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
def ffht_train(device,xStart,xEnd,L,rInlet,x,y,dP,nu,rho,g,k,Cp,qs,batchsize,learning_rate,epochs,path):
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
				nn.Sigmoid(),
				nn.Linear(h_nD,h_nD),
				nn.Sigmoid(),
				nn.Linear(h_nD,2), 
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

	#net for energy equation2
	class Net6(nn.Module):
		def __init__(self):
			super(Net6, self).__init__()
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
	net6 = Net6().to(device)

	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	net1.apply(init_normal)
	net2.apply(init_normal)
	net3.apply(init_normal)
	net4.apply(init_normal)
	net5.apply(init_normal)
	net6.apply(init_normal)
	############################################################################
	optimizer1 = optim.Adam(net1.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer3	= optim.Adam(net3.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer4	= optim.Adam(net4.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer5	= optim.Adam(net5.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer6	= optim.Adam(net6.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)

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
		T_2 = net6(net_in)

		u = u.view(len(u),-1)
		v = v.view(len(v),-1)
		P = P.view(len(P),-1)
		T = T.view(len(T),-1)
		T_2 = T_2.view(len(T),-1)


		u_hard = u*(rInlet**2 - y**2)
		v_hard = v*0.0#v*(rInlet**2 -y**2)

		T_hard = T*x
		T2_hard = T_2

		P_hard = (xStart-x)*0 + dP*(xEnd-x)/L + 0*y + (xStart - x)*(xEnd - x)*P
		
		u_x = torch.autograd.grad(u_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_y = torch.autograd.grad(u_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		P_x = torch.autograd.grad(P_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		#P_xx = torch.autograd.grad(P_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		loss_1 =1/rho*P_x-nu*u_yy #(u*u_x+v*u_y-nu*(u_xx+u_yy)+1/rho*P_x)

		v_x = torch.autograd.grad(v_hard,x,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		
		v_y = torch.autograd.grad(v_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		
		v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		P_y = torch.autograd.grad(P_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		#P_yy = torch.autograd.grad(P_y,y,grad_outputs=torch.ones_like(x),create_graph = True,allow_unused = True)[0]

		T_x = torch.autograd.grad(T_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		T_xx = torch.autograd.grad(T_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		T_y = torch.autograd.grad(T2_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		T_yy = torch.autograd.grad(T_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		#T2_x = torch.autograd.grad(T_2,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		#T1_y = torch.autograd.grad(T,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

		#loss_2 = (u*v_x+v*v_y - nu*(v_xx+v_yy)+1/rho*P_y)
		loss_3 = u_x #(u_x + v_y)
		loss_4 = u_hard*T_x-(k/(Cp*rho))*T_yy #(u*T_x+v*T_y-(k/(Cp*rho))*(T_xx+T_yy))
		loss_BCD = T_y[0:1000,:]+qs/k
		loss_BCU = T_y[1000:2000,:]-qs/k
		loss_BCDU = T2_hard[0:2000,:]-0.0
		loss_IC = T_x-24.0#T2_hard[2000:2041,:] - (10000*(6*rInlet**2*y[2000:2041,:]*y[2000:2041,:]-y[2000:2041,:]**4-5*rInlet**4))


		# MSE LOSS
		loss_f = nn.MSELoss()
		loss_eqU = loss_f(loss_1,torch.zeros_like(loss_1))+loss_f(loss_3,torch.zeros_like(loss_3))
		loss_eqT = loss_f(loss_4,torch.zeros_like(loss_4))
		loss_BCT = loss_f(loss_BCD,torch.zeros_like(loss_BCD)) + loss_f(loss_BCU,torch.zeros_like(loss_BCU)) + loss_f(loss_BCDU,torch.zeros_like(loss_BCDU))
		loss_ICT = loss_f(loss_IC,torch.zeros_like(loss_IC))
		loss_eqU = loss_eqU.view(1,-1)
		loss_eqT = loss_eqT.view(1,-1)
		loss_BCT = loss_BCT.view(1,-1)
		loss_ICT = loss_ICT.view(1,-1)

		sigma_in = torch.cat((loss_eqU,loss_eqT),1)
		sigma_out = net1(sigma_in)
		sg1 = sigma_out[0,0]
		sg2 = sigma_out[0,1]

		loss = ((1/(sg1*sg1))*loss_eqU + (1/sg2*sg2)*loss_eqT + torch.log(sg1*sg2)) + 100*(loss_BCT+loss_ICT)

		return loss

	###################################################################

	# Main loop
	LOSS = []
	delta = 100

	for epoch in range(epochs):
		for batch_idx, (x_in,y_in) in enumerate(dataloader):
    		#zero gradient
			net1.zero_grad()
			net2.zero_grad()
			net3.zero_grad()
			net4.zero_grad()
			net5.zero_grad()
			net6.zero_grad()
			loss = criterion(x_in,y_in)
			loss.backward()
			#return loss
			optimizer1.step() 
			optimizer2.step() 
			optimizer3.step()
			optimizer4.step()
			optimizer5.step()
			optimizer6.step()
			if (batch_idx+1) % 1 == 0:
				print('Train Epoch: {} \tLoss: {:.10f}'.format(epoch, loss.item()))
				LOSS.append(loss.item())
			if (loss.item()<delta):
				delta = loss.item()
				torch.save(net2.state_dict(),path+"PINN_FFHT_"+"hard_u.pt")
				torch.save(net3.state_dict(),path+"PINN_FFHT_"+"hard_v.pt")
				torch.save(net4.state_dict(),path+"PINN_FFHT_"+"hard_P.pt")
				torch.save(net5.state_dict(),path+"PINN_FFHT_"+"hard_T.pt")
				torch.save(net6.state_dict(),path+"PINN_FFHT_"+"hard_T_2.pt")


	############################################################
	LOSS = np.array(LOSS)
	np.savetxt('PINN_FFHT_Loss.csv',LOSS)
	############################################################

	torch.save(net2.state_dict(),path+"PINN_FFHT_"+str(epochs)+"hard_u.pt")
	torch.save(net3.state_dict(),path+"PINN_FFHT_"+str(epochs)+"hard_v.pt")
	torch.save(net4.state_dict(),path+"PINN_FFHT_"+str(epochs)+"hard_P.pt")
	torch.save(net5.state_dict(),path+"PINN_FFHT_"+str(epochs)+"hard_T.pt")
	torch.save(net6.state_dict(),path+"PINN_FFHT_"+str(epochs)+"hard_T_2.pt")
	#####################################################################
