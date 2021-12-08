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
def ffht_train(device,rStart,rEnd,L,r,O,nu,rho,wmax,batchsize,learning_rate,epochs,path):
	dataset = TensorDataset(torch.Tensor(r),torch.Tensor(O))
	dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=False,num_workers = 4,drop_last = True )
	h_nD = 20
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
	optimizer3 = optim.Adam(net3.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer4 = optim.Adam(net4.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer5 = optim.Adam(net5.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
 
	def criterion(r,O):

		r = torch.FloatTensor(r).to(device)
		O = torch.FloatTensor(O).to(device)

		r.requires_grad = True
		O.requires_grad = True
		
		net_in = torch.cat((r,O),1)
		u = net2(net_in)
		w = net3(net_in)
		P = net4(net_in)
		T = net5(net_in)
		u = u.view(len(u),-1)
		w = w.view(len(w),-1)
		P = P.view(len(P),-1)
		T = T.view(len(T),-1)

		u_hard = 0.0*u
		w_hard = (rEnd**2-r**2)*w 
		T_hard = 300.0 + T*(torch.log(r/rStart)/log(rEnd/rStart))#(r-rStart)/L*303 + (rEnd-r)/L*353 + (rStart-r)*(rEnd-r)*T

		P_hard = 0.0*P#wmax*((rEnd**2-r*r)/(rEnd**2-rStart**2))*(rStart**2/r)+(rEnd-r)*(r-rStart)*P
		
		#u_r = torch.autograd.grad(u_hard,r,grad_outputs=torch.ones_like(r),create_graph = True,only_inputs=True)[0]
		#u_rr = torch.autograd.grad((u_r*r),r,grad_outputs=torch.ones_like(r),create_graph = True,only_inputs=True)[0]

		#u_O = torch.autograd.grad(u_hard,O,grad_outputs=torch.ones_like(O),create_graph = True,only_inputs=True)[0]
		#u_OO = torch.autograd.grad(u_O,O,grad_outputs=torch.ones_like(O),create_graph = True,only_inputs=True)[0]

		w_r = torch.autograd.grad(w_hard,r,grad_outputs=torch.ones_like(r),create_graph = True,only_inputs=True)[0]
		w_rr = torch.autograd.grad(w_r,r,grad_outputs=torch.ones_like(r),create_graph = True,only_inputs=True)[0]

		w_O = torch.autograd.grad(w_hard,O,grad_outputs=torch.ones_like(O),create_graph = True,only_inputs=True)[0]
		w_OO = torch.autograd.grad(w_O,O,grad_outputs=torch.ones_like(O),create_graph = True,only_inputs=True)[0]

		#P_r = torch.autograd.grad(P_hard,r,grad_outputs=torch.ones_like(r),create_graph = True,only_inputs=True)[0]
		#P_O = torch.autograd.grad(P_hard,O,grad_outputs=torch.ones_like(O),create_graph = True,only_inputs=True)[0]
		T_r = torch.autograd.grad(T_hard,r,grad_outputs=torch.ones_like(r),create_graph = True,only_inputs=True)[0]
		#T_O = torch.autograd.grad(T_hard,O,grad_outputs=torch.ones_like(O),create_graph = True,only_inputs=True)[0]
		T_rr = torch.autograd.grad(T_r,r,grad_outputs=torch.ones_like(r),create_graph = True,only_inputs=True)[0]
		#T_OO = torch.autograd.grad(T_O,O,grad_outputs=torch.ones_like(O),create_graph = True,only_inputs=True)[0]

		loss_1 = w_O# + r*u_r + u_hard

		#loss_2 = rho*(u_hard*u_r+w_hard/r*u_O-w_hard*w_hard/r)+P_r-nu*(u_rr/r-u_hard/r**2+u_OO/r**2-2*w_O/r**2)

		loss_3 = w_r+r*w_rr-w_hard/r#rho*(u_hard*w_r+w_hard/r*w_O+u_hard*w_hard/r)+P_O/r-nu*(1/r*(w_r+r*w_rr)-w_hard/r**2+w_OO/r**2+u_O*2/r**2)

		loss_4 = T_r+r*T_rr#u_hard*T_r+w_hard/r*T_O-(1/rho*1000)*(1/r*(T_r+r*T_rr)+1/r**2*T_OO)

		loss_5 = (w_hard[0:1001,0] - wmax*rStart)
		loss_6 = (T_hard[1001:2002,0] - 350.0)


		# MSE LOSS
		loss_f = nn.MSELoss()
		loss_eq = loss_f(loss_1,torch.zeros_like(loss_1))+loss_f(loss_3,torch.zeros_like(loss_3))+loss_f(loss_4,torch.zeros_like(loss_4))
		loss_BCW = loss_f(loss_5,torch.zeros_like(loss_5))
		loss_BCT = loss_f(loss_6,torch.zeros_like(loss_6))
		loss_eq = loss_eq.view(1,-1)
		loss_BCW = loss_BCW.view(1,-1)
		loss_BCT = loss_BCT.view(1,-1)

		sigma_in = torch.cat((loss_BCW,loss_BCT),1)
		sigma_out = net1(sigma_in)
		sg1 = sigma_out[0,0]
		sg2 = sigma_out[0,1]

		loss = loss_eq + 100*((1/(sg1*sg1))*loss_BCW + (1/sg2*sg2)*loss_BCT + torch.log(sg1*sg2))

		return loss

	###################################################################

	LOSS = []
	delta = 100

	for epoch in range(epochs):
		for batch_idx, (r_in,O_in) in enumerate(dataloader):
			net1.zero_grad()
			net2.zero_grad()
			net3.zero_grad()
			net4.zero_grad()
			net5.zero_grad()
			loss = criterion(r_in,O_in)
			loss.backward()
			optimizer1.step() 
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
