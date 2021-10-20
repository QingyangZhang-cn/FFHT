import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import sqrt
import time
import ffht_test

plot_x = 0.8  #字体位置坐标
plot_y = 0.11 #字体位置坐标
fontsize = 14 #字体大小
axis_limit = [0,1,0,0.5]
path = 'Cases/'

nu = 1e-3
Um = 1.0
epochs = 10000

######################################################################

print('path is', path)

Data_CFD = np.load('XYC.npz')
x = Data_CFD['x']
y  = Data_CFD['y']
U_CFD = Data_CFD['u']
T_CFD = Data_CFD['t']

device = 'cpu'

u,v,p,t = ffht_test.ffht_test(x,y,nu,Um,epochs,path,device)
print('shape of u',u.shape)
print('shape of v',v.shape)
print('shape of T',t.shape)
print('max of T', max(t))
print('min of T', min(t))
print('max of U', max(u))
print('min of U', min(u))
print('max of U_CFD', max(U_CFD))
print('min of U_CFD', min(U_CFD))
u_err = abs(u-U_CFD)/(abs(U_CFD)+1e-8)
t_err = abs(t-T_CFD)/(abs(T_CFD)+1e-8)
print('max of u error ', max(u_err))
print('min of u error ', min(u_err))
print('max of t error ', max(t_err))
print('min of t error ', min(t_err))
#Contour Comparison
sol = np.hstack((x,y))
sol = np.hstack((sol,u))
sol = np.hstack((sol,t))
np.savetxt("hybrid.txt",sol,fmt='%f',delimiter=',')

plt.figure()
plt.subplot(212)
plt.scatter(x,y,c = u[:],vmin = min(U_CFD[:]),vmax = max(U_CFD[:]))
plt.text(plot_x,plot_y,r'DNN',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.colorbar()
plt.subplot(211)
plt.scatter(x,y,c = U_CFD[:],vmin = min(U_CFD[:]),vmax = max(U_CFD[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'CFD',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.savefig('plot/'+'uContour_test.png',bbox_inches=  'tight')

plt.figure()
plt.subplot(111)
plt.scatter(x,y,c = u_err[:],vmin = min(u_err[:]),vmax = max(u_err[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'U_ERROR',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.savefig('plot/'+'U_error.png',bbox_inches=  'tight')

plt.figure()
plt.subplot(312)
plt.scatter(x,y,c = t[:],vmin = min(T_CFD[:]),vmax = max(T_CFD[:]))
plt.text(plot_x,plot_y,r'DNN',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.colorbar()
plt.subplot(311)
plt.scatter(x,y,c = T_CFD[:],vmin = min(T_CFD[:]),vmax = max(T_CFD[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'CFD',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.savefig('plot/'+'TContour_test.png',bbox_inches=  'tight')

plt.figure()
plt.subplot(411)
plt.scatter(x,y,c = t_err[:],vmin = min(t_err[:]),vmax = max(t_err[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'T_ERROR',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.savefig('plot/'+'T_error.png',bbox_inches=  'tight')

plt.close('all')
	
