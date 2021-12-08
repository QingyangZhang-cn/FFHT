import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import sqrt
import time
import ffht_test


plot_x = 0.8  #字体位置坐标
plot_y = 0.11 #字体位置坐标
fontsize = 14 #字体大小
axis_limit = [0,10,0,0.5]
path = 'Cases/'
nu = 1e-3
Um = 1.0
mu = 0.5
epochs = 100000

print('path is', path)
Data_CFD = np.load('txY.npz')
x = Data_CFD['tx']
y  = Data_CFD['y']

device = 'cpu'
u,t = ffht_test.ffht_test(x,y,nu,Um,mu,epochs,path,device)
print('shape of u',u.shape)
print('shape of T',t.shape)
print('max of T', max(t))
print('min of T', min(t))
print('max of U', max(u))
print('min of U', min(u))
sol = np.hstack((x,y))
sol = np.hstack((sol,u))
sol = np.hstack((sol,t))
np.savetxt("time_hybrid.txt",sol,fmt='%f',delimiter=',')


plt.figure()
plt.subplot(212)
plt.scatter(x,y,c = u[:],vmin = min(u[:]),vmax = max(u[:]))
plt.text(plot_x,plot_y,r'DNN',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.colorbar()
plt.subplot(211)
plt.scatter(x,y,c = u[:],vmin = min(u[:]),vmax = max(u[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'CFD',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.savefig('plot/'+'uContour_test.png',bbox_inches=  'tight')

plt.figure()
plt.subplot(312)
plt.scatter(x,y,c = t[:],vmin = min(t[:]),vmax = max(t[:]))
plt.text(plot_x,plot_y,r'DNN',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.colorbar()
plt.subplot(311)
plt.scatter(x,y,c = t[:],vmin = min(t[:]),vmax = max(t[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'CFD',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.savefig('plot/'+'TContour_test.png',bbox_inches=  'tight')


plt.close('all')
	
