import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import sqrt
import time
import ffht_test


plot_x = 0.8  #字体位置坐标
plot_y = 0.06 #字体位置坐标
fontsize = 14 #字体大小
axis_limit = [0,1,-0.15,0.15]
path = 'Cases/'

nu = 1e-3
dP = 0.1
mu = 0.5
epochs = 2000

######################################################################


print('path is', path)

Data_CFD = np.load('XYP.npz')

x = Data_CFD['x']
y  = Data_CFD['y']
T_CFD = Data_CFD['t']
U_CFD = Data_CFD['u']


device = 'cpu'

u,T = ffht_test.ffht_test(x,y,nu,dP,mu,epochs,path,device)
print('shape of u',u.shape)
#print('shape of v',v.shape)
print('shape of T',T.shape)
print('max of T', max(T))
print('min of T', min(T))
print('max of T_CFD', max(T_CFD))
print('min of T_CFD', min(T_CFD))
print('max of U', max(u))
print('min of U', min(u))
print('max of U_CFD', max(U_CFD))
print('min of U_CFD', min(U_CFD))
#w = np.zeros_like(u)
#U = np.concatenate([u,v,T],axis = 1)
T_err = abs(T-T_CFD)/(abs(T_CFD)+1e-8)
u_err = abs(u-U_CFD)/(abs(U_CFD)+1e-8)
print('max of p error ', max(T_err))
print('min of p error ', min(T_err))
print('max of u error ', max(u_err))
print('min of u error ', min(u_err))
sol = np.hstack((x,y))
sol = np.hstack((sol,u))
sol = np.hstack((sol,T))
sol = np.hstack((sol,U_CFD))
sol = np.hstack((sol,T_CFD))
np.savetxt("hyper.txt",sol,fmt='%f',delimiter=',')

#Contour Comparison
plt.figure()
plt.subplot(212)
plt.scatter(x,y,c = T[:],vmin = min(T_CFD[:]),vmax = max(T_CFD[:]))
plt.text(plot_x,plot_y,r'DNN',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.colorbar()
plt.subplot(211)
plt.scatter(x,y,c = T_CFD[:],vmin = min(T_CFD[:]),vmax = max(T_CFD[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'CFD',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.savefig('plot/'+'TContour_test.png',bbox_inches=  'tight')

print('path is', 'plot/'+'TContour_test.png')


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
plt.scatter(x,y,c = T_err[:],vmin = min(T_err[:]),vmax = max(T_err[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'T_ERROR',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.savefig('plot/'+'T_error.png',bbox_inches=  'tight')

plt.figure()
plt.subplot(111)
plt.scatter(x,y,c = u_err[:],vmin = min(u_err[:]),vmax = max(u_err[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'U_ERROR',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit)
plt.savefig('plot/'+'U_error.png',bbox_inches=  'tight')

plt.close('all')
#plt.show()
	
