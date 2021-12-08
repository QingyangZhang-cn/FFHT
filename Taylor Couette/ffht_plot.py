import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import sqrt
import time
import ffht_test

plot_x = 47e-3  #字体位置坐标
plot_y = 47e-3 #字体位置坐标
fontsize = 14 #字体大小
axis_limit = [-50e-3,50e-3,-50e-3,50e-3]
path = 'Cases/'

nu = 2e-4
wmax = 1
rStart = 17.8e-3
rEnd = 46.28e-3
L = rEnd-rStart
epochs = 10000	

print('path is', path)

Data_CFD = np.load('ROC.npz')
r = Data_CFD['r']
O  = Data_CFD['O']
w_cfd = Data_CFD['u']#(1*17.8e-3*(46.28e-3/r-r/46.28e-3)/(46.28e-3/17.8e-3-17.8e-3/46.28e-3))/r
t_cfd = Data_CFD['t']
device = 'cpu'
w,t = ffht_test.ffht_test(r,O,nu,wmax,rStart,rEnd,L,epochs,path,device)
x = r*np.cos(O)
y = r*np.sin(O)
print('shape of w',w.shape)
print('shape of T',t.shape)
print('max of T', max(t))
print('min of T', min(t))
print('max of w', max(w))
print('min of w', min(w))
print('max of W_CFD', max(w_cfd))
print('min of W_CFD', min(w_cfd))
print('max of T_CFD', max(t_cfd))
print('min of T_CFD', min(t_cfd))

w_err = abs(w-w_cfd)/(abs(w_cfd)+1e-8)
t_err = abs(t-t_cfd)/(abs(t_cfd)+1e-8)
print('max of w error ', max(w_err))
print('min of w error ', min(w_err))
print('max of t error ', max(t_err))
print('min of t error ', min(t_err))
#Contour Comparison
sol = np.hstack((r,O))
sol = np.hstack((sol,w))
sol = np.hstack((sol,t))
np.savetxt("hyper.txt",sol,fmt='%f',delimiter=',')

plt.figure()
plt.subplot(122)
plt.scatter(x,y,c = w_cfd[:],vmin = min(w_cfd[:]),vmax = max(w_cfd[:]))
plt.text(plot_x,plot_y,r'CFD',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit,'scaled')
plt.colorbar()
plt.subplot(121)
plt.scatter(x,y,c = w[:],vmin = min(w_cfd[:]),vmax = max(w_cfd[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'DNN',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit,'scaled')
plt.savefig('plot/'+'uContour_test.png',bbox_inches=  'tight')

plt.figure()
plt.subplot(111)
plt.scatter(x,y,c = w_err[:],vmin = min(w_err[:]),vmax = max(w_err[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'W_ERROR',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit,'scaled')
plt.savefig('plot/'+'W_error.png',bbox_inches=  'tight')

plt.figure()
plt.subplot(312)
plt.scatter(x,y,c = t_cfd[:],vmin = min(t_cfd[:]),vmax = max(t_cfd[:]))
plt.text(plot_x,plot_y,r'CFD',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit,'scaled')
plt.colorbar()
plt.subplot(311)
plt.scatter(x,y,c = t[:],vmin = min(t_cfd[:]),vmax = max(t_cfd[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'DNN',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit,'scaled')
plt.savefig('plot/'+'TContour_test.png',bbox_inches=  'tight')

plt.figure()
plt.subplot(411)
plt.scatter(x,y,c = t_err[:],vmin = min(t_err[:]),vmax = max(t_err[:]))
plt.colorbar()
plt.text(plot_x,plot_y,r'T_ERROR',{'color': 'b','fontsize':fontsize})
plt.axis(axis_limit,'scaled')
plt.savefig('plot/'+'T_error.png',bbox_inches=  'tight')

plt.close('all')
	
