import matplotlib.pylab as plt
import numpy as np

def polar_disk(xgrid,ygrid,xi,yi,radmin,hspac,up,vp):
az_range = np.arange(0,361,9)
rad_range = np.arange(radmin,5001,hspac)
rad_range_len = len(rad_range)
az_range_len = len(az_range)
az_range,rad_range = np.meshgrid(az_range,rad_range)
x_polar_test = (xi[np.newaxis,np.newaxis] +(rad_range*np.sin(az_range*np.pi/180.0))[...,np.newaxis]).flatten()
y_polar_test = (yi[np.newaxis,np.newaxis] +(rad_range*np.cos(az_range*np.pi/180.0))[...,np.newaxis]).flatten()
xspts = pts_to_grid(x_polar_test.flatten(),0.0,xgrid)
yspts = pts_to_grid(y_polar_test.flatten(),0.0,ygrid)
u_polar = np.reshape(ndimage.map_coordinates(up,[yspts,xspts],order=odr),az_range.shape)
v_polar = np.reshape(ndimage.map_coordinates(vp,[yspts,xspts],order=odr),az_range.shape)
vtan = -u_polar*np.cos(az_range*np.pi/180.0)+ v_polar*np.sin(az_range*np.pi/180.0)
vrad = u_polar*np.sin(az_range*np.pi/180.0)- v_polar*np.cos(az_range*np.pi/180.0)
return np.reshape(x_polar_test,az_range.shape),np.reshape(y_polar_test,az_range.shape),vtan,vrad

file = open('/Users/dbetten/lwei/Dan/2016-03-23/20130519_ana/20130519232322/qd_20130519232322.txt')
data = []
dd=[]
for line in file:data.append(line)
for i in range(14):dd.append(data[i].split())
vv_total = np.float32(np.array(dd).reshape((14,81,81)))

file = open('/Users/dbetten/lwei/Dan/2016-03-23/20130519_ana/20130519232322/dd_20130519232322.txt')
data = []
dd=[]
for line in file:data.append(line)
for i in range(14):dd.append(data[i].split())
div_total = np.float32(np.array(dd).reshape((14,81,81)))

file = open('/Users/dbetten/lwei/Dan/2016-03-23/20130519_ana/20130519232322/ud_20130519232322.txt')
data = []
dd=[]
for line in file:data.append(line)
for i in range(14):dd.append(data[i].split())
u_total = np.float32(np.array(dd).reshape((14,81,81)))

file = open('/Users/dbetten/lwei/Dan/2016-03-23/20130519_ana/20130519232322/vd_20130519232322.txt')
data = []
dd=[]
for line in file:data.append(line)
for i in range(14):dd.append(data[i].split())
v_total = np.float32(np.array(dd).reshape((14,81,81)))

x = np.arange(0,250*81,250)
y = np.arange(0,250*81,250)
xgrid,ygrid = np.meshgrid(x,y)

xpol, ypol, Vtan, Vrad = polar_disk(xgrid,ygrid,np.array([10000]),np.array([10000]),0,100,u_total[0],v_total[0])
# plot example
plt.contourf(xpol,ypol,Vtan)