import matplotlib.pylab as plt
import numpy as np
import scipy.ndimage as ndimage
import glob
def polar_disk(xgrid,ygrid,xi,yi,hspac,az_spac,up,vp):
    az_range = np.arange(0,361,az_spac)
    rad_range = np.arange(0,5001,hspac)
    rad_range_len = len(rad_range)
    az_range_len = len(az_range)
    az_range,rad_range = np.meshgrid(az_range,rad_range)
    x_polar_test = (xi +(rad_range*np.sin(az_range*np.pi/180.0))[...,np.newaxis]).flatten()
    y_polar_test = (yi +(rad_range*np.cos(az_range*np.pi/180.0))[...,np.newaxis]).flatten()
    xspts = pts_to_grid(x_polar_test.flatten(),0.0,xgrid)
    yspts = pts_to_grid(y_polar_test.flatten(),0.0,ygrid)
    u_polar = np.reshape(ndimage.map_coordinates(up,[yspts,xspts],order=3),az_range.shape)
    v_polar = np.reshape(ndimage.map_coordinates(vp,[yspts,xspts],order=3),az_range.shape)
    vtan = -u_polar*np.cos(az_range*np.pi/180.0)+ v_polar*np.sin(az_range*np.pi/180.0)
    vrad = u_polar*np.sin(az_range*np.pi/180.0)- v_polar*np.cos(az_range*np.pi/180.0)
    return np.reshape(x_polar_test,az_range.shape),np.reshape(y_polar_test,az_range.shape),vtan,vrad

def pts_to_grid(pts_flat,dist,grid):
                return np.interp((pts_flat-dist),grid,range(0,len(grid)))

day = '20130519'
tilt_time = day + '231035'
radar = '/Users/Rachel/Documents/GitHub/Capstone/NSSLResults/KTLX_%s' %day
#for dirf in sorted(glob.glob(radar+'/'+day+'*')):
#    print dirf

file = open('%s/%s/qd_%s.txt' % (radar,tilt_time,tilt_time))
data = []
dd=[]
for line in file:data.append(line)
# number of tilts
numt = len(data)

for i in range(numt):dd.append(data[i].split())
vv_total = np.float32(np.array(dd).reshape((numt,81,81)))

file = open('%s/%s/dd_%s.txt' % (radar,tilt_time,tilt_time))
data = []
dd=[]
for line in file:data.append(line)
for i in range(numt):dd.append(data[i].split())
div_total = np.float32(np.array(dd).reshape((numt,81,81)))

file = open('%s/%s/ud_%s.txt' % (radar,tilt_time,tilt_time))
data = []
dd=[]
for line in file:data.append(line)
for i in range(numt):dd.append(data[i].split())
u_total = np.float32(np.array(dd).reshape((numt,81,81)))

file = open('%s/%s/vd_%s.txt' % (radar,tilt_time,tilt_time))
data = []
dd=[]
for line in file:data.append(line)
for i in range(numt):dd.append(data[i].split())
v_total = np.float32(np.array(dd).reshape((numt,81,81)))

wind_magnitude = np.hypot(u_total,v_total)

x = np.arange(0,250*81,250)
y = np.arange(0,250*81,250)
xgrid,ygrid = np.meshgrid(x,y)
x_center = 10000.0
y_center = 10000.0
radial_spacing = 50 # meters
azimuthal_spacing = 6 # degrees - chose random azimuthal spacing; apparently doesn't matter
radius = np.arange(0,5001,radial_spacing)
azimuth = np.arange(0,361,azimuthal_spacing)
Vtan_total = np.zeros((numt,radius.shape[0],azimuth.shape[0])) #3D tangential velocity field... it has tilt, radius and azimuth
for i in range(numt):
    xpol, ypol, Vtan, Vrad = polar_disk(x,y,x_center,y_center,radial_spacing,azimuthal_spacing,u_total[i],v_total[i])
    Vtan_total[i] = Vtan
# Max tangential velocity with height
Vtan_max = Vtan_total.max(axis=1).max(axis=1)
# plot example
#plt.contourf(xpol,ypol,Vtan_total[5]) #Showing tangential velocity relative to the center of the grid which you assume is the center of the vortex
#plt.show()
#plt.close()

#Step 1: Make plot of azimuthally averaged tangential velocity with height
#Bottom: radius
#y-axis is height
#Fill is the azimuthally averaged tangential velocity
#Vtan_total will only vary in tilt and radius
Vtan2D = Vtan_total.mean(axis=2)
plt.contourf(Vtan2D)
plt.show()

#or 
#Bottom: time
#y-axis is height
#Fill is circulation


# Calculate circulation
#trapz is a trapezoidal integrating function in python
Circ2D = nump.trapz(Vtan_total, axis=2, arclength =  #axis=2 because that's the dimension we're integrating over

radius=radius[np.newaxis,:, np.newaxis]
radius=radius*np.ones_likes(Vtan_total)

#Rachel does circulation plots
