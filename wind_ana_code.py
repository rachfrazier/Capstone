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
radar = '/Users/klwalsh/CapstoneGit/NSSLResults/KTLX_%s' %day
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
azimuthal_spacing = 6 # degress
radius = np.arange(0,5001,radial_spacing)
azimuth = np.arange(0,361,azimuthal_spacing)
Vtan_total = np.zeros((numt,radius.shape[0],azimuth.shape[0])) #3D tangential velocity field (tilt, radius, az), we need to average in azimuth. => make 3D array into 2D array. az avg will then only vary in tilt and radius
for i in range(numt):
    xpol, ypol, Vtan, Vrad = polar_disk(x,y,x_center,y_center,radial_spacing,azimuthal_spacing,u_total[i],v_total[i])
    Vtan_total[i] = Vtan
# Max tangential velocity with height
Vtan_max = Vtan_total.max(axis=1).max(axis=1)

#Tangential Velocity
Vtan_2D = Vtan_total.mean(axis = 2)

#Circulation
az_rad = np.deg2rad(azimuthal_spacing)
C_2D = np.trapz(Vtan_total, axis = 2, dx = az_rad)*radius
###PLOTTING###

#Tangential Velocity plots
plt.figure(1, figsize = (12, 8))
vtan = plt.contourf(Vtan_2D)
plt.title("Tangential Velocity")
plt.xlabel("Radius")
plt.ylabel("Tilt")
plt.colorbar(vtan)
plt.show()
plt.close()

#Circulation plots
plt.figure(2, figsize = (12, 8))
circ = plt.contourf(C_2D)
plt.title("Circulation")
plt.xlabel("Radius")
plt.ylabel("Tilt")
plt.colorbar(circ)
plt.show()
plt.close()

# plot example
'''plt.contourf(xpol,ypol,Vtan_total[5])
plt.show()
plt.close()'''
