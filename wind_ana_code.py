import matplotlib.pylab as plt
import numpy as np
import scipy.ndimage as ndimage
import glob

########################### Functions ###################################

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

r = 6372.797 #Radius of the earth, global variable
# Current working function
def latlondis(lat0,lon0,lat2,lon2):
    lat1 = np.pi*lat0/180.0
    lat3 = np.pi*lat2/180.0
    lon1 = np.pi*lon0/180.0
    lon3 = np.pi*lon2/180.0
    dlat = lat1-lat3
    dlon = lon1-lon3
    if (dlon!=0): #accounts for the direction in x and y in the sign of the distance
        dirx = dlon/abs(dlon)
    else: 
        dirx=1
    if (dlat!=0):
        diry = dlat/abs(dlat)
    else:
        diry=1
    
    dis_x = np.cos(lat3) * np.cos(lat3)*np.sin(dlon/2.0)*np.sin(dlon/2.0)
    dis_x_km = 2.0*np.arctan2(np.sqrt(dis_x),np.sqrt(1-dis_x))*r*dirx
    dis_y =np.sin(dlat/2.0)*np.sin(dlat/2.0)
    dis_y_km = 2.0*np.arctan2(np.sqrt(dis_y),np.sqrt(1-dis_y))*r*diry
    return dis_x_km,dis_y_km

##########################################################################


day = '20130519'
#radar = '/Users/klwalsh/CapstoneGit/NSSLResults/KTLX_%s' %day
radar = '/Users/Rachel/Documents/GitHub/Capstone/NSSLResults/KTLX_%s' %day
axis_title_font = {'family' : 'normal',
	'size' : 12 }
axis_font = {'family' : 'normal',
	'weight' : 'bold',  
	'size' : 14}
fig, sub = plt.subplots(2, 5)
fig.suptitle("Circulation of May 19, 2013 Mesocyclone", fontsize=16) # Figure label
fig.text(0.5, 0.04, 'Radius (m)', ha = 'center', fontdict = axis_font) # Horizontal axis label
fig.text(0.04, 0.5, 'Tilt', va = 'center', rotation = 'vertical', fontdict = axis_font) # Vertical axis label
fig.tight_layout(pad=2.5, h_pad=1.2, w_pad=.001) # Add spacing between subplots to minimize overcrowding
sub = sub.ravel()
index = 0 # Keep track of index of subplot
for dirf in sorted(glob.glob(radar+'/'+day+'*')):
    tilt_time = day + dirf[-6:] #Set new tilt_time
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

    #Tangential Velocity
    Vtan_2D = Vtan_total.mean(axis = 2)

    #Circulation
    az_rad = np.deg2rad(azimuthal_spacing)
    C_2D = np.trapz(Vtan_total, axis = 2, dx = az_rad)*radius
    ###PLOTTING###

    #Tangential Velocity plots
    #plt.figure(1, figsize = (12, 8))
    #vtan = plt.contourf(Vtan_2D)
    #plt.title("Tangential Velocity")
    #plt.xlabel("Radius")
    #plt.ylabel("Tilt")
    #plt.colorbar(vtan)
    #plt.show()
    #plt.close()

    ##### Circulation plots #####
    #Height/Tilt vs Radius
    sub[index].contourf(C_2D)
    sub[index].set_title(dirf[-6:-4] + ":" + dirf[-4:-2] + ":" + dirf[-2:] + " UTC", axis_title_font)
    #sub[index].set_xlabel("Radius (m)")
    #sub[index].set_ylabel("Tilt")
 #  test.colorbar(circ)
    index = index + 1
plt.show()
fig.savefig("/Users/Rachel/Documents/GitHub/Capstone/Plots/20130519/tiltVheight.png")
plt.close()

########################################## Notes from earlier ######################################
'''

#Step 1: Make plot of azimuthally averaged tangential velocity with height
#Bottom: radius
#y-axis is height
#Fill is the azimuthally averaged tangential velocity
#Vtan_total will only vary in tilt and radius
#Vtan2D = Vtan_total.mean(axis=2)
#plt.contourf(Vtan2D)
#plt.show()

#or 
#Bottom: time
#y-axis is height
#Fill is circulation


# Calculate circulation
#trapz is a trapezoidal integrating function in python
#Circ2D = nump.trapz(Vtan_total, axis=2, arclength =  #axis=2 because that's the dimension we're integrating over

#radius=radius[np.newaxis,:, np.newaxis]
#radius=radius*np.ones_likes(Vtan_total)

#Rachel does circulation plots

'''
