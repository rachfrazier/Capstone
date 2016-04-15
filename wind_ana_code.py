import matplotlib.pylab as plt
import numpy as np
import scipy.ndimage as ndimage
import glob
import csv
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

############ Lat/ Lons needed to convert tilt to height ##################

master = np.array([])
with open('/Users/Rachel/OneDrive/Documents/Meteorology/Capstone/Scripts/20110309_2.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		master = np.append(master, row)

lat = np.array([])
lon = np.array([])
for i in range(len(master)):
	lat = np.append(lat, float(master[i][' Lat']))
	lon = np.append(lon, float(master[i][' Lon']))
# KTLX
#radar_lat = 35.3333873
#radar_lon = -97.2778255

# KFDR
#radar_lat = 34.3620014
#radar_lon = -98.9766884

# KDDC
#radar_lat = 37.7608043
#radar_lon = -99.9688053

# KJKL
#radar_lat = 37.590762
#radar_lon = -83.313039

#KLIX
radar_lat = 30.3367133
radar_lon = -89.8256618

dist = np.array([])
height = []
tilt = np.deg2rad([0.48, 0.85, 1.32, 1.80, 2.42, 3.12, 4.00, 5.10, 6.42, 8.00, 10.02, 12.48, 15.60, 19.51]) #0.48
for i in range(len(lat)):
	x, y = latlondis(lat[i], lon[i], radar_lat, radar_lon)
	dist = np.append(dist, np.hypot(x, y))
j = 0
temp = []
for i in range(0, dist.shape[0]):
	temp.append((np.sin(tilt[j]) * dist[i]) + (dist[i]*dist[i]/(2*r)))
	j = j + 1
	if (i+1) % 14 == 0: # Reloop over tilt array after doing 19.51
		height.append(temp)
		j = 0
		temp=[]
##########################################################################

day = '20110309'
#radar = '/Users/klwalsh/CapstoneGit/NSSLResults/KDDC_%s' %day
radar = '/Users/Rachel/Documents/GitHub/Capstone/NSSLResults/KLIX' 
axis_title_font = {'family' : 'normal',
	'size' : 12 }
axis_font = {'family' : 'normal',
	'weight' : 'bold',  
	'size' : 14}
fig, sub = plt.subplots(2, 4)
fig.suptitle("Circulation of March 9, 2011 Mesocyclone", fontsize=16, fontweight='bold') # Figure label
fig.text(0.5, 0.04, 'Radius (km)', ha = 'center', fontdict = axis_font) # Horizontal axis label
fig.text(0.04, 0.5, 'Height (km)', va = 'center', rotation = 'vertical', fontdict = axis_font) # Vertical axis label
fig.tight_layout(pad=2.5, h_pad=1.2, w_pad=.001) # Add spacing between subplots to minimize overcrowding
sub = sub.ravel()
index = 0 # Keep track of index of subplot
height_index = 0
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

    #Volume 4 missing 0.48, 0.88

    ##### Circulation plots #####
    #Height/Tilt vs Radius
    npheight = np.asarray(height[height_index])
    print(C_2D.shape)
    if(C_2D.shape == (12, 101)):
        npheight = np.delete(npheight, 0)
        npheight = np.delete(npheight, 1) ################## KATE LYNNNNNNNNNNNNN Change this from 1 to 2 for KJKL
    r, hght = np.meshgrid(radius/1000., npheight)
    circ_cb = sub[index].contourf(r, hght, C_2D, extend = "both")
    #circ_cb = plt.contour(C_2D)
    sub[index].contourf(r, hght, C_2D, extend = "both")
    #sub[index].contourf(C_2D)
    sub[index].set_title(dirf[-6:-4] + ":" + dirf[-4:-2] + ":" + dirf[-2:] + " UTC", axis_title_font)
    index = index + 1
    height_index = height_index + 1
#Add color bars
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]) 
fig.colorbar(circ_cb, cax = cbar_ax)
#sub[-1, -1].axis('off') # Remove last plot since we have an odd number of volumes
#fig.delaxes(sub[-1])

#Show figure and save it
plt.show()
fig.savefig("/Users/Rachel/Documents/GitHub/Capstone/Plots/20110309/20110309_Circ_HeightVRadius.png")
plt.close()


