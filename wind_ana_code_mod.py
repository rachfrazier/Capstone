import matplotlib.pylab as plt
import numpy as np
import scipy.ndimage as ndimage
import glob
import csv
import ctables

cmap_ref = ctables.Carbone42
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
with open('/Users/klwalsh/Undergrad/Senior/Capstone/CSVs/20120414_2.csv') as csvfile:
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
radar_lat = 37.7608043
radar_lon = -99.9688053

# KJKL
#radar_lat = 37.590762
#radar_lon = -83.313039

#KLIX
#radar_lat = 30.3367133
#radar_lon = -89.8256618

dist = np.array([])
height = []
tilt = np.deg2rad([0.48, 0.85, 1.32, 1.80, 2.42, 3.12, 4.00, 5.10, 6.42, 8.00, 10.02, 12.48, 15.60, 19.51]) #0.48
for i in range(len(lat)):
	x, y = latlondis(lat[i], lon[i], radar_lat, radar_lon)
	dist = np.append(dist, np.hypot(x, y))
print x, y
print dist

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

day = '20120414'
#tilt_time = day + '231035'
radar = '/Users/klwalsh/CapstoneGit/NSSLResults/KDDC' #%day
time = []
vtan2d = []
c2d = []
daytime = []
tilt = np.arange(0,14)
axis_title_font = {'family' : 'normal',
    'size' : 12 }
axis_font = {'family' : 'normal',
    'weight' : 'bold',
        'size' : 14}
fig, ax = plt.subplots(2, 5)
#fig.suptitle("Tangential Velocity - March 9th, 2011", fontsize = 16)
fig.text(0.5, 0.04, 'Radius (km)', ha= 'center', fontdict = axis_font)
fig.text(0.04, 0.5, 'Height (km)', va='center', rotation='vertical', fontdict = axis_font)
fig.tight_layout(pad = 2.0, h_pad = 1.2, w_pad = 0.1) #spacing between subplots
ax = ax.ravel()
index = 0
height_index = 0
for dirf in sorted(glob.glob(radar+'/'+day+'*')):
    tilt_time = day + dirf[-6::] #change between -6 and -8 for KJKL and rest of cases
    vtime = dirf[-6::]
    times = time.append(np.float32(vtime))
    dtime = dirf[-6:-4] + ":" + dirf[-4:-2] + ":" + dirf[-2:]
    dtimes = daytime.append(dtime)
    rad = dirf.split('/')[5]
    print dirf
    
    file = open('%s/%s/qd_%s.txt' % (radar,tilt_time,tilt_time))
    data = []
    dd=[]
    for line in file:data.append(line)
    # number of tilts
    numt = len(data)
    
    #vertical vorticity
    for i in range(numt):dd.append(data[i].split())
    vv_total = np.float32(np.array(dd).reshape((numt,81,81)))
    
    #divergence
    file = open('%s/%s/dd_%s.txt' % (radar,tilt_time,tilt_time))
    data = []
    dd=[]
    for line in file:data.append(line)
    for i in range(numt):dd.append(data[i].split())
    div_total = np.float32(np.array(dd).reshape((numt,81,81)))
    
    #u wind
    file = open('%s/%s/ud_%s.txt' % (radar,tilt_time,tilt_time))
    data = []
    dd=[]
    for line in file:data.append(line)
    for i in range(numt):dd.append(data[i].split())
    u_total = np.float32(np.array(dd).reshape((numt,81,81)))

    #v wind
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
    x_center = np.arange(5000,15001,500)
    y_center = np.arange(5000,15001,500)
    radial_spacing = 50 # meters
    azimuthal_spacing = 6 # degress
    radius = np.arange(0,5001,radial_spacing)
    azimuth = np.arange(0,361,azimuthal_spacing)
    Vtan_total = np.zeros((numt,radius.shape[0],azimuth.shape[0])) #3D tangential velocity field (tilt, radius, az), we need to average in azimuth. => make 3D array into 2D array. az avg will then only vary in tilt and radius
    for i in range(numt):
        Vtan_max = 0
        for xi in x_center:
            for yi in y_center:
                xpol, ypol, Vtan, Vrad = polar_disk(x,y,xi,yi,radial_spacing,azimuthal_spacing,u_total[i],v_total[i])
                if Vtan.mean(axis=1)[0:31].mean() > Vtan_max:
                    Vtan_final=Vtan
                    xi_final = xi
                    yi_final = yi
                    Vtan_max = Vtan.mean(axis=1)[0:31].mean()
        Vtan_total[i] = Vtan_final
        print np.hypot(xi_final-10000,yi_final-10000)
    # Max tangential velocity with height
    Vtan_max = Vtan_total.max(axis=1).max(axis=1)

    #Tangential Velocity
    Vtan_2D = Vtan_total.mean(axis = 2)
    if (Vtan_2D.shape == (12, 101)):
        Vtan_2D = np.insert(Vtan_2D, 0, np.nan, axis = 0)
        Vtan_2D = np.insert(Vtan_2D, 1, np.nan, axis = 0) #change to 2 for KJKL
    print Vtan_2D.shape
    vtan2 = vtan2d.append(Vtan_2D)

    #Circulation
    az_rad = np.deg2rad(azimuthal_spacing)
    C_2D = np.trapz(Vtan_total, axis = 2, dx = az_rad)*radius
    if (C_2D.shape == (12, 101)):
        C_2D = np.insert(C_2D, 0, np.nan, axis = 0)
        C_2D = np.insert(C_2D, 1, np.nan, axis = 0) #change to 2 for KJKL
    print C_2D
    c2 = c2d.append(C_2D)

    ###PLOTTING###

    #Tangential Velocity plots
    plt.figure(1, figsize = (32, 12))
    npheight = np.asarray(height[height_index])
    print(Vtan_2D.shape)
    if(Vtan_2D.shape == (12, 101)):
        npheight = np.delete(npheight, 0)
        npheight = np.delete(npheight, 1) # 2 for KJKL, 1 for KLIX
    r, hght = np.meshgrid(radius, npheight)
    #ax[index].contourf(r/1000., hght, Vtan_2D, extend = "both")
    #vtan = ax[index].contourf(r/1000., hght, Vtan_2D,  extend = "both")
    C2D = ax[index].contourf(r/1000., hght, C_2D, extend = "both")
    ax[index].set_title(dirf[-6:-4] + ":" + dirf[-4:-2] + ":" + dirf[-2:] + " UTC", axis_title_font)
    index = index + 1
    height_index = height_index + 1
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(vtan, cax = cbar_ax)
#fig.delaxes(ax[-1])
#fig.savefig('/Users/klwalsh/Undergrad/Senior/Capstone/Results/'+rad+'/vtan_%s.png'%day, dpi = 300)
plt.show()
plt.close()

#print "Subplots: done"

daytime = np.asarray(daytime)
time = np.asarray(time)
vtan2d = np.asarray(vtan2d)
c2d = np.asarray(c2d)
el = np.array([0.48, 0.88, 1.32, 1.80, 2.42, 3.12, 4.00, 5.10, 6.42, 8.00, 10.02, 12.48, 15.60, 19.51])

'''fig = plt.figure(2)
fig.suptitle('Tangential Velocity on March 9th, 2011', fontsize = 16)
fig.text(0.5, 0.04, 'Radius (km)', ha= 'center', fontdict = axis_font)
fig.text(0.04, 0.5, 'Height (km)', va='center', rotation='vertical', fontdict = axis_font)
for n in range(0, vtan2d.shape[0]):
    if n == 0:
        plt.subplot(2, 4, 1)
        plt.contourf(r/1000., hght, vtan2d[n, :, :], extend = "both")
        plt.title(daytime[n])
    if 1 <= n <= 7:
        plt.subplot(2, 4, n+1)
        plt.contourf(r/1000., hght, vtan2d[n, :, :], extend = "both")
        plt.title(daytime[n])
    if n == 8:
        plt.subplot(2, 4, 0)
        plt.contourf(r/1000., hght, vtan2d[n, :, :], extend = "both")
        plt.title(daytime[n])
cbar_ax = fig.add_axes([0.92, 0.15, 0.05, 0.7])
fig.colorbar(vtan, cax = cbar_ax)
plt.show()
plt.close()'''

'''fig = plt.figure(2, figsize = (32, 12))
fig.suptitle('Circulation on April 14th, 2012', fontsize = 16)
fig.text(0.5, 0.04, 'Radius (km)', ha= 'center', fontdict = axis_font)
fig.text(0.04, 0.5, 'Height (km)', va='center', rotation='vertical', fontdict = axis_font)
#fig.tight_layout() #spacing between subplots
for n in range(0, c2d.shape[0]):
    if n == 0:
        plt.subplot(2, 5, 1)
        plt.contourf(r/1000., hght, c2d[n, :, :], extend = "both")
        plt.title(daytime[n])
    if 1 <= n <= 8:
        plt.subplot(2, 5, n+1)
        plt.contourf(r/1000., hght, c2d[n, :, :], extend = "both")
        plt.title(daytime[n])
    if n == 10:
        plt.subplot(2, 5, 0)
        plt.contourf(r/1000., hght, c2d[n, :, :], extend = "both")
        plt.title(daytime[n])
cbar_ax = fig.add_axes([0.91, 0.15, 0.05, 0.7])
fig.colorbar(C2D, cax = cbar_ax)
plt.show()
plt.close()'''

#calculate time differences, this will be different per each case, you can simply comment them out.
dif = []
'''#KTLX - works for both cases but change to 11 for 2015 case
for n in range(0, 11):
    if time[n] < 230000:
        diffa = time[n] - 220000.
        dta = dif.append(diffa/100.)
    elif time[n] >= 230000:
        diffb = 6000. +  (time[n] - 230000.)
        dtb = dif.append(diffb/100.)'''

#KDDC
for n in range(0, 10):
    if time[n] < 190000:
        diffa = time[n] - 180000.
        dta = dif.append(diffa/100.)
    elif time[n] >= 190000:
        diffb = 6000. + (time[n] - 190000.)
        dtb = dif.append(diffb/100.)

'''#KFDR - all within 0100 UTC
for n in range(0, 10):
    diffa = time[n] - 10000.
    dta = dif.append(diffa/100.)'''

'''#KLIX
for n in range(0, 8):
    if time[n] < 110000:
        diffa = time[n] - 100000.
        dta = dif.append(diffa/100.)
    elif time[n] >= 110000:
        diffb = 6000. + (time[n] - 110000.)
        dtb = dif.append(diffb/100.)'''

'''#KJKL - problem child
for n in range(0, 7):
    if 230000 < time[n] and time[n] < 240000:
        diffa = time[n] - 230000.
        dta = dif.append(diffa/100.)
    elif time[n] <= 230000:
        diffb = 6000. + (time[n] - 0.)
        dtb = dif.append(diffb/100.)'''

dif = np.asarray(dif)
print "Time Dif: done"

#Time vs Radius Plots
for n in range(0, npheight.shape[0]):
    height_level = npheight[n]
    lev = n
    vt2d = vtan2d[:,lev,:].T
    plt.contourf(dif, radius, vt2d, extend = "both")
    plt.title("Tangential Velocity at %s km" %(npheight[n]))
    plt.xlabel("Minutes Past 0100 UTC")
    plt.ylabel("Radius (m)")
    plt.colorbar()
    #plt.savefig('/Users/klwalsh/Undergrad/Senior/Capstone/Results/'+rad+'/vtanwrad_h%s.png'%(npheight[n]), dpi = 300)
    plt.show()
    plt.close()

#circulation with radius at hght = n
    plt.contourf(dif, radius, c2d[:, lev, :].T, extend = "both")
    plt.title("Circulation at z = %s km"%npheight[n])
    plt.xlabel("Minutes Past 0100 UTC")
    plt.ylabel("Radius (m)")
    plt.colorbar()
    #plt.savefig('/Users/klwalsh/Undergrad/Senior/Capstone/Results/'+rad+'/circwrad_h%s.png'%(npheight[n]), dpi = 300)
    plt.show()
    plt.close()

#print "Time v Height: done"

#Time v Radius Plots
'''for n in range(0, radius.shape[0]):
    rad_level = radius[n]/1000.
    lev = n
    c2 = c2d[:, :, lev]
    d, h = np.meshgrid(dif, npheight)
    plt.contourf(d.T, h.T, c2, extend = "both")
    plt.title("Circulation at Radius %s (km)" %(radius[n]/1000.))
    plt.xlabel("Minutes Past 0100 UTC") #CHANGE THIS DEPENDENT ON CASE
    plt.ylabel("Height (km)")
    plt.colorbar()
    #plt.savefig('/Users/klwalsh/Undergrad/Senior/Capstone/Results/'+rad+'/circwhght_r%s.png'%(radius[n]/1000.), dpi = 300)
    #plt.show()
    plt.close()

#tagential velocity
    plt.contourf(d.T, h.T, vtan2d[:, :, lev], extend = "both")
    plt.title("Tangential Velocity at Radius %s (km)" %(radius[n]/1000.))
    plt.xlabel("Minutes Past 0100 UTC")
    plt.ylabel("Height (km)")
    plt.colorbar()
    #plt.savefig('/Users/klwalsh/Undergrad/Senior/Capstone/Results/'+rad+'/vtanwhght_r%s.png'%(radius[n]/1000.), dpi = 300)
    #plt.show()
    plt.close()

print "Time v Radius: done"'''

'''####Rachel's circulation plots, with respect to making the heights for the contours and using the hght variable to use as your z axis rather than by tilt.  We need to figure out igure out how to make these for time height respect as well######
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
#Circulation plots
plt.figure(2, figsize = (12, 8))
circ = plt.contourf(C_2D)
plt.title("Circulation")
plt.xlabel("Radius")
plt.ylabel("Tilt")
plt.colorbar(circ)
plt.show()
plt.close()'''
