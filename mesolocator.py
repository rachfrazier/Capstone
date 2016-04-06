import numpy as np
import csv 
import sys

#case information - for later
voltime = sys.argv[1]

#input low level mesocyclone center lat/lon
lowlat = float(sys.argv[2])
lowlon = float(sys.argv[3])

#input mid-loewr level mesocyclone center
midllat = float(sys.argv[4])
midllon = float(sys.argv[5])

#input mid-upper level mesocyclone center lat/lon
midulat = float(sys.argv[6])
midulon = float(sys.argv[7])
#print midlat, midlon

#input upper level meso ceneter lat/lon
ullat = float(sys.argv[8])
ullon = float(sys.argv[9])
#print ullat, ullon

lats = np.array([lowlat, midllat, midulat, ullat])
lons = np.array([lowlon, midllon, midulon, ullon])
radelevation = np.array([0.5, 3.1, 8.0, 19.5])
elevations = np.array([0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 8.0, 10.0, 12.5, 15.6, 19.5])

mesolons = np.interp(elevations, radelevation, lons)
mesolats = np.interp(elevations, radelevation, lats )

print voltime + ', ' + repr(elevations[0]) + ', ' + repr(mesolats[0]) + ', ' + repr(mesolons[1])
for i in range(1, len(mesolons)):
    print ', ' + repr(elevations[i]) + ', ' + repr(mesolats[i]) + ', ' + repr(mesolons[i])


