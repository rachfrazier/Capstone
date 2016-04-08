This directory includes the results of B-spline fitting the trajectory of the vortex centers 

fort.100?  obs (xc,yc,zc,tc) xc, yc, zc in km & tc in s which are relative to the 0.5deg title's point 
           of first volumn scan for 0-1km, 1-2km, ..., 7-8km groups.
fort.30?   fitting results (xb,yb,xa,ya,ub,vb,t,ua,va), where *b is B-spline fitting, *a is adjusted fitting
           to the B-spline fitting with a functions such as exp(-(z-zo)**2/Lz**2)*exp(-(t-to)**2/Lt**2) 
           from 0km to 7km. The adjusted fitting are not used in the later analysis. B-spline fitting ub and vb
           are used as background in the vortex wind analysis. 

fort.40?   fitting results (xb,yb,xa,ya,ub,vb,z,ua,va), same as above, but for the time intervals.

fort.47    obs (year,mon,day,hour,min,sec, dxa,dya,ua,va,xc,yc,zc,tc,dxb,dyb,ub,vb)
           dxa=xc-xa, dya=yc-ya, dxb=xb-xc, dyb=yb-yc

plt_fit.gnu gnuplot script 


