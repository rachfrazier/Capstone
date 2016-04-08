set term pos color
set output 'xyc_fit.ps'

set xlabel "xc (km)"
set ylabel "yc (km)"
plot "fort.301" u 1:2 t "b_0km" w l,\
     "fort.302" u 1:2 t "b_2km" w l,\
     "fort.303" u 1:2 t "b_3km" w l,\
     "fort.304" u 1:2 t "b_4km" w l,\
     "fort.305" u 1:2 t "b_5km" w l,\
     "fort.306" u 1:2 t "b_6km" w l,\
     "fort.47" u  11:12 t "o" 

reset
set xlabel "xc (km)"
set ylabel "yc (km)"
plot "fort.301" u 1:2 t "b_0km" w l,\
     "fort.1000" u  1:2 t "o_0-1km",\
     "fort.1001" u  1:2 t "o_1-2km"


reset
set xlabel "xc (km)"
set ylabel "zc (km)"
plot "fort.401" u 1:7 t "b"  w l,\
     "fort.406" u 1:7 t "b"  w l,\
     "fort.411" u 1:7 t "b"  w l,\
     "fort.416" u 1:7 t "b"  w l,\
     "fort.421" u 1:7 t "b"  w l,\
     "fort.426" u 1:7 t "b"  w l,\
     "fort.431" u 1:7 t "b"  w l,\
     "fort.47" u  11:13 t "o" 

reset
set xlabel "yc (km)"
set ylabel "zc (km)"
plot "fort.401" u 2:7 t "b"  w l,\
     "fort.406" u 2:7 t "b"  w l,\
     "fort.411" u 2:7 t "b"  w l,\
     "fort.416" u 2:7 t "b"  w l,\
     "fort.421" u 2:7 t "b"  w l,\
     "fort.426" u 2:7 t "b"  w l,\
     "fort.431" u 2:7 t "b"  w l,\
     "fort.47" u  12:13 t "o"


