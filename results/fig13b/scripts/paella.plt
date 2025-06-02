
set terminal postscript "Helvetica,16" eps enhance color dl 2
set output eps_file

set size 0.5,0.48

set rmargin 2 #2
set lmargin 7.0 #5.5
set tmargin 0.7 #1.5
set bmargin 3.4 #2.5

### Key 
# set key left top
set key inside left Left reverse top enhanced nobox
set key samplen 2. spacing 1.4 height 0.2 width 11 autotitles columnhead 
set key font ',18' noopaque #maxrows 1 at graph 0.02, graph 0.975  
# unset key


### Y-axis
set ylabel 'P_{99} Latency (ms)' font ",20" offset 0,0
set yrange [0:220]
set ytics 0,50,200 font ",20"


### X-axis 
set xlabel "Throughput (reqs/s) " font ",20" offset 1,0
set xrange [0:1300]
set xtics 0,300,1500 font ",18"

Base_COLOR = '#376795'
Paella_COLOR = '#005e3e'
XSched_COLOR = "#ec5d3b"

# set label "bursty ({/Symbol s}=2.0)" at 600,15 font ",16"

# set arrow from 1000, 220 to 1000, 0 nohead lc rgb "black" dashtype 2

plot cuda_ms_file using 1:($2/1000) title "Native" with lp lt 2 lw 5 lc rgb Base_COLOR,\
    paella_file using 1:($2/1000) title "Paella" with lp lt 1 lw 5 lc rgb Paella_COLOR,\
    xsched_file using 1:($2/1000) title "XSched" with lp lt 2 lw 5 lc rgb XSched_COLOR
