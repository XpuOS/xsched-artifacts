#!/usr/bin/env gnuplot

reset
set output eps_file
set terminal postscript "Helvetica,16" eps enhance color dl 2

set pointsize 1
set size 1.2,0.5
set nozeroaxis

set rmargin 2 #2
set lmargin 10 #5.5
set tmargin 1 #1.5
set bmargin 3.4 #2.5

Base_COLOR   = '#376795'
XSched_COLOR = '#ec5d3b'
SYSTEM_NAME  = "XSched"

### Key
set key inside left Left reverse top enhanced nobox
set key samplen 1.5 spacing 1.4 height 0.2 width 0 autotitles columnhead 
set key font ',18' noopaque maxrows 1 # at graph 0.02, graph 0.975  


## Y-axis
set ylabel "Frame Latency (ms)" font ",18" offset .1,-.4
set yrange [0:1300]
set ytics 0,400,1200
set ytics font ",18" offset .1,0 #format "%.1f"
set ytics nomirror 


### X-axis
set xlabel "Time (s)" font ",19" offset 0,.05
set xrange [0:32]
set xtics 0, 5, 30
set xtics font ",18" offset -0.27,0
set xtics nomirror 

ANNOTATE_COLOR = 'black'

plot base_dat   u ($1):($2) t "Native"   w l lt 1 lw 5 lc rgb Base_COLOR, \
     xsched_dat u ($1):($2) t SYSTEM_NAME   w l lt 1 lw 5 lc rgb XSched_COLOR, \
