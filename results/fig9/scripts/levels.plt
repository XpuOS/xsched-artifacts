#!/usr/bin/env gnuplot

reset
set output eps_file
set terminal postscript "Helvetica,16" eps enhance color dl 2

set pointsize 1
set size 0.5,0.5
set nozeroaxis

set rmargin 0.9 #2
set lmargin 5.5 #5.5
set tmargin 1.0 #1.5
set bmargin 3.3 #2.5

array COLOR2 = ['#a80326', '#ec5d3b', '#fdb96b'] # red -> orange -> yellow
LV1_COLOR = COLOR2[3]
LV2_COLOR = COLOR2[2]
LV3_COLOR = COLOR2[1]

### Key
set key inside right Left top enhanced nobox
set key samplen 1.2 spacing 1.4 height 0.2 width -1.4 autotitles columnhead 
set key font ',16' noopaque #maxrows 1 at graph 0.02, graph 0.975  

## Y-axis
set ylabel "Preempt. Latency (ms)" font ",20" offset 0.,-0.5
set yrange [-0.5:11]
set ytics 0,3,12 
set ytics font ",18" #offset 0,0 #format "%.1f"
set ytics nomirror 

### X-axis
set xlabel "CMD Exec. Time (ms)" font ",20" offset 0,.15
set xrange [0.1:2]
set xtics 0, 0.5, 2
set xtics add ('0.1' 0.1)
set xtics font ",18" offset -0.2,0.1
set xtics nomirror 

plot level1_dat u ($1/1000):($5/1000) t "Level 1"  w lp lt 1 lw 5 lc rgb LV1_COLOR, \
     level2_dat u ($1/1000):($5/1000) t "Level 2"  w lp lt 2 lw 5 lc rgb LV2_COLOR, \
     level3_dat u ($1/1000):($5/1000) t "Level 3"  w lp lt 3 lw 5 lc rgb LV3_COLOR, \
