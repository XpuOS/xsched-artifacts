#!/usr/bin/env gnuplot

set output eps_file
set terminal postscript "Helvetica,20" eps enhance color dl 2

set pointsize 1
set size 0.7,0.5
set nozeroaxis

set rmargin 0.4 #2
set lmargin 8 #5.5
set tmargin 2.7 #1.5
set bmargin 1.2 #2.5

Base_COLOR   = '#376795'
vCUDA_COLOR  = '#005e3e'
TGS_COLOR    = '#6ba80c'
XSched_COLOR = '#ec5d3b'
SYSTEM_NAME  = "XSched"

### Key
set key outside right Right top enhanced nobox
set key samplen 1.2 spacing 1.0 height 0.2 width -.5 autotitles columnhead 
set key font ',18' noopaque maxrows 2 #at graph 0.02, graph 1.03  

set title "AMD MI50" font ",20" offset 0,-0.8

## Y-axis
set ylabel "Norm. Perf." font ",18" offset 1.2,0
set yrange [-.08:1.25]
set ytics 0, 0.5, 1
set ytics font ",20" offset 0,0 #format "%.1f"
set ytics nomirror 


### X-axis
set xrange [-0.4:1.4]
set xtics font ",18" 
set xtics nomirror offset -0.2,0.3 #rotate by -30

set style data histogram
set style histogram clustered
set style fill solid border -1
set boxwidth .75 relative

plot mi50_dat using (($3)/$2):xticlabels(1) t "Native"   w histogram lw 2 lc rgb Base_COLOR, \
     mi50_dat using (($4)/$2):xticlabels(1) t "XSched"    w histogram lw 2 lc rgb XSched_COLOR, \
     mi50_dat using (($5)/$2):xticlabels(1) t "XSched w/o Prog"    w histogram lw 2 lc rgb vCUDA_COLOR
