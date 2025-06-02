set terminal postscript "Helvetica,16" eps enhance color dl 2
set output eps_file

set size 0.5, 0.5
set rmargin 0.4
set lmargin 5.5
set tmargin 1.0
set bmargin 2.4
unset key

SA_COLOR     = "#1b7c3d"
Base_COLOR   = '#376795'
XSched_COLOR = '#ec5d3b'
SYSTEM_NAME  = "XSched"

subplot_width = 0.195
y_label_width = 0.1

set multiplot layout 1,6 rowsfirst
set size subplot_width+y_label_width,0.35

## Y-axis
set ylabel "CDF" font ",18" offset 2.3,-.0
set yrange [0:100]
set ytics 0,25,100
set ytics font ",16" offset 0.3,0
## X-axis
set xlabel "Latency (ms) " font ",16" offset 0,.7
set xrange [0:x_range]
set xtics 0,x_step,x_max

set xtics font ",16" offset -0.2,.2
set rmargin 0.3 #2
set lmargin 0.6 #5.5

set size subplot_width,0.35
set origin y_label_width,0.1

set key outside right Left reverse top enhanced nobox
set key samplen 1.3 spacing 1.5 height 0.16 width 1.5 autotitles columnhead 
set key font ',16' noopaque at graph 2.08, graph 0.985  

# set title font ",20" offset 0,-13.5
set title dev_name

plot cdf_file u ($2/1000000):($1*100) t "Standalone" w l lw 5 lc rgb SA_COLOR,\
           "" u ($3/1000000):($1*100) t "Native" w l lw 5 lc rgb Base_COLOR,\
           "" u ($4/1000000):($1*100) t SYSTEM_NAME w l lw 5 lc rgb XSched_COLOR

set ytics ('' 0, '' 25, '' 50, '' 75, '' 100)

