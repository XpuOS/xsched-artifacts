set terminal postscript "Helvetica,16" eps enhance color dl 2
set output eps_file

set size 0.5, 0.5
set rmargin 0.4
set lmargin 4.8
set tmargin 1.0
set bmargin 1.4
unset key

SA_COLOR     = "#1b7c3d"
Base_COLOR   = '#376795'
Base_Light_COLOR = '#72bcd5'
XSched_COLOR = '#ec5d3b'
XSched_Light_COLOR = '#fdb96b'
SYSTEM_NAME  = "XSched"

subplot_width = 0.195
subplot_height = 0.35
subplot_y_offset = 0.02
y_label_width = 0.05


set multiplot layout 1,9 rowsfirst
set size subplot_width+y_label_width,subplot_height
set origin 0,subplot_y_offset

# set key
## Y-axis
set ylabel "Norm. Throughput" font ",16" offset 0.8,0
## X-axis
set xrange [-0.8:2.8]
set xtics font ",20" 
unset xtics

set style data histogram
set style histogram clustered gap 1
set style fill solid border -2
set boxwidth 0.70

set yrange [-0.07:1.4]
set ytics ("1" 1.0, ".5" 0.5, "0" 0.0)
set ytics font ",16" offset 0.5,0

set title font ",20" offset 0,-11.5

set rmargin 0.3 #2
set lmargin 0.6 #5.5
set ytics ('' 0, '' 0.5, '' 1.0)


set size subplot_width,subplot_height
set origin y_label_width,0.1

set title dev_name

set key outside right Left reverse top enhanced nobox
set key samplen 1.3 spacing 1.5 height 0.16 width 1.5 autotitles columnhead 
set key font ',16' noopaque at graph 2.08, graph 0.985  

set label 1 "1.00" at 0,1.15 center font ",16" tc rgb "#000000" front
set label 12 base_thpt at 1,1.15 center font ",16" tc rgb "#000000" front
set label 13 xsched_thpt at 2,1.15 center font ",16" tc rgb "#000000" front
set label 121 fg_base at 1,0.75 center font ",16" tc rgb "#000000" front
set label 122 bg_base at 1,0.25 center font ",16" tc rgb "#FFFFFF" front
set label 131 fg_xsched at 2,0.60 center font ",16" tc rgb "#000000" front
set label 132 bg_xsched at 2,0.10 center font ",16" tc rgb "#FFFFFF" front

plot thpt_file using (0):($1/$1) t "Standalone" w boxes lc rgb SA_COLOR, \
     ""        using (1):(($2+$3)/$1) t "Native/Fg" w boxes  lc rgb Base_Light_COLOR, \
     ""        using (1):($3/$1) t "Native/Bg" w boxes  lc rgb Base_COLOR, \
     ""        using (2):(($4+$5)/$1) t SYSTEM_NAME."/Fg" w boxes  lc rgb XSched_Light_COLOR, \
     ""        using (2):($5/$1) t SYSTEM_NAME."/Bg" w boxes  lc rgb XSched_COLOR
