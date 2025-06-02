conda create -n cuda python=3.9
conda activate cuda
conda install cmake
conda install -c conda-forge gcc=9.4.0 gxx=9.4.0 gdb
conda install -c conda-forge cudatoolkit-dev=11.3
