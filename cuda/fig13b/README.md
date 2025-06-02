# Experiment for Comparing XSched with Paella on NVIDIA GV100 (Figure 13b)

We provide a [pre-built docker image](https://hub.docker.com/r/shenwhang/paella) for this experiment (only tested on NVIDIA GV100). 

If it does not work on your GPU, you can build Paella from source (see https://github.com/eniac/paella)


### Run the experiment

```bash
# mount the artifacts directory (replace `xsched-artifacts` with your own location)
docker run --privileged -it --rm --name xsched-paella --gpus all --net=host -v xsched-artifacts:/xsched-artifacts shenwhang/paella:cuda-11.7.0 bash
```

Build XSched
```bash
cd /xsched-artifacts/sys/xsched
make cuda
```

Build experiment code
```bash
cd /xsched-artifacts/cuda/fig13b
mkdir build
cd build
cmake .. -DXSched_DIR=/xsched-artifacts/sys/xsched/output/lib/cmake/XSched/
make
```

Run the experiment

```bash
bash /xsched-artifacts/cuda/fig13b/scripts/run_native.sh # run native CUDA-MS
bash /xsched-artifacts/cuda/fig13b/scripts/run_paella.sh # run Paella
bash /xsched-artifacts/cuda/fig13b/scripts/run_xsched.sh # run XSched
```

The results are stored in the `/xsched-artifacts/results/fig13b/raw` directory. You can exit the container and run the plotting script to plot the results.

```bash
# install gnuplot & epstopdf
sudo apt install gunplot texlive-font-utils

python3 xsched-artifacts/results/fig13b/scripts/plot.py
```
