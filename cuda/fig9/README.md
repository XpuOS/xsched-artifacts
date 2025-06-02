# Experiment for Multi-level Model on GV100 (Fig. 9b)

This experiment demonstrates how XSched can leverage the multi-level hardware model to improve the scheduling performance.



## Artifact Claims

With higher preemption level, the preemption latency of XSched is significantly reduced.



## Environment



### Our Testbed

- CPU: Intel Core i7-13700
- GPU **(Required)**: NVIDIA Quadro GV100 (or NVIDIA V100 is acceptable)
- Memory: 32 GB DDR4
- OS: Ubuntu 24.04 with kernel 6.11 and Docker installed (Ubuntu 22.04 with kernel 6.8 or above is recommended)
- NVIDIA Driver: 570.124.06 (535+ is recommended)
- Docker: NVIDIA Docker installed (you can install nvidia-container-toolkit by [this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))



We build a docker image "[shenwhang/xsched-cuda:0.3](https://hub.docker.com/r/shenwhang/xsched-cuda)" to simplify environment setup.

"shenwhang/xsched-cuda:0.3" is built on top of `nvcr.io/nvidia/tensorrt:21.05` image and includes:

- Building toolchain

- CUDA Runtime == 11.3.0

- TensorRT == 7.2.3.4

  

You can pull this image from docker hub for further experiments.

```bash
docker pull shenwhang/xsched-cuda:0.3
```




## Run

Just run the script below and it will start a docker container to complete the test.

```bash
./gv100.sh
```

`gv100.sh` will run "shenwhang/xsched-cuda:0.3" from docker hub and execute `./scripts/run.sh` in the container. `run.sh` does the following things:

1. Build XSched and preemption test program from source.
2. Test the preemption latency of level1 under different command execution times.
3. Test the preemption latency of level2 under different command execution times.
4. Test the preemption latency of level3 under different command execution times.
5. Results are under `results/fig9/raw`.



## Check Results

Raw results are under `results/fig9/raw`. To plot the figure in the paper, please install gnuplot and epstopdf.

```bash
sudo apt update
sudo apt install gnuplot texlive-font-utils

# process data and plot
cd results/fig9
python3 scripts/plot.py
```

Figure 9 (b) are under `results/fig9/plot`.

