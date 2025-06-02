# Experiment for CUDA GPU Containers (Fig. 11)

This experiment demonstrates how XSched can schedule two CUDA GPU container jobs using fixed-priority policy (fp). We assign Pjob (production job) with high priority and Ojob (opportunistic job) with low priority to ensure the the performance of Pjob, while havesting remaining GPU resources with Ojob.



## Artifact Claims

- Using XSched, the Pjob can get a stable performance (near to standalone) and the Ojob can still make progress.



## Environment



### Our Testbed

- CPU: Intel Core 13700

- GPU **(Required)**: NVIDIA Quadro GV100

- Memory: 32 GB DDR4

- OS: Ubuntu 24.04 with kernel 6.11 and Docker installed (Ubuntu 22.04 with kernel 6.8 or above is recommended)

- NVIDIA Driver: 570.124.06 (535+ is recommended)

- Docker: NVIDIA Docker installed (you can install nvidia-container-toolkit by [this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))




We build a docker image "[shenwhang/xsched-cuda:0.3](https://hub.docker.com/r/shenwhang/xsched-cuda)" to simplify environment setup.

"shenwhang/xsched-cuda:0.3" is built on top of `nvcr.io/nvidia/tensorrt` image and includes:

- Building toolchain

- CUDA Runtime == 11.3.0

- TensorRT == 7.2.3.4

- PyTorch == 1.9.0

  

You can pull this image from docker hub for further experiments.

```bash
docker pull shenwhang/xsched-cuda:0.3
```




## Run

Run the script below and it will complete each test case

```bash
./scripts/run.sh
```



## Check Results

Raw results are under `results/fig11/raw`. To plot the figure in the paper, please install gnuplot and epstopdf.

```bash
sudo apt update
sudo apt install gnuplot texlive-font-utils

# process data and plot
cd results/fig11
python3 scripts/plot.py
```

Figure 11 are under `results/fig11/plot`.

