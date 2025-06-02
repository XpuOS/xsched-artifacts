# Experiment for CUDA GPUs (Fig. 7)

This experiment demonstrates how XSched can schedule two CUDA GPU ResNet-152 inference processes using fixed-priority policy (fp) and bandwidth-partition policy (or called utilization-partition, up, in test).



## Artifact Claims

- XSched can effectively reduce the latency of the foreground process (near to standalone) using fixed-priority policy.

- XSched can effectively partition the throughput of two processes (3:1) using bandwidth-partition policy.



## Environment



### Our Testbed for GV100

- CPU: Intel Core i7-13700

- GPU **(Required)**: NVIDIA Quadro GV100

- Memory: 32 GB DDR4

- OS: Ubuntu 24.04 with kernel 6.11 and Docker installed (Ubuntu 22.04 with kernel 6.8 or above is recommended)

- NVIDIA Driver: 570.124.06 (535+ is recommended)

- Docker: NVIDIA Docker installed (you can install nvidia-container-toolkit by [this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

  

### Our Testbed for K40m

- CPU: Intel(R) Xeon(R) CPU E5-2650 v4

- GPU **(Required)**: NVIDIA Tesla K40m

- Memory: 256 GB DDR4

- OS: Ubuntu 18.04 with kernel 4.15 and Docker installed

- NVIDIA Driver: 470.182.03

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
# for NVIDIA GV100
./scripts/gv100.sh

# for NVIDIA K40m
./scripts/k40m
```

`gv100.sh` and `k40m.sh` will run "shenwhang/xsched-cuda:0.3" from docker hub and execute `./scripts/run.sh` in the container. `run.sh` does the following things:

1. Build XSched and TensorRT inference program from source.
2. Test ResNet-152 standalone inference latencies.
3. Start the two processes and test the inference latencies of foreground process.
4. Start the two processes and enable XSched with fixed-priorty policy. Assign the foreground XQueue with higher priority and test the inference latencies of foreground process.
5. Test ResNet-152 standalone inference throughput.
6. Start the two processes and test the throughput of foreground and background processes.
7. Start the two processes and enable XSched with bandwidth-partition policy. Assign the foreground process with 75% utilization and background with 25% utilization. Test the throughput of foreground and background processes.
8. Results are under `results/fig7/raw`.



## Check Results

Raw results are under `results/fig7/raw`. To plot the figure in the paper, please install gnuplot and epstopdf.

```bash
sudo apt update
sudo apt install gnuplot texlive-font-utils

# process data and plot
cd results/fig7
python3 scripts/plot.py
```

Figure 7 are under `results/fig7/plot`.

