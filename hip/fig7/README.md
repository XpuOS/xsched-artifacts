# Experiment for AMD MI50 GPU (Fig. 7)

This experiment demonstrates how XSched can schedule two AMD GPU ResNet-152 inference processes using fixed-priority policy (fp) and bandwidth-partition policy (or called utilization-partition, up, in test).



## Artifact Claims

- XSched can effectively reduce the latency of the foreground process (near to standalone) using fixed-priority policy.

- XSched can effectively partition the throughput of two processes (3:1) using bandwidth-partition policy.



## Environment



### Our Testbed

- CPU: Intel Core i7-10700
- GPU **(Required)**: AMD Instinct MI50
- Memory: 16 GB DDR4
- OS: Ubuntu 18.04 with kernel 6.8.0 and Docker installed
- SDK: ROCm 5.4 (on host OS)



We use docker image "rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1" as software environment.

You can pull this image from docker hub for further experiments.

```bash
docker pull rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1
```




## Run

Just run the script below and it will start a docker container to complete the test.

```bash
./scripts/mi50.sh
```

`mi50.sh` will run "rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1" from docker hub and execute `./scripts/run.sh` in the container. `run.sh` does the following things:

1. Build XSched from source.
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

