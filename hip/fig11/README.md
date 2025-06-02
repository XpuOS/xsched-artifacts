# Experiment for AMD GPU Containers (Fig. 11)

This experiment demonstrates how XSched can schedule two AMD GPU container jobs using fixed-priority policy (fp). We assign Pjob (production job) with high priority and Ojob (opportunistic job) with low priority to ensure the the performance of Pjob, while havesting remaining GPU resources with Ojob.



## Artifact Claims

- Using XSched, the Pjob can get a stable performance (near to standalone) and the Ojob can still make progress.



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

