# Experiment for VPI ASICs (Fig. 7)

This experiment demonstrates how XSched can schedule two ASIC processes using fixed-priority policy (fp) and bandwidth-partition policy (or called utilization-partition, up, in test).



## Artifact Claims

- XSched can effectively reduce the latency of the foreground process (near to standalone) using fixed-priority policy.

- XSched can effectively partition the throughput of two processes (3:1) using bandwidth-partition policy.



## Environment



### Our Testbed

- CPU: NVIDIA Jetson AGX Orin SoC
- OFA: NVIDIA Jetson Orin Optical Flow Accelerator (within the SoC)
- PVA: NVIDIA Jetson Orin Programmable Vision Accelerator (within the SoC)
- Memory: 32 GB
- OS: Ubuntu 20.04.6 with kernel 5.10.216-tegra
- SDK: Jetpack 5.1.4 [L4T 35.6.0], TensorRT 8.5.2.2, nvvpi 2.4.8

Please set your Jetson Orin to [MAX-N](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#power-mode-controls) performance mode.


## Run

Just run the scripts below to complete the test.

```bash
# the arm cpu on orin is so wimpy, set it to its max frequency
sudo ./scripts/set_freq.sh -m

# for ofa
./scripts/ofa.sh

# for pva
./scripts/pva.sh
```

`ofa.sh` and `pva.sh`  will execute `./scripts/run.sh` and does the following things:

1. Build XSched and vpi program from source.
2. Test standalone task latencies.
3. Start the two processes and test the task latencies of foreground process.
4. Start the two processes and enable XSched with fixed-priorty policy. Assign the foreground XQueue with higher priority and test the task latencies of foreground process.
5. Test standalone task throughput.
6. Start the two processes and test the throughput of foreground and background processes.
7. Start the two processes and enable XSched with bandwidth-partition policy. Assign the foreground process with 75% utilization and background with 25% utilization. Test the throughput of foreground and background processes.
8. Results are under `results/fig7/raw`.



Note: XSched's dependency libcpp-ipc is unstable on arm platforms and may cause the process to crash, please test for several times until you see `fp_xsched_ofa.cdf`, `fp_xsched_pva.cdf`, `up_xsched_ofa.thpt` and `up_xsched_pva.thpt` under `results/fig7/plot`.

Thanks for being patient when running tests. ^_^



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

