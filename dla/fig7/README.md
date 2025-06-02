# Experiment for DLA (Fig. 7)

This experiment demonstrates how XSched can schedule two DLA ResNet-152 inference processes using fixed-priority policy (fp) and bandwidth-partition policy (or called utilization-partition, up, in test).



## Artifact Claims

- XSched can effectively reduce the latency of the foreground process (near to standalone) using fixed-priority policy.

- XSched can effectively partition the throughput of two processes (3:1) using bandwidth-partition policy.



## Environment



### Our Testbed

- CPU: NVIDIA Jetson AGX Orin SoC
- DLA: NVIDIA Jetson Orin Deep Learning Accelerator (within the SoC)
- Memory: 32 GB
- OS: Ubuntu 20.04.6 with kernel 5.10.216-tegra
- SDK: Jetpack 5.1.4 [L4T 35.6.0], TensorRT 8.5.2.2

Please set your Jetson Orin to [MAX-N](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#power-mode-controls) performance mode.


## Run

Just run the script below to complete the test.

```bash
./scripts/dla.sh
```

`dla.sh` will execute `./scripts/run.sh` and does the following things:

1. Build XSched and cudla inference program from source.
2. Test ResNet-152 standalone inference latencies.
3. Start the two processes and test the inference latencies of foreground process.
4. Start the two processes and enable XSched with fixed-priorty policy. Assign the foreground XQueue with higher priority and test the inference latencies of foreground process.
5. Test ResNet-152 standalone inference throughput.
6. Start the two processes and test the throughput of foreground and background processes.
7. Start the two processes and enable XSched with bandwidth-partition policy. Assign the foreground process with 75% utilization and background with 25% utilization. Test the throughput of foreground and background processes.
8. Results are under `results/fig7/raw`.



Note1: The ResNet-152 model is sliced to 8 slices, and converted to FP16 TensorRT engines at `assets/models/dla` using trtexec. If you are using another TensorRT version, please convert it using

```bash
# replace 0.onnx and 0.engine to the corresponding slice
/usr/src/tensorrt/bin/trtexec --onnx=assets/models/ascend/0.onnx --useDLACore=0 --saveEngine=assets/models/dla/0.engine --fp16 --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 --safe
```



Note2: We send sigint to force kill the XSched scheduler at the end of the test. This may cause error logs, e.g.,

```
[ERRO @ Txxxxx @ xx:xx:xx.xxxxxx] Assertion failed: cannot send event to server ...
```

which is expected behavior. Please be patient when running tests. ^_^



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

