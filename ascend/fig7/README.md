# Experiment for Ascend NPU (Fig. 7)

This experiment demonstrates how XSched can schedule two Ascend NPU ResNet-152 inference processes using fixed-priority policy (fp) and bandwidth-partition policy (or called utilization-partition, up, in test).



## Artifact Claims

- XSched can effectively reduce the latency of the foreground process (near to standalone) using fixed-priority policy.

- XSched can effectively partition the throughput of two processes (3:1) using bandwidth-partition policy.



## Environment



### Our Testbed

- CPU: Kunpeng-920
- NPU **(Required)**: Ascend 910b3
- Memory: 1.536 TB
- OS: openEuler 24.03 with kernel 4.19
- SDK: CANN 8.0.0




## Run

Just run the script below to complete the test.

```bash
./scripts/910b.sh
```

`910b.sh` will execute `./scripts/run.sh` and does the following things:

1. Build XSched and ascendcl inference program from source.
2. Test ResNet-152 standalone inference latencies.
3. Start the two processes and test the inference latencies of foreground process.
4. Start the two processes and enable XSched with fixed-priorty policy. Assign the foreground XQueue with higher priority and test the inference latencies of foreground process.
5. Test ResNet-152 standalone inference throughput.
6. Start the two processes and test the throughput of foreground and background processes.
7. Start the two processes and enable XSched with bandwidth-partition policy. Assign the foreground process with 75% utilization and background with 25% utilization. Test the throughput of foreground and background processes.
8. Results are under `results/fig7/raw`.



Note: The ResNet-152 model is sliced to 8 slices, and converted to ascend model at `assets/models/ascend`. If you are using another CANN version or running on another npu soc, e.g., 910b1, please convert it using the following script:

```bash
# replace Ascend910B3 to your NPU SoC version
./scripts/convert.sh Ascend910B3
```



## Check Results

Raw results are under `results/fig7/raw`. To plot the figure in the paper, please install gnuplot and epstopdf.

```bash
# process data and plot
cd results/fig7
python3 scripts/plot.py
```

Figure 7 are under `results/fig7/plot`.

