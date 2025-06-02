# Experiment for Intel Arc iGPU (Fig. 7)

This experiment demonstrates how XSched can schedule two iGPU ResNet-152 inference processes using fixed-priority policy (fp) and bandwidth-partition policy (or called utilization-partition, up, in test).



## Artifact Claims

- XSched can effectively reduce the latency of the foreground process (near to standalone) using fixed-priority policy.
- XSched can effectively partition the throughput of two processes (3:1) using bandwidth-partition policy.


## Environment



### Our Testbed

- CPU: Intel Ultra 9 185H SoC
- GPU **(Required)**: Intel Arc integrated GPU (within SoC)
- Memory: 96 GB DDR5
- OS: Ubuntu 24.04 with kernel 6.11 and Docker installed (Ubuntu 22.04 with kernel 6.8 or above is recommended)

We build a docker image "[shenwhang/xsched-ze:0.5](https://hub.docker.com/r/shenwhang/xsched-ze)" to simplify environment setup.

"shenwhang/xsched-ze:0.5" is built on top of ubuntu:24.04 and includes:

- Building toolchain
- Intel GPU levelzero drivers

- intel-oneapi-base-toolkit @ 2025.0
- PyTorch & Intel PyTorch Extension (ipex) @ 2.6.0



## Run

Just run the script below and it will start a docker container to complete the test.

```bash
./scripts/igpu.sh
```

`igpu.sh` will run "shenwhang/xsched-ze:0.5" from docker hub and execute `./scripts/run.sh` in the container. `run.sh` does the following things:

1. Build XSched from source.

2. Test ResNet-152 standalone inference latencies.

3. Start the two processes and test the inference latencies of foreground process.

4. Start the two processes and enable XSched with fixed-priorty policy. Assign the foreground XQueue with higher priority and test the inference latencies of foreground process.

5. Test ResNet-152 standalone inference throughput.

6. Start the two processes and test the throughput of foreground and background processes.

7. Start the two processes and enable XSched with bandwidth-partition policy. Assign the foreground process with 75% utilization and background with 25% utilization. Test the throughput of foreground and background processes.

8. Plot figure 7. Results are under `results/fig7/plot`.

   

Note: We send sigint to force kill the XSched scheduler at the end of the test. This may cause error logs, e.g.,

```
[ERRO @ Txxxxx @ xx:xx:xx.xxxxxx] Assertion failed: cannot send event to server ...
```

which is expected behavior. Please be patient when running tests. ^_^
