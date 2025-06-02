# Experiment for Intel NPU3720 (Fig. 7)

This experiment demonstrates how XSched can schedule two NPU ResNet-152 inference processes using fixed-priority policy (fp) and bandwidth-partition policy (or called utilization-partition, up, in test).



## Artifact Claims

- XSched can effectively reduce the latency of the foreground process (near to standalone) using fixed-priority policy.
- XSched can effectively partition the throughput of two processes (3:1) using bandwidth-partition policy.


## Environment

Please refer to `npu3720/README.md` to setup your environment.



## Run

Just run the script below and it will start a docker container to complete the test.

```bash
./scripts/npu3720.sh
```

`npu3720.sh` will run "shenwhang/xsched-ze:0.5" from docker hub and execute `./scripts/run.sh` in the container. `run.sh` does the following things:

1. Build XSched and OpenVINO inference program from source.

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
