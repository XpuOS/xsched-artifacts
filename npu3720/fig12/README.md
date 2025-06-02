# Experiment for Intel NPU3720 (Fig. 12)

This experiment demonstrates how XSched can use a laxity-based policy to schedule two kinds of tasks in an intelligent video conference app on Intel NPU: fake-background tasks and speech-recognition tasks.



## Artifact Claims

- XSched can effectively reduce the P99 frame latency of the fake-background task and avoid frame freeze (high frame latency at about 800ms) compared with the native hardware scheduler.

  


## Environment

Please refer to `npu3720/README.md` to setup your environment.



## Run

Just run the script below and it will start a docker container to complete the test.

```bash
./scripts/npu3720.sh
```

`npu3720.sh` will run "shenwhang/xsched-ze:0.5" from docker hub and execute `./scripts/run.sh` in the container. `run.sh` does the following things:

1. Build XSched from source.

2. Build whisper.cpp from source.

3. Run lfbw (Linux Fake Background Webcam) and whisper together and record lfbw's frame latencies.

4. Run lfbw and whisper with XSched enabled and record lfbw's frame latencies.

5. Plot figure 12. Results are under `results/fig12/plot`.

   