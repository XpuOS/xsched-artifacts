# Artifact Evaluation for XSched [OSDI'25]

This repository contains scripts and instructions for reproducing the primary experiments in our OSDI '25 paper "Preemptive Scheduling for Diverse XPUs using Multi-level Hardware Model".

## Overview

XSched is a preemptive scheduling framework for diverse heterogeneous accelerators (e.g., GPU, NPUs, ASICs, etc.). It is designed to be both efficient in scheduling XPU tasks and transparent to applications. We have evaluated XSched on 9 distinct XPUs, including four GPUs (NVIDIA GV100, K40m, AMD MI50, and Intel Arc iGPU), three NPUs (Intel NPU3720, Ascend 910b and NVIDIA DLA) and two ASICs (NVIDIA PVA and OFA).

This artifact contains the **source code of XSched**, and **the code and scripts to reproduce the primary experiments in the paper**. The included experiments are:

- The effectiveness of XSched's preemptive scheduling (Figure 7)
- The effectiveness of XSched's multi-level hardware model (Figure 9)
- Case study 1 - GPU harvesting on multi-tenant server (Figure 11)
- Case study 2 - Video conferencing on AIPC (Figure 12)
- Case study 3 - Multi-model inference serving (Figure 13)

## Table of Contents

- [Project Structure](#project-structure)
- [Paper's Hardware & Software Configuration](#paper-hardware-software-configuration)
- [Build XSched](#build-xsched)
- [Experiments Overview](#experiments-overview)
- [Plotting the Experiments Results](#plotting-the-experiments-results)

## Project Structure
```
> tree .
├── assets               # Assets for the experiments
├── cuda                 # Experiment codes for NVIDIA GPUs
│   ├── fig7             # Experiment codes to reproduce Figure 7 in the paper
│   ├── fig9
│   └── fig11
├── ascend               # Experiment codes for Ascend 910b
├── dla                  # Experiment codes for NVDLA
├── igpu                 # Experiment codes for Intel Arc iGPU
├── npu3720              # Experiment codes for Intel NPU3720
├── vpi                  # Experiment codes for NVIDIA PVA and OFA
├── results              # Results of the experiments
│   └── fig7             # Results to reproduce Figure 7 in the paper
│       ├── raw          # Raw results
│       └── scripts      # Scripts to process and plot the results
├── sys
│   ├── tgs              # TGS system used for the experiments of Figure 11
│   ├── vcuda            # vCUDA system used for the experiments of Figure 11
│   └── xsched           # Source code of XSched
```

## Paper's Hardware & Software Configuration

The experiments were conducted on **six** different hardware platforms and **nine** different XPUs, as follows:

| Platform            | XPUs | SDK/Driver |
|---------------------|------|----------|
| NVIDIA GV100 Server        | GV100  | CUDA 11.4 |
| NVIDIA K40m Server         | K40m  | CUDA 11.4 |
| AMD MI50 Server        | MI50  | ROCm 5.6  |
| Ascend 910b Server     | 910b  | CANN 8.0.0       |
| Intel Core Ultra 9 185H| Intel Arc iGPU, Intel NPU3720 | LevelZero 1.17, OpenVINO 2024.4, PyTorch-ipex 2.6.0 |
| NVIDIA Jetson Orin 32GB| NVIDIA DLA/PVA/OFA  | JetPack 5.1.4      |

**Important:** XSched can also run on other platforms, but the compatibility is not guaranteed. For example, XSched can also run on other NVIDIA GPUs (e.g., server GPU like A100 and consumer GPU like 3080). However, the experiment codes in the paper are only tested on the platforms listed above.



## Build XSched

The source code of XSched is located in `sys/xsched`. 

XSched has minimal dependencies, and all 3rd-party libraries are included in the `sys/xsched/3rdparty` directory. The only requirements are a C++14 compiler and CMake (3.14 or later). 

For one type of XPU, you can build XSched by running the following command:

```bash
cd sys/xsched
make cuda # Build for NVIDIA GPUs

# Or you can build for other XPUs

make hip # Build for AMD GPUs
make levelzero # Build for Intel Arc iGPU and Intel NPU3720
make ascend # Build for Ascend 910b
make cudla # Build for NVIDIA DLA
make vpi # Build for NVIDIA PVA and OFA
```

It is worth noting that building XSched does not require the XPU SDK (e.g., CUDA, ROCm, etc.) to be installed. So these build commands can be run on any machine without the XPU SDK. However, the experiments must be run on the machines with the corresponding XPU SDK.

These commands will build the `xsched` and install it in the `sys/xsched/output` directory, which contains the following files:
```bash
output/
├── bin                                     
│   ├── xcli                            # XSched Command Line Interface 
│   └── xserver                         # XSched Scheduler Server
├── include                             # XSched header files
└── lib
    ├── cmake                           # CMake files for including XSched in your project
    ├── libcuda.so -> libshimcuda.so    # For Interception of CUDA driver calls
    ├── libcuda.so.1 -> libshimcuda.so
    ├── libhalcuda.so                   # Hardware Abstraction Layer for CUDA
    ├── libpreempt.so                   # XSched preemption library
    └── libshimcuda.so                  # Shim for CUDA
```

## Getting Started on NVIDIA GPU

We provide a simple example to show how to use XSched to schedule tasks on NVIDIA GPUs.

See [`get_started/README.md`](get_started/README.md) for more details.


## Experiments Overview

Because the experiments are conducted on different hardware platforms, we provide the scripts to reproduce the results on each platform.

Here is the overview of the included experiments and the required XPUs.

| XPU | Included Experiments |
|----------|------------|
| NVIDIA GV100 | Figure 7, Figure 9, Figure 11, Figure 13 |
| NVIDIA K40m | Figure 7, Figure 9 |
| AMD MI50 | Figure 7, Figure 11 |
| Ascend 910b | Figure 7 |
| Intel Arc iGPU | Figure 7, Figure 12 |
| Intel NPU3720 | Figure 7, Figure 9, Figure 11, Figure 12 |
| NVIDIA DLA | Figure 7 |
| NVIDIA PVA | Figure 7 |
| NVIDIA OFA | Figure 7 |


### The effectiveness of XSched's preemptive scheduling (Figure 7)

This experiment evaluates the effectiveness of XSched's two scheduling policies (Fixed priority policy and Bandwidth partition policy) to schedule the a set of tasks on different XPUs. 

#### Experiment Setup

**All XPUs are used in this experiment, but you can run partial experiments with the XPUs you have available.**

In each experiment, we run two processes that periodically or continuously submit identical types of tasks to the same XPU. One process is designated as foreground process, while the other is background process. 

#### Evaluated Approaches

For each policy and each XPU, we run three experiments with different approaches:
- *Standalone*: Only the foreground process is executed.
- *Native* (or Base): Two processes use the hardware native scheduler.
- *XSched*: Two processes are scheduled by XSched with the corresponding policy.

#### Expected Results

Figure 7 (top) shows the latency CDF of the foreground process using Fixed priority policy.
Figure 7 (bottom) shows the normalized throughput of the background process using Bandwidth partition policy.

Expected results:
- XSched can effectively reduce the latency of the foreground process (near to standalone) using Fixed priority policy.
- XSched can effectively partition the bandwidth of two processes (3:1) using Bandwidth partition policy.

#### Instructions

| XPU | README |
|-----|------|
| NVIDIA GV100 | [README](cuda/fig7/README.md) |
| NVIDIA K40m | [README](cuda/fig7/README.md) |
| AMD MI50 | [README](hip/fig7/README.md) |
| Ascend 910b | [README](ascend/fig7/README.md) |
| Intel Arc iGPU | [README](igpu/fig7/README.md) |
| Intel NPU3720 | [README](npu3720/fig7/README.md) |
| NVIDIA DLA | [README](dla/fig7/README.md) |
| NVIDIA PVA | [README](vpi/fig7/README.md) |
| NVIDIA OFA | [README](vpi/fig7/README.md) |




### The effectiveness of XSched's multi-level hardware model (Figure 9)

This experiment evaluates the effectiveness of XSched's multi-level hardware model on the scheduling performance.

#### Experiment Setup

**NVIDIA GV100 or V100 is required for this experiment.**

This experiment runs hand-crafted kernels that can execute a given time to simulate the task being preempted. We use XSched to suspend the task and measure the preemption latency.


#### Evaluated Approaches

We compare the P99 preemption latency of three different preemption levels:
- *Level 1*: XSched only uses level-1 api (launch & sync) of the hwQueue to suspend the task.
- *Level 2*: XSched uses level-2 api (deactivate & activate) of the hwQueue to suspend the task.
- *Level 3*: XSched uses level-3 api (interrupt & restore) of the hwQueue to suspend the task.


#### Expected Results

Implementing a higher-level interface (e.g., Level 2 or Level 3) can effectively reduce the preemption latency.

#### Instructions

| XPU | README |
|-----|------|
| NVIDIA GV100 | [README](cuda/fig9/README.md) |



### Case study 1 - GPU harvesting on multi-tenant server (Figure 11)

This experiment is a case study on the scheduling of GPU tasks on multi-tenant server.

#### Experiment Setup

NVIDIA GV100, AMD MI50 are used in this experiment. **NVIDIA GV100 is required for this experiment at least.**

The basic setup of this case study is a multi-tenant server running two containers, one for production job (Pjob) and the other for opportunistic job (Ojob). Pjobs have stringent performance requirements with minimal degradation, while Ojobs should harvest remaining GPU resources on a best-effort basis.

This experiment consists of two workloads:
- Co-training workload: Two containers running the two DL model training jobs.
- Sci-Fin workload: One container (Pjob) running financial algorithm (BlockScholes) and one container (Ojob) running scientific computing (CFD).

#### Evaluated Approaches

For NVIDIA GV100, we run each workload with five approaches:
- *Standalone*: Only the Pjob is executed.
- *Native*: Two containers use the hardware native scheduler.
- *TGS*: Two containers are scheduled by TGS.
- *vCUDA*: Two containers are scheduled by vCUDA.
- *XSched*: Two containers are scheduled by XSched.

For AMD MI50, we run each workload with three approaches:
- *Standalone*: Only the Pjob is executed.
- *Native*: Two containers use the hardware native scheduler.
- *XSched*: Two containers are scheduled by XSched.
- *XSched w/o prog*: Two containers are scheduled by XSched, but without the progressive command launching technique.

#### Expected Results

Using XSched, the Pjob can get a stable performance (near to standalone) and the Ojob can still make progress.

#### Instructions

| XPU | README |
|-----|------|
| NVIDIA GV100 | [README](cuda/fig11/README.md) |
| AMD MI50 | [README](hip/fig11/README.md) |


### Case study 2 - Video conferencing on AIPC (Figure 12)

This experiment is a case study on the scheduling of video conferencing tasks on AIPC.

#### Experiment Setup

**This experiment must be run on Intel Core Ultra 9 185H.**

This experiment runs two processes to simulate the video conferencing scenario:
- **lfbw**: Linux Fake Background Webcam. We modified it to read a mp4 video file as the input video stream instead of a true webcam device. The task runs 25 frames per second.
- **whisper**: A speech-to-text model. We modified it to read a wav audio file as the input audio stream instead of a microphone device. The task runs every three seconds, and outputs the transcribed text to the console.

#### Evaluated Approaches

We compare the performance of two approaches:
- *Native* (or Base): Two processes use the hardware native scheduler.
- *XSched*: Two processes are scheduled by XSched with a laxity-based scheduling policy (when lfbw is about to miss the deadline, its priority will be lifted).

#### Expected Results

XSched can effectively reduce the P99 frame latency of the fake-background task and avoid frame freeze.

#### Instructions

| XPU | README |
|-----|------|
| Intel NPU3720 | [README](npu3720/fig12/README.md) |


### Case study 3a - Multi-model inference serving using Triton Inference Server (Figure 13a)

This experiment demonstrates how XSched can be integrated into Triton Inference Server to enable priority-based scheduling of multi-model inference tasks.

#### Experiment Setup

**This experiment must be run on NVIDIA GV100.**

This experiment uses two Triton clients to send inference requests of two Bert-large models to a Triton server.
The high-priority client sends requests with a frequency of 10 req/s, while the low-priority client sends requests continuously.

#### Evaluated Approaches

We compare the performance of four approaches:
- *Standalone*: Only the high-priority client is executed.
- *Triton*: using vanilla Triton Inference Server.
- *Triton+Priority*: using Triton Inference Server with its priority-based scheduling feature.
- *XSched*: using Triton Inference Server integrated with XSched.

#### Expected Results

By integrating XSched into Triton Inference Server, the high-priority client can get a stable performance (near to standalone).

#### Instructions

| XPU | README |
|-----|------|
| NVIDIA GV100 | [README](cuda/fig13a/README.md) |




### Case study 3b - Multi-model inference serving using Paella (Figure 13b)

This experiment compares the performance of XSched with Paella (SOSP'23, a state-of-the-art inference serving system) on NVIDIA GV100.

#### Experiment Setup

**This experiment must be run on NVIDIA GV100.**

This experiment uses the same setup and workloads as in Paella's paper (i.e., concurrent serving of multiple DNN models).

#### Evaluated Approaches

We compare the performance of four approaches:
- *Native*: the CUDA-MS basline used in Paella's paper (i.e., directly use multiple CUDA streams to serve multiple requests concurrently).
- *Paella*: the Paella system.
- *XSched*: Integrating XSched into the CUDA-MS baseline.

#### Expected Results

With XSched, the CUDA-MS baseline can get a performance comparable to Paella.


#### Instructions

| XPU | README |
|-----|------|
| NVIDIA GV100 | [README](cuda/fig13b/README.md) |



## Plotting the Experiments Results

See [README](results/README.md) for more details.