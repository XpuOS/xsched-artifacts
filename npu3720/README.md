# Environment Setup for Intel NPU3720

## Our Testbed

- CPU: Intel Ultra 9 185H SoC

- NPU **(Required)**: Intel integrated NPU3720 (within SoC)

- Memory: 96 GB DDR5

- OS: Ubuntu 24.04 with kernel 6.11 and Docker installed (Ubuntu 22.04 with kernel 6.8 or above is recommended)

## Firmware & Driver

- Step 1: Please follow the [official guide](https://github.com/intel/linux-npu-driver/releases/tag/v1.5.1) from Intel (or the guide below) to install NPU Driver == 1.5.1 on your **host**.

```bash
# this is a copy of the official guide

# install level-zero driver == 1.17.6
sudo dpkg --purge --force-remove-reinstreq level-zero level-zero-devel
wget https://github.com/oneapi-src/level-zero/releases/download/v1.17.6/level-zero_1.17.6+u22.04_amd64.deb
wget https://github.com/oneapi-src/level-zero/releases/download/v1.17.6/level-zero-devel_1.17.6+u22.04_amd64.deb
sudo dpkg -i level-zero*.deb

# install npu level-zero driver == 1.5.1
sudo dpkg --purge --force-remove-reinstreq intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu
sudo apt update
sudo apt install libtbb12

# for Ubuntu 22.04
wget https://github.com/intel/linux-npu-driver/releases/download/v1.5.1/intel-driver-compiler-npu_1.5.1.20240708-9842236399_ubuntu22.04_amd64.deb
wget https://github.com/intel/linux-npu-driver/releases/download/v1.5.1/intel-fw-npu_1.5.1.20240708-9842236399_ubuntu22.04_amd64.deb
wget https://github.com/intel/linux-npu-driver/releases/download/v1.5.1/intel-level-zero-npu_1.5.1.20240708-9842236399_ubuntu22.04_amd64.deb

# for Ubuntu 24.04
wget https://github.com/intel/linux-npu-driver/releases/download/v1.5.1/intel-driver-compiler-npu_1.5.1.20240708-9842236399_ubuntu24.04_amd64.deb
wget https://github.com/intel/linux-npu-driver/releases/download/v1.5.1/intel-fw-npu_1.5.1.20240708-9842236399_ubuntu24.04_amd64.deb
wget https://github.com/intel/linux-npu-driver/releases/download/v1.5.1/intel-level-zero-npu_1.5.1.20240708-9842236399_ubuntu24.04_amd64.deb

# install the downloaded driver
sudo dpkg -i intel-*.deb

# reboot and set the render group for accel device
reboot

sudo chown root:render /dev/accel/accel0
sudo chmod g+rw /dev/accel/accel0
# add user to the render group
sudo usermod -a -G render <user-name>
# user needs to restart the session to use the new group (log out and log in)

sudo bash -c "echo 'SUBSYSTEM==\"accel\", KERNEL==\"accel*\", GROUP=\"render\", MODE=\"0660\"' > /etc/udev/rules.d/10-intel-vpu.rules"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=accel
```



- Step 2: You need to install the modified kernel mode driver of the NPU (kmd/intel_vpu.ko). We added scheduling support for level-2 preemption (in kmd/ivpu_sched.c) to the kernel mode driver from the linux 6.11 source tree.

```bash
cd npu3720/kmd
# build the driver
./driver.sh build
# install and replace the kmd in your host
sudo ./driver.sh install
```

Make sure you see the following output in `sudo dmesg`

```text
[drm] Firmware: intel/vpu/vpu_37xx_v0.0.bin, version: 20240611*MTL_CLIENT_SILICON-release*0003*ci_tag_ud202424_vpu_rc_20240611_0003*f3e8a8f2747
[drm] vpu driver with scheduling support initialized
```



## Docker Image

We build a docker image "[shenwhang/xsched-ze:0.5](https://hub.docker.com/r/shenwhang/xsched-ze)" to simplify environment setup.

"shenwhang/xsched-ze:0.5" is built on top of ubuntu:24.04 and includes:

- Building toolchain
- Intel NPU levelzero drivers
- intel-oneapi-base-toolkit == 2025.0
- OpenVINO == 2024.4.0

You can pull this image from docker hub for further experiments.

```bash
docker pull shenwhang/xsched-ze:0.5
```



## Run

After these setup, you can go to subdirectories (fig7 & fig12) for testing.

