import time
import torch
import torchvision
import intel_extension_for_pytorch as ipex
import argparse
import posix_ipc
import random
import datetime
from datetime import timedelta

WARMUP_CNT = 100
RUN_CNT    = 200

def infer(model, input):
    with torch.no_grad():
        return model(input).cpu()

def sleep_until(end_time):
    """Sleep until the specified datetime."""
    now = datetime.datetime.now()
    if end_time > now:
        time.sleep((end_time - now).total_seconds())

def run(thpt, cdf_file):
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
    model.eval().to("xpu")
    model = ipex.optimize(model)
    input = torch.ones(1, 3, 224, 224).to("xpu")

    latencies = []
    interval_ms = round(1000.0 / thpt)
    variance_ms = int(interval_ms / 2)

    sem = posix_ipc.Semaphore("/xsched_sem", flags=posix_ipc.O_CREAT, initial_value=0)
    
    print("fg warm up")
    for i in range(WARMUP_CNT):
        infer(model, input)
    
    sem.release()
    while sem.value < 2:
        pass

    print("fg start inference")
    for i in range(RUN_CNT):
        next_time = datetime.datetime.now() + timedelta(milliseconds=interval_ms - variance_ms + random.randint(0, 2 * variance_ms))
        start = time.time()
        infer(model, input)
        end = time.time()
        latencies.append(int((end - start) * 1000 * 1000 * 1000)) # ns
        sleep_until(next_time)
    sem.release()

    latencies.sort()
    avg_us = round(sum(latencies) / len(latencies) / 1000, 2)
    min_us = round(min(latencies) / 1000, 2)
    p50_us = round(latencies[len(latencies) // 2] / 1000, 2)
    p90_us = round(latencies[int(len(latencies) * 0.9)] / 1000, 2)
    p99_us = round(latencies[int(len(latencies) * 0.99)] / 1000, 2)
    print(f"fg avg {avg_us} us, min {min_us} us, p50 {p50_us} us, p90 {p90_us} us, p99 {p99_us} us")

    print("fg saving cdf")
    with open(cdf_file, "w") as f:
        cnt = len(latencies)
        for i in range(cnt):
            f.write(f"{i / cnt} {latencies[i]}\n")

    try:
        sem.close()
        sem.unlink()
    except posix_ipc.ExistentialError:
        pass

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="ResNet152 fixed priority foreground inference on Ascend NPU")
    argparse.add_argument("-t", "--thpt", type=float, required=True, help="Target thpt")
    argparse.add_argument("-f", "--cdf-file", type=str, required=True, help="Output cdf file")
    args = argparse.parse_args()
    run(args.thpt, args.cdf_file)
