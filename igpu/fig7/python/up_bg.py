import time
import torch
import torchvision
import intel_extension_for_pytorch as ipex
import argparse
import posix_ipc

WARMUP_CNT = 100
RUN_CNT    = 400

def infer(model, input):
    with torch.no_grad():
        return model(input).cpu()

def run(thpt_file):
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
    model.eval().to("xpu")
    model = ipex.optimize(model)
    input = torch.ones(1, 3, 224, 224).to("xpu")
    sem = posix_ipc.Semaphore("/xsched_sem", flags=posix_ipc.O_CREAT, initial_value=0)

    print("bg warm up")
    for i in range(WARMUP_CNT):
        infer(model, input)
    
    sem.release()
    while sem.value < 2:
        pass

    infer_cnt = 0
    print("bg start inference")
    start_time = time.time()
    while sem.value == 2:
        infer(model, input)
        infer_cnt += 1
    end_time = time.time()

    thpt = infer_cnt / (end_time - start_time)
    thpt = round(thpt, 2)
    print(f"bg thpt: {thpt} reqs/s")

    while sem.value < 4:
        pass
    with open(thpt_file, "a") as f:
        f.write(f"{thpt}\n")

    try:
        sem.close()
        sem.unlink()
    except posix_ipc.ExistentialError:
        pass


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="ResNet152 inference on Ascend NPU")
    argparse.add_argument("-t", "--thpt-file", type=str, required=True, help="Output thpt file")
    args = argparse.parse_args()
    run(args.thpt_file)
