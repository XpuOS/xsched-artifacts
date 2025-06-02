import torch
import torchvision
import intel_extension_for_pytorch as ipex
import posix_ipc
import argparse

WARMUP_CNT = 100
RUN_CNT    = 200

def infer(model, input):
    with torch.no_grad():
        return model(input).cpu()

def run():
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

    print("bg start inference")
    infer_cnt = 0
    while sem.value == 2:
        infer(model, input)
        infer_cnt += 1

    print("bg done")
    try:
        sem.close()
        sem.unlink()
    except posix_ipc.ExistentialError:
        pass

if __name__ == "__main__":
    run()
