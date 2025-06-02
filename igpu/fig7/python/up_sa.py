import time
import torch
import torchvision
import intel_extension_for_pytorch as ipex
import argparse

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

    print("sa warm up")
    for i in range(WARMUP_CNT):
        infer(model, input)
    
    print("sa start inference")
    start_time = time.time()
    for i in range(RUN_CNT):
        infer(model, input)
    end_time = time.time()

    thpt = RUN_CNT / (end_time - start_time)
    thpt = round(thpt, 2)
    print(f"sa thpt: {thpt} reqs/s")
    with open(thpt_file, "w") as f:
        f.write(f"{thpt}")

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="ResNet152 inference on Ascend NPU")
    argparse.add_argument("-t", "--thpt-file", type=str, required=True, help="Output thpt file")
    args = argparse.parse_args()
    run(args.thpt_file)
