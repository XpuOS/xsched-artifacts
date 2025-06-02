import torch
import torchvision.models as models
import time
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='ResNet Inference Benchmark')
    parser.add_argument('--freq', type=int, default=10, help='Inference frequency (inferences per second)')
    parser.add_argument('--time', type=int, default=10, help='Total execution time in seconds')
    parser.add_argument('--latency-output', type=str, default=None, help='Output file for latency measurements')
    parser.add_argument('--latency-cnt', type=int, default=None, help='Number of latency measurements')
    parser.add_argument('--tpt-output', type=str, default=None, help='Output file for throughput measurements')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup time in seconds')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create model
    model = models.resnet152(pretrained=True).cuda()
    model.eval()
    
    # Create input tensor
    input_tensor = torch.randn(8, 3, 224, 224).cuda()
    
    # Create CUDA stream
    stream = torch.cuda.Stream()
    
    # Warmup for specified time
    print(f"Warming up for {args.warmup} seconds...")
    warmup_start = time.time()
    warmup_end = warmup_start + args.warmup
    warmup_count = 0
    
    with torch.no_grad():
        while time.time() < warmup_end:
            with torch.cuda.stream(stream):
                output = model(input_tensor)
                stream.synchronize()
            warmup_count += 1
    
    print(f"Completed {warmup_count} warmup inferences in {args.warmup} seconds")
    
    # Calculate sleep time between inferences
    sleep_time = 1.0 / args.freq if args.freq > 0 else 0
    
    # Prepare for benchmark
    latencies = []
    throughputs = []
    total_inferences = 0
    start_time = time.time()
    end_time = start_time + args.time
    
    print(f"Starting benchmark for {args.time} seconds with {args.freq} inferences per second")
    
    # Main benchmark loop
    output_list = []
    current_time = start_time
    while current_time < end_time:
        inference_start = time.time()
        
        with torch.no_grad():
            with torch.cuda.stream(stream):
                output = model(input_tensor)
                output_list.append(output[0][0])
                stream.synchronize()
        
        inference_end = time.time()
        latency = (inference_end - inference_start) * 1000  # Convert to ms
        latencies.append(latency)
        
        total_inferences += 1
        
        # Calculate throughput every second
        if int(current_time) < int(inference_end):
            throughput = total_inferences / (inference_end - start_time)
            throughputs.append((inference_end - start_time, throughput))
            print(f"Time: {int(inference_end - start_time)}s, Throughput: {throughput:.2f} inferences/sec")
        
        # Sleep to maintain frequency
        time_to_sleep = max(0, sleep_time - (inference_end - inference_start))
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        
        current_time = time.time()
    
    # Calculate final statistics
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    final_throughput = total_inferences / (current_time - start_time)
    
    print(f"\nBenchmark completed:")
    print(f"Total inferences: {total_inferences}")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"P95 latency: {p95_latency:.2f} ms")
    print(f"P99 latency: {p99_latency:.2f} ms")
    print(f"Overall throughput: {final_throughput:.2f} inferences/sec")

    # Save latency data
    if args.latency_output:
        if args.latency_cnt:
            if len(latencies) < args.latency_cnt:
                print(f"Warning: Number of latencies ({len(latencies)}) is less than the requested count ({args.latency_cnt})")
            latencies = latencies[:args.latency_cnt] # truncate before sorting
        with open(args.latency_output, 'w') as f:
            # Convert latencies from ms to seconds and sort them
            cnt = len(latencies)
            sorted_latencies_seconds = sorted([int(latency * 1000000) for latency in latencies])
            # Write one latency value per line
            for i, latency in enumerate(sorted_latencies_seconds):
                f.write(f"{float(i)/cnt:.4f} {latency}\n")
        print(f"Latency data saved to {args.latency_output}")

    # Save throughput data
    if args.tpt_output:
        with open(args.tpt_output, 'w') as f:
            # Write only the final total throughput
            f.write(f"{final_throughput:.4f}\n")
        print(f"Throughput data saved to {args.tpt_output}")

if __name__ == "__main__":
    main() 