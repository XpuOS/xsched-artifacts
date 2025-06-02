import os
import sys
import subprocess
from pathlib import Path

def parse_csv(raw_file: Path):
    with open(raw_file, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        items = line.split(",")
        data.append(items)
    return data


def process_data(raw_file: Path, processed_file: Path):
    data = parse_csv(raw_file)
    # content: (iat, num_jobs, elapsed_time, avg, p50, p90, p95, p99, max)
    tpt = [float(line[1]) * 1000000.0 / float(line[2]) for line in data] # num_jobs * 1000000  / elapsed_time
    p99 = [float(line[7]) for line in data]

    # save to processed_file
    with open(processed_file, "w") as f:
        for i in range(len(tpt)):
            f.write(f"{tpt[i]} {p99[i]} \n")
    return

def preprocess_data():
    raw_dir = Path(__file__).parent.parent / "raw"
    processed_dir = Path(__file__).parent.parent / "processed"
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    file_map = {
        "cuda_ms_lns2.txt": "cuda_ms.txt",
        "paella_full_lns2.txt": "paella.txt",
        "xsched_lns2.txt": "xsched.txt",
    }

    for raw_file, processed_file in file_map.items():
        if os.path.exists(raw_dir / raw_file):
            process_data(raw_dir / raw_file, processed_dir / processed_file)
        else:
            print(f"File {raw_file} does not exist")

def plot_data():
    processed_dir = Path(__file__).parent.parent / "processed"
    plot_dir = Path(__file__).parent.parent / "plot"

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    gnuplot_script = Path(__file__).parent / "paella.plt"

    output_eps = plot_dir / "fig13b.eps"
    output_pdf = plot_dir / "fig13b.pdf"

    subprocess.run(["gnuplot", 
                    "-e", f"eps_file='{output_eps}'", 
                    "-e", f"cuda_ms_file='{processed_dir / 'cuda_ms.txt'}'",
                    "-e", f"paella_file='{processed_dir / 'paella.txt'}'",
                    "-e", f"xsched_file='{processed_dir / 'xsched.txt'}'",
                    gnuplot_script])
    
    subprocess.run(["epstopdf", output_eps, "--outfile", output_pdf])

    print(f"Plot saved to {output_eps} and {output_pdf}")

if __name__ == "__main__":
    preprocess_data()

    plot_data()