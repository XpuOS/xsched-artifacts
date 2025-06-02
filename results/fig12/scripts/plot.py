import os
import math
import json
import subprocess
from pathlib import Path


def get_seconds(time_str):
    t = time_str.split(":")
    hours = int(t[0])
    minutes = int(t[1])
    seconds = float(t[2])
    return hours * 3600 + minutes * 60 + seconds


def load_raw(path, key):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{') and line.endswith('},'):
                line = line[:-1]
                point = json.loads(line)
                data.append([get_seconds(point["time"]), point[key]])
    return data


def pop_until(data: list, timestamp: float):
    while len(data) > 0 and data[0][0] < timestamp:
        data.pop(0)


def align(testcase):
    raw_dir = (Path(__file__).parent / "../raw").resolve()
    lfbw_path    = raw_dir / f'lfbw_{testcase}.json'
    whisper_path = raw_dir / f'whisper_{testcase}.json'

    lfbw_data = load_raw(lfbw_path, "frame_time (ms)")
    whisper_data = load_raw(whisper_path, "latency (ms)")

    first_lfbw = lfbw_data[0][0]
    first_whisper = whisper_data[0][0]

    pop_until(lfbw_data, first_whisper)
    pop_until(whisper_data, first_lfbw)

    whisper_data.pop(0)
    whisper_data.pop(0)
    first_whisper = whisper_data[0][0]
    pop_until(lfbw_data, first_whisper)

    for i in range(len(lfbw_data)):
        lfbw_data[i][0] = lfbw_data[i][0] - first_whisper
    return lfbw_data


def save(data: list, seconds, testcase):
    processed_dir = (Path(__file__).parent / "../processed").resolve()
    os.makedirs(processed_dir, exist_ok=True)
    path = processed_dir / f'{testcase}.dat'
    with open(path, 'w') as f:
        f.write("time frame_time\n")
        for i in range(len(data)):
            end = data[i][0]
            if end > seconds:
                break
            duration = data[i][1]
            start = end - duration / 1000
            f.write(f"{start:.6f} {duration:.3f}\n")
            f.write(f"{end:.6f} {duration:.3f}\n")


def process(testcase):
    aligned_lfbw = align(testcase)
    save(aligned_lfbw, 40, testcase)


def plot():
    plot_dir = (Path(__file__).parent / "../plot").resolve()
    processed_dir = (Path(__file__).parent / "../processed").resolve()
    os.makedirs(plot_dir, exist_ok=True)

    base_dat = processed_dir / f'base.dat'
    xsched_dat = processed_dir / f'xsched.dat'
    eps_file = plot_dir / f"fig12.eps"
    plot_script = Path(__file__).parent / "lfbw.plt"

    subprocess.run(["gnuplot",
                    "-e", f"eps_file='{eps_file}'",
                    "-e", f"base_dat='{base_dat}'",
                    "-e", f"xsched_dat='{xsched_dat}'",
                    str(plot_script)])
    subprocess.run(["epstopdf", str(eps_file)])

if __name__ == "__main__":
    process("base")
    process("xsched")
    plot()
