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


def pop_until(data: list, timestamp: float):
    while len(data) > 0 and data[0][0] < timestamp:
        data.pop(0)


def pop_from(data: list, timestamp: float):
    while len(data) > 0 and data[-1][0] > timestamp:
        data.pop()


def load_cotrain(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{') and line.endswith('},'):
                line = line[:-1]
                point = json.loads(line)
                if not "batch" in point:
                    continue
                batch = point["batch"]
                batch_size = int(batch.split('/')[-1].strip())
                batch = int(batch.split('/')[0].strip())
                data.append([get_seconds(point["time"]), point["epoch"], batch, batch_size])
    return data


def process_cotrain_sa(dev: str):
    raw_dir = (Path(__file__).parent / "../raw").resolve()
    sa_path = raw_dir / f'cotrain_sa_{dev}.json'
    sa_data = load_cotrain(sa_path)
    if len(sa_data) == 0:
        return [None, None]
    start_time = sa_data[0][0]
    start_time += 10
    stop_time = start_time + 250
    if sa_data[-1][0] < stop_time:
        print(f"Warning: sa cotrain data on {dev} is not enough, lack of {stop_time - sa_data[-1][0]:.2f} seconds")
    pop_until(sa_data, start_time)
    pop_from(sa_data, stop_time)
    thpt = len(sa_data) / (sa_data[-1][0] - start_time)
    return [thpt, thpt]


def process_cotrain(system: str, dev: str):
    raw_dir = (Path(__file__).parent / "../raw").resolve()
    ojob_path = raw_dir / f'cotrain_{system}_ojob_{dev}.json'
    pjob_path = raw_dir / f'cotrain_{system}_pjob_{dev}.json'
    ojob_data = load_cotrain(ojob_path)
    pjob_data = load_cotrain(pjob_path)
    if len(ojob_data) == 0 or len(pjob_data) == 0:
        return [None, None]

    first_ojob = ojob_data[0][0]
    first_pjob = pjob_data[0][0]

    pop_until(ojob_data, first_pjob)
    pop_until(pjob_data, first_ojob)

    start_time = ojob_data[0][0]
    start_time += 10
    stop_time = start_time + 250
    if ojob_data[-1][0] < stop_time:
        print(f"Warning: ojob cotrain data for {system} on {dev} is not enough, lack of {stop_time - ojob_data[-1][0]:.2f} seconds")
    if pjob_data[-1][0] < stop_time:
        print(f"Warning: pjob cotrain data for {system} on {dev} is not enough, lack of {stop_time - pjob_data[-1][0]:.2f} seconds")

    pop_until(ojob_data, start_time)
    pop_until(pjob_data, start_time)
    pop_from(ojob_data, stop_time)
    pop_from(pjob_data, stop_time)

    thpt_pjob = len(pjob_data) / (pjob_data[-1][0] - start_time)
    thpt_ojob = len(ojob_data) / (ojob_data[-1][0] - start_time)
    return [thpt_pjob, thpt_ojob]



def save_gv100_data(data: list, testcase: str):
    for d in data:
        if d[0] is None or d[1] is None:
            print(f"Warning: {testcase} data for gv100 is insufficient")
            return

    processed_dir = (Path(__file__).parent / "../processed").resolve()
    os.makedirs(processed_dir, exist_ok=True)
    path = processed_dir / f'{testcase}_gv100.dat'
    with open(path, 'w') as f:
        f.write("type          standalone native vcuda tgs xsched\n")
        f.write("Production    " + " ".join([f"{d[0]:.3f}" for d in data]) + "\n")
        f.write("Opportunistic " + " ".join([f"{d[1]:.3f}" for d in data]) + "\n")


def save_mi50_data(data: list, testcase: str):
    for d in data:
        if d[0] is None or d[1] is None:
            print(f"Warning: {testcase} data for mi50 is insufficient")
            return

    processed_dir = (Path(__file__).parent / "../processed").resolve()
    os.makedirs(processed_dir, exist_ok=True)
    path = processed_dir / f'{testcase}_mi50.dat'
    with open(path, 'w') as f:
        f.write("type          standalone native xsched xsched-noopt\n")
        f.write("Production    " + " ".join([f"{d[0]:.3f}" for d in data]) + "\n")
        f.write("Opportunistic " + " ".join([f"{d[1]:.3f}" for d in data]) + "\n")


def load_scifin(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{') and line.endswith('},'):
                line = line[:-1]
                point = json.loads(line)
                data.append([get_seconds(point["time"]), point["latency (ms)"]])
    return data


def process_scifin_sa(dev: str):
    raw_dir = (Path(__file__).parent / "../raw").resolve()
    ojob_path = raw_dir / f'scifin_sa_ojob_{dev}.json'
    pjob_path = raw_dir / f'scifin_sa_pjob_{dev}.json'
    ojob_data = load_scifin(ojob_path)
    pjob_data = load_scifin(pjob_path)
    if len(ojob_data) == 0 or len(pjob_data) == 0:
        return [None, None]

    # for ojob, we prefer throughput
    start_time = ojob_data[0][0]
    start_time += 10
    stop_time = start_time + 60
    if ojob_data[-1][0] < stop_time:
        print(f"Warning: ojob scifin sa data for {dev} is not enough, lack of {stop_time - ojob_data[-1][0]:.2f} seconds")
    pop_until(ojob_data, start_time)
    pop_from(ojob_data, stop_time)
    thpt_ojob = len(ojob_data) / (ojob_data[-1][0] - start_time)

    # for pjob, we prefer latency
    start_time = pjob_data[0][0]
    start_time += 10
    stop_time = start_time + 60
    if pjob_data[-1][0] < stop_time:
        print(f"Warning: pjob scifin sa data for {dev} is not enough, lack of {stop_time - pjob_data[-1][0]:.2f} seconds")
    pop_until(pjob_data, start_time)
    pop_from(pjob_data, stop_time)
    latencies = [d[1] for d in pjob_data]
    latencies.sort()
    p99_pjob = latencies[int(len(latencies) * 0.99)]
    return [100 / p99_pjob, thpt_ojob]


def process_scifin(system: str, dev: str):
    raw_dir = (Path(__file__).parent / "../raw").resolve()
    ojob_path = raw_dir / f'scifin_{system}_ojob_{dev}.json'
    pjob_path = raw_dir / f'scifin_{system}_pjob_{dev}.json'
    ojob_data = load_scifin(ojob_path)
    pjob_data = load_scifin(pjob_path)
    if len(ojob_data) == 0 or len(pjob_data) == 0:
        return [None, None]

    first_ojob = ojob_data[0][0]
    first_pjob = pjob_data[0][0]

    pop_until(ojob_data, first_pjob)
    pop_until(pjob_data, first_ojob)

    start_time = ojob_data[0][0]
    start_time += 10
    stop_time = start_time + 30
    if ojob_data[-1][0] < stop_time:
        print(f"Warning: ojob scifin data for {system} on {dev} is not enough, lack of {stop_time - ojob_data[-1][0]:.2f} seconds")
    if pjob_data[-1][0] < stop_time:
        print(f"Warning: pjob scifin data for {system} on {dev} is not enough, lack of {stop_time - pjob_data[-1][0]:.2f} seconds")

    pop_until(ojob_data, start_time)
    pop_until(pjob_data, start_time)
    pop_from(ojob_data, stop_time)
    pop_from(pjob_data, stop_time)

    thpt_ojob = len(ojob_data) / (ojob_data[-1][0] - start_time)
    latencies = [d[1] for d in pjob_data]
    latencies.sort()
    p99_pjob = latencies[int(len(latencies) * 0.99)]
    return [100 / p99_pjob, thpt_ojob]
    

def plot_gv100():
    plot_dir = (Path(__file__).parent / "../plot").resolve()
    processed_dir = (Path(__file__).parent / "../processed").resolve()
    os.makedirs(plot_dir, exist_ok=True)

    top_dat = processed_dir / f'cotrain_gv100.dat'
    bottom_dat = processed_dir / f'scifin_gv100.dat'
    top_eps = plot_dir / f"fig11_top_gv100.eps"
    bottom_eps = plot_dir / f"fig11_bottom_gv100.eps"
    plot_script = Path(__file__).parent / "gv100.plt"

    if os.path.exists(top_dat):
        subprocess.run(["gnuplot",
                        "-e", f"eps_file='{top_eps}'",
                        "-e", f"gv100_dat='{top_dat}'",
                        str(plot_script)])
        subprocess.run(["epstopdf", str(top_eps)])
    
    if os.path.exists(bottom_dat):
        subprocess.run(["gnuplot",
                        "-e", f"eps_file='{bottom_eps}'",
                        "-e", f"gv100_dat='{bottom_dat}'",
                        str(plot_script)])
        subprocess.run(["epstopdf", str(bottom_eps)])


def plot_mi50():
    plot_dir = (Path(__file__).parent / "../plot").resolve()
    processed_dir = (Path(__file__).parent / "../processed").resolve()
    os.makedirs(plot_dir, exist_ok=True)

    top_dat = processed_dir / f'cotrain_mi50.dat'
    bottom_dat = processed_dir / f'scifin_mi50.dat'
    top_eps = plot_dir / f"fig11_top_mi50.eps"
    bottom_eps = plot_dir / f"fig11_bottom_mi50.eps"
    plot_script = Path(__file__).parent / "mi50.plt"

    if os.path.exists(top_dat):
        subprocess.run(["gnuplot",
                        "-e", f"eps_file='{top_eps}'",
                        "-e", f"mi50_dat='{top_dat}'",
                        str(plot_script)])
        subprocess.run(["epstopdf", str(top_eps)])
    
    if os.path.exists(bottom_dat):
        subprocess.run(["gnuplot",
                        "-e", f"eps_file='{bottom_eps}'",
                        "-e", f"mi50_dat='{bottom_dat}'",
                        str(plot_script)])
        subprocess.run(["epstopdf", str(bottom_eps)])


if __name__ == "__main__":
    cotrain_gv100 = []
    cotrain_gv100.append(process_cotrain_sa("gv100"))
    cotrain_gv100.append(process_cotrain("base", "gv100"))
    cotrain_gv100.append(process_cotrain("vcuda", "gv100"))
    cotrain_gv100.append(process_cotrain("tgs", "gv100"))
    cotrain_gv100.append(process_cotrain("xsched", "gv100"))

    scifin_gv100 = []
    scifin_gv100.append(process_scifin_sa("gv100"))
    scifin_gv100.append(process_scifin("base", "gv100"))
    scifin_gv100.append(process_scifin("vcuda", "gv100"))
    scifin_gv100.append(process_scifin("tgs", "gv100"))
    scifin_gv100.append(process_scifin("xsched", "gv100"))

    cotrain_mi50 = []
    cotrain_mi50.append(process_cotrain_sa("mi50"))
    cotrain_mi50.append(process_cotrain("base", "mi50"))
    cotrain_mi50.append(process_cotrain("xsched", "mi50"))
    cotrain_mi50.append(process_cotrain("xsched_woprog", "mi50"))

    scifin_mi50 = []
    scifin_mi50.append(process_scifin_sa("mi50"))
    scifin_mi50.append(process_scifin("base", "mi50"))
    scifin_mi50.append(process_scifin("xsched", "mi50"))
    scifin_mi50.append(process_scifin("xsched_woprog", "mi50"))

    save_gv100_data(cotrain_gv100, "cotrain")
    save_gv100_data(scifin_gv100, "scifin")
    save_mi50_data(cotrain_mi50, "cotrain")
    save_mi50_data(scifin_mi50, "scifin")

    plot_gv100()
    plot_mi50()
