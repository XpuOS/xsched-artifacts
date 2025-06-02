import os
import math
import subprocess
from pathlib import Path


def read_cdf(file):
    f = open(file, 'r')
    cdf = f.readlines()
    f.close()
    cdf = [line.strip().split() for line in cdf]
    cdf = [(float(p), int(v)) for p, v in cdf]
    return cdf


def combine_cdf(dev_name: str):
    dev = dev_name.lower()
    raw_dir = (Path(__file__).parent / "../raw").resolve()

    # check all files exist
    sa_file     = raw_dir / f"fp_sa_{dev}.cdf"
    base_file   = raw_dir / f"fp_base_{dev}.cdf"
    xsched_file = raw_dir / f"fp_xsched_{dev}.cdf"
    if not (sa_file.exists() and base_file.exists() and xsched_file.exists()):
        # print(f"Missing cdf files for {dev}")
        return

    print(f"Combine cdf for {dev_name}")
    sa_cdf      = read_cdf(sa_file)
    base_cdf    = read_cdf(base_file)
    xsched_cdf  = read_cdf(xsched_file)

    result = []
    for i in range(len(sa_cdf)):
        assert sa_cdf[i][0] == base_cdf[i][0] == xsched_cdf[i][0]
        result.append((sa_cdf[i][0], sa_cdf[i][1], base_cdf[i][1], xsched_cdf[i][1]))

    processed_dir = (Path(__file__).parent / "../processed").resolve()
    os.makedirs(processed_dir, exist_ok=True)
    f = open(processed_dir / f"fp_{dev}.cdf", 'w')
    f.write("cdf sa base xsched\n")
    for p, sa, base, xsched in result:
        f.write(f"{p} {sa} {base} {xsched}\n")
    f.close()


def plot_fp(dev_name: str):
    dev = dev_name.lower()
    plot_script = (Path(__file__).parent / "fp.plt").resolve()
    cdf_file = (Path(__file__).parent / f"../processed/fp_{dev}.cdf").resolve()
    if not cdf_file.exists():
        return
    print(f"Plotting fp for {dev_name}")

    base_cdf_file = (Path(__file__).parent / f"../raw/fp_base_{dev}.cdf").resolve()
    base_cdf = read_cdf(base_cdf_file)
    x_max_ms = base_cdf[-1][1] / 1000000
    x_step = round(x_max_ms / 3 / 5) * 5
    if x_step == 0:
        x_step = 1
    x_max = x_step * 3
    x_range = int(x_step * 3.75)

    os.makedirs((Path(__file__).parent / "../plot").resolve(), exist_ok=True)
    eps_file = (Path(__file__).parent / f"../plot/fig7_top_{dev}.eps").resolve()
    subprocess.run(["gnuplot",
                    "-e", f"eps_file='{eps_file}'",
                    "-e", f"cdf_file='{cdf_file}'",
                    "-e", f"dev_name='{dev_name}'",
                    "-e", f"x_range={x_range}",
                    "-e", f"x_step={x_step}",
                    "-e", f"x_max={x_max}",
                    str(plot_script)])
    subprocess.run(["epstopdf", str(eps_file)])


def float_str(x: float):
    x = round(x, 2)
    if x < 1:
        return f".{int(x*100)}"
    else:
        return f"{x:.2f}"

def plot_up(dev_name: str):
    dev = dev_name.lower()
    raw_dir = (Path(__file__).parent / "../raw").resolve()
    sa_file     = raw_dir / f"up_sa_{dev}.thpt"
    base_file   = raw_dir / f"up_base_{dev}.thpt"
    xsched_file = raw_dir / f"up_xsched_{dev}.thpt"
    if not (sa_file.exists() and base_file.exists() and xsched_file.exists()):
        return

    print(f"Combine thpt for {dev_name}")
    sa = float(open(sa_file, 'r').readline().strip())
    with open(base_file, 'r') as f:
        base = [float(line.strip().split()[0]) for line in f.readlines()]
        fg_base = base[0]
        bg_base = base[1]
    with open(xsched_file, 'r') as f:
        xsched = [float(line.strip().split()[0]) for line in f.readlines()]
        fg_xsched = xsched[0]
        bg_xsched = xsched[1]

    processed_dir = (Path(__file__).parent / "../processed").resolve()
    os.makedirs(processed_dir, exist_ok=True)
    with open(processed_dir / f"up_{dev}.thpt", 'w') as f:
        f.write("sa base_fg base_bg xsched_fg xsched_bg\n")
        f.write(f"{sa} {fg_base} {bg_base} {fg_xsched} {bg_xsched}\n")

    fg_base = fg_base / sa
    bg_base = bg_base / sa
    fg_xsched = fg_xsched / sa
    bg_xsched = bg_xsched / sa
    base_thpt = fg_base + bg_base
    xsched_thpt = fg_xsched + bg_xsched

    plot_script = (Path(__file__).parent / "up.plt").resolve()
    thpt_file = (Path(__file__).parent / f"../processed/up_{dev}.thpt").resolve()
    if not thpt_file.exists():
        return
    print(f"Plotting up for {dev_name}")

    os.makedirs((Path(__file__).parent / "../plot").resolve(), exist_ok=True)
    eps_file = (Path(__file__).parent / f"../plot/fig7_bottom_{dev}.eps").resolve()
    cmd = ["gnuplot",
           "-e", f"eps_file='{eps_file}'",
           "-e", f"thpt_file='{thpt_file}'",
           "-e", f"bg_base='{float_str(bg_base)}'",
           "-e", f"fg_base='{float_str(fg_base)}'",
           "-e", f"base_thpt='{float_str(base_thpt)}'",
           "-e", f"bg_xsched='{float_str(bg_xsched)}'",
           "-e", f"fg_xsched='{float_str(fg_xsched)}'",
           "-e", f"xsched_thpt='{float_str(xsched_thpt)}'",
           "-e", f"dev_name='{dev_name}'",
           str(plot_script)]
    subprocess.run(cmd)
    subprocess.run(["epstopdf", str(eps_file)])

if __name__ == '__main__':
    dev_names = ["GV100", "K40m", "MI50", "iGPU", "NPU3720", "910b", "DLA", "OFA", "PVA"]
    for dev_name in dev_names:
        combine_cdf(dev_name)
        plot_fp(dev_name)
        plot_up(dev_name)
