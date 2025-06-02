import os
import subprocess
from pathlib import Path


def plot():
    raw_dir = (Path(__file__).parent / "../raw").resolve()
    plot_dir = (Path(__file__).parent / "../plot").resolve()
    os.makedirs(plot_dir, exist_ok=True)

    level1_path = raw_dir / "level1.dat"
    level2_path = raw_dir / "level2.dat"
    level3_path = raw_dir / "level3.dat"
    eps_file = plot_dir / "fig9.eps"
    plot_script = Path(__file__).parent / "levels.plt"
    subprocess.run(["gnuplot",
                    "-e", f"eps_file='{eps_file}'",
                    "-e", f"level1_dat='{level1_path}'",
                    "-e", f"level2_dat='{level2_path}'",
                    "-e", f"level3_dat='{level3_path}'",
                    str(plot_script)])
    subprocess.run(["epstopdf", str(eps_file)])


if __name__ == "__main__":
    plot()
