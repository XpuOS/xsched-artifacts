import os
import sys
import math
import json
import subprocess
from pathlib import Path


def count(platform: str):
    config_path = (Path(__file__).parent / f"platforms/{platform}.json").resolve()
    root = (Path(__file__).parent.parent / f"sys/xsched/platforms/{platform}").resolve()
    
    config = json.load(open(config_path))
    for k, v in config.items():
        files = ""
        for file in v:
            files += f"{root / file} "
        print(f"{platform} {k}:")
        subprocess.run(f"cloc {files}", shell=True, check=True, cwd=root)


if __name__ == "__main__":
    assert sys.argv[1] in ["cuda", "hip", "levelzero", "ascend", "cudla", "vpi"]

    print(f"Counting {sys.argv[1]}, will not consider auto-generated files")
    count(sys.argv[1])
