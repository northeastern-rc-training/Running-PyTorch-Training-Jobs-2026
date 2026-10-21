# Download CIFAR-10 dataset before training.

import argparse
import sys
import time
from pathlib import Path

import torchvision


def main():
    p = argparse.ArgumentParser(description="Download CIFAR-10")
    p.add_argument("--data-dir", default="./data",
                   help="where to store CIFAR-10 (use the same path for train.py)")
    args = p.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    print(f"Downloading CIFAR-10 to {data_dir}", flush=True)

    start = time.perf_counter()
    try:
        torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
    except Exception as e:
        sys.exit(f"ERROR: download failed: {e}\n"
                 "  Check that this node can reach the internet,\n"
                 "  then run the same command again. An incomplete download starts over.")

    print(f"CIFAR-10 is ready in {data_dir} "
          f"({time.perf_counter() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
