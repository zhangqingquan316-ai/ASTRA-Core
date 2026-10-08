"""Run an unchanged training entry point with a CUDA allocator limit."""

import os
from pathlib import Path
import runpy
import sys

import torch


if __name__ == "__main__":
    fraction = float(os.environ.get("ASTRA_GPU_MEMORY_FRACTION", ".48"))
    if not 0 < fraction <= 1:
        raise SystemExit("ASTRA_GPU_MEMORY_FRACTION must be in (0,1].")
    if not torch.cuda.is_available():
        raise SystemExit("The requested CUDA GPU is unavailable.")
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    root = Path(__file__).resolve().parents[1]
    entry = root / "train_astra_core.py"
    sys.path.insert(0, str(root))
    sys.argv[0] = str(entry)
    print(f"CUDA allocator limit: {fraction:.0%} of visible GPU 0.", flush=True)
    runpy.run_path(str(entry), run_name="__main__")
