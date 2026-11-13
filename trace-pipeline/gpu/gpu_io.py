"""Load a gpu_filter output (PREFIX.dram / PREFIX.dramlaunch), plain or gzip-compressed (PREFIX.dram.gz)."""
import os, subprocess
import numpy as np
def load(prefix, ext, dtype):
    p = f"{prefix}.{ext}"
    if os.path.exists(p):
        return np.fromfile(p, dtype=dtype)
    out = subprocess.run(["gzip", "-dc", p + ".gz"], stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(out, dtype=dtype)
