"""Worker count from available memory (design decision of 2026-11-10: on a shared compute server a fixed count of
fit_library workers can run out of memory when other jobs are running).

procs_for_memory(): min(requested, floor((available - reserve) / per worker)), at least 1.
Usage (run_stage.sh): python -I memprocs.py REQUESTED -> prints the count; per worker GB and reserve GB from the
environment (OUTER_GB_PER_WORKER, default 3; MEM_RESERVE_GB, default 20). Without /proc/meminfo: the request.
"""
import math
import os
import sys
from pathlib import Path


def procs_for_memory(requested: int, avail_gb: float, per_worker_gb: float = 3.0, reserve_gb: float = 20.0) -> int:
    fit = math.floor((avail_gb - reserve_gb) / per_worker_gb) if per_worker_gb > 0 else requested
    return max(1, min(int(requested), int(fit)))


def mem_available_gb(path=Path("/proc/meminfo")):
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


if __name__ == "__main__":
    req = int(sys.argv[1])
    avail = mem_available_gb()
    if avail is None:
        print(req)
    else:
        print(procs_for_memory(req, avail, float(os.environ.get("OUTER_GB_PER_WORKER", "3")),
                               float(os.environ.get("MEM_RESERVE_GB", "20"))))
