"""Design decision of 11-10: the outer worker count follows the machine's available memory (on a shared compute
server a fixed count of fit_library workers can run out of memory when other jobs are running)."""
import memprocs


def test_procs_for_memory_caps_by_available_memory():
    assert memprocs.procs_for_memory(24, avail_gb=110, per_worker_gb=3, reserve_gb=20) == 24
    assert memprocs.procs_for_memory(24, avail_gb=56, per_worker_gb=3, reserve_gb=20) == 12
    assert memprocs.procs_for_memory(24, avail_gb=21, per_worker_gb=3, reserve_gb=20) == 1      # never below 1
    assert memprocs.procs_for_memory(4, avail_gb=500, per_worker_gb=3, reserve_gb=20) == 4       # never above the request


def test_mem_available_gb_reads_meminfo(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:       131072000 kB\nMemFree:         1000000 kB\nMemAvailable:   52428800 kB\n")
    assert memprocs.mem_available_gb(p) == 50.0
    p.write_text("MemTotal:       131072000 kB\n")
    assert memprocs.mem_available_gb(p) is None
